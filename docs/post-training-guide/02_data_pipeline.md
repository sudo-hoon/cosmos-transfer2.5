# 02. 데이터 로딩 및 전처리 파이프라인

이 문서는 학습 중 데이터가 디스크에서 모델 입력까지 어떻게 흘러가는지를 코드 레벨에서 상세하게 설명한다.

---

## 1. 전체 데이터 파이프라인

```
MP4 파일 (디스크)
    │
    ▼ decord.VideoReader
프레임 배열: (T, H, W, C) uint8    ← 랜덤 시작점에서 93 프레임
    │
    ▼ numpy → torch, permute
텐서: (C=3, T=93, H, W) uint8
    │
    ▼ 캡션 JSON 로드
캡션: "A street scene with..."
    │
    ▼ 컨트롤 입력 로드 (depth/seg) 또는 None
컨트롤: (C, T, H, W) 또는 None
    │
    ▼ Augmentor Chain (순차 적용)
    │  ├── ResizeLargestSideAspectPreserving
    │  ├── ReflectionPadding
    │  └── AddControlInput[Edge|Blur|Depth|Seg]
    │
    ▼ 최종 data_dict
{
  "video": (3, 93, 704, 1280) uint8,
  "control_input_edge": (3, 93, 704, 1280) uint8,
  "padding_mask": (1, 704, 1280) float,
  "t2w_qwen2p5_7b": "캡션 텍스트",
  ...
}
```

---

## 2. 프레임 샘플링 상세

> 파일: `singleview_dataset.py` → `_load_video()`

### 2.1 학습 시: 랜덤 시작점

```python
# 영상에서 93 프레임을 랜덤 위치에서 추출
total_frames = len(video_reader)      # 예: 240 프레임
required_frames = 93

# 랜덤 시작점 선택
max_start = total_frames - required_frames  # 240 - 93 = 147
start_idx = random.randint(0, max_start)    # 예: 52

# 93 프레임 추출
frame_indices = list(range(start_idx, start_idx + required_frames))
frames = video_reader.get_batch(frame_indices).asnumpy()  # (93, H, W, 3)
```

**왜 랜덤인가?**
- 같은 영상이라도 매 에폭마다 다른 구간을 사용 → 데이터 증강 효과
- 영상 전체를 골고루 학습에 활용

### 2.2 추론 시: 처음부터

```python
start_idx = 0  # 항상 0번 프레임부터
```

### 2.3 프레임 수 부족 시

93 프레임 미만인 영상은 `_load_video()`에서 실패하며, 데이터셋은 다른 영상으로 대체를 시도한다 (최대 10회 재시도).

---

## 3. 어그멘테이션 체인

> 파일: `augmentor_provider.py` → `get_video_augmentor_v2_with_control()`

### 3.1 Step 1: ResizeLargestSideAspectPreserving

**목적**: Aspect ratio를 유지하면서 타겟 해상도에 맞게 리사이즈

```python
# 타겟: (704, 1280) (720p, 16:9)
# 원본: (1080, 1920)

scaling_ratio = min(1280 / 1920, 704 / 1080)
# = min(0.667, 0.652) = 0.652

resized_h = int(0.652 * 1080) = 704
resized_w = int(0.652 * 1920) = 1252
# → (704, 1252)로 리사이즈
```

- 보간법: BICUBIC (with anti-aliasing)
- uint8 유지

### 3.2 Step 2: ReflectionPadding

**목적**: 리사이즈된 영상을 타겟 크기(704, 1280)에 맞게 중앙 정렬 + 반사 패딩

```python
# 리사이즈 후: (704, 1252)
# 타겟: (704, 1280)

padding_left = int((1280 - 1252) / 2) = 14
padding_right = 1280 - 1252 - 14 = 14

# 좌우에 14px씩 반사 패딩
# → (704, 1280) 달성
```

**부가 출력:**
- `padding_mask`: (1, 704, 1280) - 패딩된 영역 = 0, 실제 컨텐츠 = 1
- `image_size`: [H_resized, W_resized, H_original, W_original]

### 3.3 Step 3: ControlInput 어그멘터

컨트롤 타입에 따라 다른 어그멘터가 적용된다:

#### 3.3.1 Edge (Canny Edge Detection)

```python
class AddControlInputEdge:
    def __call__(self, data_dict):
        video = data_dict["video"]  # (3, T, H, W), uint8

        for t in range(T):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 학습 시: 랜덤 threshold (데이터 증강)
            t_lower = np.random.randint(20, 100)      # 예: 67
            t_diff  = np.random.randint(50, 150)       # 예: 112
            t_upper = t_lower + t_diff                  # 67 + 112 = 179

            edges = cv2.Canny(frame_gray, t_lower, t_upper)  # (H, W), 0 or 255

        # 1채널 → 3채널 확장
        control_input = edges.unsqueeze(0).expand(3, -1, -1)  # (3, T, H, W)
        data_dict["control_input_edge"] = control_input
```

**학습에서의 이점**: 매 이터레이션마다 다른 threshold로 edge를 추출하므로, 모델이 다양한 수준의 edge 정보에서 학습할 수 있다.

#### 3.3.2 Vis (Blur)

```python
class AddControlInputBlur:
    def __call__(self, data_dict):
        video = data_dict["video"]  # (3, T, H, W), uint8

        # 학습 시: 랜덤 블러 파라미터
        # 1. 다운스케일 → 블러 → 업스케일 (구조 정보 보존, 디테일 제거)
        scale_factor = random.choice([4, 8, 12, 16])
        downscaled = resize(video, H//scale_factor, W//scale_factor)
        blurred = apply_blur(downscaled)
        control_input = resize(blurred, H, W)  # 원래 크기로 복원

        data_dict["control_input_vis"] = control_input
```

#### 3.3.3 Depth (사전 계산)

```python
class AddControlInputDepth:
    def __call__(self, data_dict):
        depth = data_dict["depth"]  # (C, T, H_orig, W_orig), 사전 로드됨
        # 비디오 크기에 맞게 리사이즈
        control_input = F.interpolate(depth, size=(H, W), mode="bilinear")
        data_dict["control_input_depth"] = control_input
```

#### 3.3.4 Segmentation (사전 계산, 복잡한 처리)

```python
class AddControlInputSeg:
    def __call__(self, data_dict):
        seg_video = data_dict["segmentation"]  # 컬러 코딩된 세그맵

        # 1. 컬러 양자화 (bin_size=25)
        quantized = (seg_video / 25).int() * 25

        # 2. 첫 프레임에서 고유 색상 추출
        unique_colors = find_unique_colors(quantized[:, 0])

        # 3. 색상별 마스크 추출
        masks = []
        for color in unique_colors:
            distance = color_distance(quantized, color)
            mask = (distance < tolerance)  # tolerance=30
            if mask.sum() > min_mask_size:
                masks.append(mask)

        # 4. 마스크를 랜덤 색상으로 재착색
        colored_mask = colorize_masks(masks)  # (3, T, H, W), uint8

        data_dict["control_input_seg"] = colored_mask
```

### 3.4 마스크 자동 생성

컨트롤 마스크가 없으면 자동으로 **전체 1 마스크**가 생성된다:

```python
if f"control_input_{key}_mask" not in data_dict:
    data_dict[f"control_input_{key}_mask"] = torch.ones(1, 1, T, H, W, dtype=torch.bool)
```

---

## 4. 텍스트 인코딩 (학습 중 온라인)

> 파일: `predict2/text_encoders/text_encoder.py` → `compute_text_embeddings_online()`

### 4.1 인코딩 과정

```
캡션 텍스트 ("A bustling city...")
    │
    ▼ Qwen2.5-VL 7B Tokenizer
    │  - Chat template 적용 (system + user role)
    │  - 토큰화 → input_ids
    │  - 패딩/잘라내기 (NUM_EMBEDDING_PADDING_TOKENS까지)
    │
    ▼ Qwen2.5-VL 7B Model (frozen, no_grad)
    │  - Forward pass → hidden_states (28 layers)
    │
    ▼ Layer-wise Mean Normalization
    │  - 각 레이어별 mean normalize
    │
    ▼ FULL_CONCAT Strategy
    │  - 28 layers × 3584 dim = 100,352 dim
    │
    ▼ text_embedding: (1, seq_len, 100352), bfloat16
    │
    ▼ DiT crossattn_proj
    │  - Linear(100352, 1024) → (1, seq_len, 1024)
```

### 4.2 학습 시 텍스트 드롭아웃

CFG(Classifier-Free Guidance) 학습을 위해, **텍스트 조건을 확률적으로 드롭**한다:

```python
conditioner=dict(
    text=dict(
        dropout_rate=0.2,      # 20% 확률로 텍스트 드롭
        use_empty_string=False,
    ),
)
```

- 80%: 실제 캡션 임베딩 사용
- 20%: 텍스트 임베딩을 null/zero로 대체
- 이를 통해 모델이 텍스트 없이도 합리적인 영상을 생성할 수 있게 됨
- 추론 시 CFG의 unconditioned branch에 해당

---

## 5. 모델 입력으로의 최종 변환

### 5.1 `_normalize_video_databatch_inplace()`

배치가 모델에 들어가면 가장 먼저 이 함수가 호출된다:

```python
def _normalize_video_databatch_inplace(self, data_batch):
    # 1. 영상 정규화
    # video: uint8 [0, 255] → bfloat16 [-1, 1]
    data_batch["video"] = data_batch["video"].to(bfloat16) / 127.5 - 1.0

    # 2. 컨트롤 입력 정규화
    for key in data_batch:
        if key.startswith("control_input_"):
            if data_batch[key].dtype == torch.uint8:
                data_batch[key] = data_batch[key].to(bfloat16) / 127.5 - 1.0
            elif data_batch[key].dtype == torch.bool:
                data_batch[key] = data_batch[key].to(bfloat16)

    # 3. 시간 길이 검증
    assert data_batch["video"].shape[2] == expected_T  # state_t에 맞는지
```

### 5.2 VAE 인코딩

정규화된 영상과 컨트롤이 latent 공간으로 인코딩된다:

```python
# 영상 → latent
x0 = self.encode(normalized_video)  # (1, 3, 93, 704, 1280) → (1, 16, 24, 44, 80)

# 컨트롤 → control latent
control_latent = self.encode(normalized_control)  # 동일한 크기 변환
```

---

## 6. 데이터로더 설정

> 파일: `configs/vid2vid_transfer/defaults/dataloader_local.py`

```python
# Edge 컨트롤 데이터로더 예시
example_singleview_train_data_edge = dict(
    dataloader_train=dict(
        dataset=dict(
            _target_="SingleViewTransferDataset",
            dataset_dir="datasets/your_dataset",
            num_frames=93,
            video_size=(704, 1280),
            resolution="720",
            caption_type="t2w_qwen2p5_7b",
            control_input_type="edge",      # ← edge 온더플라이
        ),
        sampler=dict(
            _target_="InfiniteSampler",
            dataset="${dataloader_train.dataset}",
        ),
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    ),
)
```

### 주요 설정값

| 설정 | 값 | 설명 |
|------|-----|------|
| `batch_size` | 1 | GPU당 1개 영상 (gradient accumulation으로 유효 배치 확대) |
| `num_workers` | 4 | 데이터 로딩 워커 수 |
| `pin_memory` | True | GPU 전송 최적화 |
| `num_frames` | 93 | 추출할 프레임 수 |
| `video_size` | (704, 1280) | 리사이즈 타겟 |
| `caption_type` | `"t2w_qwen2p5_7b"` | Qwen2.5-VL 7B용 캡션 키 |
| `control_input_type` | 컨트롤 의존 | `"edge"`, `"depth"`, `"segcolor"`, `"vis"` |

---

## 7. 에러 처리와 복원력

### 7.1 잘못된 비디오 건너뛰기

```python
# singleview_dataset.py 내부
def __getitem__(self, index):
    for retry in range(10):  # 최대 10회 재시도
        try:
            video_path = self.video_list[index]
            frames = self._load_video(video_path)
            ...
            return data_dict
        except Exception as e:
            self._mark_bad_video(index)
            index = random.randint(0, len(self) - 1)  # 다른 영상으로 대체
    raise RuntimeError("Too many bad videos")
```

### 7.2 어그멘터 스킵

로컬 데이터셋에서는 S3/WebDataset 관련 어그멘터가 자동으로 스킵된다:

```python
SKIP_AUGMENTORS = ["video_parsing", "depth_parsing", "seg_parsing", "merge_datadict", "text_transform"]
```

이 어그멘터들은 NVIDIA 내부 S3 데이터 파이프라인 전용이므로 로컬 학습에서는 불필요하다.
