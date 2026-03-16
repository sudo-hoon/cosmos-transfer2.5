# 02. 입력 데이터 전처리

이 문서는 `generate_img2world()` 호출 시 모든 입력 데이터가 어떻게 전처리되는지를 상세하게 설명한다.

---

## 1. 입력 영상 읽기와 리사이즈

> 파일: `transfer2/inference/utils.py` → `read_and_process_video()`

### 1.1 영상 읽기

```python
input_frames, fps, aspect_ratio, original_hw = read_and_process_video(
    video_path, resolution="720", max_frames=max_frames
)
# input_frames: (C, T, H, W), dtype=uint8(torch), range [0, 255]
```

**처리 흐름:**

```
input_video.mp4 (또는 .png, .jpg)
    │
    ├─ mediapy/OpenCV로 프레임 읽기
    │   └─ 이미지: 1 프레임, fps=1
    │   └─ 영상: max_frames까지 읽기 (기본 5000)
    │
    ├─ 원본 해상도 기록: original_hw = (H_orig, W_orig)
    │
    ├─ Aspect Ratio 감지 (가장 가까운 표준 비율 선택)
    │   ├─ 16:9, 4:3, 1:1, 3:4, 9:16 중 하나
    │   └─ 감지 방법: 현재 w/h와 각 비율의 차이가 최소인 것
    │
    └─ 타겟 해상도로 리사이즈 (cv2.INTER_AREA)
```

### 1.2 해상도 매핑 테이블

`VIDEO_RES_SIZE_INFO`에서 resolution과 aspect_ratio 조합으로 정확한 (W, H)를 결정한다:

**resolution="720"** (기본값):

| Aspect Ratio | 출력 크기 (W x H) | 총 픽셀 |
|-------------|-------------------|---------|
| 16:9 | 1280 x 704 | 901,120 |
| 4:3 | 960 x 704 | 675,840 |
| 1:1 | 960 x 960 | 921,600 |
| 3:4 | 704 x 960 | 675,840 |
| 9:16 | 704 x 1280 | 901,120 |

**resolution="480"**:

| Aspect Ratio | 출력 크기 (W x H) |
|-------------|-------------------|
| 16:9 | 768 x 432 |
| 4:3 | 640 x 480 |
| 1:1 | 480 x 480 |

> **중요**: 높이와 너비는 반드시 특정 배수(spatial_compression_factor=16의 배수 관련)에 맞아야 한다. 이 테이블의 값들은 모두 이 조건을 만족한다.

### 1.3 리사이즈 방법

```python
# cv2.INTER_AREA: 영역 기반 보간 (다운스케일링에 적합)
resized_video[i] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
```

---

## 2. 텍스트 프롬프트 처리

> 파일: `transfer2/inference/utils.py` → `get_t5_from_prompt()`

### 2.1 텍스트 인코더

Cosmos Transfer 2.5는 **Qwen2.5-VL 7B (Reason1.1)** 텍스트 인코더를 사용한다.

- 28개 hidden layer의 출력을 FULL_CONCAT하여 100,352 dim 임베딩 생성
- DiT 내부에서 `crossattn_proj` (Linear 100,352→1,024)로 프로젝션
- 추론 시에는 `compute_text_embeddings_online()`으로 on-the-fly 인코딩

> **참고**: 코드에서 텍스트 임베딩 키 이름이 `t5_text_embeddings`로 되어 있지만, 이는 레거시 네이밍이다. 실제로는 Qwen2.5-VL 7B 인코더가 사용된다.

### 2.2 텍스트 임베딩 계산

```python
# 추론 시 텍스트 임베딩 계산 — Qwen2.5-VL 7B (reason1p1_7B) 온라인 인코딩
text_embeddings = model.text_encoder.compute_text_embeddings_online(
    {"ai_caption": [prompt], "images": None}, input_caption_key="ai_caption"
)
# shape: (1, seq_len, 100352), bfloat16
# 이후 DiT 내부에서 crossattn_proj(100352 → 1024)로 프로젝션
```

프롬프트는 여러 형태로 전달 가능하다:
- `str`: 단일 프롬프트 → 임베딩 계산
- `torch.Tensor`: 사전 계산된 임베딩 직접 사용
- `list[str]`: 청크별 프롬프트 → 각각 독립 계산
- `dict[int, str]`: 프레임 인덱스별 프롬프트

### 2.3 네거티브 프롬프트

일반 모델(non-distilled)에서는 CFG(Classifier-Free Guidance)를 위해 네거티브 프롬프트 임베딩도 계산한다.

```python
DEFAULT_NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics..."
```

이 임베딩은 `data_batch["neg_t5_text_embeddings"]`로 전달되어 CFG의 unconditioned branch에 사용된다 (키 이름은 레거시).

### 2.4 청크별 프롬프트

긴 영상의 경우 각 청크에 다른 프롬프트를 적용할 수 있다:

```python
# 청크별 프롬프트 리스트
prompts = ["Scene 1 description", "Scene 2 description", "Scene 3 description"]

# 또는 프레임 인덱스별 프롬프트 딕셔너리
prompts = {0: "Scene 1", 93: "Scene 2", 186: "Scene 3"}
```

청크가 프롬프트 리스트보다 많으면 **마지막 프롬프트가 반복** 사용된다:
```python
text_emb_idx = min(chunk_id, len(text_embeddings) - 1)
```

---

## 3. 이미지 컨텍스트 처리

> 파일: `transfer2/inference/utils.py` → `read_and_process_image_context()`

이미지 컨텍스트는 **스타일 레퍼런스**로 사용되며, SigLIP2 비전 인코더를 통해 임베딩된다.

### 3.1 이미지 컨텍스트 소스 우선순위

```
1. context_frame_index가 지정됨
   → 입력 영상의 해당 프레임을 이미지 컨텍스트로 사용

2. image_context_path가 지정됨
   → 별도 이미지 파일을 로드

3. 둘 다 None
   → image_context = None (이미지 컨텍스트 없이 생성)
```

### 3.2 이미지 전처리

```python
img = read_video_or_image_into_frames_BCTHW(
    img_context_path,
    H=resolution[1],  # 타겟 높이
    W=resolution[0],  # 타겟 너비
    normalize=True,    # [0,255] → [-1,1]
)[:, :, t_idx]  # 특정 프레임 선택 → (B, C, H, W)
```

### 3.3 SigLIP2 인코딩

SigLIP2 비전 인코더(`google/siglip2-so400m-patch16-naflex`)가 이미지를 토큰으로 변환한다:

```python
# 입력: (B, C, H, W), range [-1, 1]
# SigLIP2 내부에서 [0, 255]로 역변환하여 처리
inputs = processor(images=(1.0 + image_tensor) / 2.0 * 255.0, ...)
outputs = model(**inputs)
latents = outputs.last_hidden_state  # (B, 256, 1152)
```

| 항목 | 값 |
|------|-----|
| 출력 토큰 수 | 256 |
| 토큰 차원 | 1152 |
| 모델 | `google/siglip2-so400m-patch16-naflex` |

이 임베딩은 DiT의 cross-attention에 텍스트 임베딩과 함께 주입된다.

---

## 4. 컨트롤 입력 처리

> 파일: `transfer2/inference/utils.py` → `read_and_process_control_input()`
> 파일: `transfer2/datasets/augmentors/control_input.py` → `get_augmentor_for_eval()`

### 4.1 컨트롤 입력 처리 두 가지 경로

```
컨트롤 입력
    │
    ├─ 경로 A: 사전 계산 (control_path 제공)
    │   └─ 파일에서 직접 로드 & 리사이즈
    │
    └─ 경로 B: 온더플라이 계산 (control_path = None)
        ├─ depth: VideoDepthAnything 모델로 계산
        ├─ seg: GroundDINO + SAM2로 계산
        ├─ edge: Canny Edge로 계산 (augmentor에서)
        └─ vis: Bilateral Blur로 계산 (augmentor에서)
```

### 4.2 `read_and_process_control_input()` 상세

이 함수는 **depth**와 **seg** 컨트롤의 사전 계산/온더플라이 계산을 처리한다.

**Depth 온더플라이 계산:**
```python
video_np = ...  # (T, H, W, C), uint8
model = VideoDepthAnythingModel(device="cuda")
depth_maps = model.generate(video_np)
# 정규화: per-video min-max → [0, 255]
depth_normalized = (depth - d_min) / (d_max - d_min) * 255.0
# 출력: (1, T, H, W) → 3채널로 확장
```

**Segmentation 온더플라이 계산:**
```python
# SAM2 기반 비디오 세그멘테이션
seg_model = VideoSegmentationModel()
# seg_control_prompt 사용 (없으면 프롬프트 첫 128 단어)
masks = seg_model.segment(video_np, prompt=seg_control_prompt)
```

**사전 계산된 컨트롤 로드:**
```python
control_input, _, _, _ = read_and_process_video(control_path, resolution=resolution)
# 입력 영상과 동일한 리사이즈 적용
```

### 4.3 `get_augmentor_for_eval()`: Edge/Vis 온더플라이

`read_and_process_control_input()`이 반환한 후, **edge**와 **vis**는 augmentor에서 추가 처리된다:

```python
data_batch = get_augmentor_for_eval(
    data_dict=data_batch,
    input_keys=["input_video"],     # 원본 영상 키
    output_keys=hint_key,           # ["edge", "vis"] 등
    preset_edge_threshold="medium",
    preset_blur_strength="medium",
)
```

**Augmentor 처리 단계:**
1. `data_batch["input_video"]`에서 원본 영상 추출 (uint8, [0,255])
2. 해당 컨트롤 타입의 어그멘터 생성 (`use_random=False` → 결정적)
3. 컨트롤 맵 생성
4. `data_batch["control_input_edge"]` 등에 저장 (shape: `(1, 3, T, H, W)`)
5. 마스크가 없으면 전체 1 마스크 생성: `(1, 1, T, H, W)` boolean

### 4.4 Edge 프리셋 설정

| 프리셋 | Canny Low Threshold | Canny High Threshold |
|--------|--------------------|--------------------|
| very_low | 20 | 50 |
| low | 50 | 100 |
| **medium** (기본) | **100** | **200** |
| high | 200 | 300 |
| very_high | 300 | 400 |

- 낮은 threshold → 더 많은 엣지 검출 (노이즈 포함 가능)
- 높은 threshold → 더 적은 엣지 검출 (강한 경계만)

### 4.5 Blur(Vis) 프리셋 설정

| 프리셋 | Blur Factor |
|--------|-------------|
| none | 1 (원본) |
| low | 4 |
| **medium** (기본) | **2** |
| high | 1 |
| very_high | 4 |

### 4.6 컨트롤 마스크

컨트롤 마스크는 **컨트롤이 적용되는 공간-시간 영역**을 지정한다:

```
마스크 소스:
├─ mask_path: 사전 계산된 바이너리 영상 (흰색=적용, 검은색=무시)
├─ mask_prompt: SAM2에 전달할 텍스트 (예: "car building tree")
└─ None: 전체 영역에 컨트롤 적용 (기본)
```

마스크는 latent 공간에서 **control weight map**으로 변환되어 컨트롤 주입 강도를 공간적으로 조절한다.

---

## 5. 입력 정규화 정리

모든 입력은 최종적으로 모델에 전달되기 전에 `_normalize_video_databatch_inplace()`를 통해 정규화된다:

| 데이터 | 입력 형식 | 정규화 후 | 공식 |
|--------|----------|----------|------|
| 영상 (`video`/`images`) | uint8 [0, 255] | bfloat16 [-1, 1] | `x / 127.5 - 1.0` |
| 컨트롤 입력 (`control_input_*`) | uint8 [0, 255] | bfloat16 [-1, 1] | `x / 127.5 - 1.0` |
| 컨트롤 마스크 (bool) | bool | bfloat16 [0, 1] | `x.to(float)` |
| 텍스트 임베딩 | bfloat16 | bfloat16 (그대로) | - |
| 이미지 컨텍스트 | bfloat16 [-1, 1] | bfloat16 [-1, 1] (그대로) | - |

---

## 6. 데이터 배치 구성

> 파일: `inference_pipeline.py` → `_get_data_batch_input()`

최종적으로 모델에 전달되는 `data_batch` 딕셔너리의 구조:

```python
data_batch = {
    # 기본 입력
    "video": prev_output,                    # (1, 3, T, H, W), uint8 → 이전 청크 출력
    "input_video": cur_input_frames,         # (C, T, H, W), uint8 → 현재 청크 원본 영상
    "t5_text_embeddings": text_embedding,    # (1, seq_len, D), bfloat16 (키 이름은 레거시, 실제는 Qwen2.5-VL)
    "fps": torch.randint(16, 32, (1,)),      # (1,), 랜덤 FPS
    "padding_mask": torch.zeros(1, 1, H, W), # (1, 1, H, W)
    "num_conditional_frames": 0 or 1,        # 조건 프레임 수 (latent 공간 기준)
    "control_weight": [1.0],                 # 컨트롤 가중치 리스트

    # 선택적 입력
    "image_context": image_context,                 # (1, C, H, W), bfloat16
    "neg_t5_text_embeddings": neg_text_embeddings,  # (1, seq_len, D), bfloat16 (키 이름은 레거시)

    # 컨트롤 입력 (augmentor 적용 후)
    "control_input_edge": ...,               # (1, 3, T, H, W), uint8
    "control_input_edge_mask": ...,          # (1, 1, T, H, W), bool
    "control_input_depth": ...,              # ...
    # ... 등

    # 가이디드 생성 (선택적)
    "x0_spatial_condition": {
        "x0": encoded_video,                 # latent 공간 원본
        "x_sigma_mask": weight_map,          # latent 공간 마스크
        "step_threshold": 25,                # 적용 스텝 임계값
    },
}
```

> **`video` vs `input_video` 차이점:**
> - `video` (또는 `images`): 이전 청크의 **출력** 영상. 첫 청크에서는 **제로 텐서**. 모델이 조건 프레임으로 사용.
> - `input_video`: **원본 입력** 영상의 현재 청크. 온더플라이 컨트롤 생성에 사용.
