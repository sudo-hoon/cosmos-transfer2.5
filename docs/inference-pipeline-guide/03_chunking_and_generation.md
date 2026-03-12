# 03. 청크 분할과 Autoregressive 생성

이 문서는 긴 영상을 청크 단위로 분할하고, 각 청크를 생성한 뒤 합치는 전체 과정을 상세히 설명한다.

---

## 1. 왜 청크가 필요한가

Cosmos Transfer 2.5 모델은 한 번에 처리할 수 있는 프레임 수가 제한된다:

- **기본 청크 크기**: 93 프레임 (`num_video_frames_per_chunk`)
- **Latent 공간에서**: 24 토큰 (`state_t = (93 - 1) // 4 + 1 = 24`)
- 93 프레임보다 긴 영상은 **여러 청크로 분할**하여 autoregressive하게 생성

---

## 2. 청크 계산

> 파일: `inference_pipeline.py` → `_get_num_chunks()`

```python
def _get_num_chunks(self, input_frames, num_video_frames_per_chunk, num_conditional_frames):
    num_total_frames = input_frames.shape[1]                    # 전체 프레임 수
    num_frames_per_chunk = num_video_frames_per_chunk - num_conditional_frames  # 새로 생성되는 프레임 수

    if num_video_frames_per_chunk == 1:
        num_chunks = 1  # 이미지 모드
    else:
        num_generated_frames_vid2vid = num_total_frames - num_video_frames_per_chunk
        num_chunks = 1 + num_generated_frames_vid2vid // num_frames_per_chunk
        if num_generated_frames_vid2vid % num_frames_per_chunk != 0:
            num_chunks += 1

    return num_total_frames, num_chunks, num_frames_per_chunk
```

### 계산 예시

**예시 1: 200 프레임 영상, chunk=93, conditional=1**

```
num_frames_per_chunk = 93 - 1 = 92 (청크당 새로 생성되는 프레임)
num_generated_after_first = 200 - 93 = 107
추가 청크 수 = 107 // 92 = 1, 나머지 = 15 → +1
num_chunks = 1 + 1 + 1 = 3

청크 0: frames [0, 92]     → 93 프레임 처리
청크 1: frames [92, 184]   → 93 프레임 처리 (frame 92 = 조건 프레임)
청크 2: frames [184, 200]  → 16 프레임 → 패딩으로 93 프레임 맞춤
```

**예시 2: 50 프레임 영상, chunk=93**

```
50 < 93 → num_chunks = 1
입력 프레임을 93까지 패딩
```

**예시 3: 93 프레임 영상, chunk=93**

```
93 == 93 → num_chunks = 1
패딩 불필요
```

**예시 4: 이미지 (1 프레임), chunk=1**

```
num_chunks = 1 (이미지 모드)
```

---

## 3. 입력 패딩

> 파일: `inference_pipeline.py` → `_pad_input_frames()`

전체 프레임 수가 청크 크기보다 작을 때 패딩을 적용한다.

### 3.1 Reflect 패딩 (기본)

```
원본:      [F0, F1, F2, F3, F4]  (5 프레임)
필요 길이:  10 프레임

1차 반전:  [F4, F3, F2, F1]     → 4개 추가
결과:      [F0, F1, F2, F3, F4, F4, F3, F2, F1]  (9 프레임)

2차 반전:  [F1]                  → 1개 추가
결과:      [F0, F1, F2, F3, F4, F4, F3, F2, F1, F1]  (10 프레임)
```

### 3.2 Repeat 패딩 (대안)

```
원본:      [F0, F1, F2, F3, F4]  (5 프레임)
필요 길이:  10 프레임

마지막 프레임 반복:
결과:      [F0, F1, F2, F3, F4, F4, F4, F4, F4, F4]  (10 프레임)
```

> Reflect 패딩이 기본이며, 마지막 프레임이 반복되는 것보다 자연스러운 전환을 만든다.

---

## 4. 청크별 생성 루프

> 파일: `inference_pipeline.py` → `generate_img2world()` 내부 루프

```python
# 초기화
prev_output = torch.zeros_like(input_frames[:, :93]).to(uint8).cuda()[None]  # 제로 텐서
all_chunks = []

for chunk_id in range(num_chunks):
    # === 1. 현재 청크의 프레임 범위 계산 ===
    chunk_start_frame = chunk_id * num_frames_per_chunk
    chunk_end_frame = min(chunk_start_frame + num_video_frames_per_chunk, input_frames.shape[1])

    # === 2. 현재 청크 입력 슬라이스 ===
    cur_input_frames = input_frames[:, chunk_start_frame:chunk_end_frame]
    cur_input_frames = pad_to_93(cur_input_frames)  # 93 프레임으로 패딩

    # === 3. 데이터 배치 구성 ===
    data_batch = _get_data_batch_input(
        cur_input_frames,  # 현재 청크 원본 영상
        prev_output,       # 이전 청크 출력 (첫 청크: 제로 텐서)
        text_embedding,
        ...
    )

    # === 4. 컨트롤 입력 슬라이스 ===
    for k, v in control_input_dict.items():
        data_batch[k] = v[:, chunk_start_frame:chunk_end_frame]
        data_batch[k] = pad_to_93(data_batch[k])

    # === 5. 온더플라이 컨트롤 (edge, vis) ===
    data_batch = get_augmentor_for_eval(data_batch, ...)

    # === 6. 조건 프레임 수 설정 ===
    if chunk_id == 0:
        data_batch["num_conditional_frames"] = 0   # 첫 청크: 조건 없음
    else:
        data_batch["num_conditional_frames"] = 1 + (num_conditional_frames - 1) // 4
        # num_conditional_frames=1 → latent 1 토큰
        # num_conditional_frames=2 → latent 1 토큰 (temporal compression)

    # === 7. 디노이징 생성 ===
    sample = model.generate_samples_from_batch(
        data_batch, n_sample=1, guidance=guidance,
        seed=seed, num_steps=35
    )
    video = model.decode(sample)  # (1, C, T, H, W), [-1, 1]

    # === 8. 청크 결과 저장 ===
    if chunk_id == 0:
        all_chunks.append(video)                              # 전체 93 프레임
    else:
        all_chunks.append(video[:, :, num_conditional_frames:])  # 조건 프레임 제거

    # === 9. 다음 청크를 위한 조건 프레임 준비 ===
    last_frames = video[:, :, -num_conditional_frames:]  # 마지막 N 프레임
    last_frames_uint8 = (last_frames * 127.5 + 127.5).clamp(0, 255).to(uint8)
    blank_frames = torch.zeros(1, 3, 93 - num_conditional_frames, H, W, dtype=uint8)
    prev_output = torch.cat([last_frames_uint8, blank_frames], dim=2)
```

---

## 5. 프레임 오버랩 메커니즘 상세

### 5.1 오버랩은 있는가?

**핵심 답변: "겹치는 생성"이 아니라 "조건 프레임 전달"이다.**

```
청크 0 출력: [F0, F1, F2, ..., F91, F92]   ← 93 프레임 전체 저장
                                      │
                                    F92 (조건 프레임)
                                      │
                                      ▼
청크 1 입력: prev_output = [F92, 0, 0, ..., 0]  ← 1 프레임 + 92 제로 프레임
청크 1 출력: [F92', F93, F94, ..., F184]  ← 93 프레임 생성
                    └────── F92'는 제거 ──────┘
             저장: [F93, F94, ..., F184]   ← 92 프레임만 저장
```

### 5.2 오버랩 프레임의 처리

- 조건 프레임(F92')은 이전 청크의 마지막 프레임과 동일한 위치이지만, **새로 생성된 것**이다.
- 이 프레임은 **저장하지 않고 버린다** (`video[:, :, num_conditional_frames:]`).
- 따라서 **실제 출력에는 프레임 중복이 없다**.

### 5.3 컨트롤 입력의 오버랩

컨트롤 입력도 동일한 오버랩 패턴을 따른다:

```python
# 청크 0:
all_control_chunks[key].append(control_input)                    # 전체

# 청크 1+:
all_control_chunks[key].append(control_input[:, :, num_conditional_frames:])  # 조건 부분 제거
```

---

## 6. 시각적 타임라인

### 200 프레임 영상, chunk=93, conditional=1

```
원본 프레임: [0 ────────────────────────────────────── 199]

청크 0 입력범위:  [0 ──────── 92]
청크 0 출력저장:  [0 ──────── 92]  (93 프레임)

청크 1 입력범위:      [92 ─────── 184]
청크 1 조건프레임:    [92]  (이전 청크 출력의 마지막 1 프레임)
청크 1 출력저장:           [93 ─── 184]  (92 프레임)

청크 2 입력범위:               [184 ── 199 + 패딩]
청크 2 조건프레임:             [184]
청크 2 출력저장:                    [185 ── 199+α]

최종 결합: [0 ─────── 92 | 93 ───── 184 | 185 ── 199]
최종 자르기: [:200]  → 정확히 200 프레임
```

---

## 7. sigma_max: 노이즈 주입 제어

`sigma_max`가 설정되면, 입력 영상에 특정 양의 노이즈를 추가한 상태에서 디노이징을 시작한다:

```python
if sigma_max is not None:
    x0 = model.encode(normalized_input)           # 원본을 latent으로 인코딩
    x_sigma_max = model.get_x_from_clean(x0, sigma_max, seed)  # 노이즈 주입
    # x_sigma_max = x0 + sigma_max * noise
```

| sigma_max | 효과 |
|-----------|------|
| None | 순수 노이즈에서 시작 (기본) |
| 0 | 원본 영상 그대로 (노이즈 없음) |
| 50~100 | 약간의 변형 (스타일 변환) |
| 200 | 순수 노이즈와 유사 (완전 랜덤) |

---

## 8. 가이디드 생성 (Guided Generation)

가이디드 생성은 특정 영역에서 원본 영상의 구조를 보존하면서 나머지를 생성하는 기능이다.

### 8.1 마스크 처리

```python
# 마스크 읽기 (mp4 또는 npz)
guided_generation_mask = read_guided_generation_mask(
    input_path,
    foreground_labels=[1, 2, 3],  # npz에서 특정 레이블만 전경으로
    h=704, w=1280
)
# shape: (1, 3, T, H, W), range [0, 1]
```

### 8.2 Latent Weight Map 생성

마스크를 latent 공간 해상도로 변환:

```python
# 1 프레임씩 또는 4 프레임 그룹으로 처리 (temporal compression 고려)
weight_map_i = [interpolate(mask[:, :1, :1], size=(1, H_lat, W_lat))]  # 첫 프레임
for wi in range(1, T, 4):
    weight_map_i.append(interpolate(mask[:, :1, wi:wi+4], size=(1, H_lat, W_lat)))

weight_map = cat(weight_map_i, dim=2).repeat(1, C_lat, 1, 1, 1)
# shape: (1, 16, T_lat, H_lat, W_lat)
```

### 8.3 적용 방식

디노이징 루프에서 `step_threshold` 스텝까지만 가이디드 조건을 적용:

```python
x0_spatial_condition = {
    "x0": encoded_original,       # 원본의 latent
    "x_sigma_mask": weight_map,   # 어디를 보존할지
    "step_threshold": 25,         # 35스텝 중 25스텝까지 적용
}
```

---

## 9. 시드 관리

```python
# 초기 시드 설정
random.seed(seed)

# 각 청크마다 새로운 시드 파생
for chunk_id in range(num_chunks):
    seed = random.randint(0, 1000000)  # 이전 시드 기반으로 새 시드
    model.generate_samples_from_batch(..., seed=seed)
```

이렇게 하면:
- 같은 초기 시드 → 항상 같은 결과 (재현성)
- 각 청크는 서로 다른 노이즈 패턴 사용
