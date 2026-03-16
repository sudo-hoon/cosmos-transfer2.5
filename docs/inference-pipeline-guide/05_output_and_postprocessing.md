# 05. 출력 조립과 후처리

이 문서는 청크별 생성 결과를 합치고, 최종 영상을 출력하는 과정을 설명한다.

---

## 1. 청크 결과 조립

> 파일: `inference_pipeline.py:703`

### 1.1 청크 결합

```python
# 모든 청크 결합 (시간축)
full_video = torch.cat(all_chunks, dim=2)  # (1, C, T_total, H, W)

# 원본 프레임 수로 자르기
full_video = full_video[:, :, :num_total_frames, :, :]
```

**왜 잘라야 하는가?**

마지막 청크는 패딩으로 93 프레임까지 늘려서 생성했기 때문에, 실제 프레임 수보다 더 많은 프레임이 있을 수 있다.

```
예: 200 프레임 입력
청크 0: 93 프레임
청크 1: 92 프레임 (조건 1 제거)
청크 2: 92 프레임 (조건 1 제거, 패딩된 부분 포함)
합계: 93 + 92 + 92 = 277 프레임
자르기: [:200] → 200 프레임
```

### 1.2 컨트롤 비디오 결합

동일한 패턴으로 컨트롤 비디오도 결합:

```python
for key in hint_key:
    control_video_dict[key] = torch.cat(all_control_chunks[key], dim=2)
    control_video_dict[key] = control_video_dict[key][:, :, :num_total_frames, :, :]
```

---

## 2. 해상도 복원

> 파일: `transfer2/inference/utils.py` → `reshape_output_video_to_input_resolution()`

`keep_input_resolution=True` (기본값)일 때, 출력을 원본 입력 영상 해상도로 다시 리사이즈한다.

### 2.1 처리 흐름

```
생성된 영상: (1, 3, T, 704, 1280)  ← 720p, 16:9
                     │
                     ▼
          원본 해상도로 리사이즈
                     │
                     ▼
최종 출력: (1, 3, T, 1080, 1920)  ← 원본이 1920x1080이었다면
```

### 2.2 `show_control_condition`/`show_input` 처리

출력에 컨트롤 영상이나 입력 영상이 가로로 병합된 경우:

```
[control_edge | control_depth | input_video | generated_video]
      1280          1280           1280           1280
                        총 너비: 5120
```

이 경우 각 부분을 분리하여 **독립적으로 리사이즈**한 뒤 다시 병합한다:

```python
num_concatenated_parts = len(hint_key) * show_control + show_input + 1
part_width = total_width // num_concatenated_parts

parts = []
for i in range(num_concatenated_parts):
    part = full_video[:, :, :, :, i*part_width:(i+1)*part_width]
    resized = resize_video(part, original_H, original_W, cv2.INTER_LANCZOS4)
    parts.append(resized)

full_video = torch.cat(parts, dim=-1)
```

> 리사이즈 보간법: `cv2.INTER_LANCZOS4` (고품질 Lanczos 보간)

### 2.3 `keep_input_resolution=False` 일 때

리사이즈를 하지 않고 모델의 native 해상도 그대로 출력한다. 이 경우 출력 해상도는 `VIDEO_RES_SIZE_INFO` 테이블의 값이다.

---

## 3. 값 범위 변환

### 3.1 모델 출력 → 저장 가능한 형식

```python
# 모델 출력: [-1, 1] 범위
output_video = model.decode(latent)  # (1, 3, T, H, W), dtype=bfloat16

# [0, 1] 범위로 변환 (inference.py:307)
output_video = (1.0 + output_video[0]) / 2  # 배치 차원 제거, [0, 1]

# 저장 함수가 내부적으로 [0, 255]로 변환
save_img_or_video(output_video, output_path, fps=fps)
```

### 3.2 컨트롤 비디오도 동일

```python
for key in control_video_dict:
    control_video_dict[key] = (1.0 + control_video_dict[key][0]) / 2  # [0, 1]
    save_img_or_video(control_video_dict[key], f"{output_path}_control_{key}", fps=fps)
```

---

## 4. 가드레일 검사

> 파일: `cosmos_transfer2/inference.py:313`

### 4.1 비디오 가드레일

생성된 영상에 대해 안전성 검사를 수행한다:

```python
if video_guardrail_runner is not None:
    # [0, 1] → uint8 [0, 255]
    frames = (output_video * 255.0).clamp(0, 255).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

    processed_frames = run_video_guardrail(frames, video_guardrail_runner)

    if processed_frames is None:
        # 차단됨 → 생성 중단 또는 건너뛰기
        if keep_going:
            return None
        else:
            raise Exception("Guardrail blocked")
    else:
        # 통과 (일부 프레임이 수정될 수 있음)
        output_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
```

### 4.2 가드레일 순서

```
1. 텍스트 가드레일 (생성 전)
   └─ 프롬프트 + 네거티브 프롬프트 안전성 검사

2. 영상 생성

3. 비디오 가드레일 (생성 후)
   └─ 생성된 영상의 안전성 검사
   └─ 위험 콘텐츠가 있으면 차단 또는 수정
```

---

## 5. 파일 저장

### 5.1 출력 파일 구조

```
output_dir/
├── sample_name.mp4              # 생성된 영상 (또는 .jpg)
├── sample_name.txt              # 사용된 프롬프트
├── sample_name.json             # 추론 파라미터
├── sample_name_control_edge.mp4 # 엣지 컨트롤 영상
├── sample_name_control_depth.mp4# 뎁스 컨트롤 영상
├── sample_name_mask_edge.mp4    # 엣지 마스크 (있는 경우)
├── sample_name_mask_guided_generation.mp4  # 가이디드 마스크 (있는 경우)
└── config.yaml                  # 모델 설정 파일
```

### 5.2 이미지 vs 영상 판별

```python
if output_video.shape[2] == 1:
    ext = "jpg"  # 단일 프레임 → 이미지로 저장
else:
    ext = "mp4"  # 다중 프레임 → 영상으로 저장
```

---

## 6. 입력-출력 크기 관계 정리

### 6.1 해상도

| 시나리오 | 입력 크기 | 내부 처리 크기 | 출력 크기 |
|---------|----------|--------------|----------|
| `keep_input_resolution=True` (기본) | 1920x1080 | 1280x704 | **1920x1080** |
| `keep_input_resolution=False` | 1920x1080 | 1280x704 | **1280x704** |
| 입력이 이미 720p | 1280x720 | 1280x704 | **1280x720** |

> **주의**: 모델 내부 해상도(1280x704)와 입력 해상도는 다를 수 있다. aspect ratio 테이블에 의해 결정되는 정확한 해상도(예: 704 vs 720)가 다르기 때문.

### 6.2 프레임 수

| 시나리오 | 입력 프레임 | 출력 프레임 | 동일? |
|---------|------------|-----------|-------|
| 입력 ≥ 93 | N | **N** | O |
| 입력 < 93 | M | **M** | O (패딩 후 자르기) |
| 이미지 | 1 | **1** | O |

> **결론: 입력 프레임 수와 출력 프레임 수는 항상 동일하다.** 내부적으로 패딩이 적용되지만, 최종 출력에서 원본 프레임 수로 잘라낸다.

### 6.3 배치 크기

| 항목 | 값 | 설명 |
|------|-----|------|
| 입력 배치 | 1 | 항상 1 영상씩 처리 |
| 생성 배치 (`n_sample`) | 1 | 항상 1 영상 생성 |
| 다중 샘플 | 순차 | `generate()`에서 for 루프로 순차 처리 |

---

## 7. 전체 파이프라인 타이밍

벤치마크 모드(`--benchmark`)에서 측정되는 구간:

| 구간 | 내용 |
|------|------|
| `text_guardrail` | 텍스트 가드레일 검사 |
| `get_text_embeddings` | Qwen2.5-VL 7B 텍스트 인코딩 |
| `preprocessing` | 이미지 컨텍스트, 컨트롤 입력 처리 |
| `generate_chunk` | 청크당 생성 (인코딩 + 디노이징 + 디코딩) |
| `postprocessing` | 청크 결합, 해상도 복원 |
| `video_guardrail` | 비디오 가드레일 검사 |

---

## 8. 메모리 관리

```python
# 각 샘플 처리 후 GPU 캐시 비우기
torch.cuda.empty_cache()
```

### 멀티 GPU Context Parallelism

- rank 0만 출력 저장, 가드레일 검사 수행
- 다른 rank는 생성에만 참여
- 출력은 자동으로 gather됨

---

## 9. 정리: 한눈에 보는 전체 데이터 흐름

```
입력 영상 (mp4/jpg)
    │ read_and_process_video()
    │ 해상도 리사이즈, aspect ratio 감지
    ▼
input_frames: (C=3, T, H=704, W=1280), uint8 [0,255]
    │
    ├──── 텍스트: prompt → Qwen2.5-VL 7B → (1, seq_len, 100352→1024), bfloat16
    ├──── 이미지 컨텍스트: image → SigLIP2 → (1, 256, 1152), bfloat16
    ├──── 컨트롤: depth/seg → 사전계산 or 모델 → (C, T, H, W), uint8
    │                edge/vis → augmentor → (1, 3, T, H, W), uint8
    │
    ▼ 청크 분할 + for loop
    │
    ▼ _normalize_video_databatch_inplace()
    │ uint8 [0,255] → bfloat16 [-1,1]
    │
    ▼ get_data_and_condition()
    │ encode(video) → latent (1, 16, 24, 44, 80)
    │ encode(control) → control_latent
    │
    ▼ get_velocity_fn_from_batch()
    │ CFG: condition + uncondition 쌍 준비
    │
    ▼ Denoising Loop (35 steps)
    │ noise → DiT(with control hints) → velocity → scheduler.step()
    │
    ▼ decode(latent)
    │ latent (1, 16, 24, 44, 80) → video (1, 3, 93, 704, 1280), [-1,1]
    │
    ▼ 청크 조립 (조건 프레임 제거, 시간축 concat)
    │
    ▼ 원본 프레임 수로 자르기
    │
    ▼ reshape_output_video_to_input_resolution()
    │ 1280x704 → 원본 해상도 (INTER_LANCZOS4)
    │
    ▼ [-1,1] → [0,1] → uint8 [0,255]
    │
    ▼ 비디오 가드레일 검사
    │
    ▼ save_img_or_video()
    │
    ▼ output.mp4 (또는 .jpg)
```
