# Cosmos Transfer 2.5 Inference Pipeline - 전체 개요

## 문서 구성

이 가이드는 Cosmos Transfer 2.5의 `inference.py`를 통한 영상 생성 파이프라인을 상세하게 분석한 문서입니다.

| 문서 | 내용 |
|------|------|
| [01_entry_and_config.md](./01_entry_and_config.md) | 진입점, 설정, CLI 인자 구조 |
| [02_input_preprocessing.md](./02_input_preprocessing.md) | 입력 영상/이미지/텍스트 전처리, 컨트롤 입력 생성 |
| [03_chunking_and_generation.md](./03_chunking_and_generation.md) | 청크 분할, 오버랩, Autoregressive 생성 |
| [04_model_internals.md](./04_model_internals.md) | 모델 내부: 토크나이저, DiT, ControlNet, CFG |
| [05_output_and_postprocessing.md](./05_output_and_postprocessing.md) | 출력 조립, 후처리, 해상도 복원 |

---

## 아키텍처 요약

```
┌─────────────────────────────────────────────────────────────────────┐
│                    examples/inference.py (CLI 진입점)                │
│    tyro CLI → InferenceArguments 파싱 → JSON 파일에서 샘플 로드       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              cosmos_transfer2/inference.py                           │
│              Control2WorldInference 클래스                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ __init__: 체크포인트 로드, 가드레일 초기화, 파이프라인 생성    │    │
│  │ generate(): 샘플 루프 → _generate_sample()                   │    │
│  │ _generate_sample(): 가드레일 체크 → generate_img2world()     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│       transfer2/inference/inference_pipeline.py                      │
│       ControlVideo2WorldInference 클래스                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ generate_img2world():                                        │    │
│  │   1. 입력 영상 읽기 & 리사이즈                                 │    │
│  │   2. 텍스트 임베딩 계산 (Qwen2.5-VL 7B)                       │    │
│  │   3. 이미지 컨텍스트 처리 (SigLIP2)                           │    │
│  │   4. 컨트롤 입력 로드/생성 (edge, depth, seg, vis)            │    │
│  │   5. 청크 분할 계산                                           │    │
│  │   6. 청크별 반복:                                             │    │
│  │      a. 데이터 배치 구성                                      │    │
│  │      b. 컨트롤 입력 슬라이스 & 어그멘터 적용                   │    │
│  │      c. model.generate_samples_from_batch() → latent          │    │
│  │      d. model.decode(latent) → 픽셀 영상                     │    │
│  │      e. 이전 청크의 마지막 프레임 → 다음 청크 조건             │    │
│  │   7. 청크 결합 & 원본 프레임 수로 자르기                       │    │
│  │   8. 해상도 복원 (keep_input_resolution)                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│         모델 내부 (Rectified Flow + ControlNet)                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  generate_samples_from_batch():                             │     │
│  │    1. _normalize_video_databatch_inplace() [0,255]→[-1,1]   │     │
│  │    2. get_data_and_condition() → latent + 컨트롤 조건         │     │
│  │       - encode(video) → latent (Wan2.2 VAE)                 │     │
│  │       - get_control_latent(control) → 컨트롤 latent          │     │
│  │    3. get_velocity_fn_from_batch() → CFG velocity 함수       │     │
│  │    4. Denoising Loop (35 steps):                            │     │
│  │       noise → DiT(noise, control_hints, text_emb) → velocity│     │
│  │       latent = scheduler.step(velocity, t, latent)          │     │
│  │    5. Return: latent                                        │     │
│  └────────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  DiT Network (with Control Branch):                         │     │
│  │    - 메인 DiT 블록 + ControlNet 블록 (병렬 처리)             │     │
│  │    - 컨트롤 블록 → layer-wise hints 생성                    │     │
│  │    - hints가 메인 블록에 additive로 주입됨                   │     │
│  │    - control_weight로 주입 강도 조절                         │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 핵심 숫자 요약

| 항목 | 값 | 설명 |
|------|------|------|
| **기본 해상도 (720p, 16:9)** | 1280 x 704 px | 입력/출력 동일 |
| **청크당 프레임 수** | 93 frames | 기본값, `num_video_frames_per_chunk` |
| **시간 압축률** | 4x | Wan2.2 VAE temporal compression |
| **공간 압축률** | 16x | Wan2.2 VAE spatial compression (8x downsample + 2x patchify) |
| **Latent 크기 (720p, 16:9, 93f)** | 16ch x 24T x 44H x 80W | state_t=24 |
| **디노이징 스텝** | 35 steps | 기본값, `num_steps` |
| **CFG guidance** | 3 (기본) | 범위 0~7 |
| **조건 프레임** | 1 frame | 청크간 연결용, 첫 청크는 0 |
| **sigma_data** | 1.0 | Rectified flow 스케일링 |
| **배치 크기** | 1 | 항상 1 |
| **텍스트 임베딩 크기** | seq_len x 100,352 | Qwen2.5-VL 7B: 28 layers × 3584 dim, FULL_CONCAT |
| **이미지 컨텍스트 크기** | 256 x 1152 | SigLIP2 출력 토큰 |

---

## 입력 → 출력 크기 관계

```
입력 영상 (예: 1920x1080, 200 frames)
    │
    ├─ 리사이즈 → 1280x704 (720p, 16:9에 맞춤)
    │
    ├─ 청크 분할:
    │   청크 0: frames [0, 92]    → 93 frames
    │   청크 1: frames [92, 184]  → 93 frames (1 frame 오버랩)
    │   청크 2: frames [184, 200] → 패딩으로 93 frames 맞춤
    │
    ├─ 각 청크 → VAE Encode → Latent [1, 16, 24, 44, 80]
    │   ├─ DiT Denoising (35 steps)
    │   └─ VAE Decode → [1, 3, 93, 704, 1280]
    │
    ├─ 청크 결합 (오버랩 프레임 제거)
    │   └─ 원본 프레임 수(200)로 자르기
    │
    └─ 최종 출력: 1920x1080, 200 frames (keep_input_resolution=True일 때)
```

---

## 파일 위치 참조

| 역할 | 파일 경로 |
|------|-----------|
| CLI 진입점 | `examples/inference.py` |
| 설정/인자 정의 | `cosmos_transfer2/config.py` |
| 고수준 추론 클래스 | `cosmos_transfer2/inference.py` |
| 핵심 파이프라인 | `cosmos_transfer2/_src/transfer2/inference/inference_pipeline.py` |
| 유틸리티 함수 | `cosmos_transfer2/_src/transfer2/inference/utils.py` |
| Rectified Flow 모델 | `cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow.py` |
| Control VACE 모델 | `cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace.py` |
| DiT 네트워크 (ControlNet) | `cosmos_transfer2/_src/transfer2/networks/minimal_v4_lvg_dit_control_vace.py` |
| 컨트롤 입력 어그멘터 | `cosmos_transfer2/_src/transfer2/datasets/augmentors/control_input.py` |
| SigLIP2 이미지 인코더 | `cosmos_transfer2/_src/transfer2/networks/siglip2_image_context.py` |
| 토크나이저 (Wan2.2 VAE) | `cosmos_transfer2/_src/predict2/tokenizers/wan2pt2.py` |
| 해상도 매핑 테이블 | `cosmos_transfer2/_src/predict2/datasets/utils.py` |
