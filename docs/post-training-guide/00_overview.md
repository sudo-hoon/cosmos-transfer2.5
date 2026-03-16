# Cosmos Transfer 2.5 사후학습 (Post-Training) 가이드 - 전체 개요

## 문서 구성

| 문서 | 내용 |
|------|------|
| [01_data_preparation.md](./01_data_preparation.md) | 데이터 요구사항, 디렉토리 구조, 컨트롤 입력 사전 계산 |
| [02_data_pipeline.md](./02_data_pipeline.md) | 데이터 로딩, 프레임 샘플링, 어그멘테이션, 컨트롤 생성 파이프라인 |
| [03_training_theory.md](./03_training_theory.md) | Rectified Flow, ControlNet, VACE의 이론적 배경과 학습 원리 |
| [04_training_implementation.md](./04_training_implementation.md) | 코드 레벨 학습 루프, loss 계산, freeze 전략, EMA |
| [05_practical_guide.md](./05_practical_guide.md) | 실전 하이퍼파라미터, 모니터링, 체크포인트, 추론까지 |

---

## 사후학습이란?

Cosmos Transfer 2.5의 사후학습(post-training)은 **NVIDIA가 사전 학습한 ControlNet 가중치를 내 도메인 데이터에 맞게 fine-tuning**하는 것이다. 전체 모델을 처음부터 학습하는 것이 아니라:

```
사전 학습된 모델 (NVIDIA 제공)
    │
    ├── Base DiT (2B params) ── 동결 (학습 안 함)
    │
    └── ControlNet Branch ── 학습 대상
        ├── Control Blocks (4~14개) ← 이것만 학습
        ├── Control Embedder ← 이것만 학습
        └── (선택) Separate Embedders ← 이것만 학습
```

**전체 2B 파라미터 중 ControlNet 부분만 학습**하므로 효율적이다.

---

## 아키텍처 요약: 학습 시 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│  데이터셋 디렉토리                                                │
│  ├── videos/ (MP4)                                              │
│  ├── captions/ (JSON)                                           │
│  └── depth|seg/ (선택, 사전 계산)                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  SingleViewTransferDataset.__getitem__()                         │
│    1. decord로 영상 프레임 로드 (랜덤 시작점)                      │
│    2. 캡션 JSON 로드                                             │
│    3. 컨트롤 입력 로드/None                                       │
│    4. 어그멘테이션 체인:                                           │
│       ResizeLargestSide → ReflectionPadding → ControlAugmentor  │
│    5. 출력: {video, control_input_*, caption, padding_mask}      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Model.training_step(data_batch)                                 │
│    1. _normalize_video_databatch_inplace()                       │
│       uint8 [0,255] → bfloat16 [-1,1]                          │
│    2. Text Encoder (Qwen2.5-VL 7B, on-the-fly)                  │
│       캡션 → 텍스트 임베딩 (FULL_CONCAT: 28 layers × 3584)       │
│    3. VAE Encode: 영상 → latent (16ch × 24T × 44H × 80W)        │
│    4. VAE Encode: 컨트롤 → control_latent                        │
│    5. VACE 조건 마스크 생성 (랜덤 0/1/2 프레임)                    │
│    6. Rectified Flow Forward:                                    │
│       a. t ~ LogitNormal 샘플링                                  │
│       b. x_t = (1-σ)·x_0 + σ·ε  (노이즈 보간)                   │
│       c. v_target = ε - x_0  (타겟 velocity)                    │
│       d. v_pred = DiT(x_t, t, condition, control_hints)          │
│       e. loss = MSE(v_pred, v_target)                           │
│    7. Backward → ControlNet 파라미터만 업데이트                    │
│    8. EMA 업데이트                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 핵심 숫자 정리

| 항목 | 값 | 설명 |
|------|-----|------|
| **모델 크기** | 2B | 전체 파라미터 |
| **학습 가능 파라미터** | ~수백 M | ControlNet 부분만 |
| **필수 GPU** | 8x H100/A100 80GB | Context Parallelism 사용 |
| **프레임 수** | 93 | `(state_t - 1) × 4 + 1` |
| **해상도** | 704 × 1280 | 720p, 16:9 |
| **Latent 크기** | 16 × 24 × 44 × 80 | C × T × H × W |
| **배치 크기** | 1 per GPU | Gradient Accumulation = 4 |
| **유효 배치** | 4 (= 1 × 4 accum) | |
| **기본 LR** | 8.63e-5 | ≈ 2^(-14.5) |
| **Warmup** | 100~1000 steps | |
| **기본 Iteration** | 2000~5000 | |
| **텍스트 인코더** | Qwen2.5-VL 7B | On-the-fly 인코딩 |
| **CFG dropout** | 텍스트 20% | 학습 중 텍스트 드롭아웃 |
| **조건 프레임 확률** | {0: 40%, 1: 40%, 2: 20%} | VACE 조건 |
| **EMA rate** | 0.10 (Power EMA) | |
| **체크포인트 형식** | DCP (Distributed CP) | `.pt`로 변환 필요 |

---

## 필요 리소스

| 리소스 | 최소 | 권장 |
|--------|------|------|
| GPU | 8x A100 80GB | 8x H100 80GB |
| 비디오 데이터 | 50~100개 | 200~1000+개 |
| 비디오 길이 | 4초+ (93프레임@24fps) | 3초+ |
| 스토리지 | 10GB (데이터) + 50GB (체크포인트) | 100GB+ |
| 학습 시간 | ~0.7h (2000 iter, 8xA100) | ~1.5h (5000 iter) |

---

## 파일 위치 참조

| 역할 | 파일 경로 |
|------|-----------|
| 학습 진입점 | `scripts/train.py` |
| 싱글뷰 설정 래퍼 | `cosmos_transfer2/singleview_config.py` |
| 실험 설정 | `cosmos_transfer2/experiments/singleview/cosmos_singleview_example.py` |
| 베이스 실험 | `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiment/exp_large_scale.py` |
| 데이터로더 설정 | `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/defaults/dataloader_local.py` |
| 데이터셋 클래스 | `cosmos_transfer2/_src/transfer2/datasets/local_datasets/singleview_dataset.py` |
| 어그멘테이션 | `cosmos_transfer2/_src/transfer2/datasets/augmentors/control_input.py` |
| 어그멘터 프로바이더 | `cosmos_transfer2/_src/transfer2/datasets/augmentor_provider.py` |
| 모델 (Rectified Flow) | `cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow.py` |
| 모델 (Control VACE) | `cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace.py` |
| 베이스 모델 | `cosmos_transfer2/_src/predict2/models/text2world_model_rectified_flow.py` |
| DiT 네트워크 | `cosmos_transfer2/_src/transfer2/networks/minimal_v4_lvg_dit_control_vace.py` |
| 트레이너 | `cosmos_transfer2/_src/imaginaire/trainer.py` |
| EMA | `cosmos_transfer2/_src/imaginaire/utils/ema.py` |
| 텍스트 인코더 | `cosmos_transfer2/_src/predict2/text_encoders/text_encoder.py` |
| Conditioner | `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/defaults/conditioner.py` |
| VideoUFO 준비 | `scripts/prepare_videoufo_dataset.py` |
| 체크포인트 변환 | `scripts/convert_distcp_to_pt.py` |
