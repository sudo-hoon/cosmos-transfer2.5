# Post-Training 실행 가이드 - 시작부터 끝까지

## 목차

1. [전체 워크플로우 개요](#전체-워크플로우-개요)
2. [Step 1: 환경 준비](#step-1-환경-준비)
3. [Step 2: 데이터셋 배치](#step-2-데이터셋-배치)
4. [Step 3: 환경 변수 설정](#step-3-환경-변수-설정)
5. [Step 4: Post-Training 실행](#step-4-post-training-실행)
6. [Step 5: 학습 모니터링](#step-5-학습-모니터링)
7. [Step 6: 체크포인트 변환](#step-6-체크포인트-변환)
8. [Step 7: Inference (Sim2Real 변환)](#step-7-inference-sim2real-변환)
9. [Step 8: 결과 평가 및 반복](#step-8-결과-평가-및-반복)
10. [LoRA를 활용한 경량 Post-Training](#lora를-활용한-경량-post-training)
11. [트러블슈팅](#트러블슈팅)

---

## 전체 워크플로우 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Sim2Real Post-Training 워크플로우            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Step 1] 환경 준비 (GPU, 소프트웨어, 의존성)                    │
│      ↓                                                      │
│  [Step 2] 데이터셋 배치 (실제 환경 영상 + 캡션)                   │
│      ↓                                                      │
│  [Step 3] 환경 변수 설정                                      │
│      ↓                                                      │
│  [Step 4] Post-Training 실행 ← 여기서 GPU 학습이 진행됨         │
│      ↓                                                      │
│  [Step 5] 학습 모니터링 (loss, 샘플 영상 확인)                   │
│      ↓                                                      │
│  [Step 6] 체크포인트 변환 (DCP → PyTorch .pt)                  │
│      ↓                                                      │
│  [Step 7] Inference (시뮬레이션 데이터 → 사실적 데이터)           │
│      ↓                                                      │
│  [Step 8] 결과 평가 → 필요 시 데이터 추가 후 Step 4로 반복       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: 환경 준비

### 하드웨어 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| **GPU** | 8x A100 80GB | 8x H100 80GB |
| **시스템 메모리** | 256 GB | 512 GB |
| **저장 공간** | 200 GB (데이터+체크포인트) | 500 GB+ |
| **GPU 메모리** | 80 GB per GPU | 80 GB per GPU |

> **참고**: 2B 모델은 8x GPU가 최소 요구사항입니다. `context_parallel_size=8`이 기본값이며, 이는 GPU 수와 일치해야 합니다.

### 소프트웨어 설치

```bash
# 1. 저장소 클론
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5

# 2. 환경 설정 (Setup Guide 참조)
# 자세한 내용은 docs/setup.md 참조
pip install -r requirements.txt  # 또는 해당 설치 명령어

# 3. Hugging Face 로그인 (모델 다운로드용)
pip install huggingface_hub
huggingface-cli login
# 토큰 입력 프롬프트가 나타나면 HuggingFace 토큰을 입력
```

> **중요**: 상세한 환경 설정은 반드시 [Setup Guide](../setup.md)를 먼저 완료하세요.

---

## Step 2: 데이터셋 배치

### 옵션 A: VideoUFO 샘플 데이터로 테스트 (권장 - 처음 시작 시)

실제 데이터로 학습하기 전에, 먼저 제공된 샘플 데이터로 전체 파이프라인이 동작하는지 확인하세요.

```bash
# VideoUFO 데이터셋 다운로드 (128개 비디오, 테스트용)
python scripts/prepare_videoufo_dataset.py \
    --storage_dir assets/videoufo \
    --num_videos 128
```

이 스크립트는 자동으로:
1. VideoUFO 메타데이터 CSV 다운로드 (~1.1 GB)
2. 비디오가 포함된 tar 파일 다운로드 (~4 GB/tar)
3. 93프레임 이상의 비디오만 필터링하여 추출
4. 캡션 JSON 파일 생성
5. 학습에 필요한 폴더 구조로 정리

### 옵션 B: 커스텀 데이터셋 배치 (Sim2Real용)

[02. 데이터 준비 가이드](./02_data_preparation_guide.md)에 따라 준비한 데이터를 배치합니다.

```bash
# 데이터셋 디렉토리 생성
mkdir -p datasets/my_sim2real_dataset/videos
mkdir -p datasets/my_sim2real_dataset/captions

# 비디오 파일 복사
cp /path/to/your/videos/*.mp4 datasets/my_sim2real_dataset/videos/

# 캡션 파일 복사
cp /path/to/your/captions/*.json datasets/my_sim2real_dataset/captions/

# (선택) Depth/Seg control 파일 복사
mkdir -p datasets/my_sim2real_dataset/depth
cp /path/to/your/depth/*.mp4 datasets/my_sim2real_dataset/depth/
```

### 데이터 검증

```bash
# 비디오 수 확인
echo "Videos: $(ls datasets/my_sim2real_dataset/videos/*.mp4 | wc -l)"
echo "Captions: $(ls datasets/my_sim2real_dataset/captions/*.json | wc -l)"

# 샘플 캡션 확인
cat datasets/my_sim2real_dataset/captions/$(ls datasets/my_sim2real_dataset/captions/ | head -1)
```

---

## Step 3: 환경 변수 설정

```bash
# 체크포인트 및 학습 결과 저장 경로 설정
# 충분한 디스크 공간이 있는 경로를 지정하세요
export IMAGINAIRE_OUTPUT_ROOT=/path/to/outputs

# (선택) HuggingFace 캐시 경로 변경
# 기본값: ~/.cache/huggingface
export HF_HOME=/path/to/hf_cache

# (선택) Weights & Biases 설정
# 학습 메트릭을 시각화하려면 설정, 아니면 학습 시 disabled로 지정
export WANDB_API_KEY=your_api_key_here
```

### 경로 확인

```bash
# 출력 디렉토리 존재 확인
mkdir -p $IMAGINAIRE_OUTPUT_ROOT
echo "Output root: $IMAGINAIRE_OUTPUT_ROOT"
df -h $IMAGINAIRE_OUTPUT_ROOT  # 디스크 공간 확인
```

---

## Step 4: Post-Training 실행

### 4-1. Edge Control로 학습 (권장 - 전처리 불필요)

```bash
# 기본 학습 명령어 (Edge Control)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_sim2real_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

### 4-2. Depth Control로 학습

```bash
# Depth Control (depth/ 폴더에 전처리된 depth 비디오 필요)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_depth_example \
    dataloader_train.dataset.dataset_dir=datasets/my_sim2real_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

### 4-3. Segmentation Control로 학습

```bash
# Segmentation Control (seg/ 폴더에 전처리된 seg 비디오 필요)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_seg_example \
    dataloader_train.dataset.dataset_dir=datasets/my_sim2real_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

### 4-4. Visual Blur Control로 학습

```bash
# Visual Blur Control (전처리 불필요)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_vis_example \
    dataloader_train.dataset.dataset_dir=datasets/my_sim2real_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

### 주요 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `trainer.max_iter` | 5000 | 총 학습 반복 횟수 |
| `checkpoint.save_iter` | 1000 | 체크포인트 저장 간격 |
| `optimizer.lr` | 5e-5 | 학습률 |
| `scheduler.warm_up_steps` | [1000] | 학습률 웜업 스텝 |
| `job.wandb_mode` | online | W&B 모드 (online/offline/disabled) |

### 학습 반복 횟수 권장값

| 데이터 규모 | 권장 반복 횟수 | 예상 시간 (8x A100) |
|------------|-------------|-------------------|
| 50~100개 비디오 | 1,000~2,000 | ~0.5~0.7시간 |
| 200~500개 비디오 | 2,000~5,000 | ~0.8~2시간 |
| 1,000개+ 비디오 | 5,000~10,000 | ~2~5시간 |

> **팁**: 처음에는 2,000 반복으로 시작하고, 샘플 생성 결과를 확인한 후 필요 시 학습을 재개(resume)하세요.

### 첫 실행 시 주의사항

- **모델 자동 다운로드**: 첫 실행 시 HuggingFace에서 사전학습 체크포인트가 자동 다운로드됩니다 (수 GB). 네트워크 상태에 따라 시간이 걸릴 수 있습니다.
- **텍스트 인코더 다운로드**: Qwen2.5-VL-7B 모델도 자동 다운로드됩니다.
- **디스크 공간**: `HF_HOME`과 `IMAGINAIRE_OUTPUT_ROOT`에 충분한 공간이 있는지 확인하세요.

---

## Step 5: 학습 모니터링

### 터미널 출력 해석

학습 중 아래와 같은 출력이 나타납니다:

```
[Iteration 100/2000] Loss: 0.234, LR: 4.95e-05, Time: 1.23s/it
[Iteration 200/2000] Saving checkpoint...
[Iteration 500/2000] Loss: 0.156, LR: 5.00e-05, Time: 1.20s/it
```

### 정상적인 학습 지표

| 지표 | 초기 (0~200 iter) | 중간 (200~1000 iter) | 후반 (1000+ iter) |
|------|------------------|---------------------|-------------------|
| **Loss** | ~0.5 | ~0.2~0.3 | ~0.1~0.2 |
| **LR** | 점진적 증가 (warmup) | 5e-5 (최대) | 5e-5 유지 |
| **Time/iter** | ~1.2~1.5s | ~1.2s | ~1.2s |

### 문제 징후

- **Loss가 NaN/Inf**: 학습률이 너무 높거나 데이터 문제
- **Loss가 감소하지 않음**: 데이터 품질 확인, 학습률 조정 필요
- **Time/iter이 급격히 증가**: 메모리 스왑 발생 가능 (GPU OOM 직전)

### 샘플 생성 확인

학습 중 매 50 iteration마다(기본값) 샘플 비디오가 생성됩니다:

```bash
# 샘플 저장 위치
ls ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/local_single_view/*/samples/
```

샘플 비디오를 확인하여 학습이 올바르게 진행되는지 시각적으로 판단하세요.

### Weights & Biases 모니터링 (선택)

W&B를 활성화한 경우:
1. [wandb.ai](https://wandb.ai)에 접속
2. 프로젝트 대시보드에서 loss curve, 생성 샘플 등 확인

---

## Step 6: 체크포인트 변환

학습이 완료되면 DCP 형식의 체크포인트를 PyTorch 형식으로 변환해야 inference에 사용할 수 있습니다.

### 6-1. 최신 체크포인트 찾기

```bash
# 체크포인트 디렉토리 확인
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/local_single_view

# 가장 최근 학습 디렉토리 찾기
ls -la $CHECKPOINTS_DIR/

# 체크포인트 디렉토리 설정 (이름은 실행 시 생성된 것으로 대체)
TRAIN_DIR=$CHECKPOINTS_DIR/transfer2_singleview_posttrain_edge_example_*/checkpoints
```

### 6-2. DCP → PyTorch 변환

```bash
# 최신 체크포인트 iteration 확인
CHECKPOINT_ITER=$(cat $TRAIN_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$TRAIN_DIR/$CHECKPOINT_ITER

echo "Converting checkpoint: $CHECKPOINT_DIR"

# 변환 실행
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

### 6-3. 변환 결과 확인

```bash
# 생성된 파일 확인
ls -la $CHECKPOINT_DIR/

# 예상 출력:
# model.pt              - 전체 체크포인트
# model_ema_fp32.pt     - EMA 체크포인트 (FP32)
# model_ema_bf16.pt     - EMA 체크포인트 (BF16) ← Inference에 사용
```

> **Inference에는 `model_ema_bf16.pt`를 사용하세요.** EMA (Exponential Moving Average) 체크포인트가 일반적으로 더 안정적인 결과를 생성합니다.

---

## Step 7: Inference (Sim2Real 변환)

Post-Training이 완료되면, 이제 시뮬레이션 데이터를 사실적 데이터로 변환할 수 있습니다.

### 7-1. Inference 입력 준비

시뮬레이션에서 생성한 데이터로 inference 입력 JSONL 파일을 준비합니다.

```bash
# inference 입력 JSONL 파일 생성
cat > inference_input.jsonl << 'EOF'
{"input_video": "sim_data/sim_video_001.mp4", "prompt": "A realistic indoor warehouse scene with concrete floor and metal shelving", "control_type": "edge"}
{"input_video": "sim_data/sim_video_002.mp4", "prompt": "A robot arm operating in a bright industrial environment", "control_type": "edge"}
EOF
```

### 7-2. Inference 실행

```bash
# Post-Training된 모델로 Sim2Real 변환
torchrun --nproc_per_node=8 examples/inference.py \
    -i inference_input.jsonl \
    -o outputs/sim2real_results/ \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment transfer2_singleview_posttrain_edge_example
```

### 7-3. 결과 확인

```bash
# 생성된 비디오 확인
ls outputs/sim2real_results/

# 비디오 재생 (GUI 환경에서)
# 또는 원격 서버에서 로컬로 복사하여 확인
scp user@server:outputs/sim2real_results/*.mp4 ./local_results/
```

### Sim2Real Inference 팁

1. **프롬프트가 중요합니다**: Post-Training 데이터와 유사한 스타일의 프롬프트 사용
2. **Control Weight 조정**: 구조적 충실도와 시각적 품질의 균형 조절
   - `control_weight=1.0`: 원본 구조에 충실 (기본값)
   - `control_weight=0.7~0.8`: 약간의 자유도 부여
3. **Control 타입 일치**: 학습 시 사용한 Control 타입으로 inference 수행

---

## Step 8: 결과 평가 및 반복

### 평가 기준

| 평가 항목 | 확인 방법 |
|----------|----------|
| **시각적 사실성** | 생성된 영상이 실제 환경과 유사한가? |
| **구조적 일관성** | 시뮬레이션의 기하학적 구조가 보존되었는가? |
| **도메인 특성 반영** | 타겟 도메인의 특성(조명, 재질 등)이 반영되었는가? |
| **아티팩트 여부** | 비정상적인 인공물이 없는가? |
| **다운스트림 성능** | 생성 데이터로 학습한 모델의 실제 성능이 향상되었는가? |

### 결과가 만족스럽지 않을 때

| 문제 | 해결 방법 |
|------|----------|
| 전반적으로 흐릿함 | 학습 반복 횟수 증가 |
| 특정 객체가 부자연스러움 | 해당 객체가 포함된 학습 데이터 추가 |
| 조명이 부자연스러움 | 다양한 조명 조건의 데이터 추가 |
| 구조가 왜곡됨 | Depth 또는 Seg Control로 전환 |
| 도메인 특성 불일치 | 타겟 도메인 데이터 수량 증가 |

### 학습 재개 (Resume)

더 많은 학습이 필요한 경우 **동일한 명령어를 재실행**하면 자동으로 최신 체크포인트에서 재개됩니다:

```bash
# 동일한 명령어로 재실행 → 자동 resume
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_sim2real_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

---

## LoRA를 활용한 경량 Post-Training

### LoRA란?

LoRA (Low-Rank Adaptation)는 **전체 모델 가중치를 수정하지 않고**, 작은 크기의 추가 파라미터만 학습하는 기법입니다.

### LoRA vs Full Fine-tuning 비교

| 항목 | Full Fine-tuning | LoRA |
|------|-----------------|------|
| **학습 파라미터** | 2B (전체) | ~45M (~2%) |
| **GPU 메모리** | ~50 GB/GPU | ~20 GB/GPU |
| **학습 시간** | 2~4시간 | 1~2시간 |
| **필요 데이터** | 200~1,000+ | 1,000~2,000 |
| **기본 모델 보존** | 덮어씌움 | 보존 (어댑터 분리) |
| **여러 도메인** | 모델별 저장 | 어댑터만 교체 |

### LoRA 학습 실행

```bash
# LoRA Post-Training (모델 전체 대신 LoRA 어댑터만 학습)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_sim2real_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    model.config.train_architecture=lora \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

핵심 차이: `model.config.train_architecture=lora` 파라미터 추가

### LoRA 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| Rank | 16 | LoRA rank (높을수록 표현력 증가, 메모리 사용 증가) |
| Alpha | 16 | LoRA scaling factor |
| Target modules | q_proj, k_proj, v_proj, output_proj, mlp | 적용 대상 레이어 |

### 언제 LoRA를 사용할까?

- **GPU 메모리가 제한적**일 때
- **여러 도메인**에 대해 각각 다른 모델이 필요할 때 (어댑터만 교체)
- **빠른 실험 반복**이 필요할 때
- **기본 모델의 범용 능력을 보존**하고 싶을 때

---

## 트러블슈팅

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**해결 방법 (우선순위 순):**

1. `context_parallel_size` 증가 (state_t의 약수여야 함):
   ```bash
   model.config.context_parallel_size=12  # 또는 24
   ```

2. `num_frames`와 `state_t` 함께 줄이기:
   ```bash
   # 93프레임 → 77프레임
   dataloader_train.dataset.num_frames=77 model.config.state_t=20

   # 93프레임 → 61프레임
   dataloader_train.dataset.num_frames=61 model.config.state_t=16
   ```

3. LoRA 사용하기:
   ```bash
   model.config.train_architecture=lora
   ```

### Video has only X frames, need Y

```
Video has only 45 frames, need 93
```

**해결 방법:**
- 93프레임 이상의 비디오 사용 (24fps 기준 ~4초)
- 또는 `num_frames`와 `state_t`를 줄이기:
  ```bash
  # 77프레임 (≥3.2초 필요)
  dataloader_train.dataset.num_frames=77 model.config.state_t=20

  # 61프레임 (≥2.5초 필요)
  dataloader_train.dataset.num_frames=61 model.config.state_t=16
  ```

### Training loss not decreasing

**확인 사항:**
1. 학습률이 적절한가? (기본값 5e-5, 필요 시 1e-5~1e-4 범위 시도)
2. 데이터 품질이 양호한가? (해상도, 프레임 수 확인)
3. 체크포인트가 올바르게 로드되었는가? (터미널 로그 확인)
4. 데이터 수량이 충분한가? (최소 50개 이상)

### 생성 영상에 아티팩트가 있음

**해결 방법:**
1. 학습 반복 횟수 증가
2. 더 높은 품질의 학습 데이터 사용
3. Inference 시 guidance scale 조정
4. `model_ema_bf16.pt` (EMA 체크포인트) 사용 확인

### 학습이 중간에 중단됨

학습은 자동으로 최신 체크포인트에서 재개됩니다. **동일한 명령어를 다시 실행**하세요.

---

## 핵심 파일 참조

| 파일 | 용도 |
|------|------|
| `cosmos_transfer2/singleview_config.py` | 학습 설정 진입점 |
| `cosmos_transfer2/experiments/singleview/cosmos_singleview_example.py` | 실험 설정 정의 |
| `projects/cosmos/transfer2/datasets/local_datasets/singleview_dataset.py` | 데이터셋 로더 |
| `projects/cosmos/transfer2/configs/vid2vid_transfer/defaults/dataloader_local.py` | 데이터로더 설정 |
| `scripts/prepare_videoufo_dataset.py` | VideoUFO 데이터셋 준비 스크립트 |
| `scripts/convert_distcp_to_pt.py` | 체크포인트 변환 스크립트 |

---

## 다음 단계

- [04. 이론 및 학습 자료](./04_theory_and_learning.md) - Post-Training 관련 핵심 이론과 개념 정리

## 관련 문서

- [01. Post-Training 개요 및 Sim2Real 분석](./01_post_training_overview_and_sim2real.md)
- [02. 데이터 준비 가이드](./02_data_preparation_guide.md)
- [공식 Post-Training 문서](../post-training_singleview.md)
- [공식 Inference 문서](../inference.md)
