# Cosmos Transfer 2.5 Single-View Post-Training 완벽 가이드

> **목적**: 이 가이드는 NVIDIA Cosmos Transfer 2.5 모델을 나만의 데이터에 맞게 post-training(후속 학습)하는 전체 과정을 누구나 따라할 수 있도록 상세하게 설명합니다.

---

## 목차

1. [Post-Training이란?](#1-post-training이란)
2. [사전 준비사항](#2-사전-준비사항)
3. [데이터 준비](#3-데이터-준비)
4. [학습 실행](#4-학습-실행)
5. [학습 모니터링](#5-학습-모니터링)
6. [체크포인트 관리 및 추론](#6-체크포인트-관리-및-추론)
7. [실전 시나리오별 가이드](#7-실전-시나리오별-가이드)
8. [트러블슈팅](#8-트러블슈팅)
9. [FAQ](#9-faq)
10. [참고 자료](#10-참고-자료)

---

## 1. Post-Training이란?

### 1.1 개요

Post-Training은 이미 대규모 데이터로 사전 학습(pre-training)된 foundation model을 **특정 도메인이나 사용 사례에 맞게 추가 학습**하는 과정입니다.

Cosmos Transfer 2.5에서 post-training의 주요 목적:

| 목적 | 설명 | 예시 |
|------|------|------|
| **도메인 적응** | 모델을 특정 분야의 데이터에 최적화 | 의료 영상, 자율주행, 로봇 시뮬레이션 |
| **스타일 전이** | 원하는 시각적 스타일로 영상 생성 | 애니메이션 스타일, 특정 카메라 톤 |
| **품질 향상** | 특정 유형의 영상 생성 품질 개선 | 실내 장면, 야간 영상 |
| **Sim2Real** | 시뮬레이션 영상을 실제와 유사하게 변환 | 가상 환경 → 실사 |

### 1.2 Cosmos Transfer 2.5 모델 구조 (간략)

```
텍스트 프롬프트 ──→ 텍스트 인코더(Qwen2.5-VL-7B) ──→ Cross-Attention
                                                          │
제어 입력(edge/depth/seg/vis) ──→ ControlNet 브랜치 ──→ DiT 기본 모델 ──→ 생성 영상
                                                          │
노이즈 ──────────────────────────────────────────→ Diffusion Denoising
```

- **기본 모델**: Diffusion Transformer (DiT) - 2B 파라미터
- **제어 브랜치**: ControlNet 구조로 각 modality별 독립 처리
- **텍스트 인코더**: Qwen2.5-VL-7B (reason1p1_7B)

### 1.3 Post-Training vs Fine-tuning

| 항목 | Post-Training (Full) | LoRA Fine-tuning |
|------|---------------------|------------------|
| 파라미터 업데이트 | 모든 파라미터 | 일부 저랭크 파라미터만 |
| 데이터 필요량 | 많음 (200+) | 적음 (50+) |
| 학습 시간 | 길음 | 짧음 |
| 성능 | 높음 | 중간 |
| GPU 메모리 | 많음 | 적음 |

> **참고**: 현재 Cosmos Transfer 2.5의 공식 post-training 가이드는 **Full Post-Training**을 지원합니다. LoRA는 Cosmos Predict 계열에서 지원됩니다.

---

## 2. 사전 준비사항

### 2.1 하드웨어 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| **GPU** | 8x A100 80GB | 8x H100 80GB |
| **시스템 메모리** | 256GB RAM | 512GB RAM |
| **저장 공간** | 200GB (데이터+체크포인트) | 1TB+ |
| **네트워크** | 10Gbps (멀티노드 시) | InfiniBand |

> ⚠️ **중요**: 8x H100/A100 (80GB)는 2B 모델의 **최소 요구사항**입니다. GPU 메모리가 부족하면 `context_parallel_size`를 조정하거나 `num_frames`를 줄여야 합니다.

### 2.2 소프트웨어 환경 설정

#### Step 1: 기본 환경 설치

```bash
# 레포지토리 클론
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5

# 의존성 설치 (setup.md 참고)
pip install -r requirements.txt
```

#### Step 2: Hugging Face 설정

모델 체크포인트는 학습 시 자동으로 HuggingFace에서 다운로드됩니다.

```bash
# Hugging Face 로그인 (모델 다운로드에 필요)
hf auth login

# (선택) 캐시 디렉토리 변경 (기본값: ~/.cache/huggingface)
export HF_HOME=/path/to/your/hf/cache
```

#### Step 3: 출력 디렉토리 설정

```bash
# 체크포인트와 학습 결과물이 저장될 디렉토리
# 기본값: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> ⚠️ `/tmp`는 재부팅 시 삭제되므로, 반드시 영구 저장소 경로를 설정하세요.

#### Step 4: (선택) Weights & Biases 설정

학습 과정을 시각적으로 모니터링하려면 W&B를 활성화합니다.

```bash
# W&B 활성화
export WANDB_API_KEY=your_api_key_here

# 또는 비활성화 (학습 커맨드에 추가)
# job.wandb_mode=disabled
```

---

## 3. 데이터 준비

### 3.1 데이터 준비 개요

Post-training의 성공은 **데이터 품질**에 크게 좌우됩니다. NVIDIA의 표현을 빌리면:

> *"The principle of 'garbage in, garbage out' is especially relevant in post-training."*

데이터 준비의 4가지 핵심 요소:
1. **도메인 정합성** - 목표 도메인의 특성과 변화를 반영하는 데이터
2. **품질 관리** - 적극적인 필터링으로 신호 관련성 보장
3. **규모와 커버리지** - 일반화를 위한 충분한 양과 다양성
4. **형식 일관성** - 학습 파이프라인과 호환되는 구조화된 데이터

### 3.2 빠른 시작: VideoUFO 데이터셋 사용

처음 시작하는 분은 VideoUFO 데이터셋으로 빠르게 테스트할 수 있습니다.

```bash
# 128개 비디오 다운로드 (테스트용, ~5-7 GB)
python scripts/prepare_videoufo_dataset.py \
    --storage_dir assets/videoufo \
    --num_videos 128
```

**스크립트 동작 순서:**
1. VideoUFO 메타데이터 CSV 다운로드 (~1.1 GB)
2. 비디오가 포함된 tar 파일 다운로드 (~4 GB/tar)
3. 실시간으로 비디오를 추출하며 93프레임 이상인 것만 필터링
4. 요청한 수의 유효한 비디오가 확보될 때까지 추출 계속
5. 캡션 JSON 파일 자동 생성
6. 필요한 디렉토리 구조로 자동 정리

**주요 옵션:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--storage_dir` | 데이터셋 저장 경로 | (필수) |
| `--num_videos` | 보관할 비디오 수 | 8 |
| `--num_tars` | 다운로드할 tar 파일 수 | 1 |
| `--min_frames` | 비디오당 최소 프레임 수 | 93 |
| `--use_brief_caption` | 간략한 캡션만 사용 | False |
| `--skip_download` | tar 파일이 있으면 다운로드 생략 | False |

**저장 공간 참고:**

| 비디오 수 | 예상 용량 |
|-----------|----------|
| 128개 | ~5-7 GB |
| 1,000개 | ~30 GB |
| 5,000개 | ~130 GB |

### 3.3 커스텀 데이터셋 준비 (나만의 데이터)

#### 3.3.1 필수 디렉토리 구조

```
datasets/my_dataset/
├── videos/                  # 필수: 원본 비디오
│   ├── scene_001.mp4
│   ├── scene_002.mp4
│   └── ...
├── captions/                # 필수: 비디오별 캡션
│   ├── scene_001.json
│   ├── scene_002.json
│   └── ...
├── depth/                   # 선택: depth 제어용 전처리 결과
│   ├── scene_001.mp4
│   └── ...
└── seg/                     # 선택: segmentation 제어용 전처리 결과
    ├── scene_001.mp4
    └── ...
```

> ⚠️ **중요**: `videos/`의 파일명과 `captions/`의 파일명이 반드시 일치해야 합니다.
> 예: `scene_001.mp4` ↔ `scene_001.json`

#### 3.3.2 비디오 요구사항

| 항목 | 요구사항 | 비고 |
|------|---------|------|
| **형식** | MP4 | H.264 코덱 권장 |
| **해상도** | 720p (1280×704) | 16:9 비율 |
| **길이** | 3초 이상 | 93프레임 이상 확보를 위해 4초+ 권장 |
| **프레임레이트** | 10-60 fps | 24fps 이상 권장 |
| **최소 프레임 수** | 93 프레임 | `num_frames` 파라미터와 일치 |

**비디오 품질 가이드라인:**
- 흔들림이 적고 선명한 영상 사용
- 워터마크나 자막이 없는 영상 권장
- 목표 도메인과 일치하는 컨텐츠
- 다양한 장면, 각도, 조명 조건 포함

#### 3.3.3 캡션 형식

각 비디오에 대응하는 JSON 파일을 생성합니다.

```json
{
    "caption": "A busy urban intersection at dusk with cars crossing and pedestrians walking on the sidewalk. Street lights illuminate the scene with warm orange light."
}
```

**좋은 캡션 작성 팁:**
- 장면의 주요 객체, 동작, 환경을 구체적으로 설명
- 조명 조건, 시간대, 날씨 등 환경 정보 포함
- 카메라 움직임이 있다면 설명 (예: "camera slowly pans left")
- 영어로 작성 (모델이 영어 캡션으로 사전 학습됨)

**캡션 자동 생성이 필요한 경우:**
대규모 데이터셋의 경우 비디오 캡셔닝 모델을 활용할 수 있습니다. Cosmos의 데이터 큐레이션 파이프라인에서는 5초 단위 윈도우로 분할하여 캡셔닝하는 방식을 사용합니다.

### 3.4 제어 입력(Control Input) 전처리

Cosmos Transfer 2.5는 4가지 제어 modality를 지원합니다.

#### 제어 타입 비교

| 제어 타입 | 전처리 필요 | 용도 | 특징 |
|-----------|------------|------|------|
| **Edge (엣지)** ⚡ | 불필요 (자동) | 구조/형태 보존 | 빠른 시작에 최적, Canny 엣지 감지 사용 |
| **Vis (시각적 블러)** ⚡ | 불필요 (자동) | 배경/조명 보존 | 스타일 전이, 디노이징에 적합 |
| **Depth (깊이)** | VideoDepthAnything | 3D 공간 일관성 | 로봇/자율주행 등 3D 인식 필요 시 |
| **Seg (세그멘테이션)** | SAM2 | 객체 단위 제어 | 객체 교체/합성에 적합 |

#### Edge 제어 (권장 - 전처리 불필요)

Edge는 학습 중 자동으로 Canny 엣지 감지가 적용되므로 **별도의 전처리가 필요 없습니다**. 처음 시작할 때 가장 추천하는 제어 타입입니다.

#### Depth 제어 전처리

```bash
# 단일 비디오 depth 맵 생성
python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
    --input_video datasets/my_dataset/videos/scene_001.mp4 \
    --output_video datasets/my_dataset/depth/scene_001.mp4 \
    --encoder vits

# 전체 비디오 일괄 처리
for video in datasets/my_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/my_dataset/depth/$basename" \
        --encoder vits
done
```

**Depth 파라미터:**
- `--encoder vits`: 작고 빠른 모델 (권장)
- `--encoder vitl`: 크고 정확한 모델
- **출력**: 입력과 동일한 해상도/프레임 수의 그레이스케일 depth 비디오

#### Segmentation 제어 전처리

```bash
# 텍스트 프롬프트 기반 세그멘테이션 (권장)
python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --input_video datasets/my_dataset/videos/scene_001.mp4 \
    --output_video datasets/my_dataset/seg/scene_001.mp4 \
    --mode prompt \
    --prompt "person, car, vehicle, building, tree, road, sky" \
    --visualize

# 전체 비디오 일괄 처리
for video in datasets/my_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/my_dataset/seg/$basename" \
        --mode prompt \
        --prompt "person, car, vehicle, building, tree, road, sky" \
        --visualize
done
```

**세그멘테이션 모드:**

| 모드 | 사용법 | 적합한 경우 |
|------|--------|------------|
| **prompt** (권장) | `--mode prompt --prompt "person, car"` | 일반적인 객체 세그멘테이션 |
| **box** | `--mode box --box "300,0,500,400"` | 특정 영역의 객체 |
| **points** | `--mode points --points "200,300" --labels "1"` | 정확한 위치 지정 |

### 3.5 데이터셋 검증 체크리스트

학습 시작 전 다음을 확인하세요:

- [ ] `videos/` 폴더에 MP4 파일들이 존재
- [ ] `captions/` 폴더에 대응하는 JSON 파일들이 존재
- [ ] 파일명이 일치 (video1.mp4 ↔ video1.json)
- [ ] 모든 비디오가 93프레임 이상
- [ ] 캡션 JSON이 유효한 형식 (`{"caption": "..."}`)
- [ ] (depth 사용 시) `depth/` 폴더에 전처리된 MP4 존재
- [ ] (seg 사용 시) `seg/` 폴더에 전처리된 MP4 존재

---

## 4. 학습 실행

### 4.1 지원하는 실험(Experiment) 타입

| 실험명 | 제어 타입 | 전처리 | 비고 |
|--------|----------|--------|------|
| `transfer2_singleview_posttrain_edge_example` | Edge | 불필요 | ⭐ 추천 |
| `transfer2_singleview_posttrain_depth_example` | Depth | 필요 | 3D 인식 |
| `transfer2_singleview_posttrain_seg_example` | Seg | 필요 | 객체 제어 |
| `transfer2_singleview_posttrain_vis_example` | Vis | 불필요 | 스타일 전이 |

### 4.2 핵심 학습 파라미터

```python
# === 데이터셋 ===
dataset_dir = "datasets/my_dataset"         # 데이터셋 경로
num_frames = 93                              # 필수 프레임 수: (state_t-1)*4+1
video_size = (704, 1280)                     # 높이×너비, 720p 16:9

# === 모델 ===
state_t = 24                                 # 시간 축 잠재 공간 크기
context_parallel_size = 8                    # GPU 간 시퀀스 분할 수 (state_t의 약수)

# === 텍스트 인코더 ===
text_encoder_class = "reason1p1_7B"          # Qwen2.5-VL-7B
embedding_concat_strategy = "FULL_CONCAT"    # 28개 레이어 전체 사용
crossattn_proj_in_channels = 100352          # 3584 * 28
crossattn_emb_channels = 1024               # 투영 차원

# === 학습 하이퍼파라미터 ===
max_iter = 5000                              # 총 학습 반복 횟수
save_iter = 1000                             # 체크포인트 저장 주기
lr = 5e-5                                    # 학습률
warm_up_steps = 1000                         # LR 워밍업 (그래디언트 폭발 방지)
grad_accum_iter = 4                          # 그래디언트 누적 횟수
```

**파라미터 간 관계:**
```
num_frames = (state_t - 1) × 4 + 1
예: state_t=24 → num_frames = (24-1)×4+1 = 93

context_parallel_size는 반드시 state_t의 약수여야 함
예: state_t=24의 약수 = 1, 2, 3, 4, 6, 8, 12, 24
```

### 4.3 학습 실행 커맨드

#### Edge 제어 (권장 - 입문자용)

```bash
# 환경 변수 설정
export IMAGINAIRE_OUTPUT_ROOT=/path/to/outputs

# Edge 제어로 학습 시작
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

#### Depth 제어

```bash
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_depth_example \
    dataloader_train.dataset.dataset_dir=datasets/my_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled
```

#### Segmentation 제어

```bash
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_seg_example \
    dataloader_train.dataset.dataset_dir=datasets/my_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled
```

#### Visual Blur 제어

```bash
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_vis_example \
    dataloader_train.dataset.dataset_dir=datasets/my_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled
```

### 4.4 커맨드 라인 파라미터 커스터마이징

```bash
# 자주 조정하는 파라미터들
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \              # 학습 반복 횟수
    checkpoint.save_iter=500 \           # 체크포인트 저장 주기
    optimizer.lr=5e-5 \                  # 학습률
    scheduler.warm_up_steps=[1000] \     # 워밍업 스텝
    job.wandb_mode=online                # W&B 활성화
```

**필수 파라미터:**
- `dataloader_train.dataset.dataset_dir`: 데이터셋 디렉토리 경로
- `'dataloader_train.sampler.dataset=${dataloader_train.dataset}'`: 멀티 GPU 데이터 분배 연결

**선택 파라미터:**
- `trainer.max_iter`: 총 학습 반복 횟수 (기본: 5000)
- `checkpoint.save_iter`: 체크포인트 저장 빈도 (기본: 1000)
- `optimizer.lr`: 학습률 (기본: 5e-5)
- `scheduler.warm_up_steps`: LR 워밍업 스텝 (기본: [1000])
- `job.wandb_mode`: W&B 모드 (online/offline/disabled)
- `checkpoint.load_path`: 커스텀 체크포인트 경로

### 4.5 학습 시 자동으로 일어나는 일

1. **체크포인트 자동 다운로드**: HuggingFace에서 사전 학습된 모델을 자동으로 다운로드
2. **제어 입력 생성**: Edge/Vis는 학습 중 자동으로 on-the-fly 생성
3. **텍스트 인코딩**: 캡션이 Qwen2.5-VL-7B로 실시간 인코딩
4. **샘플 생성**: 50 iterations마다 샘플 영상 자동 생성
5. **체크포인트 저장**: 설정된 주기마다 DCP 형식으로 자동 저장

---

## 5. 학습 모니터링

### 5.1 터미널 출력

학습이 정상적으로 진행되면 다음과 같은 로그가 출력됩니다:

```
[Iteration 100/2000] Loss: 0.234, LR: 4.95e-05, Time: 1.23s/it
[Iteration 200/2000] Saving checkpoint...
[Iteration 250/2000] Loss: 0.189, LR: 5.00e-05, Time: 1.21s/it
```

### 5.2 정상 학습 판단 기준

| 지표 | 정상 | 비정상 |
|------|------|--------|
| **Loss** | ~0.5에서 시작 → ~0.1-0.2로 감소 | 감소하지 않거나 NaN/Inf 발생 |
| **학습률** | 워밍업 후 안정적 유지 | 급격한 변동 |
| **샘플 품질** | iteration이 진행될수록 개선 | 아티팩트 증가 또는 변화 없음 |
| **시간/반복** | 일정하게 유지 (~1.2-1.5s) | 갑자기 증가 |

### 5.3 샘플 확인

학습 중 생성되는 샘플은 다음 경로에 저장됩니다:

```
${IMAGINAIRE_OUTPUT_ROOT}/<project>/<group>/<name>/samples/
```

기본값:
```
${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/local_single_view/
    transfer2_singleview_posttrain_edge_example_<timestamp>/samples/
```

50 iterations마다 자동으로 생성되며, 학습 품질을 시각적으로 확인할 수 있습니다.

### 5.4 예상 학습 시간

| 데이터셋 크기 | 모델 | GPU | 시간/반복 | 2000 반복 총 시간 |
|-------------|------|-----|---------|----------------|
| 128 비디오 | 2B | 8x A100 | ~1.2s | ~0.7시간 |
| 1,000 비디오 | 2B | 8x A100 | ~1.5s | ~0.8시간 |

---

## 6. 체크포인트 관리 및 추론

### 6.1 체크포인트 형식

Cosmos Transfer 2.5는 두 가지 체크포인트 형식을 사용합니다:

| 형식 | 용도 | 특징 |
|------|------|------|
| **DCP (Distributed Checkpoint)** | 학습 중 저장/재개 | 다중 파일, 분산 I/O 최적화 |
| **PyTorch (.pt)** | 추론, 모델 공유 | 단일 파일, 사용 편리 |

### 6.2 체크포인트 디렉토리 구조

```
${IMAGINAIRE_OUTPUT_ROOT}/
└── cosmos_transfer2_posttrain/
    └── local_single_view/
        └── transfer2_singleview_posttrain_edge_example_2025-11-19_10-30-00/
            ├── checkpoints/
            │   ├── iter_000000500/
            │   │   ├── model/
            │   │   │   ├── .metadata
            │   │   │   └── __0_0.distcp
            │   │   ├── optim/
            │   │   ├── scheduler/
            │   │   └── trainer/
            │   ├── iter_000001000/
            │   ├── iter_000001500/
            │   ├── iter_000002000/
            │   │   └── model_ema_bf16.pt  ← 변환 후 생성
            │   └── latest_checkpoint.txt
            └── samples/
                ├── iter_000000050/
                ├── iter_000000100/
                └── ...
```

### 6.3 DCP → PyTorch 변환

학습이 완료되면 추론용으로 체크포인트를 변환합니다:

```bash
# 체크포인트 경로 설정
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_edge_example_*/checkpoints

# 최신 체크포인트 찾기
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# 변환 실행
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

**변환 결과:**
- `model_ema_bf16.pt` ← **추론에 사용** (권장)
- `model_ema_fp32.pt` ← FP32 정밀도 EMA 가중치
- `model.pt` ← 전체 체크포인트

### 6.4 학습 재개

학습이 중단된 경우, 동일한 커맨드를 다시 실행하면 **최신 체크포인트에서 자동으로 재개**됩니다.

```bash
# 동일한 커맨드 재실행 → 자동 재개
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}'
```

### 6.5 추론 (Inference)

학습된 모델로 새로운 영상을 생성합니다:

```bash
torchrun --nproc_per_node=8 examples/inference.py \
    -i assets/edge.jsonl \
    -o outputs/ \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment transfer2_singleview_posttrain_edge_example
```

---

## 7. 실전 시나리오별 가이드

### 7.1 시나리오 1: 로봇 시뮬레이션 (Sim2Real)

**목표**: 시뮬레이션 환경의 영상을 실제 환경처럼 변환

```bash
# 1. 시뮬레이션 영상과 실제 환경 영상 모두 데이터셋에 포함
datasets/robot_sim2real/
├── videos/      # 실제 환경 비디오들
├── captions/    # 캡션
└── depth/       # (선택) depth 맵

# 2. Depth 제어로 학습 (3D 공간 일관성 유지)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_depth_example \
    dataloader_train.dataset.dataset_dir=datasets/robot_sim2real \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000
```

### 7.2 시나리오 2: 자율주행 데이터 증강

**목표**: 다양한 기상 조건/시간대의 주행 영상 생성

```bash
# 1. 주행 영상 데이터셋 준비 (다양한 조건의 영상 포함)
datasets/driving/
├── videos/      # 주행 비디오들
├── captions/    # "Driving on a highway during heavy rain at night" 등
└── seg/         # (선택) 도로/차량/보행자 세그멘테이션

# 2. Edge 제어로 학습
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/driving \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000
```

### 7.3 시나리오 3: 특정 스타일의 영상 생성

**목표**: 특정 시각적 스타일의 영상 생성 (예: 애니메이션, 특정 영화 톤)

```bash
# 1. 원하는 스타일의 영상 수집
datasets/style_transfer/
├── videos/      # 원하는 스타일의 비디오들
└── captions/    # 스타일 설명 포함 캡션

# 2. Visual Blur 제어로 학습 (스타일 전이에 적합)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_vis_example \
    dataloader_train.dataset.dataset_dir=datasets/style_transfer \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=3000
```

---

## 8. 트러블슈팅

### 8.1 "CUDA out of memory"

**원인**: GPU 메모리 부족

**해결 방법 (우선순위순):**

1. `context_parallel_size` 증가 (state_t의 약수여야 함):
   ```bash
   model.config.context_parallel_size=12   # 또는 24
   ```

2. `num_frames`와 `state_t` 함께 줄이기:
   ```bash
   # 93 → 77 프레임
   dataloader_train.dataset.num_frames=77 model.config.state_t=20

   # 93 → 61 프레임
   dataloader_train.dataset.num_frames=61 model.config.state_t=16
   ```

3. 더 많은 GPU 사용하여 메모리 분산

4. 체크포인트 저장 빈도 줄이기 (`checkpoint.save_iter` 증가)

### 8.2 "Video has only X frames, need Y"

**원인**: 비디오 프레임 수 부족

**해결 방법:**
- 4초 이상의 비디오 사용 (24fps 기준 93프레임 이상)
- `num_frames`와 `state_t`를 함께 줄이기:
  ```bash
  # 77 프레임 (3.2초 이상 필요)
  dataloader_train.dataset.num_frames=77 model.config.state_t=20

  # 61 프레임 (2.5초 이상 필요)
  dataloader_train.dataset.num_frames=61 model.config.state_t=16
  ```
- 전처리 단계에서 짧은 비디오 필터링

### 8.3 학습 Loss가 감소하지 않음

**확인 사항:**
1. 학습률 확인 (너무 높거나 낮지 않은지)
2. 데이터 품질 검증 (비디오와 캡션이 올바른지)
3. 체크포인트가 정상적으로 로드되었는지 확인
4. 데이터 양 늘리기 (최소 128개 이상 권장)

### 8.4 생성 영상에 아티팩트 발생

**해결 방법:**
1. 학습을 더 오래 진행 (iteration 수 증가)
2. 고품질 학습 데이터 사용
3. 추론 시 guidance scale 조정
4. EMA 체크포인트 사용 (`model_ema_bf16.pt`)

### 8.5 캐시 디스크 공간 부족 (컨테이너 환경)

```bash
# HuggingFace 캐시 경로 변경
export HF_HOME=/mnt/large_disk/hf_cache

# 출력 디렉토리 변경
export IMAGINAIRE_OUTPUT_ROOT=/mnt/large_disk/cosmos_output
```

---

## 9. FAQ

### Q: 어떤 제어 타입으로 시작해야 하나요?
**A: Edge 제어**를 추천합니다. 전처리가 필요 없고, 빠르게 반복 실험이 가능하며, 메모리 설정이 최적화되어 있습니다.

### Q: 데이터가 얼마나 필요한가요?

| 목적 | 데이터 수 | 비고 |
|------|----------|------|
| 테스트/검증 | 50-100개 | 파이프라인 동작 확인용 |
| 기본 학습 | 200-500개 | 합리적인 결과 |
| 최적 학습 | 1,000개+ | 최상의 결과 |

### Q: context_parallel_size가 뭔가요?
시퀀스(시간 축)를 여러 GPU에 분산하여 GPU당 메모리 사용량을 줄이는 파라미터입니다.
- `state_t=24 ÷ context_parallel_size=8 = GPU당 3개의 잠재 프레임`
- 반드시 `state_t`의 약수여야 합니다.

### Q: 학습 중간에 멈추고 다시 시작할 수 있나요?
**네**, 동일한 학습 커맨드를 다시 실행하면 마지막 체크포인트에서 자동으로 재개됩니다.

### Q: 멀티노드 학습이 가능한가요?
공식 가이드에서는 단일 노드 8 GPU 설정을 기준으로 설명하고 있습니다. 멀티노드 설정은 `torchrun`의 `--nnodes`, `--node_rank`, `--master_addr` 등의 옵션을 활용할 수 있습니다.

### Q: 사전 학습된 모델은 어디서 다운로드하나요?
**자동으로 다운로드됩니다.** 학습 시작 시 HuggingFace에서 자동으로 다운로드되므로 수동 다운로드가 필요 없습니다. HuggingFace 로그인만 되어 있으면 됩니다.

---

## 10. 참고 자료

### 공식 문서
- [Cosmos Transfer 2.5 Post-Training Guide](https://docs.nvidia.com/cosmos/latest/transfer2.5/post-training/post-training_guide.html)
- [Cosmos Transfer 2.5 GitHub](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html)

### 주요 코드 파일
- 학습 실험 정의: `cosmos_transfer2/experiments/singleview/cosmos_singleview_example.py`
- 설정 래퍼: `cosmos_transfer2/singleview_config.py`
- 데이터셋 로더: `projects/cosmos/transfer2/datasets/local_datasets/singleview_dataset.py`
- 데이터로더 설정: `projects/cosmos/transfer2/configs/vid2vid_transfer/defaults/dataloader_local.py`
- VideoUFO 준비 스크립트: `scripts/prepare_videoufo_dataset.py`

### 관련 가이드
- [환경 설정 가이드](setup.md)
- [추론 가이드](inference.md)
- [멀티뷰 학습](post-training_auto_multiview.md)
- [Post-Training 개요](post-training.md)

### 모델 체크포인트
- [Cosmos-Transfer2.5-2B (HuggingFace)](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)

### 데이터셋
- [VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO) - 1M+ 비디오 (학습용 데이터셋 예시)

### 학습 자료
- [이론 및 학습 문서](study_post-training_theory_ko.md) - Post-Training의 이론적 배경과 핵심 개념
