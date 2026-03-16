# 이론 및 학습 자료 - Cosmos-Transfer2.5 Post-Training 핵심 개념

## 목차

1. [World Foundation Model (WFM)](#1-world-foundation-model-wfm)
2. [Diffusion Model 기초](#2-diffusion-model-기초)
3. [ControlNet과 Multi-ControlNet](#3-controlnet과-multi-controlnet)
4. [Latent Space와 Video Encoding](#4-latent-space와-video-encoding)
5. [FSDP와 분산 학습](#5-fsdp와-분산-학습)
6. [Context Parallelism](#6-context-parallelism)
7. [LoRA (Low-Rank Adaptation)](#7-lora-low-rank-adaptation)
8. [EMA (Exponential Moving Average)](#8-ema-exponential-moving-average)
9. [Learning Rate Warmup과 Scheduling](#9-learning-rate-warmup과-scheduling)
10. [Sim2Real Transfer Learning](#10-sim2real-transfer-learning)
11. [텍스트 인코딩과 Cross-Attention](#11-텍스트-인코딩과-cross-attention)
12. [DCP (Distributed Checkpoint)](#12-dcp-distributed-checkpoint)
13. [Control 신호 이해하기](#13-control-신호-이해하기)
14. [추가 학습 자료](#14-추가-학습-자료)

---

## 1. World Foundation Model (WFM)

### 개념

World Foundation Model(WFM)은 **실제 세계를 시뮬레이션하고 예측**하기 위해 설계된 대규모 비디오 생성 모델입니다. NVIDIA Cosmos는 이 WFM 패밀리에 속합니다.

### Cosmos 모델 패밀리 구조

```
Cosmos WFM 패밀리
├── Cosmos-Predict  : 미래 상태 예측 (비디오 → 다음 비디오)
├── Cosmos-Transfer : 스타일/도메인 전이 (Control 입력 → 비디오)
└── Cosmos-Reason   : 장면 이해 및 추론 (비디오 → 텍스트)
```

- **Cosmos-Predict2.5**: 비디오의 미래 프레임을 예측 (World Simulation)
- **Cosmos-Transfer2.5**: Control 신호를 기반으로 비디오를 변환 (Sim2Real, Style Transfer)
- **Cosmos-Reason**: 비디오 내용을 이해하고 설명 (Captioning, 추론)

### Transfer 모델의 역할

Transfer 모델은 **구조적 조건(Control)을 유지하면서 시각적 스타일을 변환**합니다. 이것이 바로 Sim2Real에 핵심적인 기능입니다.

```
[시뮬레이션 렌더링] ---(Control 신호 추출)--→ [Edge/Depth/Seg 맵]
                                                      ↓
                                             [Cosmos-Transfer2.5]
                                                      ↓
                                             [사실적 비디오 출력]
```

### 더 알아보기
- [arXiv 논문: World Simulation with Video Foundation Models](https://arxiv.org/abs/2511.00062)

---

## 2. Diffusion Model 기초

### 개념

Cosmos-Transfer2.5는 **Diffusion Model** 기반의 비디오 생성 모델입니다. Diffusion Model의 핵심 원리를 이해하면 Post-Training 과정을 더 잘 이해할 수 있습니다.

### Forward Process (노이즈 추가)

깨끗한 데이터에 점진적으로 노이즈를 추가하는 과정입니다.

```
[원본 영상] → [약간 노이즈] → [더 많은 노이즈] → ... → [순수 노이즈]
  x₀            x₁              x₂                      xₜ
```

### Reverse Process (노이즈 제거)

모델이 학습하는 것은 **노이즈를 제거하는 과정**입니다.

```
[순수 노이즈] → [노이즈 감소] → [더 깨끗] → ... → [생성된 영상]
  xₜ             xₜ₋₁            xₜ₋₂               x₀
```

### Post-Training에서의 의미

Post-Training은 모델이 **특정 도메인의 데이터에서 노이즈를 제거하는 방법**을 더 잘 학습하도록 합니다:
- 사전학습: "일반적인 비디오에서 노이즈를 제거하는 법" 학습
- Post-Training: "**특정 도메인** 비디오에서 노이즈를 제거하는 법" 추가 학습

### Training Loss와의 관계

학습 중 출력되는 **Loss 값**은 모델이 예측한 노이즈와 실제 추가된 노이즈 간의 차이입니다.
- Loss가 감소한다 = 모델이 더 정확하게 노이즈를 예측한다 = 더 나은 영상 생성

### 더 알아보기
- [Denoising Diffusion Probabilistic Models (DDPM) 원논문](https://arxiv.org/abs/2006.11239)
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

## 3. ControlNet과 Multi-ControlNet

### ControlNet 개념

ControlNet은 Diffusion Model에 **공간적 조건(spatial condition)**을 추가하는 기법입니다. 이를 통해 생성 결과의 구조를 제어할 수 있습니다.

```
일반 Diffusion Model:
[텍스트 프롬프트] → [모델] → [비디오] (구조가 매번 다름)

ControlNet 적용:
[텍스트 프롬프트] + [Control 신호] → [모델] → [비디오] (구조가 Control에 맞춤)
```

### ControlNet 아키텍처

```
                  ┌──────────────┐
[Control 입력] →  │  ControlNet  │ ──(조건 주입)──→ [메인 모델의 중간 레이어]
                  │  (복제된 인코더) │
                  └──────────────┘
                         ↑
                  사전학습 모델의 인코더를
                  복제하여 초기화
```

- ControlNet은 **사전학습된 모델의 인코더를 복제**하여 시작합니다
- 복제된 네트워크가 Control 신호를 처리하여 메인 모델에 조건을 주입합니다
- 메인 모델의 가중치는 고정하거나 함께 학습할 수 있습니다

### Multi-ControlNet

Cosmos-Transfer2.5는 **여러 종류의 Control 신호를 동시에** 처리할 수 있는 Multi-ControlNet 구조입니다.

```
[Edge 맵]   ──→ ┌─────────────────┐
[Depth 맵]  ──→ │ Multi-ControlNet│ ──→ [메인 모델] ──→ [비디오]
[Seg 맵]    ──→ │                 │
[텍스트]    ──→ └─────────────────┘
```

### Post-Training에서의 의미

Post-Training은 ControlNet 브랜치를 **사용자의 도메인에 적응**시킵니다:
- Control 신호와 실제 영상의 관계를 도메인에 맞게 재학습
- 해당 도메인에서의 시각적 표현력 향상

### 더 알아보기
- [Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet 원논문)](https://arxiv.org/abs/2302.05543)

---

## 4. Latent Space와 Video Encoding

### 왜 Latent Space를 사용하는가?

비디오는 데이터 크기가 매우 큽니다. 720p 93프레임 비디오를 직접 처리하면 계산 비용이 막대합니다.

```
원본 비디오:     1280 × 720 × 93 frames × 3 channels = ~256M pixels
Latent Space:  160 × 90 × 24 frames × channels      = 훨씬 작은 크기
```

### Latent Space 인코딩 과정

```
[원본 비디오]   ──(VAE 인코더)──→   [Latent 표현]   ──(Diffusion)──→   [변환된 Latent]
1280×720×93                        160×90×24                           160×90×24
                                                                          ↓
                                                              (VAE 디코더)
                                                                          ↓
                                                              [생성된 비디오]
                                                              1280×720×93
```

### 핵심 파라미터와의 관계

| 파라미터 | 의미 | 값 |
|---------|------|-----|
| `state_t=24` | Latent 시간 차원의 크기 | 24 latent frames |
| `num_frames=93` | 원본 비디오 프레임 수 | 93 pixel frames |
| 관계 | `num_frames = (state_t - 1) × 4 + 1` | (24-1)×4+1 = 93 |

이 수식에서 4는 **시간 축 다운샘플링 비율**입니다. VAE가 시간 축을 4배 압축합니다.

### 더 알아보기
- [Auto-Encoding Variational Bayes (VAE 원논문)](https://arxiv.org/abs/1312.6114)
- [Latent Diffusion Models (Stable Diffusion 기반 논문)](https://arxiv.org/abs/2112.10752)

---

## 5. FSDP와 분산 학습

### FSDP란?

FSDP (Fully Sharded Data Parallel)는 **대규모 모델을 여러 GPU에 걸쳐 분산 학습**하는 기법입니다. PyTorch에서 공식 지원합니다.

### 왜 FSDP가 필요한가?

2B 파라미터 모델은 하나의 GPU 메모리에 담을 수 없습니다:
- 모델 파라미터: ~4 GB (BF16)
- 옵티마이저 상태: ~16 GB
- 활성화 메모리: ~40+ GB
- **총계: 단일 GPU 80 GB로는 부족**

### FSDP 동작 방식

```
전통적 Data Parallel:
GPU 0: [전체 모델] + [데이터 1]
GPU 1: [전체 모델] + [데이터 2]  ← 모든 GPU에 전체 모델 복사
...

FSDP:
GPU 0: [모델 1/8] + [데이터 1]
GPU 1: [모델 2/8] + [데이터 2]  ← 모델을 GPU들에 분산
...
GPU 7: [모델 8/8] + [데이터 8]
```

- 각 GPU는 모델의 일부만 저장
- 계산 시 필요한 파라미터를 다른 GPU에서 가져옴 (all-gather)
- 그래디언트를 모든 GPU에 분산하여 업데이트 (reduce-scatter)

### Post-Training에서의 의미

- `--nproc_per_node=8`: 8개 GPU를 사용한 FSDP 학습
- 체크포인트는 **DCP (분산 체크포인트) 형식**으로 저장 → inference를 위해 consolidated 형식으로 변환 필요

### 더 알아보기
- [PyTorch FSDP 공식 문서](https://pytorch.org/docs/stable/fsdp.html)

---

## 6. Context Parallelism

### 개념

Context Parallelism은 **시퀀스(시간 축)를 여러 GPU에 분산**하는 기법입니다. FSDP가 모델 파라미터를 분산한다면, Context Parallelism은 **입력 데이터의 시간 축**을 분산합니다.

### 동작 방식

```
전체 시퀀스: [t1, t2, t3, t4, t5, t6, t7, t8, ... t24]  (state_t=24)

context_parallel_size=8 일 때:
GPU 0: [t1, t2, t3]     ← 3 latent frames
GPU 1: [t4, t5, t6]
GPU 2: [t7, t8, t9]
...
GPU 7: [t22, t23, t24]
```

### 핵심 제약 조건

```
state_t ÷ context_parallel_size = 정수 (나머지 0)
```

| state_t | 가능한 context_parallel_size |
|---------|---------------------------|
| 24 | 1, 2, 3, 4, 6, 8, 12, 24 |
| 20 | 1, 2, 4, 5, 10, 20 |
| 16 | 1, 2, 4, 8, 16 |

### GPU 메모리 절약 효과

`context_parallel_size`를 높이면 각 GPU가 처리하는 시퀀스 길이가 줄어 **메모리 사용량이 감소**합니다.

- `context_parallel_size=4`: GPU당 6 latent frames → 더 많은 메모리 사용
- `context_parallel_size=8`: GPU당 3 latent frames → 적당한 메모리 사용 (기본값)
- `context_parallel_size=24`: GPU당 1 latent frame → 최소 메모리, 통신 오버헤드 증가

---

## 7. LoRA (Low-Rank Adaptation)

### 핵심 아이디어

대규모 모델의 가중치 행렬 W를 직접 수정하는 대신, **작은 크기의 두 행렬(A, B)의 곱**으로 변화량을 근사합니다.

### 수학적 직관

```
기존 방식 (Full Fine-tuning):
W' = W + ΔW        (ΔW는 W와 같은 크기, 매우 큼)

LoRA 방식:
W' = W + A × B     (A, B는 작은 행렬)

예시 (W가 4096 × 4096 = 16.7M 파라미터인 경우):
- Full: ΔW = 4096 × 4096 = 16.7M 파라미터 학습
- LoRA (rank=16): A = 4096 × 16 = 65K, B = 16 × 4096 = 65K
                  → 총 130K 파라미터 학습 (128배 감소!)
```

### LoRA의 주요 파라미터

| 파라미터 | 의미 | Cosmos 기본값 |
|---------|------|-------------|
| **Rank (r)** | 저랭크 행렬의 차원. 높을수록 표현력 ↑, 메모리 ↑ | 16 |
| **Alpha (α)** | 스케일링 팩터. 보통 rank와 같거나 2배 | 16 |
| **Target Modules** | LoRA를 적용할 레이어 | q_proj, k_proj, v_proj, output_proj, mlp |

### 스케일링 공식

```
실제 적용: W' = W + (α/r) × A × B

α=16, r=16 → 스케일링 = 1.0
α=32, r=16 → 스케일링 = 2.0 (더 강한 적응)
```

### LoRA의 장점 (Post-Training 관점)

1. **메모리 효율**: GPU 메모리 ~60% 절감
2. **학습 속도**: Full Fine-tuning 대비 ~50% 빠름
3. **모듈성**: 여러 도메인에 대해 각각 LoRA 어댑터만 교체
4. **보존성**: 기본 모델의 범용 능력 보존

### 더 알아보기
- [LoRA: Low-Rank Adaptation of Large Language Models (원논문)](https://arxiv.org/abs/2106.09685)

---

## 8. EMA (Exponential Moving Average)

### 개념

EMA는 학습 과정에서 모델 파라미터의 **이동 평균**을 유지하는 기법입니다. 학습의 마지막 몇 스텝에서의 급격한 변동을 완화하여 **더 안정적인 모델**을 얻습니다.

### 수학적 정의

```
θ_ema(t) = β × θ_ema(t-1) + (1-β) × θ(t)

여기서:
- θ(t): 현재 스텝의 모델 파라미터
- θ_ema(t): 현재 EMA 파라미터
- β: decay rate (보통 0.999 ~ 0.9999)
```

### 직관적 이해

```
학습 중 파라미터 변화:
θ:     ~~~~∧∨~~~~∧∨~~~~  (노이즈가 섞인 궤적)
θ_ema: ────────────────  (부드러운 궤적)
```

EMA는 **최근 학습 결과에 가중치를 두면서도** 과거 결과를 잊지 않습니다.

### Post-Training에서의 의미

체크포인트 변환 시 생성되는 파일들:
- `model.pt`: 학습 마지막 스텝의 가중치 (노이즈 있음)
- `model_ema_bf16.pt`: EMA 가중치 (더 안정적) ← **Inference에 이것을 사용**
- `model_ema_fp32.pt`: EMA 가중치 (전체 정밀도)

---

## 9. Learning Rate Warmup과 Scheduling

### Warmup이 필요한 이유

학습 초기에 갑자기 높은 학습률을 적용하면 **그래디언트가 폭발**할 수 있습니다. 특히 사전학습된 모델을 fine-tuning할 때 주의가 필요합니다.

### Cosmos의 Warmup 스케줄

```
학습률
  ↑
5e-5 |            ┌────────────────────────────
     |           /
     |          /
     |         /
     |        /  ← Warmup 구간 (0~1000 iter)
     |       /
0    |______/
     └──────┬────┬──────────────────────── → Iteration
            0  1000                    5000
```

- **Warmup 구간 (0~1000 iter)**: 학습률이 0에서 5e-5까지 선형 증가
- **본 학습 구간 (1000+ iter)**: 학습률이 5e-5로 유지

### 파라미터

```python
lr = 5e-5                    # 최대 학습률
warm_up_steps = [1000]       # Warmup 스텝 수
```

### 학습률 조정 가이드

| 상황 | 조치 |
|------|------|
| Loss가 발산(NaN) | 학습률 감소 (예: 1e-5) |
| Loss가 매우 느리게 감소 | 학습률 증가 (예: 1e-4) |
| Loss가 초기에 급격히 진동 | Warmup 스텝 증가 (예: 2000) |

---

## 10. Sim2Real Transfer Learning

### Domain Gap 문제

시뮬레이션과 실제 환경 사이에는 **시각적 차이(Domain Gap)**가 존재합니다:

```
시뮬레이션                    실제 환경
├── 단순한 텍스처              ├── 복잡한 텍스처
├── 균일한 조명               ├── 자연스러운 조명/그림자
├── 깨끗한 경계선              ├── 노이즈, 블러
├── 완벽한 기하학              ├── 불완전한 표면
└── 제한된 다양성              └── 무한한 다양성
```

### Cosmos-Transfer2.5의 Sim2Real 접근 방식

```
┌─────────────────────────────────────────┐
│            Domain Gap 해소 전략            │
├─────────────────────────────────────────┤
│                                         │
│  1. [실제 데이터로 Post-Training]          │
│     → 모델이 타겟 도메인의 시각적 특성 학습  │
│                                         │
│  2. [시뮬레이션의 Control 신호 추출]        │
│     → 구조적 정보 (edge/depth/seg) 보존    │
│                                         │
│  3. [Inference로 변환]                    │
│     → 구조 유지 + 시각 스타일 변환          │
│                                         │
│  결과: 구조 = 시뮬레이션, 외관 = 실제 환경    │
└─────────────────────────────────────────┘
```

### 하이브리드 학습 전략 (X-Mobility 사례)

NVIDIA의 연구에 따르면, **원본 데이터 50% + Cosmos 변환 데이터 50%**의 하이브리드 학습이 가장 효과적입니다.

```
학습 데이터 구성:
├── 원본 시뮬레이션 데이터  50%  (구조적 다양성 유지)
└── Cosmos 변환 데이터     50%  (도메인 갭 해소)

→ 미션 성공률: 54% → 91% (+68.5%)
```

### 더 알아보기
- [Domain Randomization for Transferring Deep Neural Networks (Sim2Real 개요)](https://arxiv.org/abs/1710.06537)

---

## 11. 텍스트 인코딩과 Cross-Attention

### 텍스트가 비디오 생성에 미치는 영향

Cosmos-Transfer2.5는 **텍스트 설명(캡션)**을 사용하여 생성 방향을 가이드합니다.

### 인코딩 과정

```
"A robot in a warehouse"
         ↓
┌─────────────────────────┐
│  Qwen2.5-VL-7B          │  ← 텍스트 인코더
│  (reason1p1_7B)         │
│  28개 레이어의 임베딩 추출  │
└─────────────────────────┘
         ↓
[임베딩 벡터: 3584 × 28 = 100,352차원]
         ↓
┌─────────────────────────┐
│  Cross-Attention Projection │
│  100,352 → 1,024차원       │
└─────────────────────────┘
         ↓
[프로젝션된 조건 벡터] → 메인 모델의 Cross-Attention 레이어에 주입
```

### Cross-Attention 메커니즘

```
Query: 비디오 Latent 특징 (모델 내부)
Key:   텍스트 임베딩 (인코더 출력)
Value: 텍스트 임베딩 (인코더 출력)

Attention(Q, K, V) = softmax(QK^T / √d) × V
→ 비디오의 각 위치가 텍스트의 어떤 부분에 주목할지 결정
```

### Post-Training에서의 의미

- 캡션의 품질이 생성 결과에 직접적 영향
- **도메인 특화 용어**를 사용한 캡션이 더 나은 결과 생성
- 학습 시 텍스트 인코딩은 **on-the-fly**로 수행 (사전 계산 불필요)

### Cosmos의 텍스트 인코더 설정

```python
text_encoder_class = "reason1p1_7B"        # Qwen2.5-VL-7B
embedding_concat_strategy = "FULL_CONCAT"  # 28개 레이어 전체 결합
crossattn_proj_in_channels = 100352        # 3584 × 28
crossattn_emb_channels = 1024              # 프로젝션 후 차원
```

---

## 12. DCP (Distributed Checkpoint)

### DCP란?

DCP (Distributed Checkpoint)는 **분산 학습 환경에 최적화된 체크포인트 형식**입니다.

### DCP vs Consolidated Checkpoint

```
DCP 형식:                          Consolidated 형식:
checkpoints/                       model.pt (단일 파일)
├── iter_001000/
│   ├── model/
│   │   ├── .metadata
│   │   ├── __0_0.distcp           ← 파일이 여러 개로 분산
│   │   ├── __1_0.distcp
│   │   └── ...
│   ├── optim/                     ← 옵티마이저 상태 포함
│   ├── scheduler/
│   └── trainer/
└── latest_checkpoint.txt
```

| 특성 | DCP | Consolidated (.pt) |
|------|-----|-------------------|
| **파일 구조** | 다수의 분산 파일 | 단일 파일 |
| **용도** | 학습 중 저장/재개 | Inference, 모델 공유 |
| **I/O 성능** | 병렬 I/O로 빠름 | 순차 I/O |
| **FSDP 호환** | 직접 호환 | 변환 필요 |

### 변환 과정

```bash
# DCP → Consolidated 변환
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

# 결과:
# model.pt              ← 전체 가중치
# model_ema_fp32.pt     ← EMA (FP32)
# model_ema_bf16.pt     ← EMA (BF16, Inference용)
```

---

## 13. Control 신호 이해하기

### Edge (윤곽선)

```
원본 영상              Edge 맵
┌──────────┐          ┌──────────┐
│ 🏠  🌳   │    →     │ ╔══╗  ╱╲ │
│  🚗      │          │  ═══    │
│___🛣️_____│          │_________│
└──────────┘          └──────────┘
```

- **원리**: Canny Edge 또는 유사한 알고리즘으로 영상의 윤곽선 추출
- **장점**: 전처리 불필요 (on-the-fly 계산), 빠른 시작
- **적합한 경우**: 빠른 프로토타이핑, 윤곽선이 명확한 장면
- **한계**: 세밀한 구조 정보 부족

### Depth (깊이)

```
원본 영상              Depth 맵
┌──────────┐          ┌──────────┐
│ 🏠  🌳   │    →     │ ██  ▓▓  │  밝음=가까움
│  🚗      │          │  ▓▓     │  어두움=멀리
│___🛣️_____│          │░░░░░░░░░│
└──────────┘          └──────────┘
```

- **원리**: VideoDepthAnything 모델로 각 픽셀의 깊이 추정
- **장점**: 3D 공간 구조 보존, 로보틱스/자율주행에 적합
- **적합한 경우**: 정확한 3D 구조가 중요한 Sim2Real
- **전처리 필요**: DepthAnything V2 파이프라인 실행 필요

### Segmentation (분할)

```
원본 영상              Segmentation 맵
┌──────────┐          ┌──────────┐
│ 🏠  🌳   │    →     │ ██  ▒▒  │  각 객체별
│  🚗      │          │  ░░     │  고유한 색상
│___🛣️_____│          │▓▓▓▓▓▓▓▓│
└──────────┘          └──────────┘
```

- **원리**: SAM2 모델로 각 객체를 분리하여 색상 코딩
- **장점**: 객체 수준의 세밀한 제어, 어노테이션 보존에 최적
- **적합한 경우**: 객체 인식 학습용 데이터 생성
- **전처리 필요**: SAM2 파이프라인 실행 필요

### Visual Blur (시각적 흐림)

```
원본 영상              Visual Blur
┌──────────┐          ┌──────────┐
│ 🏠  🌳   │    →     │ ▒▒  ▒▒  │  전체적으로
│  🚗      │          │  ▒▒     │  흐리게 처리
│___🛣️_____│          │▒▒▒▒▒▒▒▒│
└──────────┘          └──────────┘
```

- **원리**: 원본 영상에 블러 처리
- **장점**: 전처리 불필요, 전반적 스타일 전이에 적합
- **적합한 경우**: 대략적인 구조만 유지하면 되는 경우

### Control 타입 선택 가이드 (Sim2Real 관점)

```
Sim2Real 정밀도 요구사항에 따른 선택:

낮은 정밀도 ←──────────────────────→ 높은 정밀도
  Visual Blur    Edge       Depth      Segmentation
  (스타일전이)   (윤곽선)    (3D구조)    (객체수준)
```

| 시뮬레이션 환경 | 권장 Control |
|---------------|------------|
| 로봇 매니퓰레이션 | Depth (3D 구조 보존 중요) |
| 자율주행 | Segmentation 또는 Depth |
| 실내 내비게이션 | Edge (빠른 시작) → Depth (품질 개선) |
| 스타일 전이 위주 | Visual Blur 또는 Edge |

---

## 14. 추가 학습 자료

### 논문

| 주제 | 논문 | 핵심 내용 |
|------|------|----------|
| Cosmos 모델 | [arXiv:2511.00062](https://arxiv.org/abs/2511.00062) | NVIDIA Cosmos WFM 전체 아키텍처 |
| Diffusion Models | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | DDPM 원논문 |
| ControlNet | [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) | ControlNet 원논문 |
| LoRA | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | LoRA 원논문 |
| Latent Diffusion | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | Stable Diffusion 기반 논문 |
| Sim2Real | [arXiv:1710.06537](https://arxiv.org/abs/1710.06537) | Domain Randomization |

### 온라인 자료

| 자료 | 링크 | 설명 |
|------|------|------|
| Cosmos Cookbook | [nvidia-cosmos.github.io/cosmos-cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/) | 공식 레시피 모음 |
| Cosmos 공식 문서 | [docs.nvidia.com/cosmos](https://docs.nvidia.com/cosmos/latest/) | 공식 API/가이드 문서 |
| Lil'Log Diffusion | [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Diffusion Model 튜토리얼 |
| PyTorch FSDP | [pytorch.org/docs](https://pytorch.org/docs/stable/fsdp.html) | 분산 학습 공식 문서 |
| HuggingFace | [huggingface.co/nvidia](https://huggingface.co/nvidia) | NVIDIA 모델 체크포인트 |

### 학습 순서 권장

```
1단계: 기본 개념 이해
├── Diffusion Model 기초 (Section 2)
└── Latent Space (Section 4)

2단계: 아키텍처 이해
├── ControlNet (Section 3)
└── 텍스트 인코딩 (Section 11)

3단계: 학습 기법 이해
├── FSDP와 분산 학습 (Section 5)
├── Context Parallelism (Section 6)
└── LoRA (Section 7)

4단계: 실전 최적화
├── EMA (Section 8)
├── LR Warmup (Section 9)
└── Sim2Real Transfer (Section 10)
```

---

## 관련 문서

- [01. Post-Training 개요 및 Sim2Real 분석](./01_post_training_overview_and_sim2real.md)
- [02. 데이터 준비 가이드](./02_data_preparation_guide.md)
- [03. Post-Training 실행 가이드](./03_post_training_execution_guide.md)
