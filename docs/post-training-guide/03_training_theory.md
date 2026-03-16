# 03. 학습 이론: Rectified Flow, ControlNet-VACE, 조건부 학습

이 문서는 Cosmos Transfer 2.5 사후학습의 이론적 배경을 설명한다. Flow Matching/ControlNet에 대한 기초 지식을 전제로, 이 코드베이스에서 **구체적으로 어떤 변형이 사용되는지**에 초점을 맞춘다.

---

## 1. Rectified Flow 복습과 변형

### 1.1 기본 Rectified Flow

데이터 분포 `x_0`와 노이즈 분포 `ε ~ N(0, I)` 사이의 직선 보간:

```
x_t = (1 - t) · x_0 + t · ε,    t ∈ [0, 1]
```

**Velocity field**: 이 직선의 기울기를 예측

```
v(x_t, t) = dx_t/dt = ε - x_0
```

모델은 `v_θ(x_t, t)`로 이 velocity를 근사한다.

### 1.2 Cosmos Transfer 2.5에서의 구체적 구현

#### 시간 스케일링

코드에서 `t ∈ [0, 1]`을 `timestep ∈ [0, 1000]`으로 매핑한다:

```python
# 1. LogitNormal에서 u 샘플링
u = torch.sigmoid(torch.randn(batch_size))  # u ∈ (0, 1)

# 2. Time-shift 적용 (shift = 5)
timestep = shift * u / (1 + (shift - 1) * u) * 1000
# shift=5일 때: 낮은 노이즈 레벨(t≈0 부근)에 더 많은 샘플 할당

# 3. Sigma로 변환
sigma = timestep / 1000  # σ ∈ [0, 1]
```

#### LogitNormal 시간 분포

```
u ~ sigmoid(N(0, 1))
```

이 분포의 특성:
- 중간값(t≈0.5) 부근에 가장 많은 샘플
- 극단값(t≈0, t≈1)에는 적은 샘플
- 학습 효율을 위해 "가장 어려운" 중간 노이즈 레벨에 집중

```
                ▲ 확률 밀도
                │       ╱╲
                │      ╱  ╲
                │     ╱    ╲
                │    ╱      ╲
                │   ╱        ╲
                │──╱──────────╲──▶ t
                0    0.5      1
```

Time-shift(shift=5)를 적용하면 분포가 왼쪽으로 이동하여, **덜 노이즈가 추가된 상태(낮은 t)**에서 더 많이 학습한다. 이는 디테일 복원 능력을 강화한다.

### 1.3 노이즈 보간과 타겟

```python
# Latent 공간에서의 보간
x_t = (1 - sigma) * x_0 + sigma * epsilon  # 노이즈 보간

# 타겟 velocity
v_target = epsilon - x_0  # 실제 velocity field

# 모델 예측
v_pred = model.net(x_t, timestep, condition)  # DiT가 예측한 velocity

# 손실 함수
loss = MSE(v_pred, v_target)  # L2 velocity prediction loss
```

---

## 2. ControlNet-VACE 아키텍처

### 2.1 VACE란?

**VACE** = **V**ideo **A**daptive **C**onditioning and **E**diting

일반적인 ControlNet과의 차이점:

| 특성 | 일반 ControlNet | VACE |
|------|----------------|------|
| 컨트롤 블록 수 | 모든 레이어 | 일부 레이어 (every_n개 간격) |
| 조건 프레임 | 고정 | **랜덤** (0, 1, 2 중 확률적 선택) |
| 마스크 지원 | 없음 | **공간-시간 마스크** 내장 |
| 멀티 컨트롤 | 별도 모델 | **다중 브랜치** 지원 |

### 2.2 컨트롤 블록 구성

Base DiT에 28개 블록이 있고, `vace_block_every_n=7`이면:

```
Base DiT 블록:     [B0, B1, B2, ..., B27]  (28개)
Control 블록:      [C0, C1, C2, C3]        (4개)

매핑:
C0 → B0에 hint 주입  (block 0)
C1 → B7에 hint 주입  (block 7)
C2 → B14에 hint 주입 (block 14)
C3 → B21에 hint 주입 (block 21)
```

**Control 블록의 초기화:**

사후학습 시작 시, Control 블록은 Base DiT의 가중치로 초기화된다:

```python
# "first_n" 전략 (기본):
C0 ← B0의 가중치 복사
C1 ← B1의 가중치 복사
C2 ← B2의 가중치 복사
C3 ← B3의 가중치 복사

# "spaced_n" 전략 (대안):
C0 ← B0의 가중치 복사
C1 ← B7의 가중치 복사
C2 ← B14의 가중치 복사
C3 ← B21의 가중치 복사
```

> 사전 학습된 체크포인트에서 시작하므로 이 초기화는 이미 완료된 상태이다. 하지만 새로운 학습(is_new_training=True)이면 이 과정이 실행된다.

### 2.3 Hint 주입 메커니즘

```
Control Branch:                    Main DiT Branch:

C_input → [Control Block 0] ─→ hint_0 ─→ ⊕ ─→ [Base Block 0]
    ↓                                         [Base Block 1]
    ↓                                         ...
    ↓    → [Control Block 1] ─→ hint_1 ─→ ⊕ ─→ [Base Block 7]
    ↓                                         [Base Block 8]
    ↓                                         ...
    ↓    → [Control Block 2] ─→ hint_2 ─→ ⊕ ─→ [Base Block 14]
    ↓                                         ...
    ↓    → [Control Block 3] ─→ hint_3 ─→ ⊕ ─→ [Base Block 21]
                                               ...
                                               [Base Block 27]
                                               → output
```

**주입 공식:**
```
x_out = Base_Block(x_in) + control_weight × hint
```

- `control_weight ∈ [0, 1]`: 스칼라 또는 공간-시간 가중치 맵
- 마스크가 있으면 가중치가 공간적으로 가변적

---

## 3. 조건부 프레임 학습 (VACE Conditioning)

### 3.1 학습 시 조건 프레임 랜덤 선택

모델이 다양한 조건부 시나리오에서 동작할 수 있도록, **매 이터레이션마다 조건 프레임 수를 확률적으로 선택**한다:

```python
conditional_frames_probs = {0: 0.4, 1: 0.4, 2: 0.2}

# 매 배치마다:
# 40% 확률 → 0 프레임 (비조건부: 순수 생성)
# 40% 확률 → 1 프레임 (첫 프레임을 조건으로)
# 20% 확률 → 2 프레임 (첫 2 프레임을 조건으로)
```

### 3.2 조건 마스크 생성

```python
# 예: 1 조건 프레임 선택됨
# latent shape: (1, 16, 24, 44, 80)

condition_mask = torch.zeros(1, 1, 24, 44, 80)  # 전체 0
condition_mask[:, :, :1, :, :] = 1                # 첫 latent 프레임만 1

# 이 마스크가 DiT의 입력에 추가 채널로 결합됨
# → 모델은 "어디가 조건이고 어디를 생성해야 하는지" 학습
```

### 3.3 왜 이렇게 하는가?

추론 시 다양한 시나리오를 지원하기 위해:

| 조건 프레임 수 | 추론 시 용도 |
|-------------|-------------|
| 0 | 첫 번째 청크 생성 (이전 출력 없음) |
| 1 | 두 번째+ 청크 (이전 청크의 마지막 1 프레임으로 연결) |
| 2 | 더 안정적인 시간 연속성 필요 시 |

학습에서 이 세 가지를 모두 경험하므로, 추론 시 어떤 모드로든 자연스럽게 동작한다.

---

## 4. Freeze 전략: 무엇을 학습하고 무엇을 동결하는가

### 4.1 동결 (학습하지 않는 파라미터)

```python
# Base DiT의 모든 파라미터
self.net.blocks[*]              # 28개 메인 트랜스포머 블록
self.net.x_embedder             # 입력 패치 임베더
self.net.t_embedder             # 시간 임베더
self.net.final_layer            # 최종 출력 레이어
self.tokenizer                  # VAE (인코더/디코더)
# ... 기타 base model 구성요소
```

### 4.2 학습 대상 (학습하는 파라미터)

```python
# ControlNet 브랜치
self.net.control_blocks[*]                      # 4개 컨트롤 블록
self.net.control_embedder                       # 컨트롤 입력 임베더

# 별도 임베더 (separate_embedders=True인 경우)
self.net.t_embedder_for_control_branch          # 시간 임베더 (컨트롤 전용)
self.net.t_embedding_norm_for_control_branch    # 시간 임베딩 노름
self.net.x_embedder_for_control_branch          # 입력 임베더 (컨트롤 전용)
```

### 4.3 Freeze 코드

```python
def freeze_base_model(self):
    # 1. 전체 모델 동결
    for name, param in self.net.named_parameters():
        param.requires_grad = False

    # 2. 컨트롤 블록만 해동
    if self.net.num_control_branches == 1:
        for param in self.net.control_blocks.parameters():
            param.requires_grad = True
    else:
        for nc in range(self.net.num_control_branches):
            for param in getattr(self.net, f"control_blocks_{nc}").parameters():
                param.requires_grad = True

    # 3. 컨트롤 임베더 해동
    for param in self.net.control_embedder.parameters():
        param.requires_grad = True

    # 4. (선택) 별도 임베더 해동
    if self.net.separate_embedders:
        for param in self.net.t_embedder_for_control_branch.parameters():
            param.requires_grad = True
        for param in self.net.x_embedder_for_control_branch.parameters():
            param.requires_grad = True
```

---

## 5. CFG 학습

### 5.1 학습 시 CFG 조건

매 이터레이션에서 conditioner가 condition/uncondition 쌍을 만든다:

```python
# Condition (조건부): 모든 조건 사용
condition = conditioner(data_batch, dropout_rate={text: 0.0, control: 0.0})

# Uncondition (비조건부): 텍스트만 드롭 (20% 확률)
uncondition = conditioner(data_batch, dropout_rate={text: 1.0, control: 0.0})
```

**중요**: 학습 시에는 condition과 uncondition을 별도로 계산하지 않는다. 대신 conditioner의 **드롭아웃**을 통해 CFG를 내재적으로 학습한다:

- 텍스트: 20% 확률로 null → 모델이 텍스트 없이도 학습
- 컨트롤: 드롭하지 않음 (0%) → 컨트롤은 항상 사용

### 5.2 추론 시 CFG

```python
v_cond = model(x_t, t, condition)      # 텍스트 + 컨트롤
v_uncond = model(x_t, t, uncondition)  # 텍스트 없음 + 컨트롤
v_final = v_uncond + guidance * (v_cond - v_uncond)
```

---

## 6. EMA (Exponential Moving Average)

### 6.1 Power EMA

Cosmos Transfer 2.5는 **Power EMA**를 사용한다. 일반적인 고정 β EMA와 다르게, β가 이터레이션에 따라 변한다:

```
β(t) = (1 - 1/(t+1))^(exp+1)
```

여기서 `exp`는 `rate` 파라미터(기본 0.1)로부터 계산된다.

**특성:**
- 초기(t 작을 때): β ≈ 0 → 현재 가중치에 가까움
- 후기(t 클 때): β → 1 → 변화에 둔감 (안정적)

### 6.2 EMA 업데이트 시점

```python
# 매 optimizer.step() 직전에 호출
model.on_before_zero_grad(iteration)
    → ema_updater.step(model.net, iteration)
        → ema_weight = β * ema_weight + (1-β) * current_weight
```

### 6.3 체크포인트에서의 EMA

체크포인트에는 2종류의 가중치가 저장된다:
- `model.pt`: 현재 학습 가중치
- `model_ema_bf16.pt`: EMA 가중치 ← **추론에 사용**

---

## 7. 사후학습 vs 처음부터 학습

| 측면 | 사후학습 (Fine-tuning) | 처음부터 학습 |
|------|----------------------|-------------|
| 시작점 | 사전 학습된 체크포인트 | 랜덤 초기화 |
| 학습 대상 | ControlNet 부분만 | 전체 모델 |
| Base DiT | **동결** | 학습 |
| 데이터 양 | 50~1000 영상 | 수십만~수백만 영상 |
| 이터레이션 | 2000~5000 | 수십만~수백만 |
| GPU 시간 | 1~2시간 (8xA100) | 수주~수개월 |
| 목적 | 도메인 적응 | 새 모델 학습 |

사후학습은 NVIDIA가 대규모 데이터로 사전 학습한 base model의 지식을 활용하면서, **내 도메인의 특성(스타일, 객체, 환경 등)**을 ControlNet에 학습시키는 것이다.
