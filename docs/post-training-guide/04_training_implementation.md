# 04. 코드 레벨 학습 구현

이 문서는 실제 코드에서 학습이 어떻게 구현되어 있는지를 단계별로 추적한다.

---

## 1. 학습 실행 명령어

```bash
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \
    checkpoint.save_iter=500
```

| 인자 | 설명 |
|------|------|
| `--nproc_per_node=8` | 8 GPU 사용 (Context Parallelism) |
| `--config` | Hydra 설정 파일 |
| `experiment` | 실험 이름 → 체크포인트 + 데이터로더 + 모델 설정 결정 |
| `dataloader_train.dataset.dataset_dir` | 데이터셋 경로 오버라이드 |
| `'dataloader_train.sampler.dataset=...'` | 멀티 GPU 데이터 분할을 위한 sampler 링크 |

---

## 2. 초기화 과정

### 2.1 설정 해석

```
singleview_config.py
    │
    ▼
cosmos_singleview_example.py 에서 실험 설정 로드
    │
    ├── defaults: "vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow"
    │   (base experiment: 모델 구조, 옵티마이저, 스케줄러 등 정의)
    │
    ├── data_train: "example_singleview_train_data_edge"
    │   (데이터로더 설정: SingleViewTransferDataset + 93 frames + 704x1280)
    │
    ├── checkpoint:
    │   load_path = EDGE_CHECKPOINT.s3.uri  (NVIDIA 사전학습 체크포인트)
    │   load_training_state = False  (옵티마이저 상태 로드 안 함)
    │
    └── model:
        hint_keys = "edge"
        base_load_from = None  (별도 base 모델 로드 안 함)
```

### 2.2 모델 로드

```python
# 1. 모델 아키텍처 생성
model = ControlVideo2WorldModel(config)
#   ├── net: DiT + ControlNet (모든 파라미터)
#   ├── tokenizer: Wan2.2 VAE
#   └── text_encoder: Qwen2.5-VL 7B

# 2. 사전 학습된 체크포인트 로드
model.load_state_dict(checkpoint)  # NVIDIA가 학습한 가중치

# 3. Base 모델 Freeze
model.freeze_base_model()
# → base DiT 동결, ControlNet만 requires_grad=True

# 4. Control 가중치 초기화 (이미 체크포인트에 포함되어 있으므로 스킵)
# model.copy_weights_to_control_branch()  # is_new_training=False면 스킵
```

### 2.3 옵티마이저 생성

```python
# 학습 가능한 파라미터만으로 옵티마이저 생성
optimizer = FusedAdamW(
    params=[p for p in model.net.parameters() if p.requires_grad],
    lr=8.63e-5,           # ≈ 2^(-14.5)
    weight_decay=1e-3,
    betas=(0.9, 0.999),
)
```

### 2.4 스케줄러 생성

```python
scheduler = LambdaLinearScheduler(
    f_max=[0.5],           # 최대 학습률 계수
    f_min=[0.2],           # 최소 학습률 계수
    warm_up_steps=[1000],  # 워밍업 스텝 (사후학습 예시에서는 1000)
    cycle_lengths=[5000],  # 전체 사이클 길이
)

# 실효 학습률:
# step 0:     lr = 8.63e-5 * 0.0  = 0 (워밍업 시작)
# step 500:   lr = 8.63e-5 * 0.25 = 2.16e-5 (워밍업 중간)
# step 1000:  lr = 8.63e-5 * 0.5  = 4.32e-5 (워밍업 완료, 최대)
# step 3000:  lr = 8.63e-5 * 0.35 = 3.02e-5 (감소 중)
# step 5000:  lr = 8.63e-5 * 0.2  = 1.73e-5 (최소)
```

---

## 3. 학습 루프 상세

> 파일: `imaginaire/trainer.py` → `train()`

```python
class ImaginaireTrainer:
    def train(self):
        for iteration in range(max_iter):
            # === 1. 데이터 로드 ===
            data_batch = next(dataloader_iter)
            # data_batch: {video, control_input_edge, caption, padding_mask, ...}

            # === 2. Forward + Loss ===
            output_batch, loss = model_ddp.training_step(data_batch, iteration)

            # === 3. Gradient Scaling (Mixed Precision) ===
            loss_scaled = grad_scaler.scale(loss / grad_accum_iter)
            # grad_accum_iter = 4 → 유효 배치 = 4

            # === 4. Backward ===
            loss_scaled.backward()
            # gradient가 ControlNet 파라미터에만 흐름 (나머지 frozen)

            # === 5. Gradient Accumulation 체크 ===
            if (iteration + 1) % grad_accum_iter == 0:
                # === 6. Optimizer Step ===
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # === 7. Scheduler Step ===
                scheduler.step()

                # === 8. EMA Update ===
                model.on_before_zero_grad(iteration)
                # → ema_weight = β * ema_weight + (1-β) * current_weight

                # === 9. Zero Grad ===
                optimizer.zero_grad(set_to_none=True)

            # === 10. Callbacks ===
            # 체크포인트 저장, 샘플 생성, 로깅 등
```

---

## 4. `training_step()` 내부

> 파일: `predict2/models/text2world_model_rectified_flow.py` → `forward()`

### 4.1 전체 흐름

```python
def forward(self, data_batch):
    # ===== Phase 1: 전처리 =====
    self._normalize_video_databatch_inplace(data_batch)
    # video: uint8 [0,255] → bfloat16 [-1,1]
    # control: uint8 [0,255] → bfloat16 [-1,1]

    self._augment_image_dim_inplace(data_batch)
    # 이미지면 T 차원 추가

    # ===== Phase 2: 텍스트 인코딩 (on-the-fly) =====
    if self.config.text_encoder_config.compute_online:
        text_embeddings = self.text_encoder.compute_text_embeddings_online(
            data_batch, "ai_caption"
        )
        data_batch["t5_text_embeddings"] = text_embeddings

    # ===== Phase 3: Latent 인코딩 =====
    raw_state, x0, condition = self.get_data_and_condition(data_batch)
    # x0: (B, 16, 24, 44, 80) - GT video의 latent
    # condition: 텍스트 + 컨트롤 + 조건마스크 포함

    # ===== Phase 4: 노이즈 샘플링 =====
    epsilon = torch.randn_like(x0)  # (B, 16, 24, 44, 80)

    # ===== Phase 5: 시간 샘플링 =====
    u = torch.sigmoid(torch.randn(B))  # LogitNormal
    timesteps = shift * u / (1 + (shift-1) * u) * 1000  # shift=5

    # ===== Phase 6: 노이즈 보간 =====
    sigma = timesteps / 1000  # [0, 1]
    x_t = (1 - sigma) * x0 + sigma * epsilon  # 노이즈 섞인 latent

    # ===== Phase 7: 모델 예측 =====
    v_pred = self.denoise(epsilon, x_t, timesteps, condition)
    # DiT forward: x_t → v_pred
    # 내부에서 ControlNet 블록도 실행됨

    # ===== Phase 8: Loss 계산 =====
    v_target = epsilon - x0  # 타겟 velocity
    per_instance_loss = torch.mean((v_pred - v_target) ** 2, dim=[1,2,3,4])
    # per_instance_loss: (B,)

    loss = torch.mean(per_instance_loss)  # 스칼라

    return output_batch, loss
```

### 4.2 `get_data_and_condition()` 학습 시

```python
# 1. 영상을 latent으로 인코딩
x0 = self.encode(data_batch["video"])  # VAE encode
# (1, 3, 93, 704, 1280) → (1, 16, 24, 44, 80)

# 2. 조건 프레임 마스크 생성 (랜덤)
num_cond_frames = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
condition_mask = zeros(1, 1, 24, 44, 80)
condition_mask[:, :, :num_cond_frames_latent] = 1

# 3. 컨트롤 입력을 latent으로 인코딩
control_latent = self.encode(data_batch["control_input_edge"])
# (1, 3, 93, 704, 1280) → (1, 16, 24, 44, 80)

# 4. Condition 객체 구성
condition = ControlVideo2WorldCondition(
    t5_text_embeddings=text_emb,
    condition_video_input_mask=condition_mask,
    latent_control_input=control_latent,
    control_weight=[1.0],
    ...
)
```

### 4.3 `denoise()` = DiT Forward

```python
def denoise(self, noise, x_t, timesteps, condition):
    # DiT 네트워크에 전달
    v_pred = self.net(
        x_B_C_T_H_W=x_t,                    # 노이즈 섞인 latent
        timesteps_B_T=timesteps,              # 시간 스텝
        crossattn_emb=condition.text_emb,     # 텍스트 임베딩
        latent_control_input=condition.latent_control_input,  # 컨트롤 latent
        condition_video_input_mask=condition.mask,  # VACE 마스크
        control_context_scale=condition.control_weight,  # 컨트롤 가중치
    )
    return v_pred
```

---

## 5. Gradient 흐름 분석

```
loss.backward() 시 gradient가 흐르는 경로:

loss
  │
  ▼
MSE(v_pred, v_target)
  │
  ▼
v_pred = DiT.forward(x_t, condition)
  │
  ├──→ Base DiT Blocks [B0...B27]
  │       ✗ requires_grad=False → gradient 계산 안 함
  │       (하지만 forward는 실행됨 - hint 주입 위해)
  │
  ├──→ Control Blocks [C0...C3]
  │       ✓ requires_grad=True → gradient 계산 & 가중치 업데이트
  │
  ├──→ Control Embedder
  │       ✓ requires_grad=True → gradient 계산 & 가중치 업데이트
  │
  └──→ 텍스트 인코더 (Qwen2.5-VL)
          ✗ torch.no_grad() 블록 내에서 실행 → gradient 없음
          ✗ 학습하지 않음 (frozen)
```

**메모리 효율**: Base DiT는 forward만 실행하고 backward gradient를 저장하지 않으므로, 전체 모델 학습 대비 메모리가 절약된다. 단, forward 연산 자체는 필요하다(hint 주입 시 base의 hidden state 참조).

---

## 6. 콜백 시스템

학습 중 자동으로 실행되는 콜백:

### 6.1 샘플 생성 (`every_n_sample_ema`)

```python
# 기본: 200 이터레이션마다 샘플 생성
every_n_sample_ema=dict(
    every_n=200,
    save_s3=False,
)
```

EMA 가중치로 모델을 일시적으로 전환하여 샘플을 생성하고 저장:
```
${IMAGINAIRE_OUTPUT_ROOT}/<project>/<group>/<name>/samples/iter_000200/
```

### 6.2 체크포인트 저장

```python
checkpoint=dict(
    save_iter=1000,  # 1000 이터레이션마다 저장
)
```

**DCP(Distributed Checkpoint)** 형식으로 저장:
```
checkpoints/
├── iter_000001000/
│   ├── model/          ← DCP 모델 가중치
│   ├── model_ema/      ← DCP EMA 가중치
│   ├── optimizer/      ← 옵티마이저 상태
│   └── scheduler/      ← 스케줄러 상태
└── latest_checkpoint.txt
```

### 6.3 로깅

```python
# WandB (선택적)
job.wandb_mode = "disabled"  # 또는 "online"

# 콘솔 출력
# [Iteration 100/5000] Loss: 0.234, LR: 4.95e-05, Time: 1.23s/it
```

---

## 7. 체크포인트 변환

학습 완료 후 DCP → PyTorch 형식 변환:

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/.../checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)

python scripts/convert_distcp_to_pt.py \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER
```

**생성 파일:**
- `model_ema_bf16.pt` ← **추론에 사용할 파일**
- `model_ema_fp32.pt` ← FP32 EMA 가중치
- `model.pt` ← 학습 가중치 (EMA 아님)

---

## 8. 학습 재개

같은 명령어를 다시 실행하면 자동으로 마지막 체크포인트에서 재개:

```bash
# 동일 명령어 재실행 → 자동 resume
torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    ...
```

재개 시 로드하는 것:
- 모델 가중치 (학습 중 + EMA)
- 옵티마이저 상태 (momentum 등)
- 스케줄러 상태 (현재 학습률)
- 현재 이터레이션 번호

---

## 9. Context Parallelism 동작

8 GPU에서의 데이터 분할:

```
GPU 0-7: 동일한 배치를 공유하되, latent 시퀀스를 시간축으로 분할

latent: (1, 16, 24, 44, 80)
                   ↓
GPU 0: (1, 16, 3, 44, 80)   ← latent frames [0:3]
GPU 1: (1, 16, 3, 44, 80)   ← latent frames [3:6]
GPU 2: (1, 16, 3, 44, 80)   ← latent frames [6:9]
...
GPU 7: (1, 16, 3, 44, 80)   ← latent frames [21:24]
```

- state_t=24, context_parallel_size=8 → GPU당 3 latent frames
- Attention에서 GPU간 통신 (P2P 또는 All-to-All)
- 각 GPU는 전체 시퀀스의 attention을 볼 수 있음 (메모리만 분산)

> **state_t은 context_parallel_size로 나누어 떨어져야 한다**:
> - 24 ÷ 8 = 3 ✓
> - 24 ÷ 4 = 6 ✓
> - 20 ÷ 8 = 2.5 ✗ → 20은 4나 5를 사용해야 함

---

## 10. Mixed Precision 전략

```
텐서 타입:
- 모델 가중치:     bfloat16 (기본)
- 활성화:          bfloat16
- Loss 계산:       bfloat16
- Gradient:        bfloat16
- 옵티마이저 상태:  float32 (master weights)
- EMA 가중치:      float32

# GradScaler
grad_scaler = torch.GradScaler()  # bfloat16 안정성을 위해
```

FusedAdamW의 `master_weights=True`로 옵티마이저가 FP32 마스터 가중치를 유지하여 수치 안정성을 확보한다.
