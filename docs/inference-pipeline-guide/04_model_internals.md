# 04. 모델 내부: 토크나이저, DiT, ControlNet, CFG

이 문서는 `model.generate_samples_from_batch()` 호출 시 내부적으로 발생하는 모든 일을 상세하게 설명한다.

---

## 1. 전체 모델 구성

```
ControlVideo2WorldModel (vid2vid_model_control_vace_rectified_flow.py)
    │
    ├── tokenizer: Wan2.2 VAE (인코더/디코더)
    │       - encode(): 픽셀 → latent
    │       - decode(): latent → 픽셀
    │
    ├── net: DiT with ControlNet (minimal_v4_lvg_dit_control_vace.py)
    │       - 메인 DiT 블록
    │       - ControlNet 블록 (control_blocks)
    │       - Patch Embedder, Time Embedder
    │       - Cross-attention (text + image context)
    │
    ├── text_encoder: Qwen2.5-VL 7B (Reason1.1)
    │       - 텍스트 → 임베딩 (28 layers FULL_CONCAT → 100,352 dim)
    │
    ├── conditioner: 조건 관리자
    │       - condition/uncondition 쌍 생성 (CFG용)
    │
    └── sample_scheduler: Flow Matching Scheduler
            - 시간 스텝 스케줄 생성
            - velocity → latent 업데이트
```

---

## 2. Wan2.2 VAE 토크나이저

> 파일: `predict2/tokenizers/wan2pt2.py`

### 2.1 압축 비율

| 차원 | 압축률 | 설명 |
|------|--------|------|
| **공간(H, W)** | **16x** | 8x spatial downsampling + 2x patchify |
| **시간(T)** | **4x** | `latent_T = 1 + (pixel_T - 1) // 4` |
| **채널** | 3 → 16 | RGB → 16채널 latent (state_ch) |

> 참고: 토크나이저의 native latent_ch는 48이지만, 모델이 사용하는 state_ch는 16이다. sigma_data=1.0으로 스케일링된다.

### 2.2 크기 변환 예시

**720p, 16:9, 93 프레임:**

```
픽셀 공간:  (1, 3, 93, 704, 1280)
                    │     │     │
                    ▼     ▼     ▼
Latent 공간: (1, 16, 24,  44,   80)

T: (93 - 1) / 4 + 1 = 24
H: 704 / 16 = 44
W: 1280 / 16 = 80
```

**720p, 1:1, 93 프레임:**

```
픽셀 공간:  (1, 3, 93, 960, 960)
Latent 공간: (1, 16, 24, 60, 60)
```

**이미지 (1 프레임):**

```
픽셀 공간:  (1, 3, 1, 704, 1280)
Latent 공간: (1, 16, 1, 44, 80)
```

### 2.3 인코드/디코드

```python
def encode(self, state: torch.Tensor) -> torch.Tensor:
    return self.tokenizer.encode(state) * self.sigma_data  # sigma_data = 1.0

@torch.no_grad()
def decode(self, latent: torch.Tensor) -> torch.Tensor:
    return self.tokenizer.decode(latent / self.sigma_data)
```

- `encode()`: [-1, 1] 범위의 영상을 latent으로 변환
- `decode()`: latent을 [-1, 1] 범위의 영상으로 복원
- `sigma_data = 1.0` (Rectified Flow에서)

---

## 3. `generate_samples_from_batch()` 상세

> 파일: `transfer2/models/vid2vid_model_control_vace_rectified_flow.py:330`

### 3.1 전체 흐름

```python
@torch.no_grad()
def generate_samples_from_batch(
    self, data_batch, guidance=1.5, seed=1, n_sample=1, num_steps=35, shift=5.0, **kwargs
):
    # Step 1: 데이터 정규화
    _normalize_video_databatch_inplace(data_batch)
    # uint8 [0,255] → bfloat16 [-1,1] (영상, 컨트롤 모두)

    # Step 2: 이미지 차원 보정
    _augment_image_dim_inplace(data_batch)
    # 이미지인 경우 T 차원 추가

    # Step 3: latent shape 계산
    state_shape = [state_ch, latent_T, latent_H, latent_W]
    # 예: [16, 24, 44, 80]

    # Step 4: 초기 노이즈 생성
    noise = arch_invariant_rand(seed, shape=(n_sample, *state_shape))
    # 결정적 노이즈: 시드에 의해 완전 결정

    # Step 5: velocity 함수 생성 (CFG 포함)
    velocity_fn = get_velocity_fn_from_batch(data_batch, guidance)

    # Step 6: 타임스텝 스케줄 설정
    sample_scheduler.set_timesteps(num_steps, shift=shift)
    # 예: [1000, 971, 943, ..., 0] (35 스텝)

    # Step 7: 디노이징 루프
    latents = noise
    for t in timesteps:
        velocity_pred = velocity_fn(noise, latents, t)
        latents = sample_scheduler.step(velocity_pred, t, latents)

    # Step 8: latent 반환
    return latents  # (1, 16, 24, 44, 80)
```

### 3.2 Step 3: Latent Shape 계산

```python
is_image = data_batch.get("dataset_name") == "image_data" or T_pixel == 1
if is_image:
    latent_T = 1
else:
    latent_T = tokenizer.get_latent_num_frames(T_pixel)
    # = 1 + (T_pixel - 1) // 4

latent_H = H_pixel // spatial_compression_factor  # // 16
latent_W = W_pixel // spatial_compression_factor   # // 16

state_shape = [state_ch, latent_T, latent_H, latent_W]
# state_ch = 16 (모델 설정에 의해 결정)
```

---

## 4. 데이터 조건화: `get_data_and_condition()`

> 파일: `transfer2/models/vid2vid_model_control_vace_rectified_flow.py:78`

이 함수는 입력 데이터를 **latent 조건**으로 변환한다.

### 4.1 기본 영상 조건화 (부모 클래스)

```python
# 부모 클래스에서: 입력 영상을 latent으로 인코딩
raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
# latent_state: 조건 프레임의 latent (나머지는 노이즈)
```

### 4.2 컨트롤 입력 조건화

각 컨트롤 타입(edge, depth, seg, vis)에 대해:

```python
for hint_key in self.hint_keys:
    control_input = data_batch[f"control_input_{hint_key}"]     # (1, 3, T, H, W)
    control_mask = data_batch.get(f"control_input_{hint_key}_mask")  # (1, 1, T, H, W)

    control_latent = self.get_control_latent(latent_state, control_input, control_mask)
    # control_latent: [encoded_control, mask_in_latent] 또는 [encoded_control]
```

### 4.3 `get_control_latent()` 상세

```python
def get_control_latent(self, latent_state, control_input, control_input_mask):
    if control_input is None or all(-1):
        return [zeros_like(latent_state)]  # 비활성 컨트롤

    if self.vace_has_mask:
        # 마스크 적용: 전경만 남기고 배경은 -1
        fg = (control_input + 1) / 2 * mask * 2 - 1
        latent = self.encode(fg)

        # 마스크를 latent 공간으로 변환
        # (1, 1, T, H, W) → (1, 패치크기², T_lat, H_lat, W_lat)
        mask_latent = rearrange(mask, 'b c t (h p1) (w p2) -> b (c p1 p2) t h w',
                               p1=8, p2=8)
        # 시간축 보간: pixel T → latent T
        if mask_latent.shape[2] != latent_T:
            mask_latent = interpolate(mask_latent, size=(latent_T, ...))

        return [latent, mask_latent]
    else:
        latent = self.encode(control_input)
        return [latent]
```

### 4.4 컨트롤 가중치 처리

```python
# 가중치 리스트: 각 컨트롤 타입별
control_weights = data_batch["control_weight"]  # 예: [1.0, 0.5, 0.0, 0.8]

# 마스크가 있으면 가중치를 공간-시간 맵으로 변환
if mask_exists:
    weight_map = mask * weight  # (1, 1, T_lat, H_lat, W_lat)
    # 여러 컨트롤의 가중치 합이 1.0을 넘지 않도록 cap
    total_weight = sum(all_weight_maps)
    if total_weight > 1.0:
        normalize(weight_maps)

# 모든 컨트롤 latent을 채널 축으로 결합
latent_control_input = torch.cat(all_control_latents, dim=1)
# shape: (1, 16*N_controls, T_lat, H_lat, W_lat)
# 멀티 컨트롤 예: (1, 64, 24, 44, 80) for 4 controls
```

---

## 5. Classifier-Free Guidance (CFG)

> 파일: `predict2/models/text2world_model_rectified_flow.py:442`

### 5.1 velocity 함수 생성

```python
def get_velocity_fn_from_batch(self, data_batch, guidance=1.5, is_negative_prompt=False):
    # 조건부/비조건부 조건 생성
    if is_negative_prompt:
        condition, uncondition = conditioner.get_condition_with_negative_prompt(data_batch)
        # uncondition = 네거티브 프롬프트 임베딩 사용
    else:
        condition, uncondition = conditioner.get_condition_uncondition(data_batch)
        # uncondition = null/empty 임베딩

    def velocity_fn(noise, noise_x, timestep):
        cond_v = self.denoise(noise, noise_x, timestep, condition)     # 조건부 velocity
        uncond_v = self.denoise(noise, noise_x, timestep, uncondition) # 비조건부 velocity
        velocity_pred = uncond_v + guidance * (cond_v - uncond_v)      # CFG 공식
        return velocity_pred

    return velocity_fn
```

### 5.2 CFG 공식

```
v_final = v_uncond + guidance * (v_cond - v_uncond)
```

| guidance 값 | 효과 |
|-------------|------|
| 0 | 비조건부 생성 (프롬프트 무시) |
| 1 | 조건부 생성 (CFG 없음) |
| 3 (기본) | 적당한 프롬프트 준수 |
| 7 | 강한 프롬프트 준수 (과포화 가능) |

### 5.3 Condition과 Uncondition의 차이

**Condition** (조건부):
- 텍스트: 실제 프롬프트 임베딩
- 컨트롤: 실제 컨트롤 latent + 가중치
- 이미지 컨텍스트: SigLIP2 임베딩

**Uncondition** (비조건부):
- 텍스트: 네거티브 프롬프트 임베딩 (또는 null)
- 컨트롤: 동일 (컨트롤은 항상 적용)
- 이미지 컨텍스트: 드롭아웃될 수 있음

> **Distilled 모델**: CFG가 모델에 내장되어 있어 `guidance=None`으로 실행. velocity_fn이 아닌 직접 prediction.

---

## 6. DiT 네트워크와 Control 주입

> 파일: `transfer2/networks/minimal_v4_lvg_dit_control_vace.py`

### 6.1 입력 임베딩

```python
# 1. 메인 입력 임베딩
x_input = cat([noisy_latent, condition_mask], dim=1)  # (B, 16+1, T, H, W)
x_embedded = x_embedder(x_input)  # Patch embedding → (B, T*H*W, D)

# 2. 컨트롤 입력 임베딩
control_input = pad_to_vace_channels(control_latent)  # (B, vace_channels-1, T, H, W)
control_embedded = control_embedder(control_input)     # → (B, T*H*W, D)

# 3. 시간 임베딩
t_emb = t_embedder(timestep * timestep_scale)  # (B, T, D)
t_emb = t_embedding_norm(t_emb)

# 4. 텍스트 + 이미지 컨텍스트 (cross-attention용)
crossattn = crossattn_proj(text_embedding)
if image_context:
    img_emb = img_context_proj(siglip2_embedding)
    context = (crossattn, img_emb)
else:
    context = crossattn
```

### 6.2 ControlNet 블록 처리

```
단일 컨트롤:
    control_embedded → [control_block_0] → [control_block_1] → ... → [control_block_N]
                                                                         │
                                                              hints = unbind(output)[:-1]
                                                              (레이어별 hint 리스트)

멀티 컨트롤 (branch별):
    control_0 → [control_blocks_0[i]] ─┐
    control_1 → [control_blocks_1[i]] ─┼─ weight_map 적용 → hints[i]
    control_2 → [control_blocks_2[i]] ─┤
    control_3 → [control_blocks_3[i]] ─┘
```

각 control block은:
- 자신의 hidden state (`c`)를 업데이트
- 메인 DiT의 hidden state (`x`)를 참조 (cross-reference)
- timestep, text, image context를 조건으로 사용

### 6.3 메인 DiT 블록에 Control Hint 주입

```python
for block in main_dit_blocks:
    x = block(
        x,                      # 메인 hidden state
        hints,                  # 컨트롤 hints (레이어별)
        emb=t_embedding,        # 시간 임베딩
        crossattn_emb=context,  # 텍스트 + 이미지 context
        rope_emb=rope,          # RoPE 위치 인코딩
        control_context_scale=weight,  # 컨트롤 가중치
    )
```

**Hint 주입 방식:**
- 각 메인 DiT 블록에 해당하는 hint가 **additive**하게 더해짐
- `x = x + control_context_scale * hint[layer_idx]`
- `control_context_scale`이 0이면 컨트롤 비활성화

### 6.4 최종 출력

```python
# Final layer: AdaLN → Linear projection
output = final_layer(x, t_embedding)

# Unpatchify: (B, T*H*W, D) → (B, C, T, H, W)
output = unpatchify(output)  # (B, 16, 24, 44, 80)
```

---

## 7. Rectified Flow 디노이징

### 7.1 스케줄러

```python
sample_scheduler.set_timesteps(num_steps=35, shift=5.0)
# shift: 타임스텝 분포를 조절하는 파라미터
# 높은 shift → 초기 타임스텝에 더 많은 스텝 할당
```

### 7.2 디노이징 루프

Rectified Flow에서 velocity는 `x0`에서 `x1` (노이즈)까지의 직선 경로를 나타낸다:

```
x_t = (1 - t) * x_0 + t * noise
v = dx/dt = noise - x_0
```

디노이징은 이 경로를 역추적한다:

```python
latents = noise  # t=1.0에서 시작

for t in reversed(timesteps):
    # DiT가 velocity 예측
    velocity_pred = velocity_fn(noise, latents, t)
    # velocity를 사용하여 latent 업데이트
    latents = sample_scheduler.step(velocity_pred, t, latents)

# t=0.0 도달 → clean latent
```

### 7.3 `x_sigma_max` 사용 시

`sigma_max`가 지정되면 순수 노이즈 대신 원본 기반 노이즈에서 시작:

```python
x_sigma_max = model.get_x_from_clean(x0_latent, sigma_max, seed)
# 원본 latent에 sigma_max만큼의 노이즈 추가
# sigma_max = 0: 원본 그대로
# sigma_max = 200: 거의 순수 노이즈
```

---

## 8. 조건 프레임 (VACE)

VACE(Video-Adaptive Conditioning and Editing) 메커니즘에서 조건 프레임은 마스크를 통해 처리된다:

```python
# num_conditional_frames 설정
if chunk_id == 0:
    num_cond = 0  # 첫 청크: 조건 없음
else:
    # latent 공간에서의 조건 프레임 수
    # num_conditional_frames=1 → latent 1 토큰
    num_cond = 1 + (num_conditional_frames - 1) // 4
```

**조건 마스크 생성:**
```
prev_output: [이전출력_마지막프레임, 0, 0, ..., 0]  (93 프레임)
              │                       │
              ▼                       ▼
정규화 후:    [실제값, -1, -1, ..., -1]
              │                       │
              ▼                       ▼
condition_mask: [1, 0, 0, ..., 0]     (어디가 조건인지)
```

이 마스크가 DiT의 입력에 추가 채널로 결합되어, 모델이 어떤 프레임이 조건이고 어떤 프레임을 생성해야 하는지 알 수 있다.

---

## 9. Context Parallelism (멀티 GPU)

멀티 GPU에서는 Context Parallelism으로 latent 시퀀스를 분할한다:

```
GPU 0: latent[:, :, :12, :, :]   (시간축 전반)
GPU 1: latent[:, :, 12:, :, :]   (시간축 후반)
```

- 각 GPU가 시퀀스의 일부를 처리
- Attention에서 all-to-all 또는 p2p 통신
- 최종 결과를 concatenate
