# 05. 실전 가이드: 하이퍼파라미터, 모니터링, 기대치

---

## 1. 컨트롤 타입 선택

### 1.1 어떤 컨트롤로 시작할까?

| 컨트롤 | 난이도 | 장점 | 단점 | 추천 용도 |
|--------|--------|------|------|----------|
| **Edge** | 쉬움 | 사전 계산 불필요, 빠른 반복 | 세밀한 구조 제어 제한 | 일반 씬, 첫 실험 |
| **Vis(Blur)** | 쉬움 | 사전 계산 불필요, 스타일 전환 | 구조 제어 약함 | 스타일 변환, 리마스터링 |
| **Depth** | 보통 | 3D 구조 보존 | Depth 사전 계산 필요 | 로보틱스, 3D 씬 |
| **Seg** | 어려움 | 객체별 제어 | SAM2 사전 계산 필요, 품질 의존성 | 객체 합성, 장면 편집 |

> **권장**: Edge로 시작하여 파이프라인을 검증한 뒤, 필요한 컨트롤로 전환

### 1.2 같은 데이터로 다른 컨트롤 학습

같은 데이터셋으로 여러 컨트롤 타입을 독립적으로 학습할 수 있다. 각 컨트롤 타입마다 별도의 체크포인트가 생성된다:

```bash
# Edge (데이터 준비 불필요)
... experiment=transfer2_singleview_posttrain_edge_example

# Depth (depth/ 폴더 사전 계산 필요)
... experiment=transfer2_singleview_posttrain_depth_example

# 동시에 학습 불가 - 각각 별도 실행
```

---

## 2. 하이퍼파라미터 가이드

### 2.1 이터레이션 수

| 데이터 규모 | 권장 이터레이션 | 근거 |
|------------|---------------|------|
| 50~100 영상 | 1000~2000 | 오버피팅 방지 |
| 200~500 영상 | 2000~5000 | 적절한 수렴 |
| 1000+ 영상 | 5000~10000 | 충분한 다양성 활용 |

### 2.2 학습률

기본값 `8.63e-5`는 대부분의 경우 잘 동작한다. 조절이 필요하면:

```bash
# 학습률 낮추기 (더 안정적, 느린 수렴)
optimizer.lr=5e-5

# 학습률 높이기 (빠른 수렴, 불안정 위험)
optimizer.lr=1e-4
```

### 2.3 Gradient Accumulation

기본값 `grad_accum_iter=4`:
- 실제 배치 크기: 1 (GPU당)
- 유효 배치 크기: 4 (4 이터레이션 누적)
- 더 큰 유효 배치가 필요하면 8이나 16으로 설정

```bash
trainer.grad_accum_iter=8  # 유효 배치 = 8
```

### 2.4 Warmup Steps

기본값: 1000 (사후학습 예시 기준)

```bash
# 워밍업 줄이기 (빠른 학습 시작, 짧은 학습에 적합)
scheduler.warm_up_steps=[500]

# 워밍업 늘리기 (안정적, 대규모 데이터에 적합)
scheduler.warm_up_steps=[2000]
```

### 2.5 체크포인트 저장 빈도

```bash
# 자주 저장 (디스크 사용 많음, 안전)
checkpoint.save_iter=200

# 적게 저장 (디스크 절약)
checkpoint.save_iter=1000
```

> **주의**: 체크포인트 1개가 수 GB 차지할 수 있다. 디스크 용량 고려.

---

## 3. 메모리 최적화

### 3.1 CUDA OOM 해결

**방법 1: Context Parallel 크기 늘리기**
```bash
# 8 GPU 전체를 CP에 사용 (기본)
# state_t=24 ÷ 8 = 3 frames/GPU
```

**방법 2: 프레임 수 줄이기**
```bash
# 93→77 프레임 (state_t: 24→20)
dataloader_train.dataset.num_frames=77 model.config.state_t=20

# 93→61 프레임 (state_t: 24→16)
dataloader_train.dataset.num_frames=61 model.config.state_t=16

# 93→45 프레임 (state_t: 24→12)
dataloader_train.dataset.num_frames=45 model.config.state_t=12
```

> state_t 변경 시 반드시 context_parallel_size로 나누어 떨어지는지 확인!

**방법 3: 체크포인트 빈도 줄이기**
```bash
checkpoint.save_iter=2000  # 저장 빈도 줄이기
```

### 3.2 GPU 수 변경 시

| GPU 수 | context_parallel_size | state_t=24에서 GPU당 frames |
|--------|----------------------|---------------------------|
| 4 | 4 | 6 |
| 8 | 8 | 3 |
| 12 | 12 | 2 |
| 24 | 24 | 1 |

```bash
# 4 GPU 사용 시
torchrun --nproc_per_node=4 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    model_parallel.context_parallel_size=4 \
    ...
```

---

## 4. 모니터링

### 4.1 콘솔 출력 읽기

```
[Iteration 100/5000] Loss: 0.452, LR: 4.32e-06, Time: 1.23s/it
```

| 메트릭 | 정상 범위 | 주의 필요 |
|--------|----------|----------|
| Loss | 0.5 → 0.1~0.2 | NaN/Inf, 증가 추세 |
| LR | 워밍업 후 안정 | 0이 지속되면 설정 오류 |
| Time/it | 1~2s (8xA100) | 급격한 증가 |

### 4.2 Loss 해석

```
Loss 곡선 (정상):

0.5 ─┐
     │╲
0.4 ─│  ╲
     │    ╲
0.3 ─│      ╲
     │        ─────
0.2 ─│              ──────────
     │
0.1 ─│
     └────────────────────────▶ iteration
     0    1000   2000   3000  5000
```

- **초기(0~500)**: Loss가 빠르게 감소 (워밍업 + 빠른 적응)
- **중기(500~2000)**: 점진적 감소 (도메인 적응)
- **후기(2000+)**: 수렴 (안정화)

**경고 신호:**
- Loss가 NaN/Inf: 학습률 너무 높음 → lr 줄이기
- Loss가 안 줄어듦: 데이터 문제 또는 체크포인트 미로드
- Loss가 갑자기 증가: gradient explosion → 재시작 또는 lr 줄이기

### 4.3 샘플 생성 모니터링

200 이터레이션마다 자동으로 샘플이 생성된다:

```
${IMAGINAIRE_OUTPUT_ROOT}/<project>/<group>/<name>/samples/
├── iter_000200/
│   └── sample_*.mp4
├── iter_000400/
│   └── sample_*.mp4
└── ...
```

**시간에 따른 변화 기대:**
- iter 200: 매우 노이즈함, 기본 구조만 보임
- iter 500~1000: 구조가 잡히기 시작, 색감 불안정
- iter 2000+: 도메인 특성이 반영되기 시작
- iter 5000+: 안정적인 품질

### 4.4 WandB 사용

```bash
# WandB 활성화
job.wandb_mode=online
```

로깅되는 메트릭:
- `train/loss`: 학습 loss
- `train/lr`: 현재 학습률
- `train/grad_norm`: gradient 크기
- `train/iter_time`: 이터레이션당 시간

---

## 5. 사후학습 결과로 기대할 수 있는 것

### 5.1 일반적인 기대

| 측면 | 사전 학습 모델 | 사후 학습 후 |
|------|--------------|-------------|
| 일반 영상 품질 | 우수 | 유지 또는 소폭 하락 |
| 도메인 특화 품질 | 보통 | **크게 향상** |
| 도메인 특화 객체 인식 | 제한적 | **향상** |
| 스타일 일관성 | 일반적 | **도메인에 맞춤** |
| 컨트롤 정확도 | 우수 | 도메인 내 **더 정확** |

### 5.2 적합한 사용 사례

- **자율주행**: 특정 도시/환경의 드라이빙 씬 스타일
- **로보틱스**: 특정 로봇 환경의 시각적 특성
- **영화/애니메이션**: 특정 아트 스타일
- **의료**: 특정 의료 이미징 도메인
- **건축**: 특정 건축 스타일의 렌더링

### 5.3 주의사항

- **과소학습**: 이터레이션이 너무 적으면 원본과 차이 없음
- **과적합**: 데이터가 적은데 이터레이션이 많으면 학습 데이터만 재생산
- **카타스트로픽 포겟팅**: Base DiT가 동결되어 있으므로 일반적으로 발생하지 않음. 하지만 ControlNet이 특정 도메인에 너무 특화되면 다른 도메인에서 성능 저하 가능

---

## 6. 추론 실행

### 6.1 체크포인트 변환 후 추론

```bash
# 1. DCP → PT 변환
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/.../checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
python scripts/convert_distcp_to_pt.py \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER

# 2. 추론 실행
torchrun --nproc_per_node=1 examples/inference.py \
    -i my_input.jsonl \
    -o outputs/ \
    --checkpoint-path $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model_ema_bf16.pt \
    --experiment transfer2_singleview_posttrain_edge_example \
    control:edge
```

### 6.2 추론 시 주의사항

- 반드시 **EMA 체크포인트** (`model_ema_bf16.pt`)를 사용
- `--experiment`는 학습 시 사용한 것과 동일해야 함
- 컨트롤 타입도 학습 시와 동일해야 함 (edge로 학습 → edge로 추론)

---

## 7. 실전 레시피 모음

### 7.1 빠른 검증 (30분)

```bash
# 128 영상, 500 이터레이션
python scripts/prepare_videoufo_dataset.py --storage_dir assets/videoufo --num_videos 128

torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=assets/videoufo \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=500 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

### 7.2 표준 사후학습 (1~2시간)

```bash
# 500 영상, 5000 이터레이션
torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/my_domain \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \
    checkpoint.save_iter=1000 \
    job.wandb_mode=online
```

### 7.3 고품질 사후학습 (3~5시간)

```bash
# 2000+ 영상, 10000 이터레이션
torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_depth_example \
    dataloader_train.dataset.dataset_dir=datasets/my_large_domain \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=10000 \
    checkpoint.save_iter=2000 \
    scheduler.warm_up_steps=[2000] \
    scheduler.cycle_lengths=[10000] \
    job.wandb_mode=online
```

---

## 8. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| CUDA OOM | 메모리 부족 | state_t 줄이기, CP 크기 조정 |
| "Video has only X frames" | 영상이 너무 짧음 | 93 프레임 이상 필터링 |
| Loss가 안 줄어듦 | 체크포인트 미로드 | `checkpoint.load_path` 확인 |
| Loss가 NaN | 학습률 과대 | lr 줄이기, 워밍업 늘리기 |
| 샘플이 원본과 동일 | 학습 부족 | 이터레이션 늘리기 |
| 샘플에 아티팩트 | 과적합 또는 데이터 문제 | 데이터 품질 확인, lr 줄이기 |
| 체크포인트 파일 없음 | save_iter에 도달 안 함 | save_iter 줄이기 |
| 재시작 시 처음부터 | 체크포인트 경로 불일치 | IMAGINAIRE_OUTPUT_ROOT 확인 |

---

## 9. 전체 사후학습 플로우차트

```
데이터 준비
    │
    ├── 영상 수집 (93+ frames, MP4)
    ├── 캡션 작성 (JSON)
    └── (선택) depth/seg 사전 계산
         │
         ▼
환경 설정
    │
    ├── GPU 확인 (8x H100/A100)
    ├── IMAGINAIRE_OUTPUT_ROOT 설정
    └── 패키지 설치 확인
         │
         ▼
학습 실행 ──→ 모니터링 ──→ 수렴 확인
    │              │              │
    │          Loss 곡선      샘플 품질
    │          WandB         시각적 확인
    │              │              │
    │              ▼              ▼
    │         문제 있으면     충분하면
    │         하이퍼파라미터    │
    │         조정 후 재시작    │
    │                          ▼
    │                   체크포인트 변환
    │                   DCP → model_ema_bf16.pt
    │                          │
    │                          ▼
    │                   추론 테스트
    │                   examples/inference.py
    │                          │
    │                          ▼
    └─────────────────── 만족?
                          │     │
                         Yes    No → 데이터/하이퍼파라미터 조정
                          │
                          ▼
                       배포/활용
```
