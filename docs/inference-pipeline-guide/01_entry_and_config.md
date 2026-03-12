# 01. 진입점과 설정 구조

## 1. CLI 진입점: `examples/inference.py`

추론은 아래 명령으로 실행된다:

```bash
python examples/inference.py \
    -i input_params.json \
    -o ./output \
    control:edge  # 컨트롤 타입 선택
```

### 1.1 Args 구조

```python
class Args(pydantic.BaseModel):
    input_files: list[Path]       # JSON 파라미터 파일 경로(들)
    setup: SetupArguments         # 모델 셋업 인자 (CLI에서만 제공)
    overrides: InferenceOverrides # 추론 파라미터 오버라이드
    control: ControlUnion         # 컨트롤 타입: edge | depth | vis | seg
```

**`control` 서브커맨드**: `tyro`를 사용하여 4가지 컨트롤 타입 중 하나를 선택한다.

| 서브커맨드 | 설정 클래스 | 설명 |
|-----------|------------|------|
| `control:edge` | `EdgeConfig` | Canny Edge 기반 |
| `control:depth` | `DepthConfig` | Depth Map 기반 |
| `control:vis` | `BlurConfig` | Bilateral Blur 기반 |
| `control:seg` | `SegConfig` | Segmentation 기반 |

### 1.2 실행 흐름

```python
# 1. 환경 초기화
init_environment()

# 2. CLI 인자 파싱 (tyro 사용)
args = tyro.cli(Args)

# 3. JSON 파일에서 InferenceArguments 로드
inference_samples, batch_hint_keys = InferenceArguments.from_files(args.input_files, overrides=args.overrides)

# 4. Control2WorldInference 객체 생성 (모델 로드)
inference = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)

# 5. 모든 샘플에 대해 순차 추론
inference.generate(inference_samples, output_dir=args.setup.output_dir)
```

---

## 2. SetupArguments: 모델 셋업

> 파일: `cosmos_transfer2/config.py:300`

```python
class SetupArguments(CommonSetupArguments):
    model: Literal["edge", "depth", "seg", "vis"] = "edge"
```

### 주요 필드

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `output_dir` | (필수) | 출력 디렉토리 |
| `model` | `"edge"` | 모델 변종. 각 컨트롤 타입별 체크포인트가 다름 |
| `checkpoint_path` | None | 커스텀 체크포인트 (포스트 트레이닝용) |
| `experiment` | None | 실험 이름 오버라이드 |
| `config_file` | 자동 결정 | 모델 설정 파일 경로 |
| `context_parallel_size` | WORLD_SIZE | 멀티 GPU 시 Context Parallelism 크기 |
| `disable_guardrails` | False | 안전 가드레일 비활성화 |
| `compile_tokenizer` | `"none"` | 토크나이저 컴파일 모드 (`none`/`moderate`/`aggressive`) |
| `enable_parallel_tokenizer` | False | Wan 토크나이저 Context Parallel 활성화 |
| `benchmark` | False | 벤치마크 모드 |

### 모델 변종과 체크포인트 매핑

`config.py:158`에서 `MODEL_CHECKPOINTS` 딕셔너리가 각 변종을 체크포인트 UUID로 매핑한다:

```python
MODEL_CHECKPOINTS = {
    ModelKey(variant="depth"):           "626e6618-...",
    ModelKey(variant="edge"):            "61f5694b-...",
    ModelKey(variant="seg"):             "5136ef49-...",
    ModelKey(variant="vis"):             "ba2f44f2-...",
    ModelKey(variant="auto/multiview"):  "4ecc66e9-...",
    # distilled 버전 (실험적)
    ModelKey(variant="edge", distilled=True): "41f07f13-...",
}
```

> **중요**: 단일 컨트롤 사용 시 해당 변종의 체크포인트 1개만 로드한다. 멀티 컨트롤(edge+depth 등) 사용 시 4개 모든 변종의 체크포인트를 로드하여 multi-branch 모델을 구성한다.

---

## 3. InferenceArguments: 샘플별 추론 파라미터

> 파일: `cosmos_transfer2/config.py:480`

JSON 파일로 제공되며, 각 샘플의 입력과 생성 옵션을 정의한다.

### JSON 예시

```json
{
    "name": "my_sample",
    "video_path": "./input_video.mp4",
    "prompt": "A beautiful sunset over the ocean with golden light reflecting on the waves",
    "negative_prompt": "The video captures a game playing...",
    "seed": 2025,
    "guidance": 3,
    "resolution": "720",
    "num_video_frames_per_chunk": 93,
    "num_conditional_frames": 1,
    "num_steps": 35,
    "edge": {
        "control_weight": 1.0,
        "preset_edge_threshold": "medium"
    }
}
```

### 주요 필드 상세

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | str | (필수) | 샘플 이름 (출력 파일명에 사용) |
| `video_path` | Path | (필수) | 입력 영상 경로 |
| `prompt` | str | (필수) | 텍스트 프롬프트 |
| `negative_prompt` | str | 게임/CG 관련 부정 프롬프트 | 네거티브 프롬프트 (CFG용) |
| `seed` | int | 2025 | 랜덤 시드 |
| `guidance` | int | 3 | CFG 가이던스 스케일 (0~7) |
| `resolution` | str | "720" | 출력 해상도 |
| `max_frames` | int\|None | None | 입력 영상에서 읽을 최대 프레임 수 |
| `num_video_frames_per_chunk` | int | 93 | 청크당 프레임 수 |
| `num_conditional_frames` | 0\|1\|2 | 1 | 이전 청크에서 가져올 조건 프레임 수 |
| `num_steps` | int | 35 | 디노이징 스텝 수 |
| `sigma_max` | float\|None | None | 노이즈 주입량 (None이면 순수 노이즈) |
| `image_context_path` | Path\|None | None | 이미지 컨텍스트 (스타일 레퍼런스) 경로 |
| `context_frame_index` | int\|None | None | 입력 영상에서 컨텍스트 프레임 인덱스 |
| `keep_input_resolution` | bool | True | 출력을 입력 해상도로 복원할지 |
| `show_control_condition` | bool | False | 출력에 컨트롤 영상 병합 |
| `show_input` | bool | False | 출력에 입력 영상 병합 |

### 컨트롤 설정 필드

각 컨트롤 타입별 설정:

```python
class ControlConfig(pydantic.BaseModel):
    control_path: Path | None = None        # 사전 계산된 컨트롤 영상 경로
    control_weight: float = 1.0             # 컨트롤 강도 (0.0~1.0)
    mask_path: Path | None = None           # 공간/시간 마스크 경로
    mask_prompt: str | None = None          # 마스크 생성용 프롬프트 (SAM2)
```

- **EdgeConfig**: `preset_edge_threshold` 추가 (`"very_low"` ~ `"very_high"`)
- **BlurConfig**: `preset_blur_strength` 추가 (`"very_low"` ~ `"very_high"`)
- **SegConfig**: `control_prompt` 추가 (세그멘테이션용 프롬프트)
- **DepthConfig**: 추가 필드 없음

### 컨트롤 유효성 검사

```python
# 최소 1개 컨트롤 필수
if len(self.hint_keys) == 0:
    raise ValueError("No controls provided")

# vis 컨트롤과 image_context_path는 동시 사용 불가
# (둘 다 스타일 전송 역할이므로 충돌)
if "vis" in self.hint_keys and self.image_context_path:
    raise ValueError("vis control and image_context_path conflict")
```

---

## 4. Control2WorldInference: 고수준 추론 클래스

> 파일: `cosmos_transfer2/inference.py:40`

### 4.1 초기화 (`__init__`)

```
SetupArguments + batch_hint_keys
        │
        ▼
  체크포인트 경로 결정
  (단일 컨트롤 → 1개, 멀티 컨트롤 → 4개)
        │
        ▼
  Context Parallel 초기화 (멀티 GPU 시)
        │
        ▼
  가드레일 초기화 (text + video)
        │
        ▼
  ControlVideo2WorldInference 파이프라인 생성
  (실험 설정 + 체크포인트 로드)
        │
        ▼
  Distilled 모델이면 net_fake_score = None
        │
        ▼
  토크나이저 컴파일 (선택적)
```

### 4.2 생성 루프 (`generate`)

```python
def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
    for i_sample, sample in enumerate(samples):
        output_path = self._generate_sample(sample, output_dir, sample_id=i_sample)
```

- 모든 샘플을 **순차적으로** 처리한다 (배치 크기 = 1).
- 벤치마크 모드에서는 첫 샘플을 워밍업으로 버린다.

### 4.3 단일 샘플 생성 (`_generate_sample`)

1. **텍스트 가드레일**: 프롬프트와 네거티브 프롬프트의 안전성 검사
2. **컨트롤 가중치 구성**: `hint_key` 순서에 따라 쉼표 구분 문자열 생성 (예: `"1.0,0.0,0.5,0.0"`)
3. **`generate_img2world()` 호출**: 핵심 파이프라인 실행
4. **출력 후처리**:
   - `[-1, 1]` → `[0, 1]`로 변환
   - 컨트롤 영상 별도 저장
   - 비디오 가드레일 검사
   - 최종 영상/이미지 저장

### Distilled 모델 vs 일반 모델

| 항목 | 일반 모델 | Distilled 모델 |
|------|----------|----------------|
| `negative_prompt` | 필수 (CFG용) | None (불필요) |
| `guidance` | 사용 (기본 3) | None (모델에 내장) |
| `net_fake_score` | 사용 | None으로 설정 |
| 실험 설정 | `EXPERIMENTS` 딕셔너리에서 조회 | 직접 지정 |
| Autoregressive | 지원 | 93 프레임 초과 시 경고 |
