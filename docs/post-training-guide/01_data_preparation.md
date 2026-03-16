# 01. 데이터 요구사항과 준비

---

## 1. 데이터셋 디렉토리 구조

### 1.1 기본 구조 (edge/vis 컨트롤)

Edge와 Vis(blur) 컨트롤은 **학습 중 온더플라이로 생성**되므로, 원본 영상과 캡션만 있으면 된다:

```
datasets/my_dataset/
├── videos/
│   ├── scene001.mp4
│   ├── scene002.mp4
│   └── ...
└── captions/
    ├── scene001.json    ← 파일명이 video와 동일해야 함
    ├── scene002.json
    └── ...
```

### 1.2 확장 구조 (depth/seg 컨트롤)

Depth와 Segmentation은 **사전 계산이 필요**하다:

```
datasets/my_dataset/
├── videos/
│   ├── scene001.mp4
│   └── ...
├── captions/
│   ├── scene001.json
│   └── ...
├── depth/              ← depth 컨트롤용 (사전 계산)
│   ├── scene001.mp4    ← 원본과 동일한 파일명
│   └── ...
└── seg/                ← seg 컨트롤용 (사전 계산)
    ├── scene001.mp4    ← 컬러 코딩된 세그멘테이션 맵
    └── ...
```

> **중요**: 파일명 매칭이 핵심이다. `videos/scene001.mp4`에 대응하는 캡션은 `captions/scene001.json`, 뎁스는 `depth/scene001.mp4`여야 한다.

---

## 2. 비디오 요구사항

| 항목 | 요구사항 | 설명 |
|------|---------|------|
| **포맷** | MP4 | decord로 읽을 수 있는 형식 |
| **해상도** | 720p 권장 | 더 높거나 낮아도 학습 중 자동 리사이즈 |
| **프레임 수** | **최소 93 프레임** | `state_t=24` 기준: `(24-1)×4+1=93` |
| **프레임레이트** | 10~60 fps | 24fps 권장 |
| **최소 길이** | ~4초 @24fps | 93 프레임 확보에 필요 |
| **코덱** | H.264/H.265 | decord 호환 |

### 2.1 프레임 수가 부족하면?

`state_t`와 `num_frames`를 함께 줄일 수 있다:

| state_t | num_frames | 최소 영상 길이 @24fps |
|---------|------------|---------------------|
| 24 (기본) | 93 | 3.9초 |
| 20 | 77 | 3.2초 |
| 16 | 61 | 2.5초 |
| 12 | 45 | 1.9초 |

```bash
# 예: 77 프레임으로 줄이기
torchrun ... \
    dataloader_train.dataset.num_frames=77 \
    model.config.state_t=20
```

> **주의**: `state_t`는 `context_parallel_size`로 나누어 떨어져야 한다. `state_t=20`이면 `context_parallel_size`를 4나 5로 설정.

---

## 3. 캡션 요구사항

### 3.1 JSON 형식

```json
{
    "caption": "A bustling city street at sunset with cars driving and pedestrians walking on the sidewalk. Golden light reflects off the glass buildings."
}
```

### 3.2 캡션 품질 가이드

텍스트 인코더로 **Qwen2.5-VL 7B**을 사용하므로, 상세하고 자연스러운 캡션이 효과적이다:

| 품질 | 예시 | 효과 |
|------|------|------|
| 나쁨 | `"car"` | 모델이 씬을 이해하기 어려움 |
| 보통 | `"A car driving on a road"` | 기본적인 조건부 |
| **좋음** | `"A red sports car driving along a winding mountain road at golden hour, with dramatic cloud shadows moving across the valley below"` | 풍부한 조건부 학습 |

**권장사항:**
- 영상의 시각적 내용을 상세하게 기술
- 움직임, 조명, 분위기, 객체 등 포함
- 2~5 문장 정도의 길이
- 도메인 특화 용어 포함 (예: 의료 도메인이면 의료 용어)

### 3.3 캡션이 없는 경우

캡션 파일이 없으면 기본값 `"a video"`가 사용된다. 이는 비권장이지만 동작은 한다.

---

## 4. 컨트롤 입력 사전 계산

### 4.1 Edge / Vis: 사전 계산 불필요

- **Edge**: 학습 중 Canny Edge Detection이 온더플라이로 적용됨
  - threshold가 매 이터레이션마다 랜덤으로 변함 (data augmentation 효과)
  - `t_lower ∈ [20, 100]`, `t_upper = t_lower + [50, 150]`
- **Vis(Blur)**: 학습 중 Bilateral/Gaussian Blur가 온더플라이로 적용됨
  - blur 강도와 다운스케일 비율이 랜덤

### 4.2 Depth: VideoDepthAnything으로 사전 계산

```bash
# 단일 비디오
python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
    --input_video datasets/my_dataset/videos/scene001.mp4 \
    --output_video datasets/my_dataset/depth/scene001.mp4 \
    --encoder vits

# 배치 처리
for video in datasets/my_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/my_dataset/depth/$basename" \
        --encoder vits
done
```

| 인코더 | 속도 | 정확도 | VRAM |
|--------|------|--------|------|
| `vits` | 빠름 | 보통 | 낮음 |
| `vitl` | 느림 | 높음 | 높음 |

**출력**: 원본과 동일한 해상도/프레임 수의 그레이스케일 뎁스 MP4

### 4.3 Segmentation: SAM2로 사전 계산

```bash
# 텍스트 프롬프트 기반 세그멘테이션 (권장)
python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --input_video datasets/my_dataset/videos/scene001.mp4 \
    --output_video datasets/my_dataset/seg/scene001.mp4 \
    --mode prompt \
    --prompt "person, car, vehicle, building, tree, road, sky" \
    --visualize
```

**세그멘테이션 모드:**

| 모드 | 설명 | 사용법 |
|------|------|--------|
| `prompt` | 텍스트로 객체 지정 (권장) | `--prompt "person, car"` |
| `box` | 바운딩 박스로 지정 | `--box "300,0,500,400"` |
| `points` | 포인트로 지정 | `--points "200,300" --labels "1"` |

**출력**: 각 객체가 고유 색상으로 코딩된 MP4

---

## 5. 빠른 시작: VideoUFO 데이터셋

자체 데이터가 없다면, [VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO) 데이터셋으로 시작할 수 있다:

```bash
# 128개 비디오 다운로드 (테스트용, ~5-7GB)
python scripts/prepare_videoufo_dataset.py \
    --storage_dir assets/videoufo \
    --num_videos 128

# 1000개 비디오 다운로드 (실전용, ~30GB)
python scripts/prepare_videoufo_dataset.py \
    --storage_dir assets/videoufo \
    --num_videos 1000
```

스크립트가 자동으로:
1. 메타데이터 CSV 다운로드 (~1.1 GB)
2. tar 파일에서 비디오 추출 & 93 프레임 미만 필터링
3. 캡션 JSON 생성 (brief + detailed 조합)
4. 올바른 디렉토리 구조로 정리

---

## 6. 데이터 검증 체크리스트

학습 시작 전 확인사항:

```
□ videos/ 디렉토리에 .mp4 파일 존재
□ captions/ 디렉토리에 대응하는 .json 파일 존재
□ 파일명 일치: video1.mp4 ↔ video1.json
□ 모든 비디오가 93 프레임 이상 (24fps 기준 ~4초)
□ JSON 파일에 "caption" 키 존재
□ (depth 사용 시) depth/ 디렉토리에 대응하는 .mp4 존재
□ (seg 사용 시) seg/ 디렉토리에 대응하는 .mp4 존재
```

### 프레임 수 확인 스크립트 예시

```python
import decord
import glob

for path in sorted(glob.glob("datasets/my_dataset/videos/*.mp4")):
    vr = decord.VideoReader(path)
    n_frames = len(vr)
    if n_frames < 93:
        print(f"WARNING: {path} has only {n_frames} frames (need 93)")
    else:
        print(f"OK: {path} - {n_frames} frames")
```

---

## 7. 데이터 양과 품질에 대한 기대치

| 데이터 규모 | 기대 효과 | 적합한 사용 사례 |
|------------|-----------|----------------|
| **50~100개** | 빠른 실험, 오버피팅 경향 | 개념 검증, 프로토타입 |
| **200~500개** | 적절한 일반화 시작 | 특정 도메인 적응 |
| **1000+개** | 좋은 일반화 | 프로덕션 수준 |
| **5000+개** | 최상의 품질 | 대규모 도메인 커버 |

**데이터 품질 > 데이터 양**: 적은 수의 고품질 영상이 많은 수의 저품질 영상보다 효과적이다.

### 좋은 학습 데이터의 특성

- **도메인 일관성**: 모두 비슷한 도메인의 영상 (예: 모두 드라이빙 씬, 모두 실내 씬)
- **다양한 시점/조명**: 같은 도메인 내에서도 다양한 조건
- **깨끗한 영상**: 워터마크, 자막, 심한 압축 아티팩트 없는 영상
- **상세한 캡션**: 영상 내용을 정확하게 설명하는 텍스트
