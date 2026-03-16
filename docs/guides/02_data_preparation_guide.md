# 데이터 준비 가이드 - Cosmos-Transfer2.5 Sim2Real Post-Training

## 목차

1. [데이터 준비 개요](#데이터-준비-개요)
2. [필요한 데이터 종류](#필요한-데이터-종류)
3. [데이터 수량 가이드](#데이터-수량-가이드)
4. [실제 환경 영상 촬영 가이드](#실제-환경-영상-촬영-가이드)
5. [영상 전처리](#영상-전처리)
6. [캡션(텍스트 설명) 작성](#캡션텍스트-설명-작성)
7. [Control 입력 생성 (선택사항)](#control-입력-생성-선택사항)
8. [데이터셋 폴더 구조](#데이터셋-폴더-구조)
9. [데이터 검증 체크리스트](#데이터-검증-체크리스트)

---

## 데이터 준비 개요

Sim2Real Post-Training에는 **두 종류의 데이터**가 필요합니다:

### Post-Training 단계 (학습용)
**실제 환경 영상** - 모델이 "실제 세계가 어떻게 생겼는지"를 학습하기 위한 데이터

### Inference 단계 (변환용)
**시뮬레이션 데이터** - Post-Training 완료 후, 실제처럼 변환할 시뮬레이션 데이터

> **핵심**: Post-Training은 **실제 환경 데이터**로 수행합니다. 시뮬레이션 데이터는 학습 완료 후 inference 시 입력으로 사용합니다.

---

## 필요한 데이터 종류

### 필수 데이터

| 데이터 | 형식 | 설명 |
|--------|------|------|
| **비디오 파일** | MP4 | 실제 환경에서 촬영한 영상 |
| **캡션 파일** | JSON | 각 비디오의 텍스트 설명 |

### 선택 데이터 (Control 타입에 따라)

| 데이터 | 형식 | 필요 조건 |
|--------|------|----------|
| **Depth 맵** | MP4 | Depth Control 사용 시 |
| **Segmentation 맵** | MP4 | Segmentation Control 사용 시 |

> **참고**: Edge Control과 Visual Blur Control은 전처리 없이 학습 중에 자동으로 계산됩니다. **처음 시작할 때는 Edge Control을 권장합니다.**

---

## 데이터 수량 가이드

### 권장 데이터 규모

| 규모 | 비디오 수 | 용도 | 예상 품질 |
|------|----------|------|----------|
| **최소** | 50~100개 | 빠른 검증, 프로토타이핑 | 기본적인 도메인 적응 |
| **권장** | 200~500개 | 실질적인 Post-Training | 양호한 품질 |
| **최적** | 1,000개 이상 | 프로덕션 수준 | 최상의 품질 |

### 데이터 규모별 저장 공간 예상

| 비디오 수 | 예상 용량 |
|----------|----------|
| 128개 | ~5-7 GB |
| 500개 | ~20 GB |
| 1,000개 | ~30 GB |
| 5,000개 | ~130 GB |

### 데이터 다양성이 수량보다 중요

- **다양한 시점**: 가능하면 여러 각도에서 촬영
- **다양한 조명**: 주간/야간, 실내/실외
- **다양한 시나리오**: 다양한 상황과 환경 조건
- **균등한 분포**: 특정 장면에 치우치지 않게

---

## 실제 환경 영상 촬영 가이드

### 비디오 사양 요구사항

| 항목 | 요구사항 | 세부 사항 |
|------|---------|----------|
| **형식** | MP4 | H.264/H.265 코덱 권장 |
| **해상도** | 720p 권장 | 1280×720 (16:9 비율) |
| **길이** | 최소 3초 이상 | 93프레임 이상 필요 (24fps 기준 ~4초) |
| **프레임레이트** | 10~60 fps | 24fps 또는 30fps 권장 |
| **최소 프레임 수** | 93프레임 | `num_frames=93` 설정에 맞춤 |

### 촬영 시 주의사항

#### DO (해야 할 것)
- **타겟 도메인을 충실히 반영**: 모델이 학습할 실제 환경을 다양하게 촬영
- **안정적인 촬영**: 과도한 흔들림 방지 (삼각대 또는 고정 마운트 권장)
- **충분한 길이**: 최소 4초 이상으로 촬영 (여유분 확보)
- **다양한 조건**: 조명, 날씨, 시간대 등을 다양하게
- **대상 객체 포함**: Sim2Real 변환 시 등장할 객체들이 영상에 포함되도록

#### DON'T (하지 말아야 할 것)
- 과도한 모션 블러가 있는 영상
- 해상도가 너무 낮은 영상 (480p 이하)
- 텍스트나 워터마크가 있는 영상
- 너무 짧은 영상 (3초 미만)
- 색상이 심하게 왜곡된 영상

### Sim2Real을 위한 촬영 전략

시뮬레이션에서 생성할 장면과 **유사한 환경**을 촬영하는 것이 핵심입니다.

```
예시 시나리오: 실내 로봇 환경의 Sim2Real

촬영 대상:
├── 실제 로봇 작업 공간 (다양한 각도)
├── 바닥 재질 및 텍스처
├── 조명 조건 (자연광, 형광등 등)
├── 로봇이 상호작용할 객체들
├── 벽면, 천장, 기둥 등 구조물
└── 사람이 있는/없는 환경
```

---

## 영상 전처리

### 1. 해상도 조정

720p(1280×720)가 아닌 영상은 리사이즈가 필요합니다.

```bash
# ffmpeg를 이용한 해상도 조정
# 720p로 리사이즈 (비율 유지, 패딩 없음)
ffmpeg -i input.mp4 -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2" output.mp4

# 비율이 16:9인 경우 단순 리사이즈
ffmpeg -i input.mp4 -vf "scale=1280:720" -c:a copy output.mp4
```

### 2. 프레임 수 확인 및 트리밍

```bash
# 프레임 수 확인
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 input.mp4

# 필요 시 트리밍 (처음 4초만 추출)
ffmpeg -i input.mp4 -t 4 -c copy output.mp4

# 또는 프레임 단위로 추출 (93프레임)
ffmpeg -i input.mp4 -frames:v 93 output.mp4
```

### 3. 프레임레이트 조정

```bash
# 30fps를 24fps로 변경
ffmpeg -i input.mp4 -r 24 output.mp4
```

### 4. 일괄 전처리 스크립트

```bash
#!/bin/bash
# preprocess_videos.sh
# 사용법: bash preprocess_videos.sh /path/to/raw/videos /path/to/output

INPUT_DIR=$1
OUTPUT_DIR=$2
mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video")

    # 해상도 조정 + 최소 프레임 수 확인
    frame_count=$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of csv=p=0 "$video")

    if [ "$frame_count" -ge 93 ]; then
        ffmpeg -i "$video" \
            -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2" \
            -y "$OUTPUT_DIR/$filename"
        echo "✓ Processed: $filename ($frame_count frames)"
    else
        echo "✗ Skipped: $filename (only $frame_count frames, need ≥93)"
    fi
done

echo "Done! Processed videos saved to: $OUTPUT_DIR"
```

---

## 캡션(텍스트 설명) 작성

### 캡션 파일 형식

각 비디오에 대응하는 JSON 파일을 작성합니다. **파일 이름은 비디오와 동일**해야 합니다.

```
videos/scene_001.mp4  →  captions/scene_001.json
videos/scene_002.mp4  →  captions/scene_002.json
```

### JSON 파일 형식

```json
{
    "caption": "A detailed description of the video content"
}
```

### 좋은 캡션 작성법

캡션은 모델이 장면을 이해하는 데 사용됩니다. **상세하고 정확한 설명**이 좋은 결과를 만듭니다.

#### 좋은 캡션 예시

```json
{
    "caption": "A robot arm picks up a red cube from a metal table in a well-lit industrial workspace. The background shows shelves with various objects and a concrete floor. Fluorescent lighting illuminates the scene from above."
}
```

```json
{
    "caption": "Indoor warehouse environment with concrete floors and metal shelving units. A mobile robot navigates between rows of storage racks. Natural light enters through high windows, creating shadows on the floor."
}
```

#### 나쁜 캡션 예시

```json
{
    "caption": "robot"
}
```
→ 너무 짧고 정보 부족

```json
{
    "caption": "This is a video of something happening somewhere."
}
```
→ 구체적이지 않음

### 캡션 작성 팁

1. **장면 설명**: 무엇이 보이는지 (객체, 환경, 배경)
2. **동작 설명**: 무엇이 일어나는지 (움직임, 상호작용)
3. **조명/분위기**: 조명 조건, 시간대
4. **공간 관계**: 객체의 위치, 카메라 시점
5. **재질/텍스처**: 바닥 재질, 벽면 등

### 캡션 자동 생성 활용

대량의 비디오에 대해 수동으로 캡션을 작성하기 어려운 경우:

- **Cosmos Reason 1** 모델을 활용한 자동 캡션 생성 (NVIDIA Cookbook 방식)
- **LLaVA, GPT-4V** 등 비전-언어 모델 활용
- 자동 생성 후 **수동 검수 및 보정** 권장

> **참고**: 학습 시 텍스트는 Qwen2.5-VL-7B (reason1p1_7B) 인코더로 on-the-fly 인코딩됩니다.

---

## Control 입력 생성 (선택사항)

> **Edge/Visual Blur Control을 사용하는 경우 이 단계를 건너뛰세요.** 학습 중 자동으로 계산됩니다.

### Depth Control 입력 생성

Depth Control은 3D 공간 구조를 보존하는 데 유리합니다. 로보틱스 Sim2Real에 특히 적합합니다.

```bash
# 단일 비디오의 depth 맵 생성
python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/depth/video1.mp4 \
    --encoder vits

# 전체 비디오 일괄 처리
for video in datasets/your_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/your_dataset/depth/$basename" \
        --encoder vits
done
```

**파라미터:**
- `--encoder vits`: 빠르고 가벼운 모델 (권장)
- `--encoder vitl`: 느리지만 더 정확한 모델

**출력**: 입력과 동일한 해상도/프레임 수의 그레이스케일 depth 비디오 (MP4)

### Segmentation Control 입력 생성

Segmentation Control은 객체 수준의 세분화된 제어를 제공합니다.

```bash
# 텍스트 프롬프트 방식 (권장)
python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/seg/video1.mp4 \
    --mode prompt \
    --prompt "robot, table, object, floor, wall" \
    --visualize

# 바운딩 박스 방식
python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/seg/video1.mp4 \
    --mode box \
    --box "300,0,500,400" \
    --visualize

# 포인트 방식
python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/seg/video1.mp4 \
    --mode points \
    --points "200,300" \
    --labels "1" \
    --visualize
```

**프롬프트 작성 팁** (Segmentation용):
- 장면에 등장하는 **주요 객체**를 콤마로 구분
- 시뮬레이션에서 동일하게 사용할 객체 카테고리와 **일관된 명칭** 사용
- 예: `"person, car, vehicle, building, tree, road, sky"`

---

## 데이터셋 폴더 구조

### 기본 구조 (Edge/Visual Blur Control)

```
datasets/my_sim2real_dataset/
├── videos/
│   ├── scene_001.mp4
│   ├── scene_002.mp4
│   ├── scene_003.mp4
│   └── ...
└── captions/
    ├── scene_001.json
    ├── scene_002.json
    ├── scene_003.json
    └── ...
```

### Depth Control 포함 구조

```
datasets/my_sim2real_dataset/
├── videos/
│   ├── scene_001.mp4
│   └── ...
├── captions/
│   ├── scene_001.json
│   └── ...
└── depth/
    ├── scene_001.mp4
    └── ...
```

### Segmentation Control 포함 구조

```
datasets/my_sim2real_dataset/
├── videos/
│   ├── scene_001.mp4
│   └── ...
├── captions/
│   ├── scene_001.json
│   └── ...
└── seg/
    ├── scene_001.mp4
    └── ...
```

### 중요: 파일명 매칭 규칙

| videos/ | captions/ | depth/ (선택) | seg/ (선택) |
|---------|-----------|-------------|------------|
| scene_001.mp4 | scene_001.json | scene_001.mp4 | scene_001.mp4 |
| scene_002.mp4 | scene_002.json | scene_002.mp4 | scene_002.mp4 |

**파일명에서 확장자를 제외한 이름이 정확히 일치해야 합니다.**

---

## 데이터 검증 체크리스트

데이터 준비를 완료한 후, 학습 시작 전 아래 항목을 모두 확인하세요.

### 필수 확인 항목

- [ ] **비디오 형식**: 모든 비디오가 MP4 형식인가?
- [ ] **해상도**: 720p (1280×720)로 통일되어 있는가?
- [ ] **프레임 수**: 모든 비디오가 93프레임 이상인가?
- [ ] **프레임레이트**: 10~60 fps 범위 내인가?
- [ ] **캡션 파일**: 각 비디오에 대응하는 JSON 파일이 있는가?
- [ ] **파일명 매칭**: 비디오 파일명과 캡션 파일명이 일치하는가?
- [ ] **캡션 형식**: JSON에 `"caption"` 키가 있고 텍스트가 들어있는가?
- [ ] **폴더 구조**: `videos/`와 `captions/` 디렉토리가 올바른가?

### 선택 확인 항목 (Depth/Seg Control 사용 시)

- [ ] **Depth/Seg 파일**: 각 비디오에 대응하는 control 파일이 있는가?
- [ ] **파일명 매칭**: control 파일명이 비디오 파일명과 일치하는가?
- [ ] **해상도 일치**: control 비디오의 해상도가 원본과 동일한가?
- [ ] **프레임 수 일치**: control 비디오의 프레임 수가 원본과 동일한가?

### 검증 스크립트

```bash
#!/bin/bash
# validate_dataset.sh
# 사용법: bash validate_dataset.sh datasets/my_sim2real_dataset

DATASET_DIR=$1
ERRORS=0

echo "=== Dataset Validation ==="
echo "Directory: $DATASET_DIR"
echo ""

# 1. Check directory structure
echo "[1] Checking directory structure..."
if [ ! -d "$DATASET_DIR/videos" ]; then
    echo "  ✗ ERROR: videos/ directory not found"
    ERRORS=$((ERRORS + 1))
else
    VIDEO_COUNT=$(ls "$DATASET_DIR/videos/"*.mp4 2>/dev/null | wc -l)
    echo "  ✓ videos/ directory found ($VIDEO_COUNT videos)"
fi

if [ ! -d "$DATASET_DIR/captions" ]; then
    echo "  ✗ ERROR: captions/ directory not found"
    ERRORS=$((ERRORS + 1))
else
    CAPTION_COUNT=$(ls "$DATASET_DIR/captions/"*.json 2>/dev/null | wc -l)
    echo "  ✓ captions/ directory found ($CAPTION_COUNT captions)"
fi

# 2. Check file name matching
echo ""
echo "[2] Checking file name matching..."
for video in "$DATASET_DIR/videos/"*.mp4; do
    basename=$(basename "$video" .mp4)
    if [ ! -f "$DATASET_DIR/captions/$basename.json" ]; then
        echo "  ✗ Missing caption: $basename.json"
        ERRORS=$((ERRORS + 1))
    fi
done

# 3. Check video specs
echo ""
echo "[3] Checking video specifications..."
for video in "$DATASET_DIR/videos/"*.mp4; do
    basename=$(basename "$video")
    frame_count=$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of csv=p=0 "$video" 2>/dev/null)
    resolution=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=p=0 "$video" 2>/dev/null)

    if [ "$frame_count" -lt 93 ] 2>/dev/null; then
        echo "  ✗ $basename: only $frame_count frames (need ≥93)"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "=== PASSED: All checks passed! ==="
else
    echo "=== FAILED: $ERRORS error(s) found ==="
fi
```

---

## 다음 단계

데이터 준비가 완료되었다면, [03. Post-Training 실행 가이드](./03_post_training_execution_guide.md)로 이동하여 실제 학습을 진행하세요.

## 관련 문서

- [01. Post-Training 개요 및 Sim2Real 분석](./01_post_training_overview_and_sim2real.md)
- [03. Post-Training 실행 가이드](./03_post_training_execution_guide.md)
- [04. 이론 및 학습 자료](./04_theory_and_learning.md)
