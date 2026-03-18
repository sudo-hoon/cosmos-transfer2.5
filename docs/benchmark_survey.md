# 월드 모델 / 영상 생성 모델 평가 메트릭 서베이

> 탑 학회(CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, IROS, CoRL, AAAI, SIGGRAPH) 2024-2025 논문 기반
> 작성일: 2026-03-18

---

## 목차

1. [논문별 사용 메트릭 리스트](#1-논문별-사용-메트릭-리스트)
2. [메트릭 사용 빈도 분석](#2-메트릭-사용-빈도-분석)
3. [주요 메트릭 상세 설명](#3-주요-메트릭-상세-설명)
4. [주요 벤치마크 프레임워크](#4-주요-벤치마크-프레임워크)
5. [평가 데이터셋](#5-평가-데이터셋)
6. [Best Practice: 월드 모델 평가 가이드](#6-best-practice-월드-모델-평가-가이드)

---

## 1. 논문별 사용 메트릭 리스트

### 1.1 영상 생성 모델 (Text-to-Video / Image-to-Video)

| 논문 | 학회 | 연도 | FVD | FID | IS | CLIPSIM | PSNR | SSIM | LPIPS | VBench | 기타 |
|------|------|------|-----|-----|----|---------|----- |------|-------|--------|------|
| VideoCrafter2 | CVPR | 2024 | ✓ | ✓ | ✓ | ✓ | | | | ✓ | EvalCrafter |
| DynamiCrafter | ECCV | 2024 | ✓ | | | | | | | ✓ | KVD |
| AnimateDiff | ICLR | 2024 | ✓ | | | ✓ | | | | | User study |
| CogVideoX | ICLR | 2025 | ✓ | | | | | | | ✓ | CLIP4Clip |
| Open-Sora 2.0 | arXiv | 2025 | | | | | | | | ✓ | Human pref |
| StreamingT2V | CVPR | 2025 | | | | ✓ | | | ✓ | | MAWE, SCuts |
| SVD | arXiv | 2024 | ✓ | | | | | | | | Human pref |
| Make-A-Video | arXiv | 2022 | ✓ | ✓ | ✓ | ✓ | | | | | |
| Show-1 | arXiv | 2023 | ✓ | ✓ | ✓ | ✓ | | | | | |
| VideoPoet | arXiv | 2023 | ✓ | | ✓ | ✓ | | | | | Human eval |
| LaVie | arXiv | 2023 | ✓ | | | ✓ | | | | | MOS |
| Lumiere | SIGGRAPH Asia | 2024 | ✓ | | ✓ | | | | | | Human eval |

### 1.2 자율주행 월드 모델 / Multi-View 생성

| 논문 | 학회 | 연도 | FVD | FID | mAP | NDS | mIoU | PSNR | SSIM | LPIPS | 기타 |
|------|------|------|-----|-----|-----|-----|------|------|------|-------|------|
| MagicDrive | CVPR | 2024 | | ✓ | | ✓ | ✓ (Road/Veh) | | | | |
| GenAD | CVPR (H) | 2024 | ✓ | ✓ | | | | | | | Traj Diff, CLIP-Sim |
| Drive-WM | CVPR | 2024 | ✓ | ✓ | | | | | | | |
| Panacea | CVPR | 2024 | ✓ | ✓ | ✓ | ✓ | | | | | VMS |
| DrivingGaussian | CVPR | 2024 | | | | | | ✓ | ✓ | ✓ | |
| Cam4DOcc | CVPR | 2024 | | | | | ✓ (IoU_f) | | | | |
| DriveDreamer | ECCV | 2024 | ✓ | ✓ | | | | | | | |
| WoVoGen | ECCV | 2024 | ✓ | ✓ | | | | | | | |
| DrivingDiffusion | ECCV | 2024 | ✓ | ✓ | | ✓ | ✓ | | | | |
| Street Gaussians | ECCV | 2024 | | | | | | ✓ | ✓ | ✓ | PSNR* (fg) |
| OccWorld | ECCV | 2024 | | | | | ✓ (IoU) | | | | L2 error, Collision rate |
| SV3D | ECCV | 2024 | | | | | | ✓ | ✓ | ✓ | CLIP-S |
| Vista | NeurIPS | 2024 | ✓ | ✓ | | | | | | | Traj Diff |
| Vivid-ZOO | NeurIPS | 2024 | ✓ | | | | | | | | CLIP score |
| DriveDreamer-2 | AAAI | 2025 | ✓ | ✓ | ✓ | ✓ | | | | | AMOTA/AMOTP |
| SubjectDrive | AAAI | 2025 | ✓ | ✓ | ✓ | ✓ | | | | | |
| DriveScape | CVPR | 2025 | ✓ | ✓ | | ✓ | ✓ | | | | |
| GEN3C | CVPR (H) | 2025 | | | | | | ✓ | ✓ | ✓ | TSED |
| CAT4D | CVPR | 2025 | | | | | | ✓ | ✓ | ✓ | |

### 1.3 월드 모델 (RL / Interactive / General)

| 논문 | 학회 | 연도 | HNS | IQM | FVD | FID | PSNR | SSIM | LPIPS | 기타 |
|------|------|------|-----|-----|-----|-----|------|------|-------|------|
| Genie | ICML | 2024 | | | ✓ | | ✓ | | | δt-PSNR |
| Delta-IRIS | ICML | 2024 | ✓ | ✓ | | | | | | Crafter score |
| DreamerV3 | ICLR/Nature | 2024/25 | ✓ | ✓ | | | | | | Task reward |
| R2I | ICLR | 2024 | ✓ | ✓ | | | | | | Task reward |
| UniSim | ICLR | 2024 | | | ✓ | ✓ | | | | IS, CLIP Score |
| DIAMOND | NeurIPS (S) | 2024 | ✓ | ✓ | | | | | | Per-game score |
| iVideoGPT | NeurIPS | 2024 | | | ✓ | | ✓ | ✓ | ✓ | VP2 score |
| GameNGen | ICLR | 2025 | | | ✓ | | ✓ | | ✓ | Human eval |
| GameGen-X | ICLR | 2025 | | | ✓ | ✓ | | | | TVA, SR-C/E, MS, DD |
| Cosmos | arXiv | 2025 | | | rFVD | rFID | ✓ | ✓ | | Sampson error, VBench |
| GenEx | ICLR sub | 2025 | | | ✓ | | | ✓ | ✓ | IELC |

---

## 2. 메트릭 사용 빈도 분석

아래는 위 42편 논문에서 각 메트릭이 사용된 횟수를 집계한 결과입니다.

### 2.1 전체 빈도 (높은 순)

| 순위 | 메트릭 | 사용 논문 수 | 비율 | 카테고리 |
|------|--------|------------|------|---------|
| 1 | **FVD** | 27 | 64% | 분포 품질 |
| 2 | **FID** | 20 | 48% | 분포 품질 |
| 3 | **PSNR** | 14 | 33% | 재구성 품질 |
| 4 | **SSIM** | 12 | 29% | 재구성 품질 |
| 5 | **LPIPS** | 12 | 29% | 지각 유사도 |
| 6 | **CLIPSIM / CLIP Score** | 11 | 26% | 텍스트-비디오 정합 |
| 7 | **IS (Inception Score)** | 7 | 17% | 분포 품질 |
| 8 | **VBench (16차원)** | 6 | 14% | 다차원 종합 |
| 9 | **mAP (3D detection)** | 6 | 14% | 다운스트림 태스크 |
| 10 | **NDS** | 6 | 14% | 다운스트림 태스크 |
| 11 | **mIoU (segmentation)** | 5 | 12% | 다운스트림 태스크 |
| 12 | **HNS / IQM** | 6 | 14% | RL 태스크 성능 |
| 13 | **Human Evaluation** | 8 | 19% | 주관적 품질 |
| 14 | **Trajectory Difference** | 2 | 5% | 제어 정합 |

### 2.2 도메인별 핵심 메트릭

| 도메인 | Tier 1 (거의 필수) | Tier 2 (자주 사용) | Tier 3 (보충) |
|--------|-------------------|-------------------|--------------|
| **영상 생성 (T2V/I2V)** | FVD, FID | CLIPSIM, VBench | IS, Human eval, KVD |
| **자율주행 월드 모델** | FID, FVD | mAP, NDS, mIoU | Traj Diff, AMOTA |
| **Multi-View / NVS** | PSNR, SSIM, LPIPS | FVD, FID | TSED, MEt3R, CLIP-S |
| **RL 월드 모델** | HNS, IQM | Task reward | Per-game score |
| **게임 시뮬레이션** | FVD, PSNR, LPIPS | FID, Human eval | TVA, SR |

### 2.3 시간에 따른 트렌드 변화

```
2023-2024 초반: FVD + FID + IS 가 표준 3종 세트
2024 중반:      VBench 등장 → 다차원 평가로 전환 시작
2024 후반:      FVD의 한계 공식화 (JEDi, FVMD, DEVIL)
2025:           VBench 채택 급증 + 물리/상식 평가 부상 (PhyGenBench, VBench-2.0)
                Downstream task 성능 평가 주류화 (mAP, NDS, mIoU)
```

---

## 3. 주요 메트릭 상세 설명

### 3.1 분포 기반 메트릭 (Distribution-level)

#### FVD (Fréchet Video Distance)
- **정의**: I3D 네트워크로 추출한 실제/생성 비디오 특징의 Fréchet distance
- **사용률**: 64% (가장 널리 사용)
- **알려진 문제점**:
  - I3D 특징이 Gaussian 분포를 따르지 않음 (FVD의 핵심 가정 위반) — JEDi, ICLR 2025
  - 시간적 왜곡에 둔감 (프레임 셔플링에도 FVD 변화 미미) — FVMD, ICML 2024
  - 정적 비디오에 유리한 편향 (움직임 적은 비디오가 높은 점수) — DEVIL, NeurIPS 2024
  - 수렴에 4,350+ 샘플 필요 — JEDi, ICLR 2025
- **대안**: JEDi (V-JEPA 특징 + MMD, 샘플 효율 6.25배), FVMD (모션 특화)
- **현재 위상**: 사실상 표준이지만 한계가 공식적으로 입증됨. 보고는 하되 유일한 메트릭으로 의존하지 말 것

#### FID (Fréchet Inception Distance)
- **정의**: 개별 프레임 수준의 Inception-v3 특징 Fréchet distance
- **사용률**: 48%
- **용도**: 프레임별 시각 품질 (temporal 정보 없음)
- **현재 위상**: 보조 메트릭으로 여전히 유효

#### IS (Inception Score)
- **정의**: 생성 이미지의 다양성과 품질을 Inception 분류기로 측정
- **사용률**: 17% (감소 추세)
- **현재 위상**: UCF-101 벤치마크에서 관습적으로 보고하나, 단독 사용 비권장

### 3.2 재구성 품질 메트릭 (Pixel-level)

#### PSNR (Peak Signal-to-Noise Ratio)
- **정의**: 픽셀 레벨 MSE 기반 품질 측정 (dB 단위)
- **사용률**: 33% (주로 NVS, 재구성, 게임 시뮬레이션)
- **특성**: 지각 품질과 상관이 낮을 수 있으나 재구성 정확도 측정에 표준

#### SSIM (Structural Similarity Index)
- **정의**: 구조적 유사도 (밝기, 대비, 구조)
- **사용률**: 29%

#### LPIPS (Learned Perceptual Image Patch Similarity)
- **정의**: VGG/AlexNet 특징 공간에서의 지각적 거리
- **사용률**: 29%
- **특성**: 인간 지각과의 상관이 PSNR/SSIM보다 높음

### 3.3 텍스트-비디오 정합 메트릭

#### CLIPSIM / CLIP Score
- **정의**: CLIP 임베딩 공간에서 텍스트-비디오 코사인 유사도
- **사용률**: 26%
- **한계**: 세밀한 의미론적 차이 감지에 한계

### 3.4 다운스트림 태스크 메트릭

#### mAP / NDS (3D Object Detection)
- **정의**: 생성된 데이터로 학습한 검출기의 nuScenes 성능
- **사용률**: 14% (자율주행 도메인에서는 사실상 필수)
- **평가 방법**: StreamPETR, BEVFusion, BEVFormer 등을 downstream detector로 사용
- **의의**: 생성 품질을 **실용적 가치**로 직접 측정

#### mIoU (Segmentation)
- **정의**: 도로/차량 BEV 세그멘테이션 IoU
- **사용률**: 12% (자율주행)
- **평가 방법**: CVT 등 BEV 세그멘테이션 모델 사용

### 3.5 다차원 종합 벤치마크

#### VBench (16개 차원)
- **학회**: CVPR 2024 (Highlight)
- **16개 차원**: Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality, Object Class, Multiple Objects, Human Action, Color, Spatial Relationship, Scene, Temporal Style, Appearance Style, Overall Consistency
- **채택률**: 급증 중. 40+ 모델이 리더보드에 등록
- **현재 위상**: T2V 평가의 새로운 사실상 표준

---

## 4. 주요 벤치마크 프레임워크

### 4.1 범용 영상 생성 평가

| 벤치마크 | 학회 | 핵심 기여 | 차원 수 | 채택도 |
|---------|------|----------|--------|-------|
| **VBench** | CVPR 2024 (H) | 16차원 분해, 자동 파이프라인 | 16 | ★★★★★ |
| **VBench-2.0** | arXiv 2025 | 물리·상식·해부학적 정확도 추가 | 18 | ★★★☆☆ |
| **EvalCrafter** | CVPR 2024 | 17개 메트릭 + 학습된 가중 조합 | 17 | ★★★☆☆ |
| **DEVIL** | NeurIPS 2024 | 동적 범위/제어/품질 평가 | 3 | ★★★☆☆ |
| **T2V-CompBench** | CVPR 2025 | 구성적 생성 능력 (7 카테고리) | 7 | ★★☆☆☆ |
| **Video-Bench** | CVPR 2025 | MLLM 기반 자동 평가 | - | ★★☆☆☆ |
| **VideoScore** | EMNLP 2024 | 학습된 5차원 품질 모델 | 5 | ★★★☆☆ |
| **ChronoMagic-Bench** | NeurIPS 2024 (S) | 시간 경과(time-lapse) 생성 평가 | 2 | ★★☆☆☆ |

### 4.2 월드 모델 전용 평가

| 벤치마크 | 학회 | 핵심 기여 | 채택도 |
|---------|------|----------|-------|
| **WorldScore** | ICCV 2025 | 3D/4D/비디오 통합 평가 (3000개 테스트) | ★★★☆☆ |
| **WorldModelBench** | NeurIPS 2025 D&B | 7개 응용 도메인, 56개 서브도메인, 67K 인간 라벨 | ★★★☆☆ |
| **PAI-Bench** | NVIDIA 내부 | Physical AI 도메인 스코어 + 품질 스코어 | ★★☆☆☆ |

### 4.3 물리 충실도 평가

| 벤치마크 | 학회 | 핵심 기여 | 채택도 |
|---------|------|----------|-------|
| **PhyGenBench** | ICML 2025 | 27개 물리 법칙, 계층적 VLM 평가 | ★★★☆☆ |
| **VideoPhy** | ICLR 2025 | 재료 상호작용 물리 (688 프롬프트) | ★★★☆☆ |
| **PhyWorldBench** | arXiv 2025 | 50개 시나리오 + Anti-Physics 카테고리 | ★★☆☆☆ |

### 4.4 분포 메트릭 개선 제안

| 메트릭 | 학회 | FVD 대비 개선 | 채택도 |
|--------|------|-------------|-------|
| **JEDi** | ICLR 2025 | V-JEPA 특징 + MMD, 샘플 84% 절감, 인간 정합 34%↑ | ★★★☆☆ |
| **FVMD** | ICML 2024 WS | 모션 특화 (속도+가속도 히스토그램) | ★★☆☆☆ |
| **VMBench** | ICCV 2025 | 지각 정렬 모션 메트릭 | ★★☆☆☆ |

### 4.5 Multi-View 3D 일관성

| 메트릭 | 학회 | 핵심 기여 | 채택도 |
|--------|------|----------|-------|
| **MEt3R** | CVPR 2025 | DUSt3R 기반 pose-free 3D 일관성 | ★★★☆☆ |
| **TSED** | - | Sampson epipolar distance (pose 필요) | ★★☆☆☆ |

---

## 5. 평가 데이터셋

### 5.1 영상 생성 표준 데이터셋

| 데이터셋 | 규모 | 주요 메트릭 | 현재 위상 |
|---------|------|-----------|----------|
| **UCF-101** | 13,320 비디오, 101 클래스 | FVD, IS | 레거시 표준 (여전히 널리 보고) |
| **MSR-VTT** | 10K 비디오, 200K 캡션 | CLIPSIM, FID, FVD | 레거시 표준 |
| **WebVid-2M/10M** | 웹 비디오-텍스트 쌍 | 내부 ablation용 | 훈련/ablation |

### 5.2 자율주행 데이터셋

| 데이터셋 | 규모 | 주요 용도 |
|---------|------|----------|
| **nuScenes** | 1000 시퀀스, 6카메라 | FID, FVD, mAP, NDS, mIoU (사실상 유일한 표준) |
| **Waymo Open** | 대규모 | PSNR/SSIM (NVS), zero-shot 평가 |
| **KITTI / KITTI-360** | 중규모 | PSNR/SSIM (NVS) |

### 5.3 RL / 게임 데이터셋

| 데이터셋 | 주요 메트릭 |
|---------|-----------|
| **Atari 100k** (26 games) | HNS, IQM |
| **Crafter** | Score |
| **DMC Suite** | Task reward |
| **DOOM (VizDoom)** | PSNR, LPIPS, FVD |

### 5.4 로봇 / 3D 데이터셋

| 데이터셋 | 주요 용도 |
|---------|----------|
| **BAIR** | FVD, LPIPS (로봇 pushing) |
| **RoboNet** | FVD, SSIM |
| **LIBERO / RoboCasa** | Task success rate |
| **GSO (Google Scanned Objects)** | PSNR, SSIM, LPIPS (NVS) |
| **RealEstate10K** | PSNR, SSIM, LPIPS, TSED |

---

## 6. Best Practice: 월드 모델 평가 가이드

### 6.1 결론 요약

탑 학회 논문들의 평가 방식을 종합하면, 다음과 같은 패턴이 명확합니다:

1. **FVD는 여전히 보고해야 하지만, 단독으로 신뢰하면 안 된다**
   - ICLR 2025 (JEDi)에서 3가지 핵심 결함이 공식 증명됨
   - NeurIPS 2024 (DEVIL)에서 정적 비디오 편향 입증
   - 그럼에도 42편 중 27편(64%)이 사용 → 비교 가능성을 위해 보고는 필수

2. **다차원 평가(VBench)가 새로운 표준으로 부상 중**
   - CVPR 2024 이후 급격히 채택 증가
   - CogVideoX(ICLR 2025), Open-Sora 등 최신 모델이 VBench를 1차 메트릭으로 채택

3. **도메인별 다운스트림 태스크 성능이 가장 신뢰도 높은 평가**
   - 자율주행: mAP/NDS (3D detection), mIoU (segmentation)
   - 로봇: Task success rate
   - RL: HNS/IQM

4. **물리 충실도 평가가 급부상**
   - PhyGenBench (ICML 2025), VideoPhy (ICLR 2025), VBench-2.0 모두 2024-2025 탑 학회

5. **Multi-view 3D 일관성은 MEt3R이 최선**
   - CVPR 2025, pose-free, DUSt3R 기반
   - 기존 Sampson Error, TSED는 GT pose 필요

### 6.2 도메인별 권장 평가 프로토콜

#### A. 범용 비디오 / 월드 생성 모델

```
필수 (Must-have):
├── VBench 16차원 점수 (또는 VBench-2.0)
├── FVD (UCF-101, 비교 가능성용)
├── FID (프레임 품질)
└── Human evaluation (최소 pairwise preference)

권장 (Recommended):
├── JEDi (FVD 대체/보완, ICLR 2025)
├── CLIPSIM (텍스트 정합)
├── DEVIL dynamics 메트릭 (동적 범위/제어/품질)
└── PhyGenBench 또는 VideoPhy (물리 충실도)

선택 (Optional):
├── FVMD (모션 품질 특화)
├── VideoScore (학습된 품질 모델)
├── WorldScore (3D/4D/비디오 통합 비교 시)
└── ChronoMagic-Bench (장기 변화 평가 시)
```

#### B. 자율주행 멀티뷰 월드 모델

```
필수 (Must-have):
├── FID (nuScenes)
├── FVD (nuScenes)
├── Downstream 3D Detection: mAP, NDS (StreamPETR 또는 BEVFusion)
└── Downstream BEV Segmentation: Road mIoU, Vehicle mIoU (CVT)

권장 (Recommended):
├── MEt3R (multi-view 3D 일관성, pose-free)
├── Trajectory Difference (제어 정합)
├── VMS (View Matching Score, cross-view 일관성)
└── Tracking 메트릭: AMOTA, AMOTP

선택 (Optional):
├── TSED / Sampson Error (epipolar 기하)
├── Human evaluation
└── Zero-shot 일반화: Waymo, KITTI, Cityscapes
```

#### C. Multi-View Novel View Synthesis

```
필수 (Must-have):
├── PSNR
├── SSIM
├── LPIPS
└── MEt3R (CVPR 2025, pose-free 3D 일관성)

권장 (Recommended):
├── TSED (epipolar 일관성)
├── FVD (비디오 시퀀스 시)
└── CLIP-S (의미적 일관성)
```

#### D. RL 월드 모델 (Atari, 게임)

```
필수 (Must-have):
├── Human Normalized Score (HNS)
├── Interquartile Mean (IQM)
└── Per-game raw score (최소 26 games)

권장 (Recommended):
├── PSNR / LPIPS (시각 재구성 품질)
├── FVD (생성 비디오 품질)
└── Human evaluation (실시간 플레이 구분 테스트)
```

### 6.3 메트릭 선택 의사결정 트리

```
Q1: 생성 모델의 목적은?
├── 범용 비디오 생성 → VBench + FVD + FID + Human eval
├── 자율주행 시뮬레이션 → FID/FVD + mAP/NDS/mIoU (downstream)
├── Multi-view 3D → PSNR/SSIM/LPIPS + MEt3R
├── RL 에이전트 학습 → HNS/IQM + Task reward
└── 물리 시뮬레이션 → PhyGenBench/VideoPhy + VBench-2.0

Q2: 물리적 충실도가 중요한가?
├── 예 → PhyGenBench (27 물리 법칙) 또는 VideoPhy (재료 상호작용) 추가
└── 아니오 → 기본 메트릭 세트로 충분

Q3: 장기 비디오 생성인가?
├── 예 → MAWE (StreamingT2V), RNDS (Cosmos), ChronoMagic-Bench 추가
└── 아니오 → 표준 클립 레벨 메트릭

Q4: Multi-view 일관성이 중요한가?
├── 예, GT pose 있음 → TSED + PSNR/SSIM/LPIPS
├── 예, GT pose 없음 → MEt3R (CVPR 2025)
└── 아니오 → 단일 뷰 메트릭
```

### 6.4 주의사항

1. **FVD만으로 논문을 평가하지 말 것** — 정적 비디오 생성으로 "속일 수" 있음 (DEVIL, NeurIPS 2024)
2. **IS는 보조 참고용** — 사용 감소 추세, 단독으로는 의미 없음
3. **다운스트림 태스크 성능이 가장 강력한 증거** — 특히 자율주행에서 mAP/NDS 향상이 실질적 가치 입증
4. **VBench는 T2V에 최적화** — 조건부 생성(I2V, layout-to-video)에는 VBench++ 또는 도메인 특화 벤치 필요
5. **인간 평가는 항상 포함할 것** — 자동 메트릭과 인간 판단 간 괴리가 큼 (GenAI Arena: GPT-4o도 49% 정확도)

---

## 참고문헌

### 벤치마크 / 메트릭 논문
- VBench: Huang et al., CVPR 2024 — [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf) | [GitHub](https://github.com/Vchitect/VBench)
- VBench-2.0: arXiv:2503.21755 — [Paper](https://arxiv.org/abs/2503.21755)
- EvalCrafter: Liu et al., CVPR 2024 — [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.pdf)
- JEDi: Luo et al., ICLR 2025 — [Paper](https://arxiv.org/abs/2410.05203) | [GitHub](https://github.com/oooolga/JEDi)
- FVMD: Liu et al., ICML 2024 WS — [Paper](https://arxiv.org/abs/2407.16124) | [GitHub](https://github.com/DSL-Lab/FVMD-frechet-video-motion-distance)
- DEVIL: Liao et al., NeurIPS 2024 — [Paper](https://arxiv.org/abs/2407.01094) | [GitHub](https://github.com/MingXiangL/DEVIL)
- WorldScore: Duan et al., ICCV 2025 — [Paper](https://arxiv.org/abs/2504.00983) | [GitHub](https://github.com/haoyi-duan/WorldScore)
- WorldModelBench: Li et al., NeurIPS 2025 D&B — [Paper](https://arxiv.org/abs/2502.20694) | [GitHub](https://github.com/WorldModelBench-Team/WorldModelBench)
- MEt3R: Asim et al., CVPR 2025 — [Paper](https://arxiv.org/abs/2501.06336) | [GitHub](https://github.com/mohammadasim98/met3r)
- PhyGenBench: Meng et al., ICML 2025 — [Paper](https://arxiv.org/abs/2410.05363) | [GitHub](https://github.com/OpenGVLab/PhyGenBench)
- VideoPhy: Bansal et al., ICLR 2025 — [Paper](https://arxiv.org/abs/2406.03520)
- T2V-CompBench: Sun et al., CVPR 2025 — [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_T2V-CompBench_A_Comprehensive_Benchmark_for_Compositional_Text-to-video_Generation_CVPR_2025_paper.pdf)
- ChronoMagic-Bench: PKU, NeurIPS 2024 Spotlight — [Paper](https://arxiv.org/abs/2406.18522)
- VideoScore: TIGER-AI-Lab, EMNLP 2024 — [Paper](https://arxiv.org/abs/2406.15252)
- VMBench: Ling et al., ICCV 2025 — [GitHub](https://github.com/AMAP-ML/VMBench)
- Video-Bench: Han et al., CVPR 2025 — [GitHub](https://github.com/Video-Bench/Video-Bench)

### 월드 모델 논문
- DIAMOND: Alonso et al., NeurIPS 2024 Spotlight — [Paper](https://arxiv.org/abs/2405.12399)
- GameNGen: Valevski et al., ICLR 2025 — [Paper](https://arxiv.org/abs/2408.14837)
- GameGen-X: ICLR 2025 — [OpenReview](https://openreview.net/forum?id=8VG8tpPZhe)
- Genie: Bruce et al., ICML 2024 Oral — [Paper](https://arxiv.org/abs/2402.15391)
- UniSim: ICLR 2024 — [Paper](https://arxiv.org/abs/2310.06114)
- DreamerV3: Hafner et al., Nature 2025 — [Paper](https://www.nature.com/articles/s41586-025-08744-2)
- iVideoGPT: NeurIPS 2024 — [Paper](https://arxiv.org/abs/2405.15223)
- Cosmos: NVIDIA, arXiv 2025 — [Paper](https://arxiv.org/abs/2501.03575)

### 자율주행 / Multi-View 논문
- MagicDrive: Gao et al., CVPR 2024 — nuScenes FID 16.20
- GenAD: Yang et al., CVPR 2024 Highlight — nuScenes FID 15.4, FVD 184.0
- Panacea: Wen et al., CVPR 2024 — nuScenes FVD 139, FID 16.96
- DriveDreamer: Wang et al., ECCV 2024 — nuScenes FID 52.6, FVD 452.0
- OccWorld: ECCV 2024 — IoU 29.17, mIoU 19.93
- Vista: NeurIPS 2024 — nuScenes FID 6.9, FVD 89.4
- DriveScape: Wu et al., CVPR 2025 — nuScenes FID 8.34, FVD 76.39
- GEN3C: NVIDIA, CVPR 2025 Highlight — RE10K PSNR 19.88, SSIM 0.78
- DriveDreamer-2: AAAI 2025 — nuScenes FID 11.2, FVD 55.7
- SubjectDrive: AAAI 2025 — nuScenes FID 15.98, FVD 124

### 영상 생성 논문
- CogVideoX: Tsinghua, ICLR 2025 — VBench 기반
- VideoCrafter2: CVPR 2024
- DynamiCrafter: ECCV 2024 Oral — VBench I2V 1위
- AnimateDiff: ICLR 2024
- StreamingT2V: CVPR 2025 — MAWE 10.87
- SVD: Stability AI — UCF-101 FVD 242.02
- SV3D: ECCV 2024 — GSO LPIPS 0.08, PSNR 21.26
