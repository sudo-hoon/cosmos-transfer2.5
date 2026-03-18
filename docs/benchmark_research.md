# NVIDIA Cosmos Transfer 2.5 — 벤치마크 및 평가 메트릭 조사 보고서

## 목차
1. [NVIDIA Cosmos 2.5 논문에서 사용된 메트릭](#1-nvidia-cosmos-25-논문에서-사용된-메트릭)
2. [일반 비디오 생성 품질 메트릭](#2-일반-비디오-생성-품질-메트릭)
3. [Temporal Consistency 메트릭](#3-temporal-consistency-메트릭)
4. [Multi-View Consistency 메트릭](#4-multi-view-consistency-메트릭)
5. [World Model 전용 벤치마크](#5-world-model-전용-벤치마크)
6. [종합 벤치마크 스위트](#6-종합-벤치마크-스위트)
7. [NVIDIA 자체 평가 도구](#7-nvidia-자체-평가-도구)
8. [Downstream Task 기반 평가](#8-downstream-task-기반-평가)
9. [권장 벤치마크 조합](#9-권장-벤치마크-조합)

---

## 1. NVIDIA Cosmos 2.5 논문에서 사용된 메트릭

> 논문: "World Simulation with Video Foundation Models for Physical AI" (arXiv: 2511.00062), NVIDIA 88명 저자

### 1.1 PAI-Bench (Physical AI Bench) — CVPR 2026 Accepted

NVIDIA가 Cosmos 모델 평가를 위해 개발한 전용 벤치마크 ([arXiv:2512.01989](https://arxiv.org/abs/2512.01989), [GitHub](https://github.com/SHI-Labs/physical-ai-bench)). 3가지 트랙으로 구성:

| 트랙 | 설명 |
|------|------|
| **PAI-Bench-G (Generation)** | 비디오 생성 모델 평가. Quality Score (VBench 기반 8개 T2V/I2V 메트릭) + Domain Score (Qwen3-VL-235B MLLM-as-Judge로 물리적 그럴듯함 검증) |
| **PAI-Bench-C (Conditional Control)** | Transfer 모델의 멀티모달 제어 신호 충실도, 시각적 품질, 생성 다양성 평가 |
| **PAI-Bench-U (Understanding)** | MLLM의 물리적 상식 및 체현 추론 평가 |

> **핵심 발견**: Domain Score가 Quality Score보다 ~10점 낮음 → 시각적 품질이 물리적 그럴듯함을 보장하지 않음

#### PAI-Bench-Transfer (Control 충실도)

Transfer 모델의 control signal 충실도를 측정한다.

| 메트릭 | 설명 | 방향 | Transfer1-7B | Transfer2.5-2B |
|--------|------|------|-------------|----------------|
| **Blur SSIM** | 블러 제어 입력에 대한 구조적 유사도 | ↑ Higher is better | 0.82 | **0.87** |
| **Edge F1** | 엣지 맵 제어 입력에 대한 F1 스코어 | ↑ Higher is better | 0.26 | **0.41** |
| **Depth RMSE** (si-RMSE) | 깊이 맵에 대한 Scale-invariant RMSE | ↓ Lower is better | 0.70 | **0.67** |
| **Seg mIoU** | 세그멘테이션 맵에 대한 Mean IoU | ↑ Higher is better | 0.74 | **0.76** |
| **Quality Score** (DOVER) | 전체 비디오 미적 품질 (DOVER-technical) | ↑ Higher is better | 9.24 | **9.31** |

### 1.2 RNDS (Relative Normalized DOVER Score)

- 장기 비디오 생성 시 **에러 누적(error accumulation)** 을 측정
- Autoregressive rollout 시 시간이 지남에 따라 품질이 얼마나 저하되는지를 DOVER 점수의 상대적 변화로 측정
- Transfer2.5는 4가지 control modality(edge/blur/depth/segmentation) 모두에서 Transfer1-7B 대비 낮은 에러 누적을 보임
- Closed-loop simulation 및 RL 태스크에서 핵심적 — autoregressive chunked 비디오 모델의 주요 실패 모드를 다룸

### 1.3 VideoAlign Reward Model

RLHF 스타일 VLM 기반 보상 모델. 후훈련(post-training) 시 사용하며 3가지 차원 평가:
- **Text Alignment**: 비디오가 텍스트 프롬프트와 얼마나 잘 일치하는지
- **Motion Quality**: 시간적 일관성과 사실적 모션
- **Visual Quality**: 프레임별 시각적 품질

> 훈련 시 입력당 8개 출력 생성 (20 diffusion step), GRPO 스타일 정규화로 advantage 계산

### 1.4 Physics Alignment Metrics (PhysX / Isaac Sim)

NVIDIA PhysX 및 Isaac Sim을 사용한 물리 정합성 평가:
- **8가지 통제된 시나리오**: 중력(gravity), 충돌(collision), 토크(torque), 관성(inertia) 테스트
- 표준 비디오 생성 벤치마크를 넘어서는 **월드 모델 전용 물리 메트릭**

### 1.5 Diversity-LPIPS

- 동일 제어 입력에 다른 텍스트 프롬프트를 사용했을 때 생성 결과의 다양성을 측정
- LPIPS(Learned Perceptual Image Patch Similarity) 기반
- 높을수록 더 다양한 생성 결과를 의미

### 1.4 자율주행 Multi-View 평가 메트릭

| 메트릭 | 도구 | 설명 |
|--------|------|------|
| **3D-Bbox mAP** | BEVFormer | 생성된 multi-view 영상에서 3D 객체 탐지 정확도 |
| **Lane mIoU** | LATR | 차선 탐지 정확도 |
| **Reprojection Error** | - | 다시점 간 재투영 오류 (↓) |
| **FVD** | I3D network | Fréchet Video Distance (시각+시간 품질) |
| **FID** | InceptionV3 | Fréchet Inception Distance (프레임 수준 품질) |

> Transfer2.5는 Transfer1-7B 대비 lane/cuboid detection에서 **최대 60% 개선**을 보고함.

**Cosmos-Transfer1 논문 (Table 4) 구체적 결과:**

| 설정 | 3D-Bbox mAP ↑ | Lane mIoU ↑ | Reprojection Error ↓ |
|------|---------------|------------|---------------------|
| HDMap only | 41.89 | 50.37 | 9.46 |
| LiDAR only | 46.50 | 48.19 | 8.60 |
| HDMap + LiDAR (Multi-modal) | 44.66 | 51.55 | 8.67 |

---

## 2. 일반 비디오 생성 품질 메트릭

### 2.1 FVD (Fréchet Video Distance)

- **가장 널리 사용되는 비디오 생성 메트릭**
- I3D 네트워크의 feature space에서 생성 비디오와 실제 비디오 분포 간 거리를 측정
- 시각적 품질 + 시간적 일관성을 동시에 평가
- 범위: Excellent (<100), Good (100-200), Acceptable (200-400), Poor (>400)

### 2.2 FID (Fréchet Inception Distance)

- 프레임 단위 시각적 품질/사실성 측정
- InceptionV3 네트워크의 feature space에서 분포 간 거리 계산
- 범위: Excellent (<10), Good (10-30), Acceptable (30-50), Poor (>50)

### 2.3 IS (Inception Score)

- 생성 이미지의 품질과 다양성을 동시에 측정
- 높을수록 좋음

### 2.4 LPIPS (Learned Perceptual Image Patch Similarity)

- 인간 지각과 잘 일치하는 perceptual similarity 메트릭
- 두 이미지 간 perceptual distance 측정 (낮을수록 유사)

### 2.5 SSIM (Structural Similarity Index)

- 밝기, 대비, 구조를 기반으로 이미지 유사도 측정
- 범위: 0~1, 높을수록 좋음

### 2.6 PSNR (Peak Signal-to-Noise Ratio)

- 픽셀 수준 재구성 품질 측정
- 높을수록 좋음 (보통 20-40 dB 범위)

### 2.7 CLIP Score (CLIPSim)

- CLIP 모델을 사용하여 텍스트 프롬프트와 생성 비디오 간 의미적 정합도 측정
- 텍스트-비디오 정합성 평가에 유용
- **한계**: 시간적 비일관성을 의미적 내용이 바뀌지 않는 한 감지 못함

### 2.8 DOVER (Disentangled Objective Video Quality Evaluator)

- 비디오 미적 품질과 기술적 품질을 분리하여 평가
- NVIDIA Cosmos 논문에서 Quality Score로 사용됨

---

## 3. Temporal Consistency 메트릭

### 3.1 Temporal Flickering (VBench)

- 연속 프레임 간 고주파 디테일의 불안정성(깜빡임) 감지
- 실제 비디오는 조명/카메라 흔들림으로 인해 발생하지만, 생성 비디오에서는 모델의 불완전한 시간적 일관성이 원인

### 3.2 Motion Smoothness (VBench)

- 생성 비디오의 모션이 부드럽고 물리 법칙을 따르는지 평가
- 비디오 프레임 보간 모델의 motion prior를 활용

### 3.3 Subject Consistency (VBench)

- 주체(사람, 객체)가 프레임 간에 일관된 외형을 유지하는지 측정
- DINO feature를 사용한 프레임 간 유사도 계산

### 3.4 Background Consistency (VBench)

- 배경이 프레임 간에 일관되게 유지되는지 측정
- CLIP feature를 사용한 배경 영역 유사도

### 3.5 Warping Error

- Optical flow를 이용하여 인접 프레임 간 warping 후 픽셀 차이를 측정
- 시간적 일관성의 직접적인 정량적 측정

### 3.6 Flow-based Temporal Consistency

- Optical flow의 시간적 부드러움을 측정
- 급격한 flow 변화는 시간적 비일관성을 나타냄

### 3.7 CLIP Frame Consistency

- 연속 프레임 간 CLIP embedding 유사도
- 의미적 수준의 시간적 일관성 측정
- **한계**: CLIP embedding이 coarse하여 물리적 오류를 감지하기 어려움

### 3.8 Object Permanence (WCS sub-metric)

- 객체가 프레임에서 갑자기 사라지거나 나타나는 현상을 감지
- World Consistency Score의 하위 메트릭

### 3.9 Relation Stability (WCS sub-metric)

- 객체 간 관계(위치, 크기 비율 등)의 시간적 안정성
- World Consistency Score의 하위 메트릭

### 3.10 Causal Compliance (WCS sub-metric)

- 인과 관계의 일관성 (예: 공이 벽에 부딪히면 튕겨나와야 함)
- World Consistency Score의 하위 메트릭

### 3.11 Flicker Penalty (WCS sub-metric)

- 시각적 깜빡임에 대한 페널티
- World Consistency Score의 하위 메트릭

---

## 4. Multi-View Consistency 메트릭

### 4.1 MEt3R (Measuring Multi-View Consistency in Generated Images)

- **CVPR 2025** 발표 (Max Planck Institute)
- Feature space에서 multi-view 일관성을 측정
- TSED의 한계를 보완: 부분적 비일관성도 감지
- 해상도 변화에 강건 (feature space 측정이므로)
- 에피폴라 제약 조건 만족 여부를 넘어서 실제 3D 일관성을 측정

### 4.2 TSED (Transferred Sampson Epipolar Distance)

- 생성된 이미지 쌍의 feature가 상대 카메라 포즈에 대한 에피폴라 제약을 만족하는지 측정
- **한계**: 충분한 매칭 feature를 찾으면 일관적이라 판단하여 명백한 비일관성을 놓칠 수 있음

### 4.3 Epipolar Sampson Error

- 점과 에피폴라 라인 간 거리를 정량화
- 카메라-씬 기하학적 정밀도를 반영
- VideoGPA (2026) 등에서 사용

### 4.4 MVCS (Multi-View Consistency Score)

- 크로스-프레임 재투영 정확도 평가
- 다시점 간 일관성 점수

### 4.5 3DCS (3D Consistency Score)

- 전역적 씬 통합 품질 측정
- 전체적인 3D 일관성 평가

### 4.6 Sampson Error

- 에피폴라 기하학 일관성 측정 (뷰 간)
- Cosmos 1.0 평가 프레임워크에서 사용, Cosmos 2.5에도 적용
- Camera Pose Estimation Success Rate와 함께 사용하여 다시점 영상의 기하학적 일관성 검증

### 4.7 Reprojection Error

- 다시점 영상 간 대응점의 재투영 오류
- NVIDIA Cosmos 논문에서 자율주행 multi-view 평가에 사용
- 낮을수록 좋음

### 4.7 PRISM (Pose-aware Representation for Image Synthesis Monitoring)

- Novel View Synthesis 품질을 위한 포즈 인식 triplet embedding
- 가시/비가시 영역을 구분하여 평가
- 기하학적 정합과 새로 합성된 영역의 그럴듯함을 동시 고려

### 4.8 Feature Matching Stability (VBench-2.0)

- 객체가 프레임 간에 기하학적 일관성을 유지하는지 측정
- 3D Ground Truth 없이 multi-view 일관성을 간접적으로 평가

### 4.10 Camera Pose Estimation Success Rate

- 생성된 multi-view 비디오가 기하학적으로 일관된 카메라 포즈를 가지는지 검증
- Cosmos 평가 프레임워크에서 Sampson Error와 함께 사용

### 4.11 Depth Consistency

- 여러 시점에서 추정된 깊이 맵의 일관성
- Scale-invariant depth RMSE (si-RMSE) 등 사용

---

## 5. World Model 전용 벤치마크

### 5.1 WorldModelBench (2025년 2월)

- 비디오 생성 모델의 world modeling 능력을 평가하는 전용 벤치마크
- **7개 도메인, 56개 하위 도메인, 350개 프롬프트** 포함
- 3가지 핵심 평가 기준:
  1. **Instruction Following** (0-3 점수): 지시 사항 이행도
  2. **Common Sense**: 시간적 일관성 + 미적 품질
  3. **Physical Adherence**: 5가지 물리 법칙 위반 여부 평가
- 주요 발견: 질량 보존 위반 (12%), 객체 관통 (11%), 중력 위반 (7%)이 가장 흔한 물리 오류

### 5.2 World Consistency Score (WCS) (2025년 7월)

- 비디오 생성 품질에 대한 **통합 메트릭**
- 4가지 해석 가능한 하위 구성요소:
  1. Object Permanence (객체 영속성)
  2. Relation Stability (관계 안정성)
  3. Causal Compliance (인과 준수)
  4. Flicker Penalty (깜빡임 페널티)
- 학습된 가중치로 단일 일관성 점수 생성
- 인간 판단과 정합
- VBench-2.0, EvalCrafter, LOVE 벤치마크에서 검증

### 5.3 PAIBench (Physical AI Benchmark) — NVIDIA

- NVIDIA의 자체 Physical AI 벤치마크
- PAIBench-Predict (Text2World, Image2World)
- PAIBench-Transfer (control signal 충실도)
- Cosmos-Predict2.5-2B Text2World: **0.768**, Image2World: **0.810**

---

## 6. 종합 벤치마크 스위트

### 6.1 VBench (CVPR 2024 Highlight)

**16개 평가 차원:**

| 카테고리 | 차원 |
|---------|------|
| **Quality** | Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness, Aesthetic Quality, Imaging Quality, Dynamic Degree |
| **Semantic** | Object Class, Multiple Objects, Human Action, Color, Spatial Relationship, Scene, Temporal Style, Appearance Style, Overall Consistency |

- `pip install vbench`로 설치 가능
- 인간 선호와 높은 상관관계
- GitHub: [Vchitect/VBench](https://github.com/Vchitect/VBench)

### 6.2 VBench++ (2024)

- VBench-I2V (Image-to-Video)
- VBench-Long (장기 비디오)
- VBench-Trustworthiness (공정성, 문화적 민감성, 편향, 안전성)

### 6.3 VBench-2.0 (2025년 3월)

- **본질적 충실도(Intrinsic Faithfulness)** 평가에 초점
- 5가지 핵심 차원:
  1. Human Fidelity (인체 해부학)
  2. Controllability (제어 가능성)
  3. Creativity (창의성)
  4. Physics (물리학)
  5. Commonsense (상식)
- Multi-view consistency 평가:
  - Feature Matching Stability
  - Camera Motion Speed 보정
- Human Temporal Consistency: 의복 일관성 등 VQA 기반 평가

### 6.4 EvalCrafter

- 비디오 생성 모델 평가 프레임워크
- WCS 검증에 사용됨

---

## 7. NVIDIA 자체 평가 도구

### 7.1 Cosmos Evaluator ([cosmos-evaluator](https://github.com/nvidia-cosmos/cosmos-evaluator))

자동화된 합성 비디오 품질 평가 시스템:

| 체커 | 설명 |
|------|------|
| **Hallucinated Movement Checker** | 원본과 증강 비디오 간 움직임 마스크 비교로 환각 움직임 탐지 |
| **Object Detection Correspondence** | 생성 비디오의 객체(차량, 보행자, 인프라)가 기대 위치와 일치하는지 평가 |
| **VLM Preset Verification** | VLM으로 환경 조건(날씨, 시간대, 도로 상태 등) 검증 |
| **Attribute Verification** | VLM + LLM으로 지정된 속성 존재 확인 |

- REST API 마이크로서비스 + Python API 제공
- Apache 2.0 라이선스

### 7.2 VQA Evaluator (레포 내장: `packages/cosmos-oss/vqa/`)

Cosmos Reason 모델을 사용한 VQA 기반 품질 평가:
- 키워드 기반 검증
- Must-pass 체크 시스템
- Success rate threshold (기본 80%)
- 11개 테스트 설정 YAML (multiview, robot_seg, robot_depth, car_seg, car_edge 등)

### 7.3 Cosmos Reason1 Benchmark

- 공간-시간 이해 + 물리학 추론 평가
- 로봇(RoboVQA, BridgeDataV2, RobFail), 자율주행, 인간 시연(HoloAssist) 도메인
- VQA 정확도 기반 평가

---

## 8. Downstream Task 기반 평가

World model의 실질적 유용성을 downstream task 성능으로 평가하는 방법:

### 8.1 로봇 정책 성공률

- 생성된 증강 데이터로 훈련한 로봇 정책의 실제 세계 성공률
- Cosmos Transfer 2.5: **24/30 성공** vs 최강 베이스라인 5/30
- LIBERO: **98.33%**, RoboCasa: **71.1%** (SOTA, +4%)

### 8.2 3D 객체 탐지 (BEVFormer)

- 생성된 multi-view 영상에서 BEVFormer로 3D 객체 탐지
- mAP, NDS 등으로 평가

### 8.3 차선 탐지 (LATR)

- 생성된 영상에서 LATR로 차선 탐지
- F1 Score, mIoU 등으로 평가

### 8.4 DreamGen Benchmark

- 합성 VLA(Vision-Language-Action) 훈련 데이터 품질 평가
- Instruction-following 점수 및 일반화 능력 측정 (미지의 객체, 새로운 행동, 새 환경)

### 8.5 Human Preference Evaluation

- 인간 선호도 기반 비교 평가
- Cosmos-Predict2.5-2B는 훨씬 큰 모델(Wan2.2-5B, Wan2.1-14B)과 비슷한 인간 선호도 달성 (60-85.7% 작은 모델임에도)

### 8.6 Action Policy MSE

- 레포 내 `eval.py`에 구현
- 로봇 액션 정책의 trajectory 기반 MSE 계산

---

## 9. 권장 벤치마크 조합

Cosmos Transfer 2.5 출력을 포괄적으로 평가하기 위한 권장 메트릭 조합:

### Tier 1: 기본 비디오 품질 (필수)

| 메트릭 | 측정 대상 | 도구/라이브러리 |
|--------|----------|---------------|
| FVD | 시각+시간 품질 | `torch-fidelity`, custom I3D |
| FID | 프레임 품질 | `torch-fidelity`, `clean-fid` |
| LPIPS | Perceptual similarity | `lpips` 패키지 |
| SSIM / PSNR | 픽셀 수준 품질 | `skimage`, `torchmetrics` |

### Tier 2: Temporal Consistency (필수)

| 메트릭 | 측정 대상 | 도구/라이브러리 |
|--------|----------|---------------|
| VBench Temporal Flickering | 깜빡임 | `vbench` 패키지 |
| VBench Motion Smoothness | 모션 부드러움 | `vbench` 패키지 |
| VBench Subject Consistency | 주체 일관성 | `vbench` 패키지 |
| Warping Error | 프레임 간 일관성 | Optical flow 기반 custom |
| WCS (World Consistency Score) | 물리적 일관성 | Open-source 도구 조합 |

### Tier 3: Multi-View Consistency (Multi-view 생성 시 필수)

| 메트릭 | 측정 대상 | 도구/라이브러리 |
|--------|----------|---------------|
| MEt3R | 3D 일관성 (feature space) | CVPR 2025 공개 코드 |
| Epipolar Sampson Error | 에피폴라 기하 정밀도 | OpenCV + custom |
| Reprojection Error | 재투영 오류 | OpenCV |
| Depth Consistency (si-RMSE) | 깊이 일관성 | Custom |

### Tier 4: World Model 품질 (월드 모델 평가 시 필수)

| 메트릭 | 측정 대상 | 도구/라이브러리 |
|--------|----------|---------------|
| WorldModelBench | 물리 법칙 준수 | WorldModelBench 프레임워크 |
| PAIBench-Transfer | 제어 신호 충실도 | NVIDIA 자체 |
| RNDS | 장기 품질 저하 | DOVER 기반 custom |
| VBench-2.0 Physics | 물리적 사실성 | `vbench` 패키지 |

### Tier 5: Downstream Task (응용 평가 시)

| 메트릭 | 측정 대상 | 도구 |
|--------|----------|------|
| 3D Detection mAP/NDS | 인식 모델 성능 | BEVFormer |
| Lane Detection F1/mIoU | 차선 탐지 성능 | LATR |
| Robot Policy Success Rate | 로봇 정책 성능 | 실제 환경 테스트 |
| Cosmos Evaluator Checks | 환각/대응 검증 | cosmos-evaluator |

---

---

## 10. 전체 메트릭 요약 테이블

| 카테고리 | 메트릭 |
|----------|--------|
| **전체 품질** | PAI-Bench Quality Score, PAI-Bench Domain Score |
| **이미지 품질** | FID, PSNR, SSIM, LPIPS |
| **비디오 품질** | FVD, RNDS, DOVER |
| **Temporal Consistency** | RNDS, VideoAlign (motion quality), Error Accumulation Curves, VBench Temporal Flickering/Motion Smoothness/Subject Consistency, WCS, Warping Error |
| **Multi-View / 3D Consistency** | MEt3R, Sampson Error, Reprojection Error, Pose Estimation Success Rate, PSNR/SSIM/LPIPS (view synthesis), MVCS, 3DCS |
| **Physics Alignment** | PhysX/Isaac Sim 시나리오 (gravity, collision, torque, inertia), WorldModelBench Physical Adherence |
| **Control 충실도** | SSIM (blur), F1 (edge), si-RMSE (depth), mIoU (seg) |
| **Reward Model** | VideoAlign (text alignment, motion quality, visual quality) |
| **Downstream Tasks** | BEVFormer mAP/NDS, LATR F1/mIoU, Robot Policy Success Rate, DreamGen, Human Preference |
| **World Model 전용** | WorldModelBench, WCS, PAI-Bench Domain Score, VBench-2.0 Physics/Commonsense |

---

## 참고 자료

- [Cosmos 2.5 Paper (arXiv:2511.00062)](https://arxiv.org/abs/2511.00062)
- [Cosmos-Transfer1 Paper (arXiv:2503.14492)](https://arxiv.org/abs/2503.14492)
- [VBench (CVPR 2024)](https://github.com/Vchitect/VBench)
- [VBench-2.0 (arXiv:2503.21755)](https://arxiv.org/abs/2503.21755)
- [WorldModelBench](https://worldmodelbench-team.github.io/)
- [World Consistency Score (arXiv:2508.00144)](https://arxiv.org/abs/2508.00144)
- [MEt3R — Multi-View Consistency (CVPR 2025)](https://arxiv.org/html/2501.06336)
- [Cosmos Evaluator (GitHub)](https://github.com/nvidia-cosmos/cosmos-evaluator)
- [Cosmos Cookbook (GitHub)](https://github.com/nvidia-cosmos/cosmos-cookbook)
- [NVIDIA Cosmos HuggingFace Blog](https://huggingface.co/blog/nvidia/cosmos-predict-and-transfer2-5)
- [Emergent Mind — Cosmos Transfer 2.5](https://www.emergentmind.com/topics/cosmos-transfer2-5)
- [PAI-Bench Paper (arXiv:2512.01989)](https://arxiv.org/abs/2512.01989)
- [PAI-Bench GitHub](https://github.com/SHI-Labs/physical-ai-bench)
- [VideoGPA — Geometry Priors for 3D-Consistent Video](https://arxiv.org/html/2601.23286)
- [Epipolar Geometry Improves Video Generation](https://arxiv.org/pdf/2510.21615)
- [PRISM — Pose-aware NVS Evaluation](https://arxiv.org/html/2511.12675)
- [SV4D 2.0 — Multi-View Video Diffusion](https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_SV4D_2.0_Enhancing_Spatio-Temporal_Consistency_in_Multi-View_Video_Diffusion_for_ICCV_2025_paper.pdf)
