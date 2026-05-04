# 최종 보고서: 멤리스터 기반 PIM 가속기 웨이퍼 결함 분류 파라미터 최적화

## 1. 프로젝트 개요

### 목적
VTEAM 멤리스터 기반 PIM 가속기를 활용한 웨이퍼 맵 결함 분류 시스템에서, 멤리스터 관련 파라미터를 최적화하여 분류 성능을 최대화한다.

### 시스템 구성
- **모델**: 3층 CNN (Conv2d × 3 + MaxPool × 3 + FC)
- **입력**: 64×64 웨이퍼 맵 이미지
- **출력**: 8개 결함 유형 멀티라벨 분류
- **가속기**: memtorch VTEAM 멤리스터 시뮬레이션
- **데이터**: Mixed Type Wafer Defect Datasets (38,015개)
- **결함 유형**: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-Full, Scratch, Random

---

## 2. 발견 및 수정한 문제점

### 2.1 memtorch API 버그 2건

| 원본 코드 | 문제 | 수정 |
|---|---|---|
| `adc_bitwidth=8` | memtorch에 없는 파라미터, `**kwargs`로 무시됨 | `ADC_resolution=8` |
| `r_on_variation=0.05` | VTEAM에 없는 파라미터, `**kwargs`로 무시됨 | `StochasticParameter(loc, scale)` |

### 2.2 워크플로우 오류

| 원본 | 문제 | 수정 |
|---|---|---|
| patch_model → 학습 (In-Situ) | 느리고, 노이즈에 적응하여 공정 비교 불가 | 순수 학습 → patch_model → 평가 (Ex-Situ) |
| patch_model → load_state_dict | 크로스바 전도도가 갱신되지 않음 | load_state_dict → patch_model |

### 2.3 데이터 누수
- **원본**: 밸런싱 후 train/test 분리 → 테스트 데이터 오염
- **수정**: train/test 분리 후 학습 데이터만 밸런싱

---

## 3. 최적화 과정

### 3.1 Phase 1: Ex-Situ 파라미터 탐색 (73조합 × 3회 = 219회)

4단계 순차 탐색:

| Stage | 파라미터 | 최적값 |
|---|---|---|
| A | tile_shape + ADC | None / None |
| B | 소자 변동성 | 0% / 0% |
| C | r_on / r_off | 50Ω / 500Ω |
| D | scheme, transistor 등 | 기본값 유지 |

결과: **F1 = 0.9628**

### 3.2 Phase 2: 성능 극대화

| 기법 | 설명 | 효과 |
|---|---|---|
| epoch 50 + LR scheduler | 30→50, 20 epoch마다 LR 절반 | loss 추가 감소 |
| Noise-Aware Training | 가중치에 노이즈 주입하며 미세 조정 (15 epoch) | 멤리스터 매핑 내성 향상 |
| threshold 튜닝 | 0.3~0.7 범위에서 0.05 간격 탐색 | 0.45가 최적 |

---

## 4. 최종 결과

### 4.1 성능

| 지표 | v1 (기본) | **v2 (최종)** | 향상 |
|---|---|---|---|
| **Macro F1** | 0.9628 | **0.9881** | +2.5%p |
| **Macro Precision** | 0.9909 | **0.9891** | - |
| **Macro Recall** | 0.9387 | **0.9873** | +4.9%p |
| 재현성 | ±0.0000 | ±0.0000 | - |

### 4.2 결함 유형별 성능

| Defect Type | Precision | Recall | F1 |
|---|---|---|---|
| Center | 1.00 | 1.00 | 1.00 |
| Donut | 1.00 | 1.00 | 1.00 |
| **Edge-Loc** | **0.95** | **0.95** | **0.95** |
| Edge-Ring | 0.99 | 1.00 | 0.99 |
| Loc | 1.00 | 0.99 | 0.99 |
| Near-Full | 1.00 | 0.97 | 0.99 |
| Scratch | 0.99 | 0.98 | 0.99 |
| Random | 0.99 | 1.00 | 0.99 |

모든 클래스 F1 **0.95 이상** 달성.

### 4.3 최적 설정 요약

| 항목 | 값 |
|---|---|
| 학습 방식 | Ex-Situ (순수 학습 → 멤리스터 매핑) |
| epoch | 50 + Noise-Aware 15 |
| 손실 함수 | Focal Loss (gamma=2.0) |
| LR scheduler | StepLR (step=20, gamma=0.5) |
| threshold | 0.45 |
| tile_shape | None |
| ADC_resolution | None |
| r_on / r_off | 50Ω / 500Ω |
| 변동성 | 0% |

### 4.4 핵심 발견

1. **Noise-Aware Training이 가장 효과적** — epoch 50만으로는 F1 0.9633, Noise-Aware 추가 시 0.9881
2. **threshold 0.45가 최적** — 기본값 0.5보다 낮춰서 Recall +4.9%p 향상
3. **Edge-Loc F1 0.95 달성** — 기존 0.49에서 크게 개선 (3층 CNN + Focal Loss + Noise-Aware 시너지)
4. **결함 누락률 1.3%** — Recall 0.9873 → 100개 중 1~2개만 놓침

---

## 5. 코드 구조

```
model.py               - 공통 모듈 (CNN, Dataset, FocalLoss)
step1_preprocessing.py  - 데이터 전처리
step2_train.py          - Ex-Situ 학습 (순수 CNN)
step3_evaluate.py       - Ex-Situ 평가 (멤리스터 매핑)
optimize.py             - v1 파라미터 탐색
optimize_v2.py          - v2 성능 극대화 (Noise-Aware + threshold)
```

---

## 6. 실험 환경

| 항목 | 값 |
|---|---|
| PyTorch | 2.11.0+cpu |
| memtorch | 1.1.6-cpu |
| scikit-learn | 1.8.0 |
| 데이터 | Mixed Type Wafer Defect Datasets |
