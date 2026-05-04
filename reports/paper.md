# VTEAM 멤리스터 기반 PIM 가속기의 웨이퍼 맵 결함 분류를 위한 Ex-Situ 파라미터 최적화 및 Noise-Aware Training 연구

**키워드**: 멤리스터, VTEAM, Processing-in-Memory, 웨이퍼 결함 분류, Ex-Situ, Noise-Aware Training, memtorch

---

## Abstract

본 연구에서는 VTEAM 멤리스터 모델 기반 PIM 가속기를 활용한 웨이퍼 맵 결함 분류 시스템의 멤리스터 파라미터 최적화를 수행하였다. 기존 구현에서 memtorch API 호출 오류 2건과 In-Situ 워크플로우의 논리적 오류를 발견하고 수정하였다. Ex-Situ 방식으로 재설계한 후, 73가지 파라미터 조합에 대한 체계적 탐색과 Noise-Aware Training 기법을 적용하였다. 최종적으로 Macro F1 Score 0.9881, Precision 0.9891, Recall 0.9873을 달성하였으며, 8개 결함 유형 전체에서 F1 0.95 이상의 성능을 확인하였다.

---

## 1. 서론

### 1.1 연구 배경

반도체 제조 공정에서 웨이퍼 맵 결함 패턴 분석은 수율 향상에 필수적이다[1]. 멤리스터 기반 PIM 가속기는 행렬-벡터 곱셈을 O(1)에 수행할 수 있어 CNN 추론 가속에 적합하나[2], 소자의 물리적 비이상성(variation, quantization)이 추론 정확도를 저하시킨다.

### 1.2 연구 목적

1. 기존 멤리스터 시뮬레이션 코드의 API 및 워크플로우 검증
2. Ex-Situ 방식 기반 파라미터 체계적 탐색
3. Noise-Aware Training을 통한 하드웨어 비이상성 내성 향상

---

## 2. 관련 연구

VTEAM은 전압 임계값 기반 멤리스터 모델로[3], memtorch 프레임워크에서 `patch_model`을 통해 PyTorch 모델을 멤리스터 기반으로 변환한다[4]. Ex-Situ 방식은 소프트웨어 학습 후 하드웨어 매핑을 수행하며, Noise-Aware Training은 학습 시 하드웨어 노이즈를 시뮬레이션하여 매핑 후 성능 저하를 완화하는 기법이다[5].

---

## 3. 연구 방법

### 3.1 발견된 문제점

**API 오류**: `adc_bitwidth` → `ADC_resolution`, `r_on_variation` → `StochasticParameter`. Python의 `**kwargs`로 인해 런타임 오류 없이 무시됨.

**워크플로우 오류**: `patch_model`은 호출 시점의 가중치를 크로스바 전도도로 즉시 변환하며, `forward_legacy(False)` 이후 추론은 크로스바만 사용. `load_state_dict` 이후 호출해야 학습된 가중치가 반영됨.

### 3.2 모델 구조

3층 CNN: Conv2d(1→16→32→64) + BatchNorm + ReLU + MaxPool, FC(4096→8).
손실 함수: BCE Focal Loss (gamma=2.0, alpha=0.5).

### 3.3 최적화 파이프라인

**Phase 1: Ex-Situ 파라미터 탐색**
- 순수 CNN 학습 (50 epoch, StepLR)
- 73가지 멤리스터 파라미터 조합 × 3회 반복 = 219회 평가

**Phase 2: Noise-Aware Training**
- Phase 1 모델을 기반으로 15 epoch 미세 조정
- 각 mini-batch에서 가중치에 가우시안 노이즈 주입 (σ = 1%~5%, 점진 증감)
- Forward 시 노이즈 적용, backward 후 원본 가중치에 gradient 적용

**Phase 3: Threshold 튜닝**
- 3개 모델(v1_epoch30, v2_epoch50, v2_noiseaware) × 9개 threshold(0.30~0.70)
- 총 27조합 비교, Macro F1 기준 최적 선정

---

## 4. 실험 결과

### 4.1 Phase별 성능 추이

| Phase | 모델 | Threshold | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Baseline | epoch 30 | 0.50 | 0.9628 | 0.9909 | 0.9387 |
| Phase 1 | epoch 50 | 0.50 | 0.9633 | - | - |
| Phase 2 | Noise-Aware | 0.50 | 0.9877 | - | - |
| **Phase 3** | **Noise-Aware** | **0.45** | **0.9881** | **0.9891** | **0.9873** |

### 4.2 Noise-Aware Training의 효과

| 지표 | epoch 50 (t=0.45) | Noise-Aware (t=0.45) | 차이 |
|---|---|---|---|
| F1 | 0.9659 | **0.9881** | **+2.2%p** |
| Recall | 0.9495 | **0.9873** | **+3.8%p** |

Noise-Aware Training이 가장 큰 성능 향상 요인.

### 4.3 클래스별 성능 (최종)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Center | 1.00 | 1.00 | 1.00 | 2,587 |
| Donut | 1.00 | 1.00 | 1.00 | 2,377 |
| Edge-Loc | 0.95 | 0.95 | 0.95 | 2,605 |
| Edge-Ring | 0.99 | 1.00 | 0.99 | 2,404 |
| Loc | 1.00 | 0.99 | 0.99 | 3,604 |
| Near-Full | 1.00 | 0.97 | 0.99 | 34 |
| Scratch | 0.99 | 0.98 | 0.99 | 3,713 |
| Random | 0.99 | 1.00 | 0.99 | 179 |

모든 클래스에서 F1 ≥ 0.95 달성. Edge-Loc은 기존 0.49에서 0.95로 개선.

### 4.4 Threshold 분석 (Noise-Aware 모델)

| Threshold | F1 |
|---|---|
| 0.30 | 0.9790 |
| 0.35 | 0.9835 |
| 0.40 | 0.9851 |
| **0.45** | **0.9881** |
| 0.50 | 0.9877 |
| 0.55 | 0.9867 |
| 0.60 | 0.9809 |

0.45에서 최적. 기본값 0.5 대비 Recall 향상으로 F1 +0.04%p.

---

## 5. 고찰

### 5.1 Noise-Aware Training의 메커니즘

학습 시 가중치에 가우시안 노이즈를 주입하면, 모델이 개별 가중치의 정확한 값에 의존하지 않고 전체적인 패턴에 기반하여 판단하도록 학습된다. 이는 dropout과 유사한 정규화 효과를 제공하며, 멤리스터 매핑 시 발생하는 전도도 양자화 오차에 대한 내성을 높인다.

### 5.2 최적 파라미터의 의미

타일링 없음 + 양자화 없음이 최적인 것은, Ex-Situ에서는 소프트웨어 가중치의 정밀도를 최대한 보존하는 것이 핵심임을 의미한다. 실제 하드웨어에서는 이러한 제약이 불가피하므로, Noise-Aware Training으로 내성을 키우는 것이 실용적 해결책이다.

### 5.3 연구의 한계

1. CPU 시뮬레이션이므로 실제 하드웨어 검증 미수행
2. 순차 탐색으로 전역 최적해 보장 불가
3. 단일 데이터셋/모델에 대한 결과

---

## 6. 결론

1. memtorch API 버그 2건 및 워크플로우 오류 발견·수정
2. Ex-Situ 기반 73가지 파라미터 탐색 + Noise-Aware Training + threshold 튜닝
3. **Macro F1 0.9881** (Precision 0.9891, Recall 0.9873) 달성
4. 8개 결함 유형 전체 F1 ≥ 0.95, 결함 누락률 1.3%
5. Noise-Aware Training이 멤리스터 PIM 가속기 성능 향상에 가장 효과적인 기법임을 확인

---

## References

[1] M. B. Alawieh et al., "Wafer map defect pattern classification using deep learning," IEEE Trans. Semicond. Manuf., 2020.

[2] A. Sebastian et al., "Memory devices and applications for in-memory computing," Nature Nanotechnology, vol. 15, 2020.

[3] S. Kvatinsky et al., "VTEAM: A general model for voltage-controlled memristors," IEEE TCAS-II, vol. 62, 2015.

[4] C. Lammie et al., "MemTorch: An open-source simulation framework for memristive deep learning systems," Neurocomputing, vol. 485, 2022.

[5] L. Xia et al., "Fault-tolerant training enabled by on-line fault detection for RRAM-based neural computing systems," IEEE TCAD, vol. 38, 2019.
