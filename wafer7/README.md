# Wafer7 - 연구방향 2: Class-Wise Sensitivity Analysis

이 폴더는 wafer defect class마다 memristor non-ideality에 얼마나 다르게 반응하는지 분석합니다. 같은 전체 정확도 하락이라도 Scratch, Donut, Edge-Ring 같은 클래스별 F1-score가 다르게 떨어질 수 있으므로, 단순 accuracy보다 class-wise robustness를 보는 것이 핵심입니다.

## 핵심 아이디어

wafer map defect는 클래스마다 의존하는 기하 정보가 다릅니다.

- Scratch defect는 얇고 긴 선형 구조나 edge 정보를 많이 사용합니다.
- Donut defect는 중심 주변의 ring shape와 면적 정보를 사용합니다.
- Edge 계열 defect는 wafer 바깥쪽 영역의 신호에 민감합니다.

따라서 conductance variation, quantization error, retention drift 같은 hardware noise가 모든 클래스에 동일하게 작용하지 않을 수 있습니다.

## 실행 방법

기본 실행:

```bash
python src/run_direction2.py
```

빠른 synthetic 실험:

```bash
python src/run_direction2.py --synthetic --max-samples 800 --epochs 1
```

실제 데이터 일부 실행:

```bash
python src/run_direction2.py --max-samples 2400 --epochs 3
```

## 출력되는 분석

1. `defect_geometry_summary`
   - 클래스별 sparsity, edge_ratio, center_ratio를 출력합니다.
   - 어떤 클래스가 edge 정보나 center 정보에 더 의존하는지 해석하는 데 사용합니다.

2. `condition=ideal_cnn`
   - memristor noise가 없는 기준 CNN의 per-class precision, recall, F1-score입니다.

3. Noise condition별 class-wise table
   - `conductance_variation_0.10`
   - `adc_dac_4bit`
   - `retention_drift_t10`
   - `combined_noise`

4. `confusion_matrix`
   - noise 조건에서 어떤 클래스가 어떤 클래스로 헷갈리는지 확인합니다.

5. `failure_cases`
   - 오분류된 test sample index와 true/pred class를 출력합니다.

## 결과 해석

`f1_drop`이 큰 클래스가 해당 hardware noise에 민감한 defect class입니다. 예를 들어 Scratch의 F1-score가 quantization 조건에서 크게 떨어진다면, 얇은 선형 구조가 ADC/DAC bitwidth 감소에 취약하다고 해석할 수 있습니다. 반대로 Donut은 면적과 ring shape가 비교적 두꺼운 구조이므로 conductance variation에는 더 안정적일 수 있습니다.

이 분석은 Research Direction 1의 전체 robustness 분석을 더 세밀하게 분해한 단계입니다.
