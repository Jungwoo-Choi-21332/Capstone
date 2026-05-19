# Wafer9 - 연구방향 4: Task-Aware Memristor Parameter Optimization

이 폴더는 memristor parameter를 일반 이미지 분류 기준이 아니라 wafer defect classification 성능 기준으로 최적화합니다. 기존 연구가 device-centric 관점에서 좋은 memristor 특성을 찾는 데 집중했다면, 여기서는 task-aware 관점에서 wafer map 분류에 필요한 parameter 조합을 찾습니다.

## 핵심 아이디어

같은 CNN 모델을 사용하고 memristor parameter만 바꾸면서 wafer defect classification의 accuracy와 macro F1-score가 어떻게 변하는지 분석합니다.

분석하는 parameter:

- `R_on/R_off ratio`
- conductance window
- device variation
- ADC precision

## 실행 방법

기본 실행:

```bash
python src/run_direction4.py
```

빠른 synthetic sweep:

```bash
python src/run_direction4.py --synthetic --max-samples 800 --epochs 1 --fast-sweep
```

실제 데이터 일부 실행:

```bash
python src/run_direction4.py --max-samples 2400 --epochs 3 --fast-sweep
```

## 출력 파일

실행 후 `results` 폴더에 다음 파일이 생성됩니다.

- `results/parameter_sweep.csv`
  - 모든 parameter 조합의 accuracy, macro F1, accuracy drop, F1 drop을 저장합니다.

- `results/heatmap_accuracy.csv`
  - `R_on/R_off ratio`와 ADC bitwidth를 축으로 평균 accuracy를 정리한 heatmap용 CSV입니다.

## 결과 해석

`best_parameter_set`은 wafer defect classification에서 가장 높은 accuracy와 macro F1을 보인 parameter 조합입니다. `accuracy_vs_on_off_ratio`는 on/off ratio가 커질수록 성능이 얼마나 개선되는지 보여줍니다.

예를 들어 4-bit ADC에서도 높은 on/off ratio와 낮은 variation 조건에서 성능이 유지된다면, wafer map task는 ADC precision보다 conductance margin 확보가 더 중요하다고 해석할 수 있습니다. 반대로 ADC bitwidth가 낮을 때 급격히 성능이 떨어지면 peripheral circuit precision이 핵심 병목입니다.

## 연구 기여 포인트

- wafer defect classification에 맞춘 memristor parameter 최적화
- device parameter와 task metric 사이의 관계 분석
- `R_on/R_off ratio`, conductance window, variation, ADC precision에 대한 task-aware guideline 제안
