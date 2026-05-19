# Wafer8 - 연구방향 3: Different Memristor Materials 비교

이 폴더는 서로 다른 memristor material이 wafer defect classification inference 정확도에 어떤 영향을 주는지 비교합니다. 같은 CNN 모델을 사용하되, TaOx, HfO2, IGZO, NiO의 전기적 특성을 서로 다른 device profile로 두고 weight noise, conductance quantization, retention drift, nonlinear output을 다르게 적용합니다.

## 핵심 아이디어

memristor material마다 다음 특성이 다릅니다.

- `R_on / R_off ratio`
- threshold voltage
- current-voltage nonlinearity
- switching variation
- retention drift

이 차이는 crossbar inference에서 weight programming error, conductance resolution, output distortion으로 나타나고, 결국 wafer defect classification의 accuracy와 F1-score 차이를 만들 수 있습니다.

## 비교 material

| Material | 특징 |
|---|---|
| TaOx | 높은 on/off ratio와 안정적인 oxide switching |
| HfO2 | CMOS 친화적이지만 switching variation이 상대적으로 큼 |
| IGZO | 낮은 전압 동작이 가능한 oxide semiconductor 계열 |
| NiO | 낮은 on/off ratio와 강한 nonlinearity를 갖는 조건으로 모델링 |

## 실행 방법

기본 실행:

```bash
python src/run_direction3.py
```

빠른 synthetic 실험:

```bash
python src/run_direction3.py --synthetic --max-samples 800 --epochs 1
```

실제 데이터 일부 실행:

```bash
python src/run_direction3.py --max-samples 2400 --epochs 3
```

## 출력되는 결과

코드는 먼저 ideal CNN의 `accuracy`, `macro_f1`을 출력합니다. 그 다음 material별로 다음 값을 CSV 형태로 출력합니다.

- `accuracy`
- `macro_f1`
- `accuracy_drop`
- `f1_drop`
- `R_on`
- `R_off`
- `on_off_ratio`
- `Vth`
- `nonlinearity`
- `switching_noise`
- `retention_nu`
- `sensitivity_index`

## 결과 해석

예를 들어 TaOx가 가장 높은 정확도를 보이면 높은 on/off ratio와 낮은 switching noise가 wafer classification에 유리하다는 해석을 할 수 있습니다. 반대로 NiO가 낮은 성능을 보이면 낮은 conductance margin과 강한 nonlinearity가 inference robustness를 떨어뜨린다고 볼 수 있습니다.

이 연구방향은 단순히 “어떤 재료가 좋다”를 넘어서, device physics parameter가 neural network inference metric과 어떻게 연결되는지 보여주는 것이 목적입니다.
