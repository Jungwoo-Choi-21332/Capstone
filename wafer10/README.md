# Wafer10 - 연구방향 5: Hardware-Aware / Noise-Aware Training

이 폴더는 memristor hardware non-ideality를 학습 중에 미리 주입해 wafer defect classification 모델의 hardware robustness를 높이는 실험입니다.

## 핵심 아이디어

일반적인 흐름은 다음과 같습니다.

```text
Train ideal CNN -> Map to hardware -> Accuracy degradation
```

이 연구방향은 대신 다음 흐름을 사용합니다.

```text
Train CNN with hardware noise injection -> Map to hardware -> Improved robustness
```

즉, 학습 중에 conductance variation, quantization noise, drift, stuck-at fault 영향을 일부 경험하게 해서 실제 hardware mapping 이후 accuracy drop을 줄이는 것이 목표입니다.

## 주입하는 hardware noise

- conductance variation
- ADC/DAC quantization noise
- retention drift
- stuck-at faults

## 실행 방법

기본 실행:

```bash
python src/run_direction5.py
```

빠른 synthetic 실험:

```bash
python src/run_direction5.py --synthetic --max-samples 800 --epochs 1
```

hardware noise 조건 변경:

```bash
python src/run_direction5.py --variation 0.10 --adc-bits 4 --stuck-rate 0.02
```

## 출력되는 결과

코드는 두 가지 training method를 비교합니다.

| Training Method | 의미 |
|---|---|
| Ideal Training | 깨끗한 CNN 학습 후 hardware noise를 사후 적용 |
| Noise-Aware Training | 학습 중 hardware noise를 주입한 뒤 hardware noise 적용 |

출력 표에는 다음 값이 포함됩니다.

- `ideal_accuracy`
- `hardware_accuracy`
- `accuracy_drop`
- `ideal_macro_f1`
- `hardware_macro_f1`
- `f1_drop`
- `robustness_gain_drop_reduction`

## 결과 해석

`accuracy_drop`이 작을수록 hardware mapping 이후 성능 손실이 적다는 뜻입니다. 예를 들어 Ideal Training의 drop이 12%이고 Noise-Aware Training의 drop이 3%라면, noise-aware training이 memristive wafer defect classifier의 hardware robustness를 크게 개선했다고 해석할 수 있습니다.

이 연구방향은 Research Direction 1~4에서 분석한 hardware non-ideality를 학습 단계로 끌어와 성능 저하를 줄이는 마무리 단계입니다.
