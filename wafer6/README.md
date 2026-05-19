# Wafer6 - 연구방향 1: Memristor 비이상성 환경에서 Wafer Defect Classification의 Robustness 분석

이 폴더는 wafer defect classification 모델이 memristor 기반 crossbar inference 환경에서 얼마나 견고한지 분석하는 코드입니다. MNIST나 CIFAR-10과 달리 wafer map은 sparse pattern, geometric defect structure, binary-like distribution을 가지므로 memristor hardware noise에 대한 민감도가 다르게 나타날 수 있습니다.

## 핵심 아이디어

일반적인 CNN 추론 결과를 기준선으로 두고, memristor non-idealities를 단계적으로 주입한 crossbar inference 결과와 비교합니다. 이를 통해 wafer defect classification이 어떤 hardware imperfection에 가장 취약한지 확인합니다.

## 분석하는 비이상성

- Conductance variation: memristor conductance programming 오차를 weight multiplicative noise로 모델링합니다.
- ADC/DAC quantization: 입력 전압과 출력 전류 변환의 bitwidth 제한을 activation/logit quantization으로 모델링합니다.
- Retention drift: 시간이 지남에 따라 conductance가 감소하는 현상을 weight scaling으로 모델링합니다.
- Stuck-at faults: 일부 cell이 stuck-on 또는 stuck-off 상태가 되는 현상을 weight 강제 치환으로 모델링합니다.
- Nonlinear I-V behavior: crossbar 출력의 비선형성을 activation compression으로 모델링합니다.
- Crossbar size: 큰 layer를 여러 tile로 나누는 경우의 누적 손실을 간단한 tile penalty로 반영합니다.

## 실행 방법

기본 실행:

```bash
python src/run_direction1.py
```

빠른 synthetic 실험:

```bash
python src/run_direction1.py --synthetic --max-samples 800 --epochs 1
```

실제 데이터 일부만 사용:

```bash
python src/run_direction1.py --max-samples 2400 --epochs 3
```

MemTorch의 `patch_model`을 실제로 호출해 VTEAM memristor crossbar module로 변환하는 최소 예시는 다음 파일입니다.

```bash
python src/memtorch_crossbar_example.py --synthetic --max-samples 400 --epochs 1
```

## 출력되는 실험

1. `ideal_cnn`
   - hardware non-ideality가 없는 PyTorch CNN 기준 정확도입니다.

2. `conductance_variation_sweep`
   - conductance variation을 `0.00`, `0.03`, `0.05`, `0.10`, `0.15`로 증가시키며 정확도 하락을 확인합니다.

3. `adc_bitwidth_sweep`
   - ADC/DAC bitwidth를 `4-bit`, `6-bit`, `8-bit`로 바꾸며 quantization 영향도를 확인합니다.

4. `crossbar_size_comparison`
   - `32x32`, `64x64`, `128x128`, `256x256` crossbar tile 설정을 비교합니다.

5. `fault_and_drift_analysis`
   - retention drift, stuck-at fault, nonlinear I-V, combined worst case를 비교합니다.

## MemTorch 관련 메모

`run_direction1.py`는 sweep 실험을 안정적으로 반복하기 위해 PyTorch CNN에 memristor non-ideality를 명시적으로 주입하는 방식으로 작성되어 있습니다. 실행 시작 시 `memtorch_available=True/False`를 출력하므로 현재 환경에서 MemTorch 설치 여부를 확인할 수 있습니다.

`memtorch_crossbar_example.py`는 `memtorch.mn.patch_model`을 사용해 `SmallWaferCNN`의 `Conv2d`, `Linear` layer를 실제 MemTorch memristive module로 변환합니다. 본격 실험에서는 이 예시를 기반으로 ADC bitwidth, tile shape, device model parameter를 sweep하면 됩니다.

## 연구 결과 해석

이 실험에서 중요한 값은 단순히 최고 정확도가 아니라 `ideal_cnn` 대비 각 non-ideality 조건의 accuracy drop입니다. 예를 들어 conductance variation에서는 정확도가 유지되지만 4-bit ADC에서 크게 떨어진다면, wafer map classification은 weight programming error보다 peripheral circuit quantization에 더 민감하다는 결론을 세울 수 있습니다.
