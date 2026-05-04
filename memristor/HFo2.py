import torch
import numpy as np
import matplotlib.pyplot as plt

import memtorch
from memtorch.bh.memristor import VTEAM
import memtorch.mn as mn

#explain#
#HfO2 memristor has characteristics that move Oxygen vacany -> counductive filament -> LRS(resistance low)
#Occurence set(HRS -> LRS) / reset(LRS -> HRS) follow voltage polarity

############################################
# 1. HfO2 memristor 파라미터
############################################

params = {
    "r_on": 1e2,
    "r_off": 1e6,

    "v_on": -1.0,
    "v_off": 1.0,

    "k_on": -1e-2,
    "k_off": 1e-3,

    "alpha_on": 3,
    "alpha_off": 1,

    "x_on": 0,
    "x_off": 1
}

############################################
# 2. I-V hysteresis 계산
############################################

memristor = VTEAM(**params)

time = np.linspace(0, 1, 1500)
voltage = 1.2 * np.sin(2 * np.pi * time)

current = []

for v in voltage:
    memristor.simulate(torch.tensor([v], dtype=torch.float32))

    R = memristor.get_resistance()

    I = v / R

    current.append(I)

############################################
# 3. PyTorch Linear layer 생성
############################################

torch_layer = torch.nn.Linear(32, 16, bias=False)

############################################
# 4. Memristor crossbar로 변환
############################################

mem_layer = mn.Linear(
    linear_layer=torch_layer,
    memristor_model=VTEAM,
    memristor_model_params=params
)

############################################
# 5. forward 연산
############################################

x = torch.randn(1, 32)

y = mem_layer(x)

############################################
# 6. 결과 출력
############################################

print("Crossbar output shape:", y.shape)
print("Example output:\n", y)

############################################
# 7. I-V plot
############################################

plt.figure()

plt.plot(voltage, current)

plt.title("HfO2 Memristor I-V Hysteresis")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")

plt.show()