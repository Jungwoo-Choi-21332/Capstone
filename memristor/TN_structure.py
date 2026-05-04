import torch
import numpy as np
import matplotlib.pyplot as plt

import memtorch
from memtorch.bh.memristor import VTEAM
import memtorch.mn as mn

############################################
# 1. TN 구조 memristor 파라미터
# (TiN/HfO2/TiN stack)
############################################

params_TN = {

    # TiN electrode → 비교적 안정적인 switching
    "r_on": 5e1,
    "r_off": 5e5,

    "v_on": -0.9,
    "v_off": 0.9,

    "k_on": -5e-3,
    "k_off": 5e-4,

    "alpha_on": 2,
    "alpha_off": 2,

    "x_on": 0,
    "x_off": 1
}

############################################
# 2. I-V hysteresis 계산
############################################

memristor_TN = VTEAM(**params_TN)

time = np.linspace(0,1,1500)

voltage = 1.0*np.sin(2*np.pi*time)

current_TN = []

for v in voltage:

    memristor_TN.simulate(
        torch.tensor([v],dtype=torch.float32)
    )

    R = memristor_TN.get_resistance()

    I = v/R

    current_TN.append(I)

############################################
# 3. PyTorch Linear layer 생성
############################################

torch_layer_TN = torch.nn.Linear(
    32,
    16,
    bias=False
)

############################################
# 4. Memristor crossbar 변환
############################################

mem_layer_TN = mn.Linear(

    linear_layer=torch_layer_TN,

    memristor_model=VTEAM,

    memristor_model_params=params_TN
)

############################################
# 5. forward 계산
############################################

x = torch.randn(1,32)

y_TN = mem_layer_TN(x)

############################################
# 6. 출력
############################################

print("\nTN structure crossbar")

print("output shape:",y_TN.shape)

print("example output:\n",y_TN)

############################################
# 7. I-V plot
############################################

plt.figure()

plt.plot(voltage,current_TN)

plt.title("TiN/HfO2/TiN I-V hysteresis")

plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")

plt.show()