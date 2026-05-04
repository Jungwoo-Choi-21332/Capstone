import torch
import numpy as np
import matplotlib.pyplot as plt

import memtorch
from memtorch.bh.memristor import VTEAM
import memtorch.mn as mn

############################################
# 1. Ta2O5 memristor 파라미터
############################################

params_Ta2O5 = {

    # Ta2O5는 보통 높은 on/off ratio
    "r_on": 2e1,
    "r_off": 2e6,

    "v_on": -0.8,
    "v_off": 0.8,

    "k_on": -8e-3,
    "k_off": 2e-3,

    "alpha_on": 1.8,
    "alpha_off": 1.2,

    "x_on": 0,
    "x_off": 1
}

############################################
# 2. I-V hysteresis 계산
############################################

memristor_Ta2O5 = VTEAM(**params_Ta2O5)

time = np.linspace(0,1,1500)

voltage = 1.0*np.sin(2*np.pi*time)

current_Ta2O5 = []

for v in voltage:

    memristor_Ta2O5.simulate(
        torch.tensor([v],dtype=torch.float32)
    )

    R = memristor_Ta2O5.get_resistance()

    I = v/R

    current_Ta2O5.append(I)

############################################
# 3. PyTorch Linear layer 생성
############################################

torch_layer_Ta2O5 = torch.nn.Linear(
    32,
    16,
    bias=False
)

############################################
# 4. memristor crossbar 변환
############################################

mem_layer_Ta2O5 = mn.Linear(

    linear_layer=torch_layer_Ta2O5,

    memristor_model=VTEAM,

    memristor_model_params=params_Ta2O5
)

############################################
# 5. forward 연산
############################################

x = torch.randn(1,32)

y_Ta2O5 = mem_layer_Ta2O5(x)

############################################
# 6. 출력
############################################

print("\nTa2O5 crossbar")

print("output shape:", y_Ta2O5.shape)

print("example output:\n", y_Ta2O5)

############################################
# 7. I-V plot
############################################

plt.figure()

plt.plot(voltage,current_Ta2O5)

plt.title("Ta2O5 memristor I-V hysteresis")

plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")

plt.show()