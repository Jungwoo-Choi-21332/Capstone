import torch
import numpy as np
import matplotlib.pyplot as plt

import memtorch
from memtorch.bh.memristor import VTEAM
import memtorch.mn as mn

############################################
# ZnO parameter
############################################

params_ZnO = {

    "r_on": 8e1,
    "r_off": 8e5,

    "v_on": -0.7,
    "v_off": 0.7,

    "k_on": -6e-3,
    "k_off": 1e-3,

    "alpha_on": 2.2,
    "alpha_off": 1.3,

    "x_on": 0,
    "x_off": 1
}

############################################
# I-V
############################################

mem_ZnO = VTEAM(**params_ZnO)

time = np.linspace(0,1,1500)
voltage = 0.9*np.sin(2*np.pi*time)

current_ZnO = []

for v in voltage:

    mem_ZnO.simulate(
        torch.tensor([v],dtype=torch.float32)
    )

    R = mem_ZnO.get_resistance()

    current_ZnO.append(v/R)

############################################
# crossbar
############################################

torch_layer_ZnO = torch.nn.Linear(32,16,bias=False)

mem_layer_ZnO = mn.Linear(

    linear_layer=torch_layer_ZnO,

    memristor_model=VTEAM,

    memristor_model_params=params_ZnO
)

x = torch.randn(1,32)

y_ZnO = mem_layer_ZnO(x)

print("\nZnO crossbar")
print(y_ZnO.shape)

############################################
# plot
############################################

plt.figure()

plt.plot(voltage,current_ZnO)

plt.title("ZnO I-V")

plt.xlabel("Voltage")
plt.ylabel("Current")

plt.show()