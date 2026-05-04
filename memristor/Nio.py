import torch
import numpy as np
import matplotlib.pyplot as plt

import memtorch
from memtorch.bh.memristor import VTEAM
import memtorch.mn as mn

#Nickel oxide memristors are promising Random access meemory
#High on/off ratio(10^3 ~ 10^5)
#exhibiting bipolar and unipolar switching
#image hyperlink
#https://www.researchgate.net/figure/Memristor-characteristics-of-NiO-investigated-with-CAFM-a-Schematic-of-sample-layout_fig15_301549902

############################################
# NiO parameter
############################################

params_NiO = {

    # resistance range
    "Ron": 500,          # ohm
    "Roff": 5e4,         # ohm

    # threshold voltage
    "V_on": 0.9,         # SET 시작
    "V_off": -1.1,       # RESET 시작

    # switching speed parameter
    "k_on": 5e-3,
    "k_off": -5e-3,

    # nonlinearity
    "alpha_on": 1.8,
    "alpha_off": 2.2,

    # state variable range
    "D": 10e-9,

    # window function
    "p": 2
}

############################################
# I-V
############################################

mem_NiO = VTEAM(**params_NiO)

time = np.linspace(0,1,1500)
voltage = 1.2*np.sin(2*np.pi*time)

current_NiO = []

for v in voltage:

    mem_NiO.simulate(
        torch.tensor([v],dtype=torch.float32)
    )

    R = mem_NiO.get_resistance()

    current_NiO.append(v/R)

############################################
# crossbar
############################################

torch_layer_NiO = torch.nn.Linear(32,16,bias=False)

mem_layer_NiO = mn.Linear(

    linear_layer=torch_layer_NiO,

    memristor_model=VTEAM,

    memristor_model_params=params_NiO
)

x = torch.randn(1,32)

y_NiO = mem_layer_NiO(x)

print("\nNiO crossbar")

print(y_NiO.shape)

############################################
# plot
############################################

plt.figure()

plt.plot(voltage,current_NiO)

plt.title("NiO I-V")

plt.xlabel("Voltage")

plt.ylabel("Current")

plt.show()