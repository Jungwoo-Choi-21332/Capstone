# 🧠 Memristor Parameter Mapping for MemTorch

This repository documents how parameters extracted from a memristor modeling paper can be interpreted and used within **MemTorch**.

> Based on parameter extraction methodology from the uploaded paper 

---

## 📌 Overview

Memristor-based neural networks use **conductance (G)** instead of traditional weights.

In MemTorch:

* **Weight ↔ Conductance**
* **State variable (x) ↔ Normalized weight**

This document explains how physical memristor parameters map to MemTorch simulation parameters.

---

## ⚙️ Core Concept

The memristor current model:

```math
i(t) = g_{max} \cdot v(t) \cdot x(t) + g_{min} \cdot sinh(b \cdot v(t)) \cdot (1 - x(t))
```

* `x(t)` controls interpolation between ON/OFF states
* Conductance is bounded between `gmin` and `gmax`

---

## 🔄 Parameter Mapping (Paper → MemTorch)

| Paper Parameter | Meaning              | MemTorch Equivalent         |
| --------------- | -------------------- | --------------------------- |
| `gmax`          | Maximum conductance  | `G_on`, `Gmax`              |
| `gmin`          | Minimum conductance  | `G_off`, `Gmin`             |
| `x`             | Internal state (0~1) | Weight                      |
| `Vp`, `Vn`      | Switching thresholds | `V_set`, `V_reset`          |
| `Ap`, `An`      | Switching speed      | Learning rate / sensitivity |
| `xp`, `xn`      | Saturation boundary  | Nonlinear update behavior   |
| `b`             | I-V nonlinearity     | Nonlinearity factor         |
| Noise terms     | Device variation     | Stochastic update           |

---

## 📊 Weight Representation

Weights are derived from conductance:

```math
w = \frac{G - G_{min}}{G_{max} - G_{min}}
```

* `x ≈ w`
* `x ∈ [0,1]`

---

## ⚡ I-V Characteristics

Two regimes are modeled:

### ON State (Linear)

```math
i_{on} = g_{max} \cdot v
```

### OFF State (Nonlinear)

```math
i_{off} = g_{min} \cdot sinh(b \cdot v)
```

* `b ↑` → stronger nonlinearity
* affects inference accuracy in analog computing

---

## 🔁 State Update Dynamics

State evolution:

```math
\frac{dx}{dt} = g(V(t)) \cdot f(x(t))
```

### Components:

* `g(V)` → threshold-dependent switching
* `f(x)` → boundary effects (slows near 0 or 1)

---

## 🚧 Switching Behavior

| Parameter  | Role                      |
| ---------- | ------------------------- |
| `Vp`, `Vn` | When switching starts     |
| `Ap`, `An` | How fast switching occurs |
| `xp`, `xn` | Where saturation begins   |

---

## 🎯 Physical Interpretation

This model captures:

* Threshold-based switching
* Nonlinear I-V behavior
* State saturation
* Device variability

---

## 🌪 Device Variability

The paper introduces Gaussian noise on:

* `Ap`, `An`
* `Vth,p`, `Vth,n`

This models:

* Cycle-to-cycle variation
* Device-to-device variation

---

## 🧠 Key Insight

> MemTorch parameters are not arbitrary —
> they represent **physical device behavior translated into neural network weight dynamics**

---

## 🚀 Summary

| Category        | Meaning              |
| --------------- | -------------------- |
| Conductance     | Weight range         |
| State variable  | Weight               |
| Threshold       | Update trigger       |
| Switching speed | Learning rate        |
| Nonlinearity    | Analog distortion    |
| Noise           | Hardware variability |

---

## 🔬 Why This Matters

Understanding this mapping enables:

* More realistic neuromorphic simulations
* Hardware-aware neural network training
* Accurate crossbar modeling

---

## 📎 Future Work

* [ ] Implement MemTorch device class using extracted parameters
* [ ] Simulate crossbar-based inference
* [ ] Analyze training accuracy vs nonlinearity
* [ ] Study noise impact on convergence

---

## 📚 Reference

* Memristor Model Optimization Based on Parameter Extraction from Device Characterization Data
  (See uploaded PDF)

---

## 🙌 Acknowledgment

This project is based on understanding and adapting physical memristor models for use in deep learning frameworks.

---
