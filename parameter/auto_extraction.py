#This code is based on Paper "Memristor  Model Optimization Based on Parameter Extraction from Device-
#-Characterization Data

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp


@dataclass
class MemristorParams:
    Vth_p: float
    Vth_n: float
    gmin: float
    gmax: float
    b: float
    Ap: float
    An: float
    xp: float
    xn: float


def safe_conductance(V, I, eps=1e-9):
    return I / np.where(np.abs(V) < eps, np.sign(V) * eps + eps, V)


def sinh_model(V, gmin, b):
    return gmin * np.sinh(b * V)


def extract_thresholds(V, I):
    dV = np.gradient(V)
    dIdV = np.gradient(I, V, edge_order=1)

    pos_region = (V > 0) & (dV > 0)
    neg_region = (V < 0) & (dV < 0)

    Vth_p = V[pos_region][np.argmax(dIdV[pos_region])]
    Vth_n = V[neg_region][np.argmin(dIdV[neg_region])]

    return Vth_p, Vth_n


def fit_stable_states(V, I, Vth_p, Vth_n):
    G = np.abs(safe_conductance(V, I))

    high_mask = G >= np.percentile(G, 80)
    low_mask = G <= np.percentile(G, 30)

    # ON state: i = gmax * v
    V_on = V[high_mask]
    I_on = I[high_mask]
    gmax = np.sum(V_on * I_on) / np.sum(V_on ** 2)

    # OFF state: i = gmin * sinh(bv)
    V_off = V[low_mask]
    I_off = I[low_mask]

    popt, _ = curve_fit(
        sinh_model,
        V_off,
        I_off,
        p0=[np.min(G[G > 0]), 2.0],
        maxfev=20000
    )

    gmin, b = popt
    return abs(gmin), abs(gmax), abs(b)


def estimate_dynamic_params(V, I, gmin, gmax):
    G = np.abs(safe_conductance(V, I))
    dGdt = np.gradient(G)  # uniform sampling assumed

    pos_idx = np.argmax(dGdt)
    neg_idx = np.argmin(dGdt)

    gpk_p = abs(dGdt[pos_idx])
    gpk_n = abs(dGdt[neg_idx])

    denom = max(gmax - gmin, 1e-12)

    Ap = gpk_p / denom
    An = gpk_n / denom

    # point after max switching speed
    slow_p_idx = min(pos_idx + 1, len(G) - 1)
    slow_n_idx = min(neg_idx + 1, len(G) - 1)

    xp = (G[slow_p_idx] - gmin) / denom
    xn = (G[slow_n_idx] - gmin) / denom

    xp = float(np.clip(xp, 0.0, 1.0))
    xn = float(np.clip(xn, 0.0, 1.0))

    return Ap, An, xp, xn


def extract_memristor_params(csv_path, voltage_col="V", current_col="I"):
    df = pd.read_csv(csv_path)

    V = df[voltage_col].to_numpy(dtype=float)
    I = df[current_col].to_numpy(dtype=float)

    Vth_p, Vth_n = extract_thresholds(V, I)
    gmin, gmax, b = fit_stable_states(V, I, Vth_p, Vth_n)
    Ap, An, xp, xn = estimate_dynamic_params(V, I, gmin, gmax)

    return MemristorParams(
        Vth_p=Vth_p,
        Vth_n=Vth_n,
        gmin=gmin,
        gmax=gmax,
        b=b,
        Ap=Ap,
        An=An,
        xp=xp,
        xn=xn
    )


def model_current(V, x, p: MemristorParams):
    ion = p.gmax * V
    ioff = p.gmin * np.sinh(p.b * V)
    return ion * x + ioff * (1 - x)


def g_voltage(V, p: MemristorParams):
    if V > p.Vth_p:
        return p.Ap * (np.exp(V) - np.exp(p.Vth_p))
    elif V < p.Vth_n:
        return -p.An * (np.exp(-V) - np.exp(-p.Vth_n))
    else:
        return 0.0


def f_window(x, V, p: MemristorParams):
    x = np.clip(x, 0.0, 1.0)

    if V > 0:
        if x >= p.xp:
            return 1.0
        return np.exp(-((p.xp - x) / max(p.xp, 1e-9)))
    else:
        if x <= 1 - p.xn:
            return 1.0
        return np.exp(-((x - (1 - p.xn)) / max(p.xn, 1e-9)))


def simulate_iv(V_waveform, p: MemristorParams, dt=1e-3, x0=0.5):
    x = x0
    I_out = []
    X_out = []

    for V in V_waveform:
        dxdt = g_voltage(V, p) * f_window(x, V, p)
        x = np.clip(x + dxdt * dt, 0.0, 1.0)

        I_out.append(model_current(V, x, p))
        X_out.append(x)

    return np.array(I_out), np.array(X_out)


if __name__ == "__main__":
    params = extract_memristor_params("IV_charaterization/fig.10", voltage_col="V", current_col="I")
    print(params)

    df = pd.read_csv("iv_data.csv")
    V = df["V"].to_numpy()

    I_model, X_model = simulate_iv(V, params)

    out = pd.DataFrame({
        "V": V,
        "I_model": I_model,
        "x": X_model
    })

    out.to_csv("model_result.csv", index=False)
    print("Saved: model_result.csv")