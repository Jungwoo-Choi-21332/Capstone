import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "Wafer_Map_Datasets.npz"


@dataclass(frozen=True)
class MaterialProfile:
    name: str
    r_on: float
    r_off: float
    v_th: float
    nonlinearity: float
    switching_noise: float
    retention_nu: float
    description: str


MATERIALS = [
    MaterialProfile("TaOx", 1.0e3, 1.0e6, 0.55, 1.04, 0.035, 0.020, "high Roff/Ron, stable oxide switching"),
    MaterialProfile("HfO2", 1.5e3, 5.0e5, 0.75, 1.12, 0.060, 0.035, "CMOS-friendly but more variable switching"),
    MaterialProfile("IGZO", 2.0e3, 8.0e5, 0.45, 1.07, 0.040, 0.025, "oxide semiconductor, low-voltage operation"),
    MaterialProfile("NiO", 8.0e2, 1.5e5, 0.90, 1.20, 0.090, 0.050, "lower on/off ratio and stronger nonlinearity"),
]


class SmallWaferCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(24 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def make_synthetic(n=1600, size=52, seed=29):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size), dtype=np.float32)
    labels = np.arange(n, dtype=np.int64) % 8
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    mask = rr < size * 0.46
    for i, cls in enumerate(labels):
        wafer = rng.normal(0.02, 0.04, (size, size))
        if cls == 0:
            wafer[(xx - center) ** 2 + (yy - center) ** 2 < (size * 0.13) ** 2] += 0.9
        elif cls == 1:
            wafer[(rr > size * 0.17) & (rr < size * 0.27)] += 0.9
        elif cls == 2:
            wafer[(rr > size * 0.36) & (xx < center)] += 0.9
        elif cls == 3:
            wafer[(rr > size * 0.34) & mask] += 0.9
        elif cls == 4:
            cy, cx = rng.integers(14, size - 14, 2)
            wafer[cy - 5:cy + 6, cx - 5:cx + 6] += 0.9
        elif cls == 5:
            for _ in range(12):
                cy, cx = rng.integers(8, size - 8, 2)
                wafer[cy - 2:cy + 3, cx - 2:cx + 3] += 0.8
        elif cls == 6:
            offset = rng.integers(-8, 8)
            wafer[np.abs((yy - xx) - offset) < 2] += 0.9
        else:
            wafer[mask] += 0.65
        images[i] = np.clip(wafer * mask, 0, 1)
    return images, labels


def load_data(path, max_samples=None, synthetic=False):
    if synthetic or not path.exists():
        return make_synthetic(n=max_samples or 1600)
    with np.load(path, allow_pickle=True) as data:
        x = data["arr_0"].astype(np.float32)
        y_raw = data["arr_1"]
    y = np.argmax(y_raw, axis=1) if y_raw.ndim == 2 else y_raw.astype(np.int64)
    if max_samples and len(x) > max_samples:
        x, y = balanced_subset(x, y, max_samples)
    x = x / max(float(x.max()), 1.0)
    return np.clip(x, 0.0, 1.0), y


def balanced_subset(x, y, max_samples, seed=29):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_samples // len(classes))
    chosen = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        chosen.extend(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False))
    if len(chosen) < max_samples:
        rest = np.setdiff1d(np.arange(len(y)), np.array(chosen), assume_unique=False)
        chosen.extend(rng.choice(rest, size=min(max_samples - len(chosen), len(rest)), replace=False))
    chosen = np.array(chosen[:max_samples])
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


def split_data(x, y, test_ratio=0.2, seed=29):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_ratio))
    return x[idx[:cut]], x[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def make_loader(x, y, batch_size=64, shuffle=False):
    xt = torch.tensor(x[:, None, :, :], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


def train_model(model, loader, epochs=3, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total = 0.0
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        print(f"epoch={epoch}, train_loss={total / len(loader.dataset):.4f}")


def apply_material_nonidealities(model, material, seed=29):
    rng = np.random.default_rng(seed)
    patched = copy.deepcopy(model)
    on_off_ratio = material.r_off / material.r_on
    ratio_penalty = 1.0 / np.sqrt(max(on_off_ratio, 1.0))
    threshold_penalty = max(0.0, material.v_th - 0.5) * 0.03
    sigma = material.switching_noise + ratio_penalty + threshold_penalty
    drift = (1.0 + 10.0) ** (-material.retention_nu)

    with torch.no_grad():
        for name, param in patched.named_parameters():
            if "weight" not in name:
                continue
            w = param.data
            noise = torch.tensor(rng.normal(1.0, sigma, size=w.shape), dtype=w.dtype, device=w.device)
            w *= noise * drift
            conductance_levels = min(256, max(16, int(np.log2(on_off_ratio + 1) * 24)))
            w.copy_(quantize_tensor(w, conductance_levels))
    return patched


def quantize_tensor(x, levels):
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin).clamp_min(1e-6)
    return torch.round((x - xmin) / scale * (levels - 1)) / (levels - 1) * scale + xmin


@torch.no_grad()
def evaluate(model, loader, material=None, device="cpu"):
    model.eval()
    y_true = []
    y_pred = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        if material is not None:
            logits = nonlinear_output(logits, material.nonlinearity)
            logits = apply_readout_noise(logits, material)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return accuracy(y_true, y_pred), macro_f1(y_true, y_pred), y_true, y_pred


def nonlinear_output(logits, alpha):
    signed = torch.sign(logits)
    return signed * torch.pow(torch.abs(logits).clamp_min(1e-8), alpha)


def apply_readout_noise(logits, material):
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(int(material.v_th * 1000) + int(material.r_on))
    on_off = material.r_off / material.r_on
    noise_std = material.switching_noise * 0.35 + 0.03 / np.log10(on_off)
    gain = (1.0 + 10.0) ** (-material.retention_nu)
    noise = torch.randn(logits.shape, generator=generator, device=logits.device) * noise_std
    return logits * gain + noise


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def sensitivity_index(material):
    on_off = material.r_off / material.r_on
    return (
        material.switching_noise * 3.0
        + material.retention_nu * 2.0
        + abs(material.nonlinearity - 1.0)
        + max(0.0, 1.0 / np.log10(on_off))
        + material.v_th * 0.02
    )


def main():
    parser = argparse.ArgumentParser(description="Research direction 3: comparison of memristor materials")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[Direction 3] Comparison of different memristor materials")
    x, y = load_data(args.data, args.max_samples, args.synthetic)
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = SmallWaferCNN(num_classes=int(y.max()) + 1)
    train_model(model, train_loader, epochs=args.epochs, device=args.device)
    ideal_acc, ideal_f1, _, _ = evaluate(model.to(args.device), test_loader, device=args.device)
    print(f"ideal_cnn,accuracy={ideal_acc:.4f},macro_f1={ideal_f1:.4f}")

    print("\nmaterial,accuracy,macro_f1,accuracy_drop,f1_drop,Ron,Roff,on_off_ratio,Vth,nonlinearity,switching_noise,retention_nu,sensitivity_index")
    results = []
    for material in MATERIALS:
        material_model = apply_material_nonidealities(model, material).to(args.device)
        acc, f1, _, _ = evaluate(material_model, test_loader, material=material, device=args.device)
        results.append((material.name, acc, f1))
        print(
            f"{material.name},{acc:.4f},{f1:.4f},{ideal_acc - acc:.4f},{ideal_f1 - f1:.4f},"
            f"{material.r_on:.1f},{material.r_off:.1f},{material.r_off / material.r_on:.1f},"
            f"{material.v_th:.3f},{material.nonlinearity:.3f},{material.switching_noise:.3f},"
            f"{material.retention_nu:.3f},{sensitivity_index(material):.4f}"
        )

    best = max(results, key=lambda row: row[1])
    worst = min(results, key=lambda row: row[1])
    print(f"\nbest_material={best[0]}, best_accuracy={best[1]:.4f}, best_macro_f1={best[2]:.4f}")
    print(f"worst_material={worst[0]}, worst_accuracy={worst[1]:.4f}, worst_macro_f1={worst[2]:.4f}")

    print("\nphysics_interpretation")
    for material in MATERIALS:
        print(f"{material.name}: {material.description}")


if __name__ == "__main__":
    main()
