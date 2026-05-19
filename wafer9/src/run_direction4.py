import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "Wafer_Map_Datasets.npz"
RESULT_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass(frozen=True)
class ParameterSet:
    on_off_ratio: float
    conductance_window: float
    device_variation: float
    adc_bits: int


class SmallWaferCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(24 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, adc_bits=None):
        x = maybe_quantize(x, adc_bits)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = maybe_quantize(x, adc_bits)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return maybe_quantize(self.fc2(x), adc_bits)


def maybe_quantize(x, bits):
    if bits is None:
        return x
    levels = 2**bits - 1
    xmin = x.amin(dim=tuple(range(1, x.ndim)), keepdim=True)
    xmax = x.amax(dim=tuple(range(1, x.ndim)), keepdim=True)
    scale = (xmax - xmin).clamp_min(1e-6)
    return torch.round((x - xmin) / scale * levels) / levels * scale + xmin


def make_synthetic(n=1600, size=52, seed=41):
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


def balanced_subset(x, y, max_samples, seed=41):
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


def split_data(x, y, test_ratio=0.2, seed=41):
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


def apply_memristor_params(model, params, seed=41):
    rng = np.random.default_rng(seed)
    patched = copy.deepcopy(model)
    ratio_noise = 1.0 / np.sqrt(params.on_off_ratio)
    window_noise = 0.015 / max(params.conductance_window, 1e-6)
    sigma = params.device_variation + ratio_noise + window_noise
    conductance_levels = min(512, max(8, int(np.log2(params.on_off_ratio + 1) * params.conductance_window * 32)))

    with torch.no_grad():
        for name, param in patched.named_parameters():
            if "weight" not in name:
                continue
            w = param.data
            noise = torch.tensor(rng.normal(1.0, sigma, size=w.shape), dtype=w.dtype, device=w.device)
            w *= noise
            w.copy_(quantize_tensor(w, conductance_levels))
    return patched


def quantize_tensor(x, levels):
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin).clamp_min(1e-6)
    return torch.round((x - xmin) / scale * (levels - 1)) / (levels - 1) * scale + xmin


@torch.no_grad()
def evaluate(model, loader, adc_bits=None, device="cpu"):
    model.eval()
    y_true = []
    y_pred = []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb, adc_bits=adc_bits).argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean()), macro_f1(y_true, y_pred)


def macro_f1(y_true, y_pred):
    scores = []
    for cls in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def build_sweep(fast=False):
    ratios = [50, 100, 300, 1000] if fast else [30, 50, 100, 300, 1000, 3000]
    windows = [0.5, 1.0, 2.0] if fast else [0.25, 0.5, 1.0, 2.0]
    variations = [0.03, 0.08] if fast else [0.01, 0.03, 0.05, 0.08]
    adc_bits = [4, 6, 8]
    return [ParameterSet(r, w, v, b) for r in ratios for w in windows for v in variations for b in adc_bits]


def save_results(rows):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_path = RESULT_DIR / "parameter_sweep.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    heatmap_path = RESULT_DIR / "heatmap_accuracy.csv"
    ratios = sorted({row["on_off_ratio"] for row in rows})
    adc_bits = sorted({row["adc_bits"] for row in rows})
    with heatmap_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["on_off_ratio"] + [f"adc_{b}bit" for b in adc_bits])
        for ratio in ratios:
            line = [ratio]
            for bits in adc_bits:
                vals = [row["accuracy"] for row in rows if row["on_off_ratio"] == ratio and row["adc_bits"] == bits]
                line.append(round(float(np.mean(vals)), 6))
            writer.writerow(line)
    return sweep_path, heatmap_path


def print_guidelines(best, rows):
    print("\ntask_aware_parameter_guidelines")
    print(f"recommended_on_off_ratio={best['on_off_ratio']}")
    print(f"recommended_conductance_window={best['conductance_window']}")
    print(f"recommended_device_variation<={best['device_variation']}")
    print(f"recommended_adc_bits={best['adc_bits']}")

    by_ratio = {}
    for row in rows:
        by_ratio.setdefault(row["on_off_ratio"], []).append(row["accuracy"])
    print("\naccuracy_vs_on_off_ratio")
    for ratio, vals in sorted(by_ratio.items()):
        print(f"ratio={ratio}, mean_accuracy={np.mean(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Research direction 4: task-aware memristor parameter optimization")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--fast-sweep", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[Direction 4] Task-aware memristor parameter optimization")
    x, y = load_data(args.data, args.max_samples, args.synthetic)
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = SmallWaferCNN(num_classes=int(y.max()) + 1)
    train_model(model, train_loader, epochs=args.epochs, device=args.device)
    ideal_acc, ideal_f1 = evaluate(model.to(args.device), test_loader, device=args.device)
    print(f"ideal_cnn,accuracy={ideal_acc:.4f},macro_f1={ideal_f1:.4f}")

    rows = []
    for params in build_sweep(fast=args.fast_sweep):
        noisy_model = apply_memristor_params(model, params).to(args.device)
        acc, f1 = evaluate(noisy_model, test_loader, adc_bits=params.adc_bits, device=args.device)
        row = {
            "on_off_ratio": params.on_off_ratio,
            "conductance_window": params.conductance_window,
            "device_variation": params.device_variation,
            "adc_bits": params.adc_bits,
            "accuracy": acc,
            "macro_f1": f1,
            "accuracy_drop": ideal_acc - acc,
            "f1_drop": ideal_f1 - f1,
        }
        rows.append(row)
        print(
            f"ratio={params.on_off_ratio}, window={params.conductance_window}, "
            f"variation={params.device_variation}, adc={params.adc_bits}, "
            f"accuracy={acc:.4f}, macro_f1={f1:.4f}"
        )

    best = max(rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    print(
        f"\nbest_parameter_set: ratio={best['on_off_ratio']}, window={best['conductance_window']}, "
        f"variation={best['device_variation']}, adc={best['adc_bits']}, "
        f"accuracy={best['accuracy']:.4f}, macro_f1={best['macro_f1']:.4f}"
    )
    sweep_path, heatmap_path = save_results(rows)
    print(f"saved_parameter_sweep={sweep_path}")
    print(f"saved_accuracy_heatmap_csv={heatmap_path}")
    print_guidelines(best, rows)


if __name__ == "__main__":
    main()
