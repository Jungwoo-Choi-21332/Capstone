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


@dataclass
class HardwareNoise:
    conductance_variation: float = 0.08
    adc_bits: int = 6
    dac_bits: int = 6
    retention_time: float = 10.0
    retention_nu: float = 0.03
    stuck_fault_rate: float = 0.01
    seed: int = 53


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


def make_synthetic(n=1600, size=52, seed=53):
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


def balanced_subset(x, y, max_samples, seed=53):
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


def split_data(x, y, test_ratio=0.2, seed=53):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_ratio))
    return x[idx[:cut]], x[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def make_loader(x, y, batch_size=64, shuffle=False):
    xt = torch.tensor(x[:, None, :, :], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


def train_model(model, loader, epochs, noise_aware=False, noise=None, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if noise_aware:
                xb = maybe_quantize(xb, noise.dac_bits)
                logits = forward_with_training_noise(model, xb, noise)
            else:
                logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        mode = "noise_aware" if noise_aware else "ideal"
        print(f"{mode}_epoch={epoch}, train_loss={total / len(loader.dataset):.4f}")


def forward_with_training_noise(model, x, noise):
    inject_weight_noise_(model, noise, scale=0.35)
    logits = model(x, adc_bits=noise.adc_bits)
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(noise.seed)
    logit_noise = torch.randn(logits.shape, generator=generator, device=logits.device)
    logits = logits + logit_noise * (noise.conductance_variation * 0.25 + 0.01)
    drift_gain = (1.0 + noise.retention_time) ** (-noise.retention_nu)
    return logits * drift_gain


def inject_weight_noise_(model, noise, scale=1.0):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            param.add_(torch.randn_like(param) * noise.conductance_variation * scale * param.std().clamp_min(1e-6))


def map_to_hardware(model, noise):
    rng = np.random.default_rng(noise.seed)
    patched = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in patched.named_parameters():
            if "weight" not in name:
                continue
            w = param.data
            variation = torch.tensor(rng.normal(1.0, noise.conductance_variation, size=w.shape), dtype=w.dtype, device=w.device)
            w *= variation
            w *= (1.0 + noise.retention_time) ** (-noise.retention_nu)
            if noise.stuck_fault_rate > 0:
                mask = torch.tensor(rng.random(w.shape) < noise.stuck_fault_rate, device=w.device)
                stuck_sign = torch.tensor(rng.choice([-1.0, 1.0], size=w.shape), dtype=w.dtype, device=w.device)
                stuck_value = w.abs().max().clamp_min(1e-6) * stuck_sign
                w[mask] = stuck_value[mask]
            w.copy_(quantize_tensor(w, 2**noise.adc_bits))
    return patched


def quantize_tensor(x, levels):
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin).clamp_min(1e-6)
    return torch.round((x - xmin) / scale * (levels - 1)) / (levels - 1) * scale + xmin


@torch.no_grad()
def evaluate(model, loader, hardware_noise=None, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    macro_true = []
    macro_pred = []
    adc_bits = None if hardware_noise is None else hardware_noise.adc_bits
    for xb, yb in loader:
        xb = xb.to(device)
        if hardware_noise is not None:
            xb = maybe_quantize(xb, hardware_noise.dac_bits)
        pred = model(xb, adc_bits=adc_bits).argmax(dim=1).cpu().numpy()
        correct += int((pred == yb.numpy()).sum())
        total += len(yb)
        macro_true.extend(yb.numpy().tolist())
        macro_pred.extend(pred.tolist())
    return correct / max(total, 1), macro_f1(np.array(macro_true), np.array(macro_pred))


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


def compare_training_methods(ideal_model, noise_model, test_loader, noise, device):
    rows = []
    for name, model in [("Ideal Training", ideal_model), ("Noise-Aware Training", noise_model)]:
        ideal_acc, ideal_f1 = evaluate(model.to(device), test_loader, device=device)
        hardware_model = map_to_hardware(model, noise).to(device)
        hw_acc, hw_f1 = evaluate(hardware_model, test_loader, hardware_noise=noise, device=device)
        rows.append(
            {
                "method": name,
                "ideal_accuracy": ideal_acc,
                "hardware_accuracy": hw_acc,
                "accuracy_drop": ideal_acc - hw_acc,
                "ideal_macro_f1": ideal_f1,
                "hardware_macro_f1": hw_f1,
                "f1_drop": ideal_f1 - hw_f1,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Research direction 5: hardware-aware noise-aware training")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variation", type=float, default=0.08)
    parser.add_argument("--adc-bits", type=int, default=6)
    parser.add_argument("--stuck-rate", type=float, default=0.01)
    args = parser.parse_args()

    print("[Direction 5] Hardware-aware / noise-aware training")
    noise = HardwareNoise(
        conductance_variation=args.variation,
        adc_bits=args.adc_bits,
        dac_bits=args.adc_bits,
        stuck_fault_rate=args.stuck_rate,
    )
    x, y = load_data(args.data, args.max_samples, args.synthetic)
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    ideal_model = SmallWaferCNN(num_classes=int(y.max()) + 1)
    noise_model = SmallWaferCNN(num_classes=int(y.max()) + 1)

    train_model(ideal_model, train_loader, args.epochs, noise_aware=False, noise=noise, device=args.device)
    train_model(noise_model, train_loader, args.epochs, noise_aware=True, noise=noise, device=args.device)

    rows = compare_training_methods(ideal_model, noise_model, test_loader, noise, args.device)
    print("\ntraining_method,ideal_accuracy,hardware_accuracy,accuracy_drop,ideal_macro_f1,hardware_macro_f1,f1_drop")
    for row in rows:
        print(
            f"{row['method']},{row['ideal_accuracy']:.4f},{row['hardware_accuracy']:.4f},"
            f"{row['accuracy_drop']:.4f},{row['ideal_macro_f1']:.4f},"
            f"{row['hardware_macro_f1']:.4f},{row['f1_drop']:.4f}"
        )

    ideal_drop = rows[0]["accuracy_drop"]
    noise_drop = rows[1]["accuracy_drop"]
    reduction = 0.0 if ideal_drop <= 0 else (ideal_drop - noise_drop) / ideal_drop
    print(f"\nrobustness_gain_drop_reduction={reduction:.4f}")
    print(
        f"hardware_noise: variation={noise.conductance_variation}, adc_bits={noise.adc_bits}, "
        f"retention_time={noise.retention_time}, stuck_fault_rate={noise.stuck_fault_rate}"
    )


if __name__ == "__main__":
    main()
