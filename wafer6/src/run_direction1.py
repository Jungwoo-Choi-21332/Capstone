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
class NonIdealityConfig:
    conductance_variation: float = 0.0
    adc_bits: int | None = None
    dac_bits: int | None = None
    retention_time: float = 0.0
    retention_nu: float = 0.03
    stuck_fault_rate: float = 0.0
    nonlinear_alpha: float = 1.0
    crossbar_rows: int = 128
    crossbar_cols: int = 128
    seed: int = 42


class SmallWaferCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(24 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, quantize_bits=None, nonlinear_alpha=1.0):
        x = maybe_quantize(x, quantize_bits)
        x = F.relu(self.conv1(x))
        x = maybe_nonlinear(x, nonlinear_alpha)
        x = F.max_pool2d(x, 2)
        x = maybe_quantize(x, quantize_bits)
        x = F.relu(self.conv2(x))
        x = maybe_nonlinear(x, nonlinear_alpha)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = maybe_nonlinear(x, nonlinear_alpha)
        x = self.fc2(x)
        return maybe_quantize(x, quantize_bits)


def maybe_quantize(x, bits):
    if bits is None or bits <= 0:
        return x
    levels = 2**bits - 1
    xmin = x.amin(dim=tuple(range(1, x.ndim)), keepdim=True)
    xmax = x.amax(dim=tuple(range(1, x.ndim)), keepdim=True)
    scale = (xmax - xmin).clamp_min(1e-6)
    return torch.round((x - xmin) / scale * levels) / levels * scale + xmin


def maybe_nonlinear(x, alpha):
    if abs(alpha - 1.0) < 1e-6:
        return x
    signed = torch.sign(x)
    return signed * torch.pow(torch.abs(x).clamp_min(1e-8), alpha)


def make_synthetic(n=1600, size=52, seed=7):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    wafer_mask = rr < size * 0.46

    for i in range(n):
        cls = i % 8
        labels[i] = cls
        wafer = rng.normal(0.02, 0.04, (size, size))
        if cls == 0:
            wafer += rng.normal(0.02, 0.02, wafer.shape)
        elif cls == 1:
            wafer[(rr > size * 0.35) & wafer_mask] += 1.0
        elif cls == 2:
            wafer[:, size // 2 - 2:size // 2 + 2] += 1.0
        elif cls == 3:
            wafer[size // 2 - 2:size // 2 + 2, :] += 1.0
        elif cls == 4:
            wafer[np.abs(xx - yy) < 2] += 1.0
        elif cls == 5:
            wafer[(xx - center) ** 2 + (yy - center) ** 2 < (size * 0.14) ** 2] += 1.0
        elif cls == 6:
            wafer[(xx < size * 0.30) & wafer_mask] += 1.0
        else:
            for _ in range(9):
                cy, cx = rng.integers(8, size - 8, 2)
                wafer[cy - 2:cy + 3, cx - 2:cx + 3] += 1.0
        images[i] = np.clip(wafer * wafer_mask, 0, 1)
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
    x = normalize_binary_like(x)
    return x, y


def normalize_binary_like(x):
    x = x.astype(np.float32)
    xmax = max(float(x.max()), 1.0)
    return np.clip(x / xmax, 0.0, 1.0)


def balanced_subset(x, y, max_samples, seed=7):
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


def split_data(x, y, test_ratio=0.2, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_ratio))
    return x[idx[:cut]], x[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def make_loader(x, y, batch_size=64, shuffle=False):
    xt = torch.tensor(x[:, None, :, :], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, epochs=3, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
        print(f"epoch={epoch}, train_loss={total_loss / len(train_loader.dataset):.4f}")


@torch.no_grad()
def evaluate(model, loader, config=None, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    quant_bits = None if config is None else config.adc_bits
    nonlinear_alpha = 1.0 if config is None else config.nonlinear_alpha
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if config and config.dac_bits:
            xb = maybe_quantize(xb, config.dac_bits)
        logits = model(xb, quantize_bits=quant_bits, nonlinear_alpha=nonlinear_alpha)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += len(xb)
    return correct / max(total, 1)


def perturb_for_crossbar(model, config):
    rng = np.random.default_rng(config.seed)
    patched = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in patched.named_parameters():
            if "weight" not in name:
                continue
            w = param.data
            if config.conductance_variation > 0:
                w *= torch.tensor(
                    rng.normal(1.0, config.conductance_variation, size=w.shape),
                    dtype=w.dtype,
                    device=w.device,
                )
            if config.retention_time > 0:
                drift = (1.0 + config.retention_time) ** (-config.retention_nu)
                w *= drift
            if config.stuck_fault_rate > 0:
                mask = torch.tensor(rng.random(w.shape) < config.stuck_fault_rate, device=w.device)
                signs = torch.tensor(rng.choice([-1.0, 1.0], size=w.shape), dtype=w.dtype, device=w.device)
                stuck_value = w.abs().max().clamp_min(1e-6) * signs
                w[mask] = stuck_value[mask]
            tile_penalty = estimate_crossbar_tile_penalty(w.shape, config.crossbar_rows, config.crossbar_cols)
            if tile_penalty > 0:
                w *= 1.0 - tile_penalty
    return patched


def estimate_crossbar_tile_penalty(weight_shape, rows, cols):
    out_dim = weight_shape[0]
    in_dim = int(np.prod(weight_shape[1:]))
    tiles = int(np.ceil(out_dim / rows) * np.ceil(in_dim / cols))
    return min(0.12, max(0, tiles - 1) * 0.003)


def run_experiment(model, test_loader, label, config, device):
    crossbar_model = perturb_for_crossbar(model, config).to(device)
    acc = evaluate(crossbar_model, test_loader, config=config, device=device)
    print(
        f"{label}, accuracy={acc:.4f}, variation={config.conductance_variation}, "
        f"adc_bits={config.adc_bits}, dac_bits={config.dac_bits}, retention_time={config.retention_time}, "
        f"stuck_fault_rate={config.stuck_fault_rate}, nonlinear_alpha={config.nonlinear_alpha}, "
        f"crossbar={config.crossbar_rows}x{config.crossbar_cols}"
    )
    return acc


def memtorch_status():
    try:
        import memtorch

        return True, getattr(memtorch, "__version__", "unknown")
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(
        description="Research direction 1: robustness of wafer defect CNN under memristor non-idealities"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    memtorch_ok, memtorch_info = memtorch_status()
    print("[Direction 1] Robustness of wafer defect classification under memristor non-idealities")
    print(f"memtorch_available={memtorch_ok}, memtorch_info={memtorch_info}")

    x, y = load_data(args.data, args.max_samples, args.synthetic)
    num_classes = int(np.max(y)) + 1
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = SmallWaferCNN(num_classes=num_classes)
    train_model(model, train_loader, epochs=args.epochs, device=args.device)
    ideal_acc = evaluate(model.to(args.device), test_loader, device=args.device)
    print(f"ideal_cnn, accuracy={ideal_acc:.4f}")

    print("\nconductance_variation_sweep")
    for variation in [0.00, 0.03, 0.05, 0.10, 0.15]:
        run_experiment(
            model,
            test_loader,
            f"variation_{variation:.2f}",
            NonIdealityConfig(conductance_variation=variation, adc_bits=8, dac_bits=8),
            args.device,
        )

    print("\nadc_bitwidth_sweep")
    for bits in [4, 6, 8]:
        run_experiment(
            model,
            test_loader,
            f"adc_{bits}bit",
            NonIdealityConfig(conductance_variation=0.05, adc_bits=bits, dac_bits=bits),
            args.device,
        )

    print("\ncrossbar_size_comparison")
    for rows, cols in [(32, 32), (64, 64), (128, 128), (256, 256)]:
        run_experiment(
            model,
            test_loader,
            f"crossbar_{rows}x{cols}",
            NonIdealityConfig(conductance_variation=0.05, adc_bits=8, dac_bits=8, crossbar_rows=rows, crossbar_cols=cols),
            args.device,
        )

    print("\nfault_and_drift_analysis")
    scenarios = [
        ("retention_t10", NonIdealityConfig(retention_time=10, adc_bits=8, dac_bits=8)),
        ("stuck_1pct", NonIdealityConfig(stuck_fault_rate=0.01, adc_bits=8, dac_bits=8)),
        ("stuck_5pct", NonIdealityConfig(stuck_fault_rate=0.05, adc_bits=8, dac_bits=8)),
        ("nonlinear_iv", NonIdealityConfig(nonlinear_alpha=0.85, adc_bits=8, dac_bits=8)),
        (
            "combined_worst_case",
            NonIdealityConfig(
                conductance_variation=0.10,
                adc_bits=4,
                dac_bits=4,
                retention_time=10,
                stuck_fault_rate=0.02,
                nonlinear_alpha=0.85,
                crossbar_rows=64,
                crossbar_cols=64,
            ),
        ),
    ]
    for label, config in scenarios:
        run_experiment(model, test_loader, label, config, args.device)


if __name__ == "__main__":
    main()
