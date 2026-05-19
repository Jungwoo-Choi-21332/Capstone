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
CLASS_NAMES = {
    0: "Center",
    1: "Donut",
    2: "Edge-Loc",
    3: "Edge-Ring",
    4: "Loc",
    5: "Random",
    6: "Scratch",
    7: "Near-full",
}


@dataclass
class NoiseConfig:
    name: str
    conductance_variation: float = 0.0
    adc_bits: int | None = None
    dac_bits: int | None = None
    retention_time: float = 0.0
    retention_nu: float = 0.03
    seed: int = 77


class SmallWaferCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(24 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, quantize_bits=None):
        x = maybe_quantize(x, quantize_bits)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = maybe_quantize(x, quantize_bits)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return maybe_quantize(self.fc2(x), quantize_bits)


def maybe_quantize(x, bits):
    if bits is None or bits <= 0:
        return x
    levels = 2**bits - 1
    xmin = x.amin(dim=tuple(range(1, x.ndim)), keepdim=True)
    xmax = x.amax(dim=tuple(range(1, x.ndim)), keepdim=True)
    scale = (xmax - xmin).clamp_min(1e-6)
    return torch.round((x - xmin) / scale * levels) / levels * scale + xmin


def make_synthetic(n=1600, size=52, seed=17):
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
            wafer[(xx - center) ** 2 + (yy - center) ** 2 < (size * 0.13) ** 2] += 0.9
        elif cls == 1:
            wafer[(rr > size * 0.17) & (rr < size * 0.27)] += 0.9
        elif cls == 2:
            wafer[(rr > size * 0.36) & (xx < center)] += 0.9
        elif cls == 3:
            wafer[(rr > size * 0.34) & wafer_mask] += 0.9
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
            wafer[wafer_mask] += 0.65
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
    x = x / max(float(x.max()), 1.0)
    return np.clip(x, 0.0, 1.0), y


def balanced_subset(x, y, max_samples, seed=17):
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


def split_data(x, y, test_ratio=0.2, seed=17):
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
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        print(f"epoch={epoch}, train_loss={total / len(loader.dataset):.4f}")


def perturb_model(model, config):
    rng = np.random.default_rng(config.seed)
    patched = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in patched.named_parameters():
            if "weight" not in name:
                continue
            w = param.data
            if config.conductance_variation > 0:
                noise = torch.tensor(rng.normal(1.0, config.conductance_variation, size=w.shape), dtype=w.dtype)
                w *= noise.to(w.device)
            if config.retention_time > 0:
                w *= (1.0 + config.retention_time) ** (-config.retention_nu)
    return patched


@torch.no_grad()
def predict(model, loader, config=None, device="cpu"):
    model.eval()
    y_true = []
    y_pred = []
    bits = None if config is None else config.adc_bits
    for xb, yb in loader:
        xb = xb.to(device)
        if config and config.dac_bits:
            xb = maybe_quantize(xb, config.dac_bits)
        logits = model(xb, quantize_bits=bits)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_scores(cm):
    rows = []
    for cls in range(cm.shape[0]):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append((cls, precision, recall, f1, int(cm[cls, :].sum())))
    return rows


def print_class_table(title, scores, baseline=None):
    print(f"\n{title}")
    print("class_id,class_name,precision,recall,f1,f1_drop,support")
    for cls, precision, recall, f1, support in scores:
        base_f1 = None if baseline is None else baseline[cls][3]
        drop = 0.0 if base_f1 is None else base_f1 - f1
        print(
            f"{cls},{CLASS_NAMES.get(cls, f'class-{cls}')},"
            f"{precision:.4f},{recall:.4f},{f1:.4f},{drop:.4f},{support}"
        )


def geometry_features(images, labels):
    rows = []
    h, w = images.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    edge = rr > min(h, w) * 0.35
    center = rr < min(h, w) * 0.18
    for cls in np.unique(labels):
        cls_img = images[labels == cls]
        binary = cls_img > 0.5
        sparsity = binary.reshape(len(cls_img), -1).mean(axis=1).mean()
        edge_ratio = binary[:, edge].mean()
        center_ratio = binary[:, center].mean()
        rows.append((int(cls), float(sparsity), float(edge_ratio), float(center_ratio)))
    return rows


def print_failure_cases(y_true, y_pred, max_cases=20):
    print("\nfailure_cases")
    print("index,true_class,pred_class")
    misses = np.where(y_true != y_pred)[0][:max_cases]
    for idx in misses:
        print(f"{idx},{CLASS_NAMES.get(int(y_true[idx]), y_true[idx])},{CLASS_NAMES.get(int(y_pred[idx]), y_pred[idx])}")


def run_condition(model, loader, config, baseline_scores, num_classes, device):
    noisy_model = perturb_model(model, config).to(device)
    y_true, y_pred = predict(noisy_model, loader, config=config, device=device)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    scores = per_class_scores(cm)
    print_class_table(f"condition={config.name}", scores, baseline=baseline_scores)
    print("confusion_matrix")
    print(cm)
    print_failure_cases(y_true, y_pred)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Research direction 2: class-wise sensitivity under memristor noise")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[Direction 2] Class-wise sensitivity analysis under memristor non-idealities")
    x, y = load_data(args.data, args.max_samples, args.synthetic)
    num_classes = int(y.max()) + 1
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    print("\ndefect_geometry_summary")
    print("class_id,class_name,sparsity,edge_ratio,center_ratio")
    for cls, sparsity, edge_ratio, center_ratio in geometry_features(x_test, y_test):
        print(f"{cls},{CLASS_NAMES.get(cls, f'class-{cls}')},{sparsity:.4f},{edge_ratio:.4f},{center_ratio:.4f}")

    model = SmallWaferCNN(num_classes=num_classes)
    train_model(model, train_loader, epochs=args.epochs, device=args.device)
    y_true, y_pred = predict(model.to(args.device), test_loader, device=args.device)
    ideal_cm = confusion_matrix(y_true, y_pred, num_classes)
    baseline_scores = per_class_scores(ideal_cm)
    print_class_table("condition=ideal_cnn", baseline_scores)
    print("confusion_matrix")
    print(ideal_cm)

    conditions = [
        NoiseConfig(name="conductance_variation_0.10", conductance_variation=0.10, adc_bits=8, dac_bits=8),
        NoiseConfig(name="adc_dac_4bit", conductance_variation=0.03, adc_bits=4, dac_bits=4),
        NoiseConfig(name="retention_drift_t10", retention_time=10, adc_bits=8, dac_bits=8),
        NoiseConfig(name="combined_noise", conductance_variation=0.10, adc_bits=4, dac_bits=4, retention_time=10),
    ]
    for config in conditions:
        run_condition(model, test_loader, config, baseline_scores, num_classes, args.device)


if __name__ == "__main__":
    main()
