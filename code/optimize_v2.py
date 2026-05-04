"""
Optimization pipeline v2 for wafer defect classification.

Improvements over the original script:
1. Uses a validation split for threshold/model selection instead of tuning on test data.
2. Wraps each phase in functions so the script is easier to maintain and rerun.
3. Tracks averaged validation metrics across repeated memristor evaluations.
4. Separates final test reporting from model-selection logic.
"""

import json
import os
from pathlib import Path

import memtorch
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from model import FocalLoss, WaferCNN, WaferDataset


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data" / "wafer_data_64.pth"
MODEL_V1_PATH = BASE_DIR / "software_model.pth"
MODEL_V2_PATH = BASE_DIR / "software_model_v2.pth"
MODEL_V2_NOISE_PATH = BASE_DIR / "software_model_v2_noiseaware.pth"
RESULT_JSON_PATH = RESULTS_DIR / "optimize_v2_result.json"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
NOISE_EPOCHS = 15
VAL_RATIO = 0.1
NUM_EVAL = 3
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
NOISE_LEVELS = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.05, 0.05, 0.05, 0.04, 0.04,
    0.03, 0.03, 0.02, 0.02, 0.01,
]

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_DEVICE = torch.device("cpu")

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
    ]
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    data = torch.load(DATA_PATH, weights_only=False)
    classes = list(data["classes"])
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"], classes


def build_validation_split(x_train, y_train, val_ratio=VAL_RATIO, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    num_samples = x_train.shape[0]
    num_val = max(1, int(num_samples * val_ratio))
    perm = torch.randperm(num_samples, generator=generator)
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    return (
        x_train[train_idx],
        y_train[train_idx],
        x_train[val_idx],
        y_train[val_idx],
    )


def create_train_loader(x, y):
    dataset = WaferDataset(x, y, TRAIN_TRANSFORM)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def create_eval_loader(x, y):
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=False)


def train_base_model(x_train, y_train, classes):
    print("=" * 60)
    print("Phase 1: train baseline CNN for 50 epochs")
    print("=" * 60)

    if MODEL_V2_PATH.exists():
        print(f"Skipping training, checkpoint already exists: {MODEL_V2_PATH.name}")
        return

    set_seed(SEED)
    model = WaferCNN(num_classes=len(classes)).to(TRAIN_DEVICE)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loader = create_train_loader(x_train, y_train)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(TRAIN_DEVICE)
            y_batch = y_batch.to(TRAIN_DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Loss: {total_loss / len(loader):.4f} | LR: {lr:.6f}"
        )

    torch.save(model.cpu().state_dict(), MODEL_V2_PATH)
    print(f"Saved {MODEL_V2_PATH.name}")


def inject_weight_noise(model, noise_std):
    original_weights = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            original_weights[name] = param.detach().clone()
            noise = torch.randn_like(param) * param.detach().abs() * noise_std
            param.add_(noise)
    return original_weights


def restore_weights(model, original_weights):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_weights:
                param.copy_(original_weights[name])


def train_noise_aware_model(x_train, y_train, classes):
    print("\n" + "=" * 60)
    print("Phase 2: noise-aware fine-tuning")
    print("=" * 60)

    if MODEL_V2_NOISE_PATH.exists():
        print(f"Skipping training, checkpoint already exists: {MODEL_V2_NOISE_PATH.name}")
        return

    if not MODEL_V2_PATH.exists():
        raise FileNotFoundError(f"Missing base checkpoint: {MODEL_V2_PATH}")

    set_seed(SEED)
    model = WaferCNN(num_classes=len(classes)).to(TRAIN_DEVICE)
    model.load_state_dict(torch.load(MODEL_V2_PATH, map_location=TRAIN_DEVICE, weights_only=True))

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loader = create_train_loader(x_train, y_train)

    for epoch, noise_std in enumerate(NOISE_LEVELS[:NOISE_EPOCHS], start=1):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(TRAIN_DEVICE)
            y_batch = y_batch.to(TRAIN_DEVICE)
            optimizer.zero_grad()

            original_weights = inject_weight_noise(model, noise_std)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            restore_weights(model, original_weights)

            optimizer.step()
            total_loss += loss.item()

        print(
            f"  Epoch {epoch:2d}/{NOISE_EPOCHS} | "
            f"Loss: {total_loss / len(loader):.4f} | Noise: {noise_std:.2%}"
        )

    torch.save(model.cpu().state_dict(), MODEL_V2_NOISE_PATH)
    print(f"Saved {MODEL_V2_NOISE_PATH.name}")


def build_patched_model(model_path, num_classes):
    model = WaferCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=EVAL_DEVICE, weights_only=True))
    model.eval()
    patched = memtorch.patch_model(
        model,
        memristor_model=memtorch.bh.memristor.VTEAM,
        memristor_model_params={"r_on": 50, "r_off": 500},
    )
    patched.eval()
    return patched


def predict_multilabel(patched_model, loader, threshold):
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(EVAL_DEVICE)
            pred = (torch.sigmoid(patched_model(x_batch)) > threshold).int()
            preds_list.extend(pred.cpu().numpy())
            labels_list.extend(y_batch.cpu().numpy())
    return np.array(labels_list), np.array(preds_list)


def evaluate_thresholds(model_path, model_name, x_eval, y_eval, classes):
    loader = create_eval_loader(x_eval, y_eval)
    best_candidate = None

    print(f"\n--- {model_name} ---")
    for threshold in THRESHOLDS:
        reports = []
        f1s = []

        for _ in range(NUM_EVAL):
            patched = build_patched_model(model_path, len(classes))
            labels, preds = predict_multilabel(patched, loader, threshold)
            report = classification_report(
                labels,
                preds,
                target_names=classes,
                zero_division=0,
                output_dict=True,
            )
            reports.append(report)
            f1s.append(report["macro avg"]["f1-score"])

        avg_precision = float(np.mean([r["macro avg"]["precision"] for r in reports]))
        avg_recall = float(np.mean([r["macro avg"]["recall"] for r in reports]))
        avg_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        print(
            f"  threshold={threshold:.2f} | "
            f"Val F1={avg_f1:.4f} | Std={std_f1:.4f}"
        )

        candidate = {
            "model": model_name,
            "model_path": str(model_path),
            "threshold": threshold,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "f1_std": std_f1,
            "all_f1": [float(x) for x in f1s],
        }

        if best_candidate is None or candidate["f1"] > best_candidate["f1"]:
            best_candidate = candidate

    return best_candidate


def evaluate_final(model_path, threshold, x_test, y_test, classes):
    loader = create_eval_loader(x_test, y_test)
    reports = []
    final_labels = None
    final_preds = None

    for _ in range(NUM_EVAL):
        patched = build_patched_model(model_path, len(classes))
        labels, preds = predict_multilabel(patched, loader, threshold)
        report = classification_report(
            labels,
            preds,
            target_names=classes,
            zero_division=0,
            output_dict=True,
        )
        reports.append(report)
        final_labels = labels
        final_preds = preds

    test_summary = {
        "precision": float(np.mean([r["macro avg"]["precision"] for r in reports])),
        "recall": float(np.mean([r["macro avg"]["recall"] for r in reports])),
        "f1": float(np.mean([r["macro avg"]["f1-score"] for r in reports])),
        "f1_std": float(np.std([r["macro avg"]["f1-score"] for r in reports])),
    }

    final_report = classification_report(
        final_labels,
        final_preds,
        target_names=classes,
        zero_division=0,
    )
    return test_summary, final_report


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test, classes = load_data()
    x_fit, y_fit, x_val, y_val = build_validation_split(x_train, y_train)

    print(f"Training device: {TRAIN_DEVICE}")
    print(f"Evaluation device: {EVAL_DEVICE}")
    print(f"Train split: {len(x_fit)} | Validation split: {len(x_val)} | Test split: {len(x_test)}")

    train_base_model(x_fit, y_fit, classes)
    train_noise_aware_model(x_fit, y_fit, classes)

    print("\n" + "=" * 60)
    print("Phase 3: model comparison and threshold search on validation set")
    print("=" * 60)

    model_candidates = {
        "v1_epoch30": MODEL_V1_PATH,
        "v2_epoch50": MODEL_V2_PATH,
        "v2_noiseaware": MODEL_V2_NOISE_PATH,
    }

    best_overall = None
    for model_name, model_path in model_candidates.items():
        if not model_path.exists():
            print(f"  {model_name}: checkpoint not found, skipping")
            continue
        candidate = evaluate_thresholds(model_path, model_name, x_val, y_val, classes)
        if best_overall is None or candidate["f1"] > best_overall["f1"]:
            best_overall = candidate

    if best_overall is None:
        raise RuntimeError("No model checkpoints were available for evaluation.")

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Best Model:     {best_overall['model']}")
    print(f"  Best Threshold: {best_overall['threshold']}")
    print(f"  Val F1:         {best_overall['f1']:.4f}")
    print(f"  Val Precision:  {best_overall['precision']:.4f}")
    print(f"  Val Recall:     {best_overall['recall']:.4f}")

    test_summary, final_report = evaluate_final(
        Path(best_overall["model_path"]),
        best_overall["threshold"],
        x_test,
        y_test,
        classes,
    )

    print("\n[Test Classification Report]")
    print(final_report)
    print("Test macro averages:")
    print(f"  Precision: {test_summary['precision']:.4f}")
    print(f"  Recall:    {test_summary['recall']:.4f}")
    print(f"  F1:        {test_summary['f1']:.4f}")
    print(f"  F1 Std:    {test_summary['f1_std']:.4f}")

    result_payload = {
        "selection": best_overall,
        "test_summary": test_summary,
        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "noise_epochs": NOISE_EPOCHS,
            "val_ratio": VAL_RATIO,
            "thresholds": THRESHOLDS,
            "num_eval": NUM_EVAL,
        },
    }

    with RESULT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {RESULT_JSON_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
