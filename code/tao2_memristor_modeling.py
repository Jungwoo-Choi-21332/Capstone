import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.map.Input import naive_scale

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# 1. Dataset
# =========================
class WaferMapDataset(Dataset):
    def __init__(self, npz_path="./archive/Wafer_Map_Datasets.npz"):
        data = np.load(npz_path, allow_pickle=True)
        print("NPZ keys:", data.files)

        if "arr_0" in data.files and "arr_1" in data.files:
            x, y = data["arr_0"], data["arr_1"]
        elif "x" in data.files and "y" in data.files:
            x, y = data["x"], data["y"]
        elif "X" in data.files and "Y" in data.files:
            x, y = data["X"], data["Y"]
        else:
            raise KeyError(f"Unknown keys: {data.files}")

        x = np.asarray(x).astype(np.float32)

        if x.ndim == 3:
            x = x[:, None, :, :]
        elif x.ndim == 4 and x.shape[-1] == 1:
            x = np.transpose(x, (0, 3, 1, 2))

        if x.max() > 1:
            x = x / x.max()

        y = np.asarray(y)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataloaders(path, batch_size=32):
    dataset = WaferMapDataset(path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =========================
# 2. CNN Model
# =========================
class WaferCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================
# 3. Training (ideal CNN)
# =========================
def train_cnn(model, loader, device, epochs=3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {ep+1}] Loss: {total_loss:.4f}")


# =========================
# 4. TaOx VTEAM parameters (논문 기반)
# =========================
taox_params = {
    "time_series_resolution": 1e-6,
    "r_on": 4.8e3,
    "r_off": 7.0e5,
    "d": 25e-9,
    "v_on": -0.6,
    "v_off": 0.78,
    "k_on": -0.1,
    "k_off": 1e-8,
    "alpha_on": 18,
    "alpha_off": 2,
}


# =========================
# 5. MemTorch 변환
# =========================
def build_memristor_model(cnn):
    mem_model = patch_model(
        copy.deepcopy(cnn),

        memristor_model=memtorch.bh.memristor.VTEAM,
        memristor_model_params=taox_params,

        module_parameters_to_patch=[nn.Conv2d, nn.Linear],

        mapping_routine=naive_map,
        scheme=memtorch.bh.Scheme.DoubleColumn,
        transistor=True,

        tile_shape=(128, 128),

        max_input_voltage=0.2,
        scaling_routine=naive_scale,
    )

    return mem_model


# =========================
# 6. Evaluation
# =========================
def evaluate(model, loader, device, criterion=None, model_name="Model", plot_cm=False):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_batches = len(loader)

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            pred = torch.argmax(out, dim=1)

            total_loss += loss.item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # 👉 진행률 출력 (한 줄 갱신)
            print(f"\rProgress: {i+1}/{total_batches} ({(i+1)/total_batches*100:.1f}%)", end="")

    print()  # 줄 정리

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n[{model_name}]")
    print(f"Test Loss      : {avg_loss:.4f}")
    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print(f"F1 Macro       : {f1_macro:.4f}")
    print(f"F1 Weighted    : {f1_weighted:.4f}")

    print("\nClassification Report")
    print(classification_report(all_labels, all_preds))

    if plot_cm:
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm)
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.colorbar()

        num_classes = cm.shape[0]
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center")

        plt.tight_layout()
        plt.show()

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
# =========================
# 7. Main
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_dataloaders(
        "./archive/Wafer_Map_Datasets.npz"
    )

    criterion = nn.CrossEntropyLoss()

    cnn = WaferCNN().to(device)
    train_cnn(cnn, train_loader, device)

    ideal_result = evaluate(
        cnn,
        test_loader,
        device,
        criterion=criterion,
        model_name="Ideal CNN",
        plot_cm=False
    )

    mem_cnn = build_memristor_model(cnn).to(device)

    mem_result = evaluate(
        mem_cnn,
        test_loader,
        device,
        criterion=criterion,
        model_name="TaOx Memristor CNN",
        plot_cm=True   # 맨 마지막에 confusion map 출력
    )