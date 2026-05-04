# -*- coding: utf-8 -*-
"""
Silicon Wafer Defect Classification with MemTorch
워크플로:
  1. CNN 학습 (표준 PyTorch)  — WM811K 웨이퍼 결함 9-class
  2. patch_model()  → 멤리스터 크로스바 어레이로 변환 (VTEAM 모델)
  3. tune_()        → 각 레이어 출력 스케일 보정 (필수 단계)
  4. 변환 전후 정확도 비교 및 그래프 출력
"""

import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import memtorch
from memtorch.bh.memristor.VTEAM import VTEAM
from memtorch.mn import patch_model

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEVICE       = torch.device("cpu")
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-3
DATA_DIR     = "./data/WM811k_Dataset"
RESULT_IMG   = "wafer_memtorch_result.png"
TRAIN_RATIO  = 0.8

torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# Dataset  (32×32 RGB, 9 classes)
# ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

print("Loading WM811K dataset...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
class_names  = full_dataset.classes
n_classes    = len(class_names)
print(f"  Classes ({n_classes}): {class_names}")
print(f"  Total images: {len(full_dataset)}")

# Train/test split
n_train = int(len(full_dataset) * TRAIN_RATIO)
n_test  = len(full_dataset) - n_train
train_base, test_dataset = random_split(
    full_dataset, [n_train, n_test],
    generator=torch.Generator().manual_seed(42)
)
print(f"  Train: {n_train}  |  Test: {n_test}")

# Apply augmentation to train split only
class AugmentedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):  return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # img is already a tensor from base transform; convert back to PIL for aug
        img_pil = transforms.functional.to_pil_image(
            (img * 0.5 + 0.5).clamp(0, 1)
        )
        return self.transform(img_pil), label

train_dataset = AugmentedSubset(train_base, train_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────
# Model  (CNN for 32×32 RGB → 9 classes)
# ──────────────────────────────────────────────
class WaferCNN(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 16×16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def evaluate(model, loader, label=""):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100.0 * correct / total
    if label:
        print(f"  {label}: {correct}/{total} = {acc:.2f}%")
    return acc

def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, 100.0 * correct / total

def per_class_accuracy(model, loader, n_classes=9):
    correct = [0] * n_classes
    total   = [0] * n_classes
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for c in range(n_classes):
                mask        = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum().item()
                total[c]   += mask.sum().item()
    return [100.0 * correct[c] / total[c] if total[c] > 0 else 0.0
            for c in range(n_classes)]

# ──────────────────────────────────────────────
# 1. Train
# ──────────────────────────────────────────────
print("\n" + "="*55)
print(" STEP 1 : Training CNN on WM811K")
print("="*55)

model     = WaferCNN(n_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses, train_accs, test_accs_history = [], [], []
best_acc  = 0.0
best_state = None

for ep in range(1, EPOCHS + 1):
    loss, acc = train_epoch(model, train_loader, optimizer, criterion, ep)
    scheduler.step()
    test_acc = evaluate(model, test_loader)
    train_losses.append(loss)
    train_accs.append(acc)
    test_accs_history.append(test_acc)
    if test_acc > best_acc:
        best_acc   = test_acc
        best_state = copy.deepcopy(model.state_dict())
    if ep % 5 == 0 or ep == 1:
        print(f"  Epoch {ep:2d}/{EPOCHS}  loss={loss:.4f}  "
              f"train={acc:.1f}%  test={test_acc:.1f}%  (best={best_acc:.1f}%)")

# Restore best weights
model.load_state_dict(best_state)
print(f"\nBest test accuracy: {best_acc:.2f}%")

print("\nEvaluating standard model...")
acc_before = evaluate(model, test_loader, "Standard CNN")

# ──────────────────────────────────────────────
# 2. Convert to memristive model
# ──────────────────────────────────────────────
print("\n" + "="*55)
print(" STEP 2 : Converting to Memristive Crossbar (VTEAM)")
print("="*55)

memristive_model = copy.deepcopy(model)

print("Patching model layers...")
t0 = time.time()
memristive_model = patch_model(
    memristive_model,
    memristor_model=VTEAM,
    memristor_model_params={},
    module_parameters_to_patch=[nn.Linear, nn.Conv2d],
    transistor=True,
    use_bindings=False,
    verbose=False,
)
print(f"  Patching done in {time.time()-t0:.1f}s")

# tune_(): 크로스바 출력(전도도 스케일) → 가중치 스케일 선형 보정
print("Tuning layers (output scale calibration)...")
t0 = time.time()
memristive_model.tune_()
print(f"  Tuning done in {time.time()-t0:.1f}s")

# ──────────────────────────────────────────────
# 3. Evaluate memristive model
# ──────────────────────────────────────────────
print("\nEvaluating memristive model...")
acc_after = evaluate(memristive_model, test_loader, "Memristive CNN (VTEAM + tuned)")

# ──────────────────────────────────────────────
# 4. Per-class accuracy
# ──────────────────────────────────────────────
print("\n" + "="*55)
print(" STEP 3 : Per-class accuracy comparison")
print("="*55)

class_acc_before = per_class_accuracy(model,             test_loader, n_classes)
class_acc_after  = per_class_accuracy(memristive_model,  test_loader, n_classes)

short = ["Center","Donut","EdgeL","EdgeR","Local","Scratch","nearFull","None","Random"]
print(f"\n{'Class':<12} {'Standard':>10} {'Memristive':>12} {'Diff':>8}")
print("-" * 46)
for i, name in enumerate(class_names):
    diff = class_acc_after[i] - class_acc_before[i]
    print(f"  {name:<10} {class_acc_before[i]:>9.1f}%  {class_acc_after[i]:>9.1f}%  {diff:>+7.1f}%")

# ──────────────────────────────────────────────
# 5. Weight / conductance distributions
# ──────────────────────────────────────────────
def get_weights(m):
    w = []
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w.append(mod.weight.data.cpu().numpy().flatten())
    return np.concatenate(w)

def get_conductances(m):
    g = []
    for mod in m.modules():
        if hasattr(mod, "crossbars") and len(mod.crossbars) >= 2:
            G_pos = mod.crossbars[0].conductance_matrix.cpu().detach().numpy().flatten()
            G_neg = mod.crossbars[1].conductance_matrix.cpu().detach().numpy().flatten()
            g.append(G_pos - G_neg)
    return np.concatenate(g) if g else np.array([0.0])

weights_before = get_weights(model)
conductances   = get_conductances(memristive_model)

# ──────────────────────────────────────────────
# 6. Confusion matrix
# ──────────────────────────────────────────────
def confusion_matrix(m, loader, n):
    cm = np.zeros((n, n), dtype=int)
    m.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            preds = m(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            for t, p in zip(labels.numpy(), preds):
                cm[t][p] += 1
    return cm

cm_before = confusion_matrix(model,             test_loader, n_classes)
cm_after  = confusion_matrix(memristive_model,  test_loader, n_classes)

# ──────────────────────────────────────────────
# 7. Plot results (2×4 grid)
# ──────────────────────────────────────────────
print("\n" + "="*55)
print(" STEP 4 : Generating plots")
print("="*55)

fig = plt.figure(figsize=(20, 12))
fig.suptitle("Silicon Wafer Defect Classification with MemTorch (VTEAM)",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.40)

short_labels = [c[:7] for c in class_names]  # shorten for tick labels

# ── 7-1. Training curves ─────────────────────
ax1 = fig.add_subplot(gs[0, 0])
epochs_x = range(1, EPOCHS + 1)
ax1.plot(epochs_x, train_losses, color="#2196F3", linewidth=1.8, label="Train Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="#2196F3")
ax1.tick_params(axis="y", labelcolor="#2196F3")
ax1b = ax1.twinx()
ax1b.plot(epochs_x, train_accs,       "--", color="#FF5722", linewidth=1.8, label="Train Acc")
ax1b.plot(epochs_x, test_accs_history, ":",  color="#4CAF50", linewidth=1.8, label="Test Acc")
ax1b.set_ylabel("Accuracy (%)", color="#555")
ax1.set_title("Training Curves", fontweight="bold")
lines = (ax1.get_legend_handles_labels()[0] +
         ax1b.get_legend_handles_labels()[0])
lbs   = (ax1.get_legend_handles_labels()[1] +
         ax1b.get_legend_handles_labels()[1])
ax1.legend(lines, lbs, fontsize=7, loc="center right")
ax1.grid(True, alpha=0.3)

# ── 7-2. Overall accuracy comparison ─────────
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(["Standard\nCNN", "Memristive\nCNN\n(VTEAM)"],
               [acc_before, acc_after],
               color=["#4CAF50", "#FF9800"], width=0.45,
               edgecolor="white", linewidth=1.5)
for bar, acc in zip(bars, [acc_before, acc_after]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax2.set_ylim(0, 115)
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Overall Accuracy\n(test set)", fontweight="bold")
ax2.grid(True, axis="y", alpha=0.3)
drop = acc_after - acc_before
ax2.annotate(f"Diff: {drop:+.2f}%",
             xy=(0.5, min(acc_before, acc_after) - 6),
             xycoords=("axes fraction", "data"), ha="center",
             fontsize=10, fontweight="bold",
             color="#E91E63" if drop < 0 else "#4CAF50")

# ── 7-3. Per-class accuracy comparison ───────
ax3 = fig.add_subplot(gs[0, 2])
x, w = np.arange(n_classes), 0.35
ax3.bar(x - w/2, class_acc_before, w, label="Standard",   color="#4CAF50", alpha=0.85)
ax3.bar(x + w/2, class_acc_after,  w, label="Memristive", color="#FF9800", alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax3.set_ylabel("Accuracy (%)"); ax3.set_ylim(0, 115)
ax3.set_title("Per-class Accuracy", fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(True, axis="y", alpha=0.3)

# ── 7-4. Accuracy change per class ───────────
ax4 = fig.add_subplot(gs[0, 3])
diffs = np.array(class_acc_after) - np.array(class_acc_before)
colors_d = ["#4CAF50" if d >= 0 else "#F44336" for d in diffs]
ax4.bar(range(n_classes), diffs, color=colors_d, edgecolor="white", linewidth=1.2)
ax4.axhline(0, color="black", linewidth=1)
ax4.set_xticks(range(n_classes))
ax4.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax4.set_ylabel("Acc change (%)"); ax4.set_title("Accuracy Change per Class\n(Mem − Standard)", fontweight="bold")
ax4.grid(True, axis="y", alpha=0.3)

# ── 7-5. Weight distribution ─────────────────
ax5 = fig.add_subplot(gs[1, 0])
ax5.hist(weights_before, bins=80, color="#2196F3", alpha=0.8, edgecolor="none")
ax5.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax5.set_xlabel("Weight"); ax5.set_ylabel("Count")
ax5.set_title("Weight Distribution\n(Standard CNN)", fontweight="bold")
ax5.grid(True, alpha=0.3)

# ── 7-6. Conductance distribution ────────────
ax6 = fig.add_subplot(gs[1, 1])
ax6.hist(conductances, bins=80, color="#FF9800", alpha=0.8, edgecolor="none")
ax6.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax6.set_xlabel("G⁺ − G⁻ (S)"); ax6.set_ylabel("Count")
ax6.set_title("Conductance Distribution\n(Memristive CNN)", fontweight="bold")
ax6.grid(True, alpha=0.3)

# ── 7-7. Confusion matrix — standard ─────────
ax7 = fig.add_subplot(gs[1, 2])
im7 = ax7.imshow(cm_before, cmap="Blues", aspect="auto")
ax7.set_xticks(range(n_classes)); ax7.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
ax7.set_yticks(range(n_classes)); ax7.set_yticklabels(short_labels, fontsize=7)
ax7.set_title("Confusion Matrix\n(Standard CNN)", fontweight="bold")
ax7.set_xlabel("Predicted"); ax7.set_ylabel("True")
for i in range(n_classes):
    for j in range(n_classes):
        val = cm_before[i, j]
        if val > 0:
            ax7.text(j, i, str(val), ha="center", va="center",
                     fontsize=7, color="white" if val > cm_before.max()*0.5 else "black")
plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

# ── 7-8. Confusion matrix — memristive ───────
ax8 = fig.add_subplot(gs[1, 3])
im8 = ax8.imshow(cm_after, cmap="Oranges", aspect="auto")
ax8.set_xticks(range(n_classes)); ax8.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
ax8.set_yticks(range(n_classes)); ax8.set_yticklabels(short_labels, fontsize=7)
ax8.set_title("Confusion Matrix\n(Memristive CNN)", fontweight="bold")
ax8.set_xlabel("Predicted"); ax8.set_ylabel("True")
for i in range(n_classes):
    for j in range(n_classes):
        val = cm_after[i, j]
        if val > 0:
            ax8.text(j, i, str(val), ha="center", va="center",
                     fontsize=7, color="white" if val > cm_after.max()*0.5 else "black")
plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

plt.savefig(RESULT_IMG, dpi=150, bbox_inches="tight")
print(f"  Plot saved -> {RESULT_IMG}")

# ──────────────────────────────────────────────
# 8. Summary
# ──────────────────────────────────────────────
print("\n" + "="*55)
print(" SUMMARY")
print("="*55)
print(f"  memtorch version   : {memtorch.__version__}")
print(f"  torch version      : {torch.__version__}")
print(f"  Dataset            : WM811K Silicon Wafer ({len(full_dataset)} images, {n_classes} classes)")
print(f"  Train / Test split : {n_train} / {n_test}")
print(f"  Training epochs    : {EPOCHS}")
print(f"  Standard CNN acc   : {acc_before:.2f}%")
print(f"  Memristive CNN acc : {acc_after:.2f}%")
print(f"  Accuracy diff      : {acc_after - acc_before:+.2f}%")
print(f"  Result image       : {RESULT_IMG}")
print("="*55)
