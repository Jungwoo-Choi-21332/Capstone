"""
Step 2: Ex-Situ 학습 — 순수 CNN만 학습 (멤리스터 없이)
- 멤리스터 없이 빠르게 학습 → 가중치 저장
- Step 3에서 멤리스터 매핑 후 평가 (Ex-Situ 방식)
- Focal Loss (Edge-Loc 등 어려운 클래스 집중)
- 30 epoch (충분한 수렴)
"""
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import WaferCNN, WaferDataset, FocalLoss

np.random.seed(42)
torch.manual_seed(42)

# --- 데이터 로드 ---
data = torch.load("../data/wafer_data_64.pth", weights_only=False)
X_train, y_train, classes = data['X_train'], data['y_train'], data['classes']
print(f"학습 데이터: {len(X_train)}개, 클래스: {len(classes)}개")

# --- 학습 설정 ---
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
])

model = WaferCNN(num_classes=len(classes))
criterion = FocalLoss(gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=LR)
loader = DataLoader(WaferDataset(X_train, y_train, transform), batch_size=BATCH_SIZE, shuffle=True)

# --- 학습 ---
print(f"[Ex-Situ] 순수 CNN 학습 ({EPOCHS} epochs, Focal Loss)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "../software_model.pth")
print("저장 완료: ../software_model.pth")
