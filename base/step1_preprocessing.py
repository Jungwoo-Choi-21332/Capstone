"""
Step 1: 데이터 전처리
- 52x52 → 64x64 리사이징
- Train/Test 분리 후 학습 데이터만 밸런싱 (데이터 누수 방지)
- 재현 가능 (시드 고정)
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from model import CLASSES
import os

np.random.seed(42)
torch.manual_seed(42)

# --- 데이터 로드 ---
for path in ["../../archive/Wafer_Map_Datasets.npz",
             "../archive/Wafer_Map_Datasets.npz"]:
    if os.path.exists(path):
        raw = np.load(path)
        break
else:
    raise FileNotFoundError("Wafer_Map_Datasets.npz를 찾을 수 없습니다.")

X_raw, y_raw = raw['arr_0'], raw['arr_1']

# --- 64x64 리사이징 ---
X = F.interpolate(
    torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1) / 2.0,
    size=(64, 64), mode='bilinear', align_corners=False
)
y = torch.tensor(y_raw, dtype=torch.float32)
print(f"리사이징: {X_raw.shape[1]}x{X_raw.shape[2]} → 64x64 ({len(X)}개)")

# --- Train/Test 분리 먼저 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 학습 데이터만 밸런싱 (테스트는 원본 분포 유지) ---
is_normal = (y_train.sum(dim=1) == 0)
X_normal, y_normal = X_train[is_normal], y_train[is_normal]
X_defect, y_defect = X_train[~is_normal], y_train[~is_normal]

n_sample = min(len(X_normal), int(len(X_defect) * 1.5))
idx = np.random.choice(len(X_normal), n_sample, replace=False)

X_train = torch.cat([X_normal[idx], X_defect])
y_train = torch.cat([y_normal[idx], y_defect])

print(f"학습: {len(X_train)}개 (정상 {n_sample} + 결함 {len(X_defect)})")
print(f"테스트: {len(X_test)}개 (원본 분포)")

# --- 저장 ---
os.makedirs("../data", exist_ok=True)
torch.save({
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': np.array(CLASSES)
}, "../data/wafer_data_64.pth")
print("저장 완료: ../data/wafer_data_64.pth")
