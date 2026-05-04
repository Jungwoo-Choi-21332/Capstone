"""
공통 모듈: CNN 모델 + 데이터셋 + 손실 함수
step2, step3에서 공통으로 사용
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# 결함 유형 8개
CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-Full', 'Scratch', 'Random']


class WaferCNN(nn.Module):
    """3층 CNN — 64x64 웨이퍼 맵 멀티라벨 분류"""
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2: 32x32 → 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3: 16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))


class WaferDataset(Dataset):
    """웨이퍼 맵 데이터셋 (증강 지원)"""
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.transform = X, y, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class FocalLoss(nn.Module):
    """BCE Focal Loss — 어려운 샘플(Edge-Loc 등)에 자동 가중"""
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
