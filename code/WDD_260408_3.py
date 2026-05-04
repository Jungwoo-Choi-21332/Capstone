import numpy as np
import torch
from sklearn.model_selection import train_test_split

# 데이터 로드 (52x52 해상도) [cite: 14]
raw_data = np.load("../archive (1)/Wafer_Map_Datasets.npz")
X_raw = raw_data['arr_0']
y_raw = raw_data['arr_1'] # 8차원 원-핫 라벨 [cite: 15]

# 정규화 및 텐서 변환
X = torch.FloatTensor(X_raw.astype(np.float32) / 2.0).unsqueeze(1)
y = torch.FloatTensor(y_raw)

# 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PDF 기준 8대 결함 명칭 [cite: 20]
all_classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-Full', 'Scratch', 'Random']

torch.save({
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': np.array(all_classes)
}, "multi_wafer_data.pth")
print("✅ 모든 유형을 포함한 데이터 전처리 완료!")

