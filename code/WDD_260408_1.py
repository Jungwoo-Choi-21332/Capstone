import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import memtorch
import numpy as np

# 1. 데이터 로드 (X_test, y_test 정의)
print("📦 검증용 데이터 로딩 중...")
try:
    data = torch.load("processed_mixed_wafer_data.pth", weights_only=False)
    X_test = data['X_test']
    y_test = data['y_test']
    classes = data['classes']
    print(f" 데이터 로드 완료! (Test samples: {len(X_test)})")
except FileNotFoundError:
    print(" 'multi_wafer_data.pth' 파일을 찾을 수 없습니다. Step1을 먼저 실행하세요.")
    exit()

# 2. 모델 구조 정의 (Step 2와 100% 일치해야 함)
class MultiLabelWaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelWaferCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.fc = nn.Linear(32 * 12 * 12, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 3. MemTorch 매핑 및 가중치 로드
print("⚙️ 멤리스터 모델 구성 및 가중치 로드 중...")
model = MultiLabelWaferCNN(len(classes))
memristive_model = memtorch.patch_model(model,
                                        memristor_model=memtorch.bh.memristor.VTEAM,
                                        memristor_model_params={},
                                        tile_shape=(128, 128),
                                        adc_bitwidth=8,
                                        dac_bitwidth=8)

try:
    memristive_model.load_state_dict(torch.load("memristor_wafer_model.pth", weights_only=False))
    memristive_model.eval()
    print("✅ 모델 가중치 로드 성공!")
except FileNotFoundError:
    print(" 'multi_memristor_model.pth'가 없습니다. Step2 학습을 먼저 완료하세요.")
    exit()

# 4. 예측 수행
all_preds = []
all_labels = []
threshold = 0.5

print("🧐 성능 평가 시작...")
with torch.no_grad():
    for inputs, labels in DataLoader(TensorDataset(X_test, y_test), batch_size=32):
        outputs = memristive_model(inputs)
        predicted = (torch.sigmoid(outputs) > threshold).int()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 5. 결과 출력
print("\n📊 [Mixed-type 결함 정밀 분석 보고서]")
print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

# 6. 스크래치(Scratch) 집중 분석 시각화
scratch_idx = list(classes).index('Scratch')
scratch_cm = multilabel_confusion_matrix(all_labels, all_preds)[scratch_idx]

plt.figure(figsize=(5, 4))
sns.heatmap(scratch_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Pred: No', 'Pred: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title('Scratch Detection Confusion Matrix')
plt.show()