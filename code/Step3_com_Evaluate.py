import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import memtorch
import numpy as np

# 1. 경로 자동 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "processed_mixed_wafer_data.pth")
model_weights_path = os.path.join(current_dir, "memristor_wafer_model.pth")

# 2. 데이터 및 모델 로드
print("📦 검증용 데이터 로딩 중...")
try:
    data = torch.load(data_path, weights_only=False)
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes'] # ['Normal', 'Single-type', 'Mixed-type']
    print(f"✅ 데이터 로드 성공! (Test: {len(X_test)}개)")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
    exit()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 모델 구조 정의 (Step 2와 동일)
class WaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(WaferCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.fc = nn.Linear(16 * 15 * 15, num_classes)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = WaferCNN(len(classes))

# 4. MemTorch 매핑 및 가중치 로드
print("⚡ 멤리스터 모델 구성 및 가중치 입히는 중...")
reference_memristor = memtorch.bh.memristor.VTEAM
memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params={},
                                        tile_shape=(128, 128))

try:
    memristive_model.load_state_dict(torch.load(model_weights_path, weights_only=False))
    memristive_model.eval() 
    print(f"✅ 학습된 모델 로드 완료!")
except FileNotFoundError:
    print(f"❌ 모델 가중치 파일을 찾을 수 없습니다: {model_weights_path}")
    exit()

# 5. 예측 수행
print("🧐 멤리스터 모델 성능 테스트 시작 (Normal vs Single vs Mixed)...")
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = memristive_model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 6. 결과 분석 및 시각화
print("\n📝 [최종 분류 보고서]")

# 클래스 이름을 문자열로 변환 (TypeError 방지)
target_names = [str(c) for c in classes]

# f1-score, accuracy 등이 포함된 보고서 출력
print(classification_report(all_labels, all_preds, target_names=target_names))

# --- 혼동 행렬(Confusion Matrix) 시각화 ---
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds)

# 히트맵 그리기
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')

plt.title('Wafer Defect Classification (Memristor-based PIM)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 결과 그래프 저장
result_img_path = os.path.join(current_dir, "final_evaluation_result.png")
plt.savefig(result_img_path)
print(f"\n📊 결과 그래프가 저장되었습니다: {result_img_path}")
plt.show()