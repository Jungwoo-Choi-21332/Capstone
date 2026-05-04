import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import memtorch

# [중요] 현재 파일이 있는 폴더 경로 자동 인식
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "processed_mixed_wafer_data.pth")
model_save_path = os.path.join(current_dir, "memristor_wafer_model.pth")

# 1. 전처리된 데이터 불러오기
try:
    data = torch.load(data_path, weights_only=False)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    print(f"🚀 학습 데이터 로드 완료! (Train: {len(X_train)}개, Classes: {len(classes)}개)")
    print(f"📊 클래스 구성: {classes}")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {data_path}\nStep1을 먼저 실행했는지 확인해 주세요.")
    exit()

# 2. 모델 정의 (3클래스 분류를 위해 num_classes 사용)
class WaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(WaferCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2) # 32x32 -> 15x15
        self.fc = nn.Linear(16 * 15 * 15, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# num_classes는 데이터에서 가져온 classes의 길이(3)가 됩니다.
model = WaferCNN(len(classes))

# 3. MemTorch 매핑 (멤리스터 PIM 구조로 변환)
print("⚡ 멤리스터 크로스바 매핑 중...")
reference_memristor = memtorch.bh.memristor.VTEAM
memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params={},
                                        tile_shape=(128, 128), 
                                        adc_bitwidth=8, 
                                        dac_bitwidth=8)
print("✅ 멤리스터 모델 매핑 성공!")

# 4. 학습 설정
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 속도를 위해 배치 사이즈 64로 업

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(memristive_model.parameters(), lr=0.001)

# 5. 학습 루프
epochs = 10  # 3클래스라 조금 더 학습시키기 위해 10회로 설정
print(f"🏋️ 3-Class 학습 시작 (Normal vs Single vs Mixed)...")

memristive_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # 멤리스터 기반 추론 및 학습
        outputs = memristive_model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Acc: {accuracy:.2f}%")

# 6. 학습된 모델 저장 (절대 경로 사용)
torch.save(memristive_model.state_dict(), model_save_path)
print(f"\n✅ 학습 완료! 모델이 저장되었습니다: {model_save_path}")