import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import memtorch

data = torch.load("multi_wafer_data.pth", weights_only=False)
X_train, y_train, classes = data['X_train'], data['y_train'], data['classes']

class MultiLabelWaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelWaferCNN, self).__init__()
        # 52x52 입력 특징을 골고루 뽑기 위해 은닉층 2개 유지
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.fc = nn.Linear(32 * 12 * 12, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = MultiLabelWaferCNN(len(classes))
memristive_model = memtorch.patch_model(model,
                                        memristor_model=memtorch.bh.memristor.VTEAM,
                                        memristor_model_params={},
                                        tile_shape=(128, 128),
                                        adc_bitwidth=8,
                                        dac_bitwidth=8)

print("⚡ 멤리스터 모델 매핑 성공!")

# 모든 유형에 대해 평등하게 손실 계산 (균형 학습)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(memristive_model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
for epoch in range(15):
    memristive_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(memristive_model(inputs), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} 진행 중...")

torch.save(memristive_model.state_dict(), "multi_memristor_model.pth")