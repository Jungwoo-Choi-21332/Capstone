import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
# 사진을 보니 archive 폴더가 한 단계 위에 있으므로 경로를 맞춰줍니다.
lswmd_path = os.path.join(current_dir,"LSWMD.pkl")
mixed_path = os.path.join(current_dir,"Wafer_Map_Datasets.npz")

# 2. 전처리 함수 정의
def preprocess_wafer_maps(wafer_list):
    processed = []
    for m in wafer_list:
        # 사이즈 통일 (32, 32)
        res = resize(m, (32, 32), preserve_range=True, order=0)
        processed.append(res / 2.0) # 정규화
    return np.array(processed)

# --- (A) LSWMD.pkl에서 정상(None) 데이터 가져오기 ---
print("📦 LSWMD.pkl에서 정상 데이터 추출 중...")
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

with open(lswmd_path, 'rb') as f:
    df_lswmd = pickle.load(f, encoding='latin1')

# 'none' 데이터만 10,000개 샘플링
df_none = df_lswmd[df_lswmd['failureType'].apply(lambda x: len(x)>0 and x[0][0]=='none')].sample(10000, random_state=42)
none_images = preprocess_wafer_maps(df_none['waferMap'].values)
none_labels = np.zeros(len(none_images)) # 정상은 라벨 0

# --- (B) npz 파일에서 불량 데이터 가져오기 ---
print("📦 npz 파일에서 불량 데이터 추출 중...")
npz_data = np.load(mixed_path, allow_pickle=True)
images_all = npz_data['arr_0']
labels_all = npz_data['arr_1']

# Single-type(0)과 Mixed-type(1)을 각각 10,000개씩 추출 (균형 유지)
idx_single = np.where(labels_all == 0)[0][:10000]
idx_mixed = np.where(labels_all == 1)[0][:10000]

single_images = preprocess_wafer_maps(images_all[idx_single])
single_labels = np.ones(len(single_images)) # 단일 불량은 라벨 1

mixed_images = preprocess_wafer_maps(images_all[idx_mixed])
mixed_labels = np.full(len(mixed_images), 2) # 복합 불량은 라벨 2

# --- (C) 데이터 통합 ---
print("🔄 모든 데이터 통합 중...")
X_final = np.concatenate([none_images, single_images, mixed_images], axis=0)
y_final = np.concatenate([none_labels, single_labels, mixed_labels], axis=0)

# 텐서 변환 (Batch, Channel, H, W)
X_tensor = torch.FloatTensor(X_final).unsqueeze(1)
y_tensor = torch.LongTensor(y_final)

# 3. 학습/검증 분리 및 저장
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)

classes = np.array(['Normal', 'Single-type', 'Mixed-type'])

data_to_save = {
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': classes
}

save_path = os.path.join(current_dir, "processed_mixed_wafer_data.pth")
torch.save(data_to_save, save_path)

print(f"\n✅ 전처리 및 병합 완료!")
print(f"📊 최종 데이터 구성: Normal(0) {len(none_labels)}개, Single(1) {len(single_labels)}개, Mixed(2) {len(mixed_labels)}개")
print(f"📍 저장 위치: {save_path}")