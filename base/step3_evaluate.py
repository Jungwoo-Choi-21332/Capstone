"""
Step 3: Ex-Situ 평가 — 학습된 가중치 → 멤리스터 매핑 → 평가
- 가중치 먼저 로드 → patch_model 적용 (올바른 Ex-Situ 순서)
- ADC_resolution (대문자, 올바른 memtorch API)
- StochasticParameter (올바른 소자 변동성)
- 소자별 비교: selected_material 변경으로 TiO2/HfO2/TaO2 비교

사용법:
  1. step1, step2를 먼저 실행
  2. 아래 설정 섹션에서 소자/파라미터 변경
  3. step3 실행
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import memtorch
from memtorch.bh.StochasticParameter import StochasticParameter
import numpy as np
import json
import os

from model import WaferCNN

# ================================================================
#  설정 — 여기만 수정하세요
# ================================================================
selected_material = "TaO2"     # "TiO2", "HfO2", "TaO2"
TILE_SHAPE = (128, 128)        # 크로스바 타일 크기
ADC_RES = 8                    # ADC 해상도 (None = 양자화 없음)
NUM_EVAL = 5                   # 평가 반복 횟수 (변동성 반영)
# ================================================================

# --- 소자별 물리 파라미터 ---
MATERIALS = {
    "TiO2": {  # 높은 변동성
        'r_on': StochasticParameter(loc=100, scale=10.0),    # 10%
        'r_off': StochasticParameter(loc=5000, scale=750.0), # 15%
    },
    "HfO2": {  # 중간 변동성
        'r_on': StochasticParameter(loc=100, scale=5.0),     # 5%
        'r_off': StochasticParameter(loc=5000, scale=500.0), # 10%
    },
    "TaO2": {  # 낮은 변동성 (최적화된 소자)
        'r_on': StochasticParameter(loc=100, scale=2.0),     # 2%
        'r_off': StochasticParameter(loc=5000, scale=200.0), # 4%
    },
}

# --- 데이터 로드 ---
data = torch.load("../data/wafer_data_64.pth", weights_only=False)
X_test, y_test, classes = data['X_test'], data['y_test'], data['classes']

os.makedirs("../results", exist_ok=True)
print(f"테스트: {len(X_test)}개 | 소자: {selected_material}")
print(f"tile: {TILE_SHAPE} | ADC: {ADC_RES}")

# --- Ex-Situ 평가 (N회 반복) ---
results = []

for run in range(NUM_EVAL):
    # 매번 새 모델 → 가중치 로드 → patch_model (변동성이 매번 다르게 적용)
    model = WaferCNN(num_classes=len(classes))
    model.load_state_dict(torch.load("../software_model.pth", weights_only=True))

    patch_args = {
        'memristor_model': memtorch.bh.memristor.VTEAM,
        'memristor_model_params': MATERIALS[selected_material],
        'tile_shape': TILE_SHAPE,
    }
    if ADC_RES is not None:
        patch_args['ADC_resolution'] = ADC_RES
        patch_args['quant_method'] = 'linear'

    patched = memtorch.patch_model(model, **patch_args)
    patched.eval()

    preds_list, labels_list = [], []
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(X_test, y_test), batch_size=32):
            pred = (torch.sigmoid(patched(x)) > 0.5).int()
            preds_list.extend(pred.cpu().numpy())
            labels_list.extend(y.cpu().numpy())

    preds, lbls = np.array(preds_list), np.array(labels_list)
    rpt = classification_report(lbls, preds, target_names=classes, zero_division=0, output_dict=True)
    r = {k: rpt['macro avg'][k] for k in ['f1-score', 'precision', 'recall']}
    results.append(r)
    print(f"  Run {run+1}/{NUM_EVAL}: F1={r['f1-score']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

# --- 결과 요약 ---
f1s = [r['f1-score'] for r in results]
print(f"\n{'='*60}")
print(f"[{selected_material}] 결과 ({NUM_EVAL}회)")
print(f"{'='*60}")
print(f"  F1:        {np.mean(f1s):.4f} (±{np.std(f1s):.4f})")
print(f"  Precision: {np.mean([r['precision'] for r in results]):.4f}")
print(f"  Recall:    {np.mean([r['recall'] for r in results]):.4f}")
print(f"  Best:      {max(f1s):.4f} / Worst: {min(f1s):.4f}")

# --- 상세 결과 ---
mcm = multilabel_confusion_matrix(lbls, preds)
print(f"\n{'Defect':<12} | {'Acc':>7} | {'Prec':>6} | {'Recall':>6}")
print("-" * 42)
for i, c in enumerate(classes):
    tn, fp, fn, tp = mcm[i].ravel()
    total = tn + fp + fn + tp
    print(f"{c:<12} | {(tp+tn)/total*100:>6.1f}% | {tp/(tp+fp) if tp+fp else 0:>5.2f} | {tp/(tp+fn) if tp+fn else 0:>5.2f}")

print(f"\n{classification_report(lbls, preds, target_names=classes, zero_division=0)}")

# --- 저장 ---
json.dump({
    'material': selected_material, 'tile': str(TILE_SHAPE), 'adc': ADC_RES,
    'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s)),
    'all_f1': [float(x) for x in f1s],
}, open(f"../results/eval_{selected_material}.json", 'w'), indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, t in enumerate(['Edge-Loc', 'Scratch']):
    idx = list(classes).index(t)
    sns.heatmap(mcm[idx], annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    axes[i].set_title(f'{t} ({selected_material})')
plt.tight_layout()
plt.savefig(f"../results/confusion_{selected_material}.png", dpi=150)
print("저장 완료!")
