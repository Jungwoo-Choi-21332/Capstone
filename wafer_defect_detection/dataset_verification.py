import numpy as np
from collections import Counter

npz_path = "Wafer_Map_Datasets.npz"

data = np.load(npz_path, allow_pickle=True)

print("NPZ keys:", data.files)

# X, y 자동 탐색
if "arr_0" in data.files and "arr_1" in data.files:
    X = data["arr_0"]
    y = data["arr_1"]
elif "x" in data.files and "y" in data.files:
    X = data["x"]
    y = data["y"]
elif "X" in data.files and "Y" in data.files:
    X = data["X"]
    y = data["Y"]
else:
    raise KeyError(f"X/y key를 찾을 수 없습니다. keys={data.files}")

print("X shape:", X.shape)
print("y shape:", y.shape)

# one-hot → index
if y.ndim > 1:
    y = np.argmax(y, axis=1)

y = np.asarray(y).astype(int)

# 기존 wafer defect class
base_classes = [
    "Normal",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full"
]

counts = Counter(y)
total = len(y)

# Multi 포함 집계용
aggregated_counts = {}

print("\n===== Raw Class Distribution =====")
for class_idx in sorted(counts.keys()):
    if class_idx < len(base_classes):
        name = base_classes[class_idx]
    else:
        name = "Multi"

    count = counts[class_idx]
    ratio = count / total * 100

    print(f"{class_idx:2d} | {name:10s} | {count:6d} | {ratio:6.2f}%")

    # Multi 포함 aggregate
    if name not in aggregated_counts:
        aggregated_counts[name] = 0
    aggregated_counts[name] += count


print("\n===== Aggregated Distribution (Multi 포함) =====")
for name, count in aggregated_counts.items():
    ratio = count / total * 100
    print(f"{name:10s} | {count:6d} | {ratio:6.2f}%")

print("\nTotal samples:", total)