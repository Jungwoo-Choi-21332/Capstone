import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


# ---------------------------
# 1. Path setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "input")

file_path = os.path.join(DATA_DIR, "uci-secom.csv")


# ---------------------------
# 2. Load data
# ---------------------------
df = pd.read_csv(file_path)

# Drop non-numeric columns
df = df.select_dtypes(include=[np.number])

# Split
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

y = np.where(y == -1, 0, 1)

print("Original shape:", X.shape)


# ---------------------------
# 3. Missing value imputation
# ---------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# ---------------------------
# 4. Remove low-variance features
# ---------------------------
selector = VarianceThreshold(threshold=1e-5)
X = selector.fit_transform(X)

print("After variance threshold:", X.shape)


# ---------------------------
# 5. Feature scaling
# ---------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ---------------------------
# 6. Dimensionality reduction (PCA)
# ---------------------------
pca = PCA(n_components=50)
X = pca.fit_transform(X)

print("After PCA:", X.shape)


# ---------------------------
# 7. Handle class imbalance (SMOTE)
# ---------------------------
smote = SMOTE()
X, y = smote.fit_resample(X, y)

print("After SMOTE:", X.shape)


# ---------------------------
# 8. Convert to sequences
# ---------------------------
def create_sequences(X, y, seq_len=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


seq_len = 10
X_seq, y_seq = create_sequences(X, y, seq_len)

print("Final sequence shape:", X_seq.shape)
print("Final label shape:", y_seq.shape)


# ---------------------------
# Optional: function wrapper
# ---------------------------
def load_preprocessed_data(seq_len=10):
    return X_seq, y_seq