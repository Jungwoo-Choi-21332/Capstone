from data_analysis import load_preprocessed_data
from ESN_model import ESN

from sklearn.metrics import accuracy_score, f1_score


# ---------------------------
# 1. Load data
# ---------------------------
X_seq, y_seq = load_preprocessed_data(seq_len=10)

print("Data loaded:", X_seq.shape)


# ---------------------------
# 2. Train/Test split
# ---------------------------
split = int(0.8 * len(X_seq))

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


# ---------------------------
# 3. Initialize ESN
# ---------------------------
model = ESN(
    input_dim=X_seq.shape[2],
    reservoir_size=200,
    spectral_radius=0.9,
    leaking_rate=0.3,
    reg=1e-6
)


# ---------------------------
# 4. Train
# ---------------------------
model.fit(X_train, y_train)


# ---------------------------
# 5. Predict
# ---------------------------
y_pred = model.predict_class(X_test)


# ---------------------------
# 6. Evaluation
# ---------------------------
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", acc)
print("F1-score:", f1)