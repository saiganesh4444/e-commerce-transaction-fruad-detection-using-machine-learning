import joblib
import numpy as np
import pandas as pd
from utils.data_generator import load_or_create_dataset

THRESHOLD = 0.50

print("Loading models...")
ann = joblib.load("model_artifacts/fraud_ann_model.joblib")
dnn = joblib.load("model_artifacts/fraud_dnn_model.joblib")
lda = joblib.load("model_artifacts/lda.joblib")
preprocessor = joblib.load("model_artifacts/preprocessor.joblib")
print("Models loaded")

df = load_or_create_dataset("transaction.csv")
X = df.drop(columns=["is_fraud"])

X_proc = preprocessor.transform(X)
try:
    X_proc = X_proc.toarray()
except Exception:
    pass
X_lda = lda.transform(X_proc)
print("Data preprocessed")

ann_proba = ann.predict_proba(X_lda)[:, 1]
dnn_proba = dnn.predict_proba(X_lda)[:, 1]

final_proba = (ann_proba + dnn_proba) / 2.0
final_pred = (final_proba >= THRESHOLD).astype(int)

df["fraud_probability"] = np.round(final_proba, 4)
df["fraud_prediction"] = final_pred

OUTFILE = "fraud_predictions_final_v2.csv"
df.to_csv(OUTFILE, index=False)
print("Ensemble predictions saved ->", OUTFILE)
