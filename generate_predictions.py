import joblib
import pandas as pd
from utils.data_generator import load_or_create_dataset

print("Loading model & preprocessor...")
ann = joblib.load("model_artifacts/fraud_ann_model.joblib")
lda = joblib.load("model_artifacts/lda.joblib")
preprocessor = joblib.load("model_artifacts/preprocessor.joblib")

print("Loading data...")
df = load_or_create_dataset("transaction.csv")

X = df.drop(columns=["is_fraud"])
X_proc_raw = preprocessor.transform(X)
if hasattr(X_proc_raw, "toarray"):
    X_proc_raw = X_proc_raw.toarray()
X_proc = lda.transform(X_proc_raw)

print("Predicting...")
ann_prob = ann.predict_proba(X_proc)[:, 1]

THRESHOLD = 0.50
y_pred = (ann_prob >= THRESHOLD).astype(int)

df["fraud_probability"] = ann_prob
df["fraud_prediction"] = y_pred

df.to_csv("fraud_predictions_final_v2.csv", index=False)
print("Predictions saved: fraud_predictions_final_v2.csv")
