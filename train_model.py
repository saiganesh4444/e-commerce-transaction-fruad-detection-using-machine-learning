# ================================
# train_model.py
# HYBRID FRAUD MODEL TRAINING
# ANN + DNN + SMOTE + LDA
# ================================

import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.data_generator import (
    FEATURE_CAT_COLS,
    FEATURE_NUM_COLS,
    load_or_create_dataset,
)

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "transaction.csv"
OUT_DIR = "model_artifacts"
MAX_ROWS = 200_000


def train_pipeline(csv_path: str = CSV_PATH, out_dir: str = OUT_DIR, max_rows: int = MAX_ROWS) -> dict:
    """Train the hybrid ANN + DNN + SMOTE + LDA fraud detection pipeline."""
    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset...")
    df = load_or_create_dataset(csv_path, n_rows=10000)

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"])

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    numeric_tf = Pipeline([("scaler", StandardScaler())])
    categorical_tf = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=20))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_tf, FEATURE_NUM_COLS),
        ("cat", categorical_tf, FEATURE_CAT_COLS),
    ])

    print("Preprocessing features...")
    X_sparse = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, f"{out_dir}/preprocessor.joblib")

    X_dense = X_sparse.toarray() if hasattr(X_sparse, "toarray") else X_sparse

    # -----------------------------
    # TRAIN / TEST SPLIT (BEFORE SMOTE TO PREVENT LEAKAGE)
    # -----------------------------
    X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(
        X_dense,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # -----------------------------
    # SMOTE
    # -----------------------------
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_raw, y_train_raw)

    # -----------------------------
    # LDA
    # -----------------------------
    print("Applying LDA...")
    lda = LinearDiscriminantAnalysis()
    X_train = lda.fit_transform(X_res, y_res)
    X_val = lda.transform(X_val_raw)
    joblib.dump(lda, f"{out_dir}/lda.joblib")

    print("Shape after LDA:", X_train.shape)

    # -----------------------------
    # ANN MODEL (Shallow Neural Network: 128 -> 64 -> 1)
    # -----------------------------
    print("Training ANN...")
    ann = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=256,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
    )
    ann.fit(X_train, y_res)
    joblib.dump(ann, f"{out_dir}/fraud_ann_model.joblib")

    # -----------------------------
    # DNN MODEL (Deep Neural Network: 256 -> 128 -> 64 -> 1)
    # -----------------------------
    print("Training DNN...")
    dnn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=2e-3,
        batch_size=256,
        max_iter=80,
        early_stopping=True,
        n_iter_no_change=6,
        random_state=42,
    )
    dnn.fit(X_train, y_res)
    joblib.dump(dnn, f"{out_dir}/fraud_dnn_model.joblib")

    print("TRAINING COMPLETE — ANN + DNN + SMOTE + LDA")

    # ================================
    # ENSEMBLE EVALUATION
    # ================================
    print("Generating Evaluation Metrics...")
    ann_prob = ann.predict_proba(X_val)[:, 1]
    dnn_prob = dnn.predict_proba(X_val)[:, 1]
    y_pred_prob = (ann_prob + dnn_prob) / 2.0
    y_pred = (y_pred_prob >= 0.50).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_val, y_pred_prob)), 4),
        "train_samples_after_smote": int(len(X_train)),
        "val_samples": int(len(X_val)),
    }

    with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # ================================
    # CONFUSION MATRIX
    # ================================
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "#111827"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix (Ensemble)", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Predicted", color="#9ca3af")
    ax.set_ylabel("Actual", color="#9ca3af")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"], color="#e5e7eb")
    ax.set_yticklabels(["Legit", "Fraud"], color="#e5e7eb")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=120, facecolor="#0e1117")
    plt.close()

    # ================================
    # ROC CURVE
    # ================================
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="#0e1117")
    ax.set_facecolor("#111827")
    ax.plot(fpr, tpr, color="#38bdf8", lw=2.5, label=f"Ensemble AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", lw=1.2)
    ax.set_xlabel("False Positive Rate", color="#9ca3af")
    ax.set_ylabel("True Positive Rate", color="#9ca3af")
    ax.set_title("ROC Curve", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="#d1d5db")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png", dpi=120, facecolor="#0e1117")
    plt.close()

    # Also save default scored predictions if not present
    X_all_proc = preprocessor.transform(X)
    if hasattr(X_all_proc, "toarray"):
        X_all_proc = X_all_proc.toarray()
    X_all = lda.transform(X_all_proc)
    all_prob = (ann.predict_proba(X_all)[:, 1] + dnn.predict_proba(X_all)[:, 1]) / 2.0
    scored = df.copy()
    scored["fraud_probability"] = np.round(all_prob, 4)
    scored["fraud_prediction"] = (all_prob >= 0.50).astype(int)
    scored.to_csv("fraud_predictions_final_v2.csv", index=False)

    print("Metrics:", metrics)
    print("ROC Curve & Confusion Matrix saved inside model_artifacts/")
    return metrics


if __name__ == "__main__":
    train_pipeline()
