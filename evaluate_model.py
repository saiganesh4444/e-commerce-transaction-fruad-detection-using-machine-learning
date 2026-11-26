import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

print("Loading predictions...")
df = pd.read_csv("fraud_predictions_final_v2.csv")

y_true = df["is_fraud"]
y_pred = df["fraud_prediction"]
y_prob = df["fraud_probability"]

print("\n===== MODEL PERFORMANCE (HYBRID ENSEMBLE) =====")
print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_true, y_pred, zero_division=0), 4))
print("F1 Score :", round(f1_score(y_true, y_pred, zero_division=0), 4))
print("AUC      :", round(roc_auc_score(y_true, y_prob), 4))

cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("model_artifacts/confusion_matrix.png", dpi=120)
plt.close()

fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")
plt.savefig("model_artifacts/roc_curve.png", dpi=120)
plt.close()

print("Evaluation completed")
