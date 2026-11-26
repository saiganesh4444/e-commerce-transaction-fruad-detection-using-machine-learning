# 🛒 E-Commerce Transaction Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://e-commerce-transaction-fruad-detection-using-machine-learning.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ANN%20%2B%20DNN%20%2B%20LDA-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SMOTE](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-22C55E)](https://imbalanced-learn.org/)

A hybrid machine learning fraud analytics platform combining **SMOTE** class balancing, **Linear Discriminant Analysis (LDA)** feature projection, and an **ANN + DNN Neural Ensemble** with an interactive **Streamlit + Plotly** dark-mode dashboard.

🌐 **Live Working URL:** [https://e-commerce-transaction-fruad-detection-using-machine-learning.streamlit.app/](https://e-commerce-transaction-fruad-detection-using-machine-learning.streamlit.app/)

---

## ✨ Key Features

- **Hybrid Ensemble Pipeline**: Combines a Shallow ANN `(128 → 64)` and a Deep DNN `(256 → 128 → 64)` trained on SMOTE-balanced, LDA-projected transaction features.
- **Executive Analytics Dashboard**: Real-time KPI cards, hourly fraud rate trends, device & location risk breakdowns, and Category × Payment Method heatmaps.
- **Interactive Real-Time Risk Scoring**: Live transaction predictor with quick scenario presets, Plotly risk gauge, individual ANN/DNN scores, and explainable risk signals.
- **In-App Training & Evaluation**: Re-train the full pipeline directly from the UI and inspect live Confusion Matrix & ROC Curve artifacts.
- **Transaction Monitor**: Filterable, color-coded transaction log with CSV export.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Raw Transactions] --> B[StandardScaler + OneHotEncoder]
    B --> C[SMOTE Oversampling]
    C --> D[LDA Feature Projection]
    D --> E[Shallow ANN 128→64]
    D --> F[Deep DNN 256→128→64]
    E --> G[Ensemble Probability Fusion]
    F --> G
    G --> H{Risk ≥ Threshold?}
    H -->|Yes| I[🚨 Fraud Alert]
    H -->|No| J[✅ Legitimate]
```

---

## 🚀 Quick Start

### Option 1: Access Live Application
Directly access the hosted application without installing anything locally:
👉 **[https://e-commerce-transaction-fruad-detection-using-machine-learning.streamlit.app/](https://e-commerce-transaction-fruad-detection-using-machine-learning.streamlit.app/)**

### Option 2: Run Locally
```bash
# 1. Clone repository
git clone https://github.com/saiganesh4444/e-commerce-transaction-fruad-detection-using-machine-learning.git
cd e-commerce-transaction-fruad-detection-using-machine-learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train models on transaction.csv or built-in synthetic data
python train_model.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```text
├── app.py                     # Professional Streamlit + Plotly dashboard
├── train_model.py             # Hybrid ANN + DNN + SMOTE + LDA training script
├── evaluate_model.py          # Model evaluation & metric plots
├── generate_predictions.py    # Batch prediction generator
├── predict_ensemble.py        # Ensemble batch scoring
├── fix_predictions.py         # Prediction label fixer utility
├── merge_labels.py            # Label merge utility
├── top_fraud.py               # Top high-risk fraud cases exporter
├── utils/
│   ├── __init__.py
│   └── data_generator.py      # Synthetic e-commerce transaction generator
├── model_artifacts/           # Saved preprocessor, LDA, ANN/DNN models & plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── fraud_ann_model.joblib
│   ├── fraud_dnn_model.joblib
│   ├── lda.joblib
│   ├── preprocessor.joblib
│   └── metrics.json
├── .streamlit/
│   └── config.toml            # Dark fintech theme configuration
└── requirements.txt           # Python dependencies
```

---

## 📊 Dataset

The original large dataset is hosted on Google Drive:
[Download Dataset](https://drive.google.com/file/d/1UTq2urzXKbjdskdpjVoDqMnfsfIBQ-Vk/view?usp=sharing)

If `transaction.csv` is not present locally, the application automatically generates a realistic synthetic e-commerce dataset so all analytics, training, and real-time prediction features work out-of-the-box.
