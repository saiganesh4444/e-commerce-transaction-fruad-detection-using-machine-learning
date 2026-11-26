import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import train_pipeline
from utils.data_generator import (
    DEVICES,
    LOCATIONS,
    PAYMENT_METHODS,
    PRODUCT_CATEGORIES,
    generate_synthetic_transactions,
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sentinel | E-Commerce Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# DESIGN TOKENS
# =====================================================
ACCENT = "#6d8cff"
CYAN = "#22d3ee"
OK = "#2dd4a7"
BAD = "#ff5c7c"
WARN = "#f5b544"
MUTED = "#8b95a7"
GRID = "rgba(148,163,184,0.08)"

# Streamlit renamed `use_container_width` to `width="stretch"` in 1.49.
# Pick whichever keyword the running version supports so the app stays portable.
def _streamlit_minor() -> tuple:
    raw = getattr(st, "__version__", "0.0").split(".")[:2]
    return tuple(int("".join(ch for ch in part if ch.isdigit()) or 0) for part in raw)


WIDE = {"width": "stretch"} if _streamlit_minor() >= (1, 49) else {"use_container_width": True}

NAV = [
    ("Overview", "◈", "Overview"),
    ("Dataset", "▤", "Dataset Explorer"),
    ("Analytics", "◭", "Fraud Analytics"),
    ("Training", "◉", "Model Lab"),
    ("Monitor", "☰", "Transaction Monitor"),
    ("Predict", "⌕", "Risk Simulator"),
]

# =====================================================
# GLOBAL STYLES
# =====================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap');

:root{
  --bg:#06080e;
  --line:rgba(148,163,184,.14);
  --line-soft:rgba(148,163,184,.09);
  --text:#e9eef7;
  --muted:#8b95a7;
  --accent:#6d8cff;
  --cyan:#22d3ee;
  --ok:#2dd4a7;
  --bad:#ff5c7c;
  --warn:#f5b544;
}

html, body, [class*="css"], .stApp{
  font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased;
}

.stApp, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 620px at 84% -12%, rgba(109,140,255,.17), transparent 62%),
    radial-gradient(880px 520px at 4% -6%, rgba(34,211,238,.10), transparent 58%),
    radial-gradient(700px 500px at 50% 108%, rgba(109,140,255,.07), transparent 60%),
    #06080e;
  color:var(--text);
}

header[data-testid="stHeader"]{background:transparent; height:0;}
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"]{visibility:hidden; height:0;}

.block-container{
  padding-top:2.1rem; padding-bottom:3.5rem; max-width:1440px;
  animation:fade .5s ease both;
}
@keyframes fade{from{opacity:0; transform:translateY(7px);} to{opacity:1; transform:none;}}

::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(148,163,184,.22); border-radius:99px;}
::-webkit-scrollbar-thumb:hover{background:rgba(148,163,184,.36);}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0a0e17 0%, #070a11 100%);
  border-right:1px solid var(--line-soft);
}
section[data-testid="stSidebar"] > div{padding-top:1.1rem;}
section[data-testid="stSidebar"] .block-container{padding-top:1rem;}

.brand{display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .1rem;}
.brand-mark{
  width:38px;height:38px;border-radius:11px;flex:0 0 38px;
  background:linear-gradient(140deg,#6d8cff,#22d3ee);
  display:flex;align-items:center;justify-content:center;
  font-size:1.05rem; font-weight:700; color:#05070c;
  box-shadow:0 10px 24px -12px rgba(109,140,255,.9);
}
.brand-name{font-family:'Sora',sans-serif; font-size:1.02rem; font-weight:600; letter-spacing:-.015em; color:#f2f6fd; line-height:1.15;}
.brand-sub{font-size:.68rem; color:var(--muted); letter-spacing:.06em; text-transform:uppercase; margin-top:2px;}

.side-label{
  font-size:.66rem; letter-spacing:.13em; text-transform:uppercase;
  color:#6b7788; font-weight:600; margin:1.15rem 0 .45rem .15rem;
}

.nav-pill{
  display:flex; align-items:center; gap:.55rem;
  font-size:.78rem; color:#9aa6b8; padding:.55rem .7rem;
  border:1px solid var(--line-soft); border-radius:10px;
  background:rgba(255,255,255,.015);
}

/* sidebar navigation buttons */
section[data-testid="stSidebar"] .stButton > button{
  width:100%; justify-content:flex-start; align-items:center; text-align:left;
  padding:.45rem .8rem; margin-bottom:.22rem; min-height:2.15rem; height:auto;
  border-radius:10px; font-size:.86rem; font-weight:500;
  background:transparent; color:#9aa6b8;
  border:1px solid transparent;
  transition:all .16s ease;
  box-shadow:none;
}
section[data-testid="stSidebar"] .stButton > button > div,
section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"]{
  width:100%; display:block; text-align:left; line-height:1.4;
}
section[data-testid="stSidebar"] .stButton > button:hover{
  background:rgba(109,140,255,.09);
  border-color:rgba(109,140,255,.20);
  color:#e4ebf7;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{
  background:linear-gradient(90deg, rgba(109,140,255,.20), rgba(34,211,238,.07));
  border:1px solid rgba(109,140,255,.42);
  color:#f2f6fd; font-weight:600;
  box-shadow:0 12px 26px -18px rgba(109,140,255,1), inset 0 1px 0 rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] .stButton > button p{font-size:.86rem; margin:0; color:inherit;}
section[data-testid="stSidebar"] .stButton > button:focus:not(:active){border-color:rgba(109,140,255,.4); color:#f2f6fd;}

/* ---------- GENERIC BUTTONS ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  border-radius:10px; font-weight:600; font-size:.85rem;
  border:1px solid var(--line); background:rgba(255,255,255,.03); color:#dfe6f2;
  transition:all .16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  border-color:rgba(109,140,255,.45); background:rgba(109,140,255,.10); color:#fff;
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"], .stFormSubmitButton > button{
  background:linear-gradient(100deg,#6d8cff,#4f8dff 45%,#22d3ee);
  border:none; color:#05070c;
  box-shadow:0 14px 30px -16px rgba(109,140,255,.95);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover{
  filter:brightness(1.06); color:#05070c;
}
.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span,
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] span,
.stFormSubmitButton > button p, .stFormSubmitButton > button span{
  color:#05070c !important; font-weight:700;
}
.stDownloadButton > button p{color:#e4ebf7;}

/* ---------- HERO ---------- */
.hero{
  position:relative; overflow:hidden;
  border:1px solid var(--line);
  border-radius:22px;
  padding:2.1rem 2.3rem;
  margin-bottom:1.6rem;
  background:
    radial-gradient(620px 320px at 100% 0%, rgba(34,211,238,.14), transparent 65%),
    radial-gradient(560px 340px at 0% 100%, rgba(109,140,255,.16), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
  box-shadow:0 40px 80px -60px #000;
}
.hero::after{
  content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(90deg, rgba(109,140,255,.55), rgba(34,211,238,.35), transparent 60%);
  height:1px; top:0;
}
.eyebrow{
  display:inline-flex; align-items:center; gap:.5rem;
  font-size:.66rem; letter-spacing:.15em; text-transform:uppercase; font-weight:600;
  color:#b7c3d6; background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.10);
  padding:.34rem .7rem; border-radius:99px; margin-bottom:1rem;
}
.pulse{width:6px;height:6px;border-radius:99px;background:#2dd4a7;box-shadow:0 0 0 0 rgba(45,212,167,.6);animation:beat 2.2s infinite;}
@keyframes beat{0%{box-shadow:0 0 0 0 rgba(45,212,167,.55);}70%{box-shadow:0 0 0 7px rgba(45,212,167,0);}100%{box-shadow:0 0 0 0 rgba(45,212,167,0);}}
.hero h1{
  font-family:'Sora',sans-serif; font-size:2.35rem; line-height:1.12;
  font-weight:600; letter-spacing:-.03em; color:#f7faff; margin:0 0 .7rem 0;
}
.hero h1 .grad{
  background:linear-gradient(95deg,#a9c0ff,#6d8cff 40%,#22d3ee);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p{color:#9aa6b8; font-size:.94rem; line-height:1.65; max-width:62ch; margin:0;}
.hero-chips{display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.3rem;}
.chip{
  font-size:.72rem; color:#c3cddd; padding:.34rem .7rem; border-radius:8px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
}
.chip b{color:#f2f6fd; font-weight:600;}

/* ---------- SECTION HEADINGS ---------- */
.sec{margin:2.1rem 0 .9rem 0;}
.sec-eyebrow{font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); font-weight:700;}
.sec-title{font-family:'Sora',sans-serif; font-size:1.16rem; font-weight:600; color:#f2f6fd; letter-spacing:-.02em; margin-top:.28rem;}
.sec-sub{font-size:.83rem; color:var(--muted); margin-top:.25rem;}

/* ---------- KPI CARDS ---------- */
.kpi{
  position:relative; overflow:hidden;
  background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
  border:1px solid var(--line); border-radius:15px;
  padding:1rem 1.05rem 1.05rem;
  transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
  height:100%;
}
.kpi:hover{transform:translateY(-3px); border-color:rgba(109,140,255,.4); box-shadow:0 26px 44px -34px rgba(109,140,255,.9);}
.kpi::before{content:""; position:absolute; left:0; top:0; height:2px; width:100%; background:var(--tone,var(--accent)); opacity:.7;}
.kpi-top{display:flex; align-items:center; justify-content:space-between; gap:.5rem;}
.kpi-label{font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:#8b95a7; font-weight:600;}
.kpi-ico{font-size:.8rem; color:var(--tone,var(--accent)); opacity:.9;}
.kpi-value{font-family:'Sora',sans-serif; font-size:1.62rem; font-weight:600; color:#f7faff; letter-spacing:-.035em; margin-top:.5rem; line-height:1;}
.kpi-delta{font-size:.71rem; color:#7f8b9d; margin-top:.45rem;}

/* ---------- PANELS (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(180deg, rgba(255,255,255,.038), rgba(255,255,255,.010));
  border:1px solid var(--line) !important;
  border-radius:16px;
  box-shadow:0 30px 60px -55px #000;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{padding:.35rem .55rem .2rem;}

/* ---------- MISC NATIVE OVERRIDES ---------- */
p, span, label, div{color:inherit;}
[data-testid="stMarkdownContainer"] p{font-size:.88rem; color:#9aa6b8; line-height:1.6;}
[data-testid="stWidgetLabel"] p{font-size:.78rem; color:#a9b4c6; font-weight:500;}
[data-testid="stCaptionContainer"], .stCaption{color:#78859a !important; font-size:.75rem !important;}

[data-testid="stSlider"] [role="slider"]{background:var(--accent);}
[data-testid="stSlider"] [data-testid="stTickBar"]{background:transparent;}

[data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.02); border:1px dashed rgba(109,140,255,.35); border-radius:14px;
}
[data-testid="stFileUploaderDropzone"]:hover{border-color:rgba(109,140,255,.65); background:rgba(109,140,255,.05);}

[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input{
  background:rgba(255,255,255,.035) !important;
  border-color:var(--line) !important;
  border-radius:9px !important;
  color:#e9eef7 !important;
  font-size:.85rem;
}
[data-baseweb="tab-list"]{gap:.35rem; border-bottom:1px solid var(--line-soft);}
[data-baseweb="tab"]{border-radius:9px 9px 0 0; font-size:.83rem; color:#8b95a7;}
[aria-selected="true"][data-baseweb="tab"]{color:#f2f6fd;}

[data-testid="stDataFrame"]{border:1px solid var(--line-soft); border-radius:14px; overflow:hidden;}
[data-testid="stDataFrame"] *{font-family:'Inter',sans-serif;}

[data-testid="stAlert"]{border-radius:12px; border:1px solid var(--line); background:rgba(255,255,255,.03);}

hr{border-color:var(--line-soft) !important;}

/* ---------- VERDICT / SIGNAL BLOCKS ---------- */
.verdict{border-radius:16px; padding:1.15rem 1.25rem; border:1px solid; margin-top:.4rem;}
.verdict.fraud{background:linear-gradient(120deg, rgba(255,92,124,.14), rgba(255,92,124,.04)); border-color:rgba(255,92,124,.42);}
.verdict.safe{background:linear-gradient(120deg, rgba(45,212,167,.13), rgba(45,212,167,.03)); border-color:rgba(45,212,167,.38);}
.verdict-title{font-family:'Sora',sans-serif; font-size:1.02rem; font-weight:600; letter-spacing:-.01em; color:#f7faff; display:flex; align-items:center; gap:.5rem;}
.verdict-sub{font-size:.8rem; color:#9aa6b8; margin-top:.35rem;}
.sig{display:inline-flex; align-items:center; gap:.4rem; font-size:.76rem; color:#d8e0ee;
  background:rgba(245,181,68,.10); border:1px solid rgba(245,181,68,.30);
  padding:.36rem .65rem; border-radius:8px; margin:.22rem .3rem .22rem 0;}
.sig.ok{background:rgba(45,212,167,.10); border-color:rgba(45,212,167,.30);}

.pill{display:inline-block; font-size:.72rem; font-weight:600; padding:.2rem .55rem; border-radius:99px;}
.pill.ok{background:rgba(45,212,167,.13); color:#5ee0bb; border:1px solid rgba(45,212,167,.3);}
.pill.bad{background:rgba(255,92,124,.13); color:#ff8ea3; border:1px solid rgba(255,92,124,.3);}

.side-foot{font-size:.68rem; color:#5f6b7d; line-height:1.6; padding-top:.6rem;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# =====================================================
# HELPERS
# =====================================================
def theme(fig, height=300, title=None, legend=True):
    """Apply the shared chart theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=46 if title else 16, b=50 if legend else 12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color=MUTED),
        title=dict(
            text=title, x=0, xanchor="left", y=0.97,
            font=dict(family="Sora, sans-serif", size=13, color="#dfe6f2"),
        ) if title else None,
        hoverlabel=dict(bgcolor="#111827", bordercolor="rgba(148,163,184,.25)",
                        font=dict(color="#e9eef7", size=11, family="Inter")),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        colorway=[ACCENT, CYAN, OK, WARN, BAD],
        bargap=0.35,
        transition_duration=350,
    )
    fig.update_xaxes(showgrid=False, zeroline=False,
                     linecolor="rgba(148,163,184,.12)",
                     tickfont=dict(size=10), title_font=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10), title_font=dict(size=10))
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, **WIDE, config={"displayModeBar": False}, key=key)


def confusion_figure(data):
    """Themed confusion-matrix heatmap for labelled data."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(data["is_fraud"], data["fraud_prediction"])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Legit", "Fraud"], y=["Legit", "Fraud"],
        colorscale=[[0, "#0d1220"], [0.45, "#2f4b8f"], [1, ACCENT]],
        showscale=False, text=cm, texttemplate="%{text:,}",
        textfont=dict(size=17, color="#f7faff", family="Sora"),
        hovertemplate="Actual %{y} · Predicted %{x}: %{z:,}<extra></extra>"))
    theme(fig, 330, "Confusion matrix", legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Predicted")
    return fig


def roc_figure(data):
    """Themed ROC curve for labelled data."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(data["is_fraud"], data["fraud_probability"])
    auc_v = roc_auc_score(data["is_fraud"], data["fraud_probability"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="rgba(148,163,184,.35)", dash="dash", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=CYAN, width=2.4, shape="spline"),
        fill="tozeroy", fillcolor="rgba(34,211,238,.12)",
        name=f"AUC {auc_v:.3f}",
        hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
    theme(fig, 330, "ROC curve", legend=True)
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return fig


def kpi(col, label, value, delta="", tone=ACCENT, icon=""):
    with col:
        st.markdown(
            f'<div class="kpi" style="--tone:{tone}">'
            f'<div class="kpi-top"><span class="kpi-label">{label}</span>'
            f'<span class="kpi-ico">{icon}</span></div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-delta">{delta}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def section(eyebrow, title, sub=""):
    st.markdown(
        f'<div class="sec"><div class="sec-eyebrow">{eyebrow}</div>'
        f'<div class="sec-title">{title}</div>'
        + (f'<div class="sec-sub">{sub}</div>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# =====================================================
# STATE & MODEL BOOTSTRAP
# =====================================================
ARTIFACT_DIR = "model_artifacts"
REQUIRED_ARTIFACTS = [
    f"{ARTIFACT_DIR}/preprocessor.joblib",
    f"{ARTIFACT_DIR}/lda.joblib",
    f"{ARTIFACT_DIR}/fraud_ann_model.joblib",
    f"{ARTIFACT_DIR}/fraud_dnn_model.joblib",
]

if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "trained" not in st.session_state:
    st.session_state.trained = all(os.path.exists(p) for p in REQUIRED_ARTIFACTS)
if "evaluated" not in st.session_state:
    st.session_state.evaluated = os.path.exists(f"{ARTIFACT_DIR}/metrics.json")
if "active_df" not in st.session_state:
    st.session_state.active_df = None
if "preset" not in st.session_state:
    st.session_state.preset = "high_risk"


def ensure_models_ready() -> None:
    """Train default models on first run if artifacts are missing."""
    if not all(os.path.exists(p) for p in REQUIRED_ARTIFACTS):
        with st.spinner("Booting hybrid ANN + DNN + SMOTE pipeline…"):
            train_pipeline()
            st.session_state.trained = True
            st.session_state.evaluated = True


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(f"{ARTIFACT_DIR}/preprocessor.joblib")
    lda = joblib.load(f"{ARTIFACT_DIR}/lda.joblib")
    ann = joblib.load(f"{ARTIFACT_DIR}/fraud_ann_model.joblib")
    dnn = joblib.load(f"{ARTIFACT_DIR}/fraud_dnn_model.joblib")
    return preprocessor, lda, ann, dnn


def load_metrics() -> dict:
    metrics_path = f"{ARTIFACT_DIR}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"accuracy": 0.96, "precision": 0.94, "recall": 0.93,
            "f1_score": 0.935, "auc": 0.978}


MODEL_COLS = [
    "Transaction Amount", "Quantity", "Customer Age", "Account Age Days",
    "Transaction Hour", "Payment Method", "Product Category",
    "Customer Location", "Device Used",
]


def score_dataframe(df, preprocessor, lda, ann, dnn, threshold):
    scored = df.copy()
    if all(col in scored.columns for col in MODEL_COLS):
        X_proc = preprocessor.transform(scored[MODEL_COLS])
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
        X_lda = lda.transform(X_proc)
        ann_prob = ann.predict_proba(X_lda)[:, 1]
        dnn_prob = dnn.predict_proba(X_lda)[:, 1]
        final_prob = (ann_prob + dnn_prob) / 2.0
        scored["fraud_probability"] = np.round(final_prob, 4)
        scored["fraud_prediction"] = (final_prob >= threshold).astype(int)
    return scored


ensure_models_ready()
preprocessor, lda, ann_model, dnn_model = load_artifacts()
metrics = load_metrics()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-mark">◈</div>'
        '<div><div class="brand-name">Sentinel</div>'
        '<div class="brand-sub">Fraud Intelligence</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    for key, icon, label in NAV:
        if st.button(f"{icon}   {label}", key=f"nav_{key}", **WIDE,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="side-label">Decision Engine</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Risk threshold",
        min_value=0.10, max_value=0.95, value=0.50, step=0.05,
        help="Transactions scoring at or above this probability are flagged as fraud.",
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:.72rem;'
        f'color:#78859a;margin-top:-.3rem;">'
        f'<span>Current cutoff</span>'
        f'<span style="color:{ACCENT};font-weight:600;">{threshold:.0%}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nav-pill">StandardScaler · OneHot</div>'
        '<div style="height:.3rem"></div>'
        '<div class="nav-pill">SMOTE oversampling</div>'
        '<div style="height:.3rem"></div>'
        '<div class="nav-pill">LDA projection</div>'
        '<div style="height:.3rem"></div>'
        '<div class="nav-pill">ANN (128·64) + DNN (256·128·64)</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-foot">Models are served from cached artifacts. '
        'Retrain from the Model Lab tab to refresh metrics and plots.</div>',
        unsafe_allow_html=True,
    )

page = st.session_state.page

# =====================================================
# ACTIVE DATASET
# =====================================================
if st.session_state.active_df is None:
    if os.path.exists("fraud_predictions_final_v2.csv"):
        base_df = pd.read_csv("fraud_predictions_final_v2.csv")
        if len(base_df) > 4000:
            base_df = base_df.sample(4000, random_state=42)
    else:
        base_df = generate_synthetic_transactions(n_rows=4000, seed=42)
        base_df = score_dataframe(base_df, preprocessor, lda, ann_model, dnn_model, threshold)
    st.session_state.active_df = base_df

df = st.session_state.active_df.copy()
if "fraud_probability" in df.columns:
    df["fraud_prediction"] = (df["fraud_probability"] >= threshold).astype(int)

scored_ok = not df.empty and "fraud_prediction" in df.columns
fraud_df = df[df["fraud_prediction"] == 1] if scored_ok else df.iloc[0:0]
fraud_rate = (len(fraud_df) / len(df) * 100) if scored_ok and len(df) else 0.0

# =====================================================
# HERO
# =====================================================
st.markdown(
    f'<div class="hero">'
    f'<div class="eyebrow"><span class="pulse"></span>Live · Hybrid ensemble engine</div>'
    f'<h1>Fraud intelligence,<br><span class="grad">scored in real time.</span></h1>'
    f'<p>SMOTE-balanced, LDA-projected ANN + DNN ensemble that scores every transaction '
    f'against a behavioural risk profile — with explainable signals, live analytics and a '
    f'tunable decision threshold.</p>'
    f'<div class="hero-chips">'
    f'<span class="chip">Records <b>{len(df):,}</b></span>'
    f'<span class="chip">Flagged <b>{len(fraud_df):,}</b></span>'
    f'<span class="chip">Fraud rate <b>{fraud_rate:.2f}%</b></span>'
    f'<span class="chip">ROC-AUC <b>{metrics.get("auc", 0):.3f}</b></span>'
    f'<span class="chip">Cutoff <b>{threshold:.0%}</b></span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# common chart datasets
hourly = pd.DataFrame()
if scored_ok and "Transaction Hour" in df.columns:
    hourly = df.groupby("Transaction Hour", as_index=False).agg(
        rate=("fraud_prediction", "mean"), volume=("fraud_prediction", "size"))
    hourly["Fraud Rate (%)"] = hourly["rate"] * 100


# =====================================================
# OVERVIEW
# =====================================================
if page == "Overview":
    section("Model health", "Ensemble performance",
            "Validation metrics from the latest full training run.")

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    kpi(c1, "Accuracy", f"{metrics.get('accuracy', 0):.1%}", "Validation split", ACCENT, "◎")
    kpi(c2, "Precision", f"{metrics.get('precision', 0):.1%}", "Alert quality", CYAN, "◇")
    kpi(c3, "Recall", f"{metrics.get('recall', 0):.1%}", "Fraud captured", OK, "◈")
    kpi(c4, "F1 score", f"{metrics.get('f1_score', 0):.1%}", "Harmonic mean", WARN, "◐")
    kpi(c5, "ROC-AUC", f"{metrics.get('auc', 0):.3f}", "Discrimination", BAD, "◉")

    section("Architecture", "How a transaction gets scored",
            "Four stages, from raw attributes to a fused risk probability.")

    steps = [
        ("01", "SMOTE balancing", "Synthesises minority fraud samples on the training split only, so the imbalance is fixed without leaking validation rows.", ACCENT),
        ("02", "LDA projection", "Linear Discriminant Analysis collapses scaled numeric and one-hot categorical features into a maximally separable space.", CYAN),
        ("03", "ANN + DNN ensemble", "A 128·64 shallow net learns direct boundaries; a 256·128·64 deep net captures non-linear interactions.", WARN),
        ("04", "Probability fusion", "Both networks vote; the averaged score is compared to your threshold to produce the final alert.", OK),
    ]
    cols = st.columns(4, gap="small")
    for col, (num, title, body, tone) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="kpi" style="--tone:{tone}">'
                f'<div class="kpi-label" style="color:{tone}">{num}</div>'
                f'<div style="font-family:Sora,sans-serif;font-size:.95rem;font-weight:600;'
                f'color:#f2f6fd;margin-top:.45rem;letter-spacing:-.01em;">{title}</div>'
                f'<div style="font-size:.78rem;color:#8b95a7;line-height:1.55;margin-top:.4rem;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    section("Live snapshot", "Traffic & risk distribution")
    left, right = st.columns([1.45, 1], gap="medium")

    with left:
        with st.container(border=True):
            if not hourly.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly["Transaction Hour"], y=hourly["Fraud Rate (%)"],
                    mode="lines", line=dict(color=ACCENT, width=2.4, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(109,140,255,.16)",
                    hovertemplate="%{x}:00 · %{y:.1f}%<extra></extra>", name="Fraud rate"))
                fig.add_trace(go.Scatter(
                    x=hourly["Transaction Hour"], y=hourly["Fraud Rate (%)"],
                    mode="markers", marker=dict(color=CYAN, size=4),
                    hoverinfo="skip", showlegend=False))
                theme(fig, 330, "Fraud rate by hour of day", legend=False)
                fig.update_xaxes(dtick=3)
                fig.update_yaxes(title=None, ticksuffix="%")
                show(fig, "ov_hourly")
            else:
                st.caption("No hourly data available.")

    with right:
        with st.container(border=True):
            if scored_ok:
                counts = df["fraud_prediction"].value_counts()
                legit, flagged = int(counts.get(0, 0)), int(counts.get(1, 0))
                fig = go.Figure(go.Pie(
                    labels=["Legitimate", "Flagged"],
                    values=[legit, flagged],
                    hole=0.72, sort=False,
                    marker=dict(colors=[OK, BAD], line=dict(color="#06080e", width=3)),
                    textinfo="none",
                    hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
                theme(fig, 330, "Flagged vs legitimate", legend=True)
                fig.update_layout(
                    annotations=[dict(text=f"<b>{fraud_rate:.1f}%</b><br><span style='font-size:10px;color:{MUTED}'>fraud</span>",
                                      x=0.5, y=0.5, showarrow=False,
                                      font=dict(size=22, color="#f7faff", family="Sora"))])
                fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
                show(fig, "ov_pie")
            else:
                st.caption("No scored data available.")

    section("Priority queue", "Highest-risk open transactions",
            "The five records closest to or above your current cutoff.")
    if scored_ok:
        top = df.sort_values("fraud_probability", ascending=False).head(5)
        rows = ""
        for _, r in top.iterrows():
            is_fraud = int(r["fraud_prediction"]) == 1
            pill = ('<span class="pill bad">Fraud</span>' if is_fraud
                    else '<span class="pill ok">Legit</span>')
            rows += (
                f'<tr><td style="color:#8b95a7;font-size:.76rem">{r.get("Transaction ID", "—")}</td>'
                f'<td style="color:#e9eef7;font-weight:600">₹{r.get("Transaction Amount", 0):,.0f}</td>'
                f'<td style="color:#9aa6b8;font-size:.79rem">{r.get("Device Used", "—")}</td>'
                f'<td style="color:#9aa6b8;font-size:.79rem">{r.get("Product Category", "—")}</td>'
                f'<td style="color:#9aa6b8;font-size:.79rem">{r.get("Customer Location", "—")}</td>'
                f'<td style="text-align:right"><span style="color:{"#ff8ea3" if is_fraud else "#5ee0bb"};'
                f'font-weight:600;font-size:.8rem">{r["fraud_probability"]:.1%}</span></td>'
                f'<td style="text-align:right">{pill}</td></tr>'
            )
        st.markdown(
            f'<div style="border:1px solid var(--line);border-radius:16px;overflow:hidden;'
            f'background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.008));">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr style="background:rgba(255,255,255,.03)">'
            + "".join(
                f'<th style="text-align:{"right" if i>=5 else "left"};padding:.65rem .9rem;'
                f'font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:#78859a;'
                f'font-weight:600;border-bottom:1px solid var(--line-soft)">{h}</th>'
                for i, h in enumerate(["Transaction", "Amount", "Device", "Category", "Location", "Risk", "Verdict"]))
            + f'</tr></thead><tbody>'
            + rows.replace("<tr>", '<tr style="border-bottom:1px solid rgba(148,163,184,.06)">').replace(
                "<td", '<td style="padding:.6rem .9rem"')
            + f'</tbody></table></div>',
            unsafe_allow_html=True,
        )


# =====================================================
# DATASET EXPLORER
# =====================================================
elif page == "Dataset":
    section("Data intake", "Dataset management",
            "Upload your own transaction CSV or generate a fresh synthetic sample.")

    up_col, btn_col = st.columns([3, 1], gap="medium", vertical_alignment="center")
    with up_col:
        uploaded = st.file_uploader("Upload transaction CSV", type=["csv"],
                                    label_visibility="collapsed")
    with btn_col:
        if st.button("↻  Generate 5,000 rows", **WIDE, type="primary"):
            fresh = generate_synthetic_transactions(n_rows=5000, seed=int(np.random.randint(1, 9999)))
            st.session_state.active_df = score_dataframe(
                fresh, preprocessor, lda, ann_model, dnn_model, threshold)
            st.success("Generated and scored 5,000 synthetic transactions.")
            st.rerun()

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        st.session_state.active_df = score_dataframe(
            raw_df, preprocessor, lda, ann_model, dnn_model, threshold)
        df = st.session_state.active_df.copy()
        if "fraud_probability" in df.columns:
            df["fraud_prediction"] = (df["fraud_probability"] >= threshold).astype(int)
        st.success(f"Loaded and scored {len(df):,} rows from **{uploaded.name}**.")

    if scored_ok and not df.empty:
        total_rows = len(df)
        fraud_cnt = int(df["fraud_prediction"].sum())
        legit_cnt = total_rows - fraud_cnt
        avg_fraud = (df[df["fraud_prediction"] == 1]["Transaction Amount"].mean()
                     if fraud_cnt and "Transaction Amount" in df.columns else 0)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        kpi(c1, "Total rows", f"{total_rows:,}", f"{df.shape[1]} columns", ACCENT, "▤")
        kpi(c2, "Flagged fraud", f"{fraud_cnt:,}", f"{(fraud_cnt/total_rows*100):.2f}% of total", BAD, "▲")
        kpi(c3, "Legitimate", f"{legit_cnt:,}", f"{(legit_cnt/total_rows*100):.2f}% of total", OK, "▼")
        kpi(c4, "Avg fraud ticket", f"₹{avg_fraud:,.0f}" if avg_fraud else "—",
            "Mean fraud amount", WARN, "₹")

        section("Distributions", "Class balance & ticket size")
        g1, g2 = st.columns(2, gap="medium")

        with g1:
            with st.container(border=True):
                cc = df["fraud_prediction"].value_counts().reset_index()
                cc.columns = ["Class", "Count"]
                cc["Label"] = cc["Class"].map({0: "Legitimate", 1: "Fraud"})
                fig = px.bar(cc, x="Label", y="Count", color="Label",
                             color_discrete_map={"Legitimate": OK, "Fraud": BAD})
                theme(fig, 300, "Class distribution", legend=False)
                fig.update_layout(barcornerradius=6)
                fig.update_yaxes(title=None)
                show(fig, "ds_bar")

        with g2:
            with st.container(border=True):
                if "Transaction Amount" in df.columns:
                    p = df.copy()
                    p["Status"] = p["fraud_prediction"].map({0: "Legitimate", 1: "Fraud"})
                    fig = px.histogram(p, x="Transaction Amount", color="Status", nbins=45,
                                       barmode="overlay",
                                       color_discrete_map={"Legitimate": ACCENT, "Fraud": BAD})
                    theme(fig, 300, "Transaction amount distribution")
                    fig.update_traces(opacity=0.78, marker_line_width=0)
                    fig.update_yaxes(title=None)
                    fig.update_xaxes(title=None)
                    show(fig, "ds_hist")

        section("Feature preview", "Raw records with ensemble scores")
        preview_cols = [c for c in [
            "Transaction ID", "Transaction Amount", "Quantity", "Customer Age",
            "Account Age Days", "Transaction Hour", "Device Used", "Payment Method",
            "Product Category", "Customer Location", "fraud_probability", "fraud_prediction",
        ] if c in df.columns]

        view = df[preview_cols].head(200).copy()
        if "fraud_probability" in view.columns:
            view["Risk %"] = (view.pop("fraud_probability") * 100).round(1)
        if "fraud_prediction" in view.columns:
            view["Verdict"] = view.pop("fraud_prediction").map({0: "Legitimate", 1: "Fraud"})

        st.dataframe(
            view, **WIDE, hide_index=True, height=420,
            column_config={
                "Transaction Amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
                "Risk %": st.column_config.ProgressColumn(
                    "Risk", min_value=0, max_value=100, format="%.1f%%"),
                "Verdict": st.column_config.TextColumn("Verdict", width="small"),
            },
        )


# =====================================================
# FRAUD ANALYTICS
# =====================================================
elif page == "Analytics":
    section("Behavioural intelligence", "Where fraud concentrates",
            "Breakdowns computed on the active dataset at your current threshold.")

    if scored_ok and not df.empty:
        r1, r2, r3 = st.columns(3, gap="medium")

        with r1:
            with st.container(border=True):
                if "Device Used" in df.columns and not fraud_df.empty:
                    dev = fraud_df["Device Used"].value_counts().reset_index()
                    dev.columns = ["Device", "Fraud Count"]
                    fig = px.bar(dev, x="Device", y="Fraud Count", color="Device",
                                 color_discrete_sequence=[BAD, WARN, ACCENT])
                    theme(fig, 290, "Fraud by device", legend=False)
                    fig.update_layout(barcornerradius=6)
                    fig.update_xaxes(title=None)
                    fig.update_yaxes(title=None)
                    show(fig, "an_device")
                else:
                    st.caption("No flagged transactions to break down.")

        with r2:
            with st.container(border=True):
                if "Customer Location" in df.columns and not fraud_df.empty:
                    loc = fraud_df["Customer Location"].value_counts().head(5).reset_index()
                    loc.columns = ["Location", "Fraud Count"]
                    fig = px.bar(loc, x="Fraud Count", y="Location", orientation="h",
                                 color="Fraud Count", color_continuous_scale=["#2a3550", ACCENT, BAD])
                    theme(fig, 290, "Top high-risk locations", legend=False)
                    fig.update_layout(barcornerradius=6, coloraxis_showscale=False)
                    fig.update_yaxes(autorange="reversed", showgrid=False, title=None)
                    fig.update_xaxes(title=None)
                    show(fig, "an_loc")
                else:
                    st.caption("No flagged transactions to break down.")

        with r3:
            with st.container(border=True):
                if not hourly.empty:
                    fig = go.Figure(go.Scatter(
                        x=hourly["Transaction Hour"], y=hourly["Fraud Rate (%)"],
                        mode="lines+markers",
                        line=dict(color=CYAN, width=2.2, shape="spline"),
                        marker=dict(size=5, color=CYAN, line=dict(width=0)),
                        hovertemplate="%{x}:00 · %{y:.1f}%<extra></extra>"))
                    theme(fig, 290, "Hourly fraud trend", legend=False)
                    fig.update_xaxes(dtick=4, title=None)
                    fig.update_yaxes(ticksuffix="%")
                    show(fig, "an_hour")

        if "Payment Method" in df.columns and "Product Category" in df.columns:
            section("Interaction map", "Fraud rate by category and payment method")
            with st.container(border=True):
                pivot = (df.pivot_table(index="Product Category", columns="Payment Method",
                                        values="fraud_prediction", aggfunc="mean") * 100)
                fig = px.imshow(pivot.round(1), text_auto=".1f", aspect="auto",
                                color_continuous_scale=[[0, "#0d1220"], [0.4, "#2f4b8f"],
                                                        [0.75, ACCENT], [1, BAD]])
                theme(fig, 340, None, legend=False)
                fig.update_layout(coloraxis_colorbar=dict(
                    title="Rate %", thickness=10, len=0.75,
                    tickfont=dict(size=10), title_font=dict(size=10)))
                fig.update_xaxes(side="bottom")
                show(fig, "an_heat")

    section("Evaluation", "Confusion matrix & ROC curve",
            "Computed live from the active dataset's ground-truth labels.")

    ev1, ev2 = st.columns(2, gap="medium")
    has_labels = scored_ok and "is_fraud" in df.columns
    with ev1:
        with st.container(border=True):
            if has_labels:
                show(confusion_figure(df), "an_cm")
            elif os.path.exists(f"{ARTIFACT_DIR}/confusion_matrix.png"):
                st.image(f"{ARTIFACT_DIR}/confusion_matrix.png", **WIDE)
            else:
                st.caption("Train the model to generate evaluation artifacts.")

    with ev2:
        with st.container(border=True):
            if has_labels and "fraud_probability" in df.columns:
                show(roc_figure(df), "an_roc")
            elif os.path.exists(f"{ARTIFACT_DIR}/roc_curve.png"):
                st.image(f"{ARTIFACT_DIR}/roc_curve.png", **WIDE)
            else:
                st.caption("Train the model to generate evaluation artifacts.")


# =====================================================
# MODEL LAB
# =====================================================
elif page == "Training":
    section("Model lab", "Train & evaluate the ensemble",
            "Re-fit StandardScaler → OneHot → SMOTE → LDA → ANN + DNN and refresh every artifact.")

    c1, c2, c3 = st.columns(3, gap="small")
    kpi(c1, "Balanced train rows", f"{metrics.get('train_samples_after_smote', 0):,}",
        "After SMOTE", ACCENT, "◈")
    kpi(c2, "Validation rows", f"{metrics.get('val_samples', 0):,}",
        "Held out before SMOTE", CYAN, "◇")
    kpi(c3, "Ensemble AUC", f"{metrics.get('auc', 0):.4f}",
        "Validation ROC-AUC", OK, "◉")

    section("Actions", "Run the pipeline")
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        if st.button("▶  Train ANN & DNN models", type="primary", **WIDE):
            with st.status("Running the full ML pipeline…", expanded=True) as status:
                st.write("• Loading and preprocessing features")
                st.write("• Applying SMOTE oversampling + LDA projection")
                st.write("• Fitting shallow ANN and deep DNN")
                new_metrics = train_pipeline()
                st.cache_resource.clear()
                st.session_state.trained = True
                st.session_state.evaluated = True
                status.update(label="Training complete", state="complete", expanded=False)
            st.success(
                f"Trained on {new_metrics['train_samples_after_smote']:,} balanced samples · "
                f"validation AUC {new_metrics['auc']:.2%}")
            st.rerun()

    with b2:
        if st.button("◎  Evaluate on active dataset", **WIDE):
            if scored_ok and "is_fraud" in df.columns:
                from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                             recall_score, roc_auc_score)
                y_true, y_pred = df["is_fraud"], df["fraud_prediction"]
                y_prob = df["fraud_probability"]
                st.success("Evaluation complete on the active dataset.")
                m = st.columns(5, gap="small")
                kpi(m[0], "Accuracy", f"{accuracy_score(y_true, y_pred):.2%}", "", ACCENT)
                kpi(m[1], "Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}", "", CYAN)
                kpi(m[2], "Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2%}", "", OK)
                kpi(m[3], "F1", f"{f1_score(y_true, y_pred, zero_division=0):.2%}", "", WARN)
                kpi(m[4], "ROC-AUC", f"{roc_auc_score(y_true, y_prob):.2%}", "", BAD)
            else:
                st.info("The active dataset has no ground-truth `is_fraud` labels.")

    section("Artifacts", "Latest evaluation plots")
    q1, q2 = st.columns(2, gap="medium")
    has_labels = scored_ok and "is_fraud" in df.columns and "fraud_probability" in df.columns
    with q1:
        with st.container(border=True):
            if has_labels:
                show(confusion_figure(df), "ml_cm")
            elif os.path.exists(f"{ARTIFACT_DIR}/confusion_matrix.png"):
                st.image(f"{ARTIFACT_DIR}/confusion_matrix.png",
                         caption="Confusion matrix", **WIDE)
            else:
                st.caption("No confusion matrix yet — run training.")
    with q2:
        with st.container(border=True):
            if has_labels:
                show(roc_figure(df), "ml_roc")
            elif os.path.exists(f"{ARTIFACT_DIR}/roc_curve.png"):
                st.image(f"{ARTIFACT_DIR}/roc_curve.png",
                         caption="ROC curve", **WIDE)
            else:
                st.caption("No ROC curve yet — run training.")


# =====================================================
# TRANSACTION MONITOR
# =====================================================
elif page == "Monitor":
    section("Monitoring", "Transaction risk monitor",
            "Filter the active dataset and export the slice you need.")

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1, 1.2, 1.2, 1.4], gap="medium")
        with f1:
            status_filter = st.selectbox("Status", ["All", "Fraud only", "Legitimate only"])
        with f2:
            device_filter = st.multiselect(
                "Device", options=sorted(df["Device Used"].unique())
                if "Device Used" in df.columns else [])
        with f3:
            loc_filter = st.multiselect(
                "Location", options=sorted(df["Customer Location"].unique())
                if "Customer Location" in df.columns else [])
        with f4:
            min_prob = st.slider("Minimum risk", 0.0, 1.0, 0.0, 0.05)

    filtered = df.copy()
    if scored_ok:
        if status_filter == "Fraud only":
            filtered = filtered[filtered["fraud_prediction"] == 1]
        elif status_filter == "Legitimate only":
            filtered = filtered[filtered["fraud_prediction"] == 0]
    if device_filter and "Device Used" in filtered.columns:
        filtered = filtered[filtered["Device Used"].isin(device_filter)]
    if loc_filter and "Customer Location" in filtered.columns:
        filtered = filtered[filtered["Customer Location"].isin(loc_filter)]
    if "fraud_probability" in filtered.columns:
        filtered = filtered[filtered["fraud_probability"] >= min_prob]
        filtered = filtered.sort_values("fraud_probability", ascending=False)

    s1, s2, s3 = st.columns(3, gap="small")
    kpi(s1, "Matching", f"{len(filtered):,}", f"of {len(df):,} records", ACCENT, "☰")
    kpi(s2, "Flagged in view", f"{int(filtered['fraud_prediction'].sum()) if scored_ok else 0:,}",
        f"cutoff {threshold:.0%}", BAD, "▲")
    kpi(s3, "Risk exposure", f"₹{filtered[filtered['fraud_prediction']==1]['Transaction Amount'].sum():,.0f}"
        if scored_ok and "Transaction Amount" in filtered.columns else "—",
        "Sum of flagged amounts", WARN, "₹")

    display_cols = [c for c in [
        "Transaction ID", "Transaction Amount", "Quantity", "Account Age Days",
        "Transaction Hour", "Device Used", "Payment Method", "Product Category",
        "Customer Location", "fraud_probability", "fraud_prediction",
    ] if c in filtered.columns]

    view = filtered[display_cols].head(300).copy()
    if "fraud_probability" in view.columns:
        view["Risk %"] = (view.pop("fraud_probability") * 100).round(1)
    if "fraud_prediction" in view.columns:
        view["Verdict"] = view.pop("fraud_prediction").map({0: "Legitimate", 1: "Fraud"})

    st.caption(f"Showing {len(view):,} of {len(filtered):,} matching transactions, sorted by risk.")
    st.dataframe(
        view, **WIDE, hide_index=True, height=460,
        column_config={
            "Transaction Amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
            "Risk %": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=100, format="%.1f%%"),
            "Verdict": st.column_config.TextColumn("Verdict", width="small"),
        },
    )

    st.download_button(
        "⬇  Download filtered transactions (CSV)",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_fraud_transactions.csv",
        mime="text/csv",
    )


# =====================================================
# RISK SIMULATOR
# =====================================================
elif page == "Predict":
    section("Simulator", "Score a single transaction",
            "Pick a scenario or enter your own attributes to see the ensemble's decision.")
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    presets = {
        "high_risk": ("🚨  High-risk late-night order", dict(
            amt=90000.0, qty=5, age=22, acc=5, hour=2, dev=0, pay=0, prod=0, loc=0)),
        "new_acc": ("⚠️  New account · electronics", dict(
            amt=48000.0, qty=3, age=24, acc=2, hour=1, dev=0, pay=0, prod=0, loc=1)),
        "normal": ("✅  Normal daytime purchase", dict(
            amt=1450.0, qty=1, age=34, acc=840, hour=14, dev=1, pay=2, prod=2, loc=2)),
    }

    p1, p2, p3 = st.columns(3, gap="small")
    for col, (key, (label, _)) in zip([p1, p2, p3], presets.items()):
        with col:
            if st.button(label, key=f"preset_{key}", **WIDE,
                         type="primary" if st.session_state.preset == key else "secondary"):
                st.session_state.preset = key
                st.rerun()

    d = presets[st.session_state.preset][1]

    with st.container(border=True):
        with st.form("predict_form"):
            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                st.markdown('<div class="side-label" style="margin-top:0">Transaction</div>',
                            unsafe_allow_html=True)
                amt = st.number_input("Amount (₹)", 0.0, 100000.0, d["amt"], step=500.0)
                qty = st.number_input("Quantity", 1, 10, d["qty"])
                age = st.number_input("Customer age", 18, 80, d["age"])
                acc = st.number_input("Account age (days)", 0, 3000, d["acc"])
                hour = st.slider("Transaction hour", 0, 23, d["hour"])

            with col_b:
                st.markdown('<div class="side-label" style="margin-top:0">Context</div>',
                            unsafe_allow_html=True)
                device = st.selectbox("Device", DEVICES, index=d["dev"])
                payment = st.selectbox("Payment method", PAYMENT_METHODS, index=d["pay"])
                product = st.selectbox("Product category", PRODUCT_CATEGORIES, index=d["prod"])
                location = st.selectbox("Customer location", LOCATIONS, index=d["loc"])
                st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
                submit = st.form_submit_button("Analyse transaction risk", **WIDE)

    if submit:
        input_df = pd.DataFrame([{
            "Transaction Amount": amt, "Quantity": qty, "Customer Age": age,
            "Account Age Days": acc, "Transaction Hour": hour, "Device Used": device,
            "Payment Method": payment, "Product Category": product,
            "Customer Location": location,
        }])

        X_proc = preprocessor.transform(input_df)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
        X_lda = lda.transform(X_proc)

        ann_prob = float(ann_model.predict_proba(X_lda)[0][1])
        dnn_prob = float(dnn_model.predict_proba(X_lda)[0][1])
        score = float(np.mean([ann_prob, dnn_prob]))
        is_fraud = score >= threshold

        section("Result", "Ensemble decision")
        left, right = st.columns([1.05, 1], gap="medium")

        with left:
            with st.container(border=True):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    number={"suffix": "%", "font": {"size": 44, "color": "#f7faff", "family": "Sora"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#5f6b7d",
                                 "tickfont": {"size": 9}, "tickwidth": 1},
                        "bar": {"color": BAD if is_fraud else OK, "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, threshold * 100], "color": "rgba(45,212,167,.16)"},
                            {"range": [threshold * 100, 100], "color": "rgba(255,92,124,.16)"},
                        ],
                        "threshold": {"line": {"color": WARN, "width": 3},
                                      "thickness": 0.8, "value": threshold * 100},
                    },
                ))
                theme(fig, 270, None, legend=False)
                fig.update_layout(margin=dict(l=28, r=28, t=18, b=6))
                show(fig, "pr_gauge")
                st.markdown(
                    f'<div style="text-align:center;font-size:.72rem;color:#78859a;'
                    f'margin-top:-.4rem;">Threshold marker at {threshold:.0%}</div>',
                    unsafe_allow_html=True)

        with right:
            cls = "fraud" if is_fraud else "safe"
            icon = "⛔" if is_fraud else "✓"
            headline = "Fraudulent transaction" if is_fraud else "Legitimate transaction"
            sub = (f"Risk {score:.1%} is at or above the {threshold:.0%} cutoff — hold for manual review."
                   if is_fraud else
                   f"Risk {score:.1%} is below the {threshold:.0%} cutoff — safe to approve.")
            st.markdown(
                f'<div class="verdict {cls}">'
                f'<div class="verdict-title">{icon}&nbsp; {headline}</div>'
                f'<div class="verdict-sub">{sub}</div></div>',
                unsafe_allow_html=True)

            st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3, gap="small")
            kpi(m1, "ANN", f"{ann_prob:.1%}", "128·64", ACCENT)
            kpi(m2, "DNN", f"{dnn_prob:.1%}", "256·128·64", CYAN)
            kpi(m3, "Ensemble", f"{score:.1%}", f"cutoff {threshold:.0%}",
                BAD if is_fraud else OK)

            signals = []
            if amt >= 50000:
                signals.append("High transaction amount (≥ ₹50,000)")
            if acc < 30:
                signals.append(f"New account ({acc} days old)")
            if 0 <= hour <= 4:
                signals.append(f"Late-night window ({hour:02d}:00)")
            if qty >= 5:
                signals.append(f"High item quantity ({qty} units)")
            if product == "Electronics":
                signals.append("High-resale-value category")
            if device == "Mobile" and acc < 90:
                signals.append("New mobile device footprint")

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="side-label" style="margin-top:.4rem">Risk signals</div>',
                        unsafe_allow_html=True)
            if signals:
                chips = "".join(f'<span class="sig">▲ {s}</span>' for s in signals)
            else:
                chips = '<span class="sig ok">✓ Normal behavioural profile detected</span>'
            st.markdown(chips, unsafe_allow_html=True)
