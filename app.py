import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
    page_title="Sentinel — Fraud Control Sheet",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# DESIGN TOKENS
# =====================================================
INK = "#12110f"
PAPER = "#efece2"
PAPER_2 = "#f7f5ec"
MUTED = "#77746a"
RULE = "#c9c4b4"
RED = "#c8341f"
GREEN = "#1c6b4a"
AMBER = "#a9660f"
GRID = "rgba(18,17,15,0.08)"

# Streamlit renamed `use_container_width` to `width="stretch"` in 1.49.
# Pick whichever keyword the running version supports so the app stays portable.
def _streamlit_minor() -> tuple:
    raw = getattr(st, "__version__", "0.0").split(".")[:2]
    return tuple(int("".join(ch for ch in part if ch.isdigit()) or 0) for part in raw)


WIDE = {"width": "stretch"} if _streamlit_minor() >= (1, 49) else {"use_container_width": True}


# =====================================================
# TABLE OVERFLOW NOTE
# =====================================================
# The sideways-scroll note follows the grid's real overflow, not the viewport
# width: a wide window with the sidebar open can still be too narrow for the
# column set, and a narrow one fits once the sidebar is collapsed.
_OVERFLOW_WATCHER = """<script>
(function () {
  var doc;
  try { doc = window.parent.document; } catch (e) { return; }
  var FOLLOWING = 4;
  function scrollerOf(frame) {
    var sc = frame.querySelector('.dvn-scroller');
    if (sc) return sc;
    var kids = frame.querySelectorAll('*');
    for (var i = 0; i < kids.length; i++) {
      var el = kids[i];
      if (el.clientWidth && /^(auto|scroll)$/.test(getComputedStyle(el).overflowX)) return el;
    }
    return null;
  }
  function frameAfter(note) {
    var frames = doc.querySelectorAll('[data-testid="stDataFrame"]');
    for (var i = 0; i < frames.length; i++) {
      if (note.compareDocumentPosition(frames[i]) & FOLLOWING) return frames[i];
    }
    return null;
  }
  function sync() {
    var notes = doc.querySelectorAll('.df-more');
    for (var i = 0; i < notes.length; i++) {
      var note = notes[i];
      var frame = frameAfter(note);
      var sc = frame && scrollerOf(frame);
      var over = !!sc && sc.scrollWidth - sc.clientWidth > 1;
      if (over) {
        if (note.getAttribute('data-overflow') !== '1') note.setAttribute('data-overflow', '1');
      } else if (note.hasAttribute('data-overflow')) {
        note.removeAttribute('data-overflow');
      }
    }
  }
  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    setTimeout(function () { pending = false; sync(); }, 80);
  }
  try { new MutationObserver(schedule).observe(doc.body, { childList: true, subtree: true }); } catch (e) {}
  window.parent.addEventListener('resize', schedule);
  setInterval(sync, 700);
  sync();
})();
</script>"""


def overflow_note():
    """Mark where the sideways-scroll note sits; it stays hidden until the grid overflows."""
    st.markdown('<div class="df-more">Scroll the table sideways for the remaining columns →</div>',
                unsafe_allow_html=True)


def watch_table_overflow():
    """Reveal the note above this table only while its grid really scrolls sideways."""
    components.html(_OVERFLOW_WATCHER, height=0)

NAV = [
    ("Overview", "01"),
    ("Dataset", "02"),
    ("Analytics", "03"),
    ("Training", "04"),
    ("Monitor", "05"),
    ("Predict", "06"),
]
PAGE_NUM = {key: num for key, num in NAV}
PAGE_TITLE = {
    "Overview": "Overview",
    "Dataset": "Dataset",
    "Analytics": "Analytics",
    "Training": "Model Lab",
    "Monitor": "Monitor",
    "Predict": "Predict",
}
PAGE_DESC = {
    "Dataset": "Upload a transaction file or generate a synthetic sample, then inspect "
               "class balance and ticket size before scoring.",
    "Analytics": "Where fraud concentrates — by device, location, hour, product category "
                 "and payment method, at your current cutoff.",
    "Training": "Re-fit the scaler, SMOTE balance, LDA projection and both networks, "
                "then review the evaluation plots.",
    "Monitor": "Filter the active dataset by status, device, location and minimum risk, "
               "then export the exact slice you need.",
    "Predict": "Score a single transaction through the ensemble and read the verdict, "
               "model votes and behavioural risk signals.",
}

# =====================================================
# GLOBAL STYLES
# =====================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --paper:#efece2;
  --paper2:#f7f5ec;
  --ink:#12110f;
  --ink-soft:#3a3833;
  --muted:#77746a;
  --rule:#c9c4b4;
  --red:#c8341f;
  --green:#1c6b4a;
  --amber:#a9660f;
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"]{
  font-family:'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  -webkit-font-smoothing:antialiased;
}

.stApp, [data-testid="stAppViewContainer"]{
  background:
    linear-gradient(rgba(18,17,15,.028) 1px, transparent 1px) 0 0 / 100% 28px,
    var(--paper);
  color:var(--ink);
}

header[data-testid="stHeader"]{
  background:transparent; min-height:0 !important; height:0; overflow:visible; pointer-events:none;
}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], [data-testid="stMainMenuButton"],
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"]{display:none !important;}
header[data-testid="stHeader"] [data-testid="stToolbar"]{
  display:flex !important; align-items:center; gap:0; width:auto; height:auto;
  min-height:0; padding:0; margin:0; background:transparent;
}
[data-testid="stHeaderActionElements"]{display:none !important;}
/* reopen control — only rendered while the sidebar is collapsed */
[data-testid="stExpandSidebarButton"]{
  display:inline-flex !important; align-items:center; justify-content:center;
  position:fixed; top:12px; right:12px; z-index:1000; pointer-events:auto;
  width:40px; height:32px; padding:0; margin:0;
  background:var(--ink); color:var(--paper); border:1px solid var(--ink);
  border-radius:0; box-shadow:none;
  transition:background .12s ease, border-color .12s ease;
}
[data-testid="stExpandSidebarButton"]:hover{background:var(--red); border-color:var(--red);}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] span{
  color:var(--paper) !important; font-size:18px !important; line-height:1;
}
[data-testid="stExpandSidebarButton"]::after{
  content:"INDEX"; position:absolute; right:calc(100% + 7px); top:50%;
  transform:translateY(-50%); padding:.14rem .32rem;
  font-size:.54rem; letter-spacing:.18em; color:var(--ink);
  background:var(--paper); border:1px solid var(--rule); white-space:nowrap;
  opacity:0; pointer-events:none; transition:opacity .12s ease;
}
[data-testid="stExpandSidebarButton"]:hover::after{opacity:1;}
body:has([data-testid="stSidebar"][data-collapsed="false"]) [data-testid="stExpandSidebarButton"]{
  display:none !important;
}

html{scroll-behavior:smooth;}
::selection{background:var(--red); color:var(--paper);}
:focus-visible{outline:2px solid var(--red); outline-offset:2px;}
.block-container{
  padding-top:1.2rem; padding-bottom:1rem; max-width:1400px;
  animation:sheet .35s ease both;
}
@keyframes sheet{from{opacity:0;} to{opacity:1;}}

::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:var(--paper2);}
::-webkit-scrollbar-thumb{background:var(--rule); border:2px solid var(--paper2);}
::-webkit-scrollbar-thumb:hover{background:var(--ink);}

/* ---------- TOP BAR / MASTHEAD ---------- */
.topbar{
  display:flex; flex-wrap:wrap; justify-content:space-between; gap:.6rem;
  border-top:2px solid var(--ink); border-bottom:1px solid var(--ink);
  padding:.4rem 0; margin-bottom:0;
  font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-soft);
}
.topbar b{color:var(--ink); font-weight:600;}
.topbar .blink{display:inline-block; width:6px; height:6px; background:var(--red); margin-right:.35rem;}

.masthead{
  position:relative; overflow:hidden; border:1px solid var(--ink); background:var(--paper2);
  padding:1.5rem 1.6rem 1.1rem; margin-bottom:0;
}
.masthead::before, .masthead::after{
  content:"+"; position:absolute; font-size:.8rem; color:var(--ink); line-height:1;
}
.masthead::before{top:4px; left:5px;}
.masthead::after{bottom:2px; right:6px;}
.mast-eyebrow{
  display:flex; flex-wrap:wrap; justify-content:space-between; gap:.5rem;
  font-size:.6rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted);
  border-bottom:1px dashed var(--rule); padding-bottom:.5rem; margin-bottom:1rem;
}
.masthead h1{
  font-family:'IBM Plex Mono', monospace; font-weight:600;
  font-size:2.6rem; line-height:1.04; letter-spacing:-.045em;
  color:var(--ink); margin:0 0 .55rem 0; text-transform:uppercase;
  text-wrap:balance;
}
.masthead h1 .slash{color:var(--red);}
.mast-role{
  font-size:.78rem; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-soft);
  margin-bottom:1.15rem;
}
.mast-role b{color:var(--ink);}
.mast-lede{
  font-size:.82rem; line-height:1.75; color:var(--ink-soft);
  max-width:74ch; margin:0 0 1.2rem 0;
}
.mast-meta{
  display:grid; grid-template-columns:repeat(5,minmax(0,1fr));
  border-top:1px solid var(--ink); margin:0 -1.6rem;
}
.mast-cell{padding:.6rem 1.6rem .1rem; border-right:1px solid var(--rule);}
.mast-cell:last-child{border-right:none;}
.mast-k{display:block; font-size:.57rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-bottom:.2rem;}
.mast-v{display:block; font-size:.82rem; font-weight:600; color:var(--ink); letter-spacing:.02em;}
.mast-v.red{color:var(--red);}

.ticker{
  overflow:hidden; border-bottom:1px solid var(--ink); margin-bottom:0;
  padding:.4rem 0; background:transparent;
}
.ticker-track{
  display:flex; width:max-content; gap:2.5rem; white-space:nowrap;
  font-size:.6rem; letter-spacing:.18em; text-transform:uppercase; color:var(--ink-soft);
  animation:ticker 42s linear infinite;
}
@keyframes ticker{from{transform:translateX(0);} to{transform:translateX(-50%);}}
.ticker:hover .ticker-track{animation-play-state:paused;}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"]{
  background:var(--paper2);
  border-right:1px solid var(--ink);
}
section[data-testid="stSidebar"] > div{padding-top:.9rem;}
section[data-testid="stSidebar"] .block-container{padding-top:.4rem;}
[data-testid="stSidebarHeader"]{height:0 !important; min-height:0 !important; padding:0 !important;}
[data-testid="stSidebarCollapseButton"]{
  position:absolute !important; top:16px; right:14px; z-index:6;
}

.brand{display:flex; align-items:stretch; gap:.65rem; padding:0 0 1rem; border-bottom:1px solid var(--ink); margin-bottom:.9rem;}
.brand-mark{
  width:42px;height:42px;flex:0 0 42px; border:1px solid var(--ink);
  display:flex;align-items:center;justify-content:center;
  font-size:.82rem; font-weight:600; letter-spacing:.04em; color:var(--ink);
  background:var(--paper);
}
.brand-name{font-size:.92rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--ink); line-height:1.2;}
.brand-sub{font-size:.58rem; color:var(--muted); letter-spacing:.15em; text-transform:uppercase; margin-top:3px;}

.side-label{
  font-size:.57rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--muted); font-weight:500; margin:1.3rem 0 .5rem 0;
  display:flex; justify-content:space-between; align-items:baseline;
}
.side-label span{color:var(--ink); font-weight:600;}

/* sidebar navigation buttons */
section[data-testid="stSidebar"] .stButton > button{
  width:100%; justify-content:flex-start; align-items:center; text-align:left;
  padding:.5rem .65rem; margin-bottom:0; min-height:2.05rem; height:auto;
  border-radius:0; font-family:'IBM Plex Mono', monospace;
  font-size:.7rem; font-weight:500; letter-spacing:.09em; text-transform:uppercase;
  background:transparent; color:var(--ink-soft);
  border:none; border-bottom:1px dashed var(--rule); border-left:2px solid transparent;
  transition:background .12s ease, color .12s ease;
  box-shadow:none;
}
section[data-testid="stSidebar"] .stButton > button > div,
section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"]{
  width:100%; display:block; text-align:left; line-height:1.3;
}
section[data-testid="stSidebar"] .stButton > button:hover{
  background:rgba(18,17,15,.06); color:var(--ink); border-left-color:var(--red);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{
  background:var(--ink); color:var(--paper); border-left:2px solid var(--red);
  border-bottom:1px solid var(--ink); font-weight:600;
}
section[data-testid="stSidebar"] .stButton > button p{font-size:.7rem; margin:0; color:inherit; letter-spacing:inherit;}
section[data-testid="stSidebar"] .stButton > button:focus:not(:active){border-color:var(--ink); color:var(--ink);}

.cutoff-box{
  border:1px solid var(--ink); background:var(--paper);
  padding:.55rem .65rem; display:flex; justify-content:space-between; align-items:baseline;
  font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
}
.cutoff-box b{font-size:1.05rem; color:var(--ink); letter-spacing:-.02em;}

.pipe-row{
  display:flex; justify-content:space-between; gap:.5rem;
  border-bottom:1px dotted var(--rule); padding:.34rem 0;
  font-size:.62rem; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-soft);
}
.pipe-row em{font-style:normal; color:var(--muted);}

.side-foot{
  font-size:.6rem; color:var(--muted); line-height:1.75; letter-spacing:.04em;
  padding-top:.9rem; margin-top:1rem; border-top:1px solid var(--rule);
  text-transform:uppercase;
}

/* sidebar collapse control */
[data-testid="stSidebarCollapseButton"] button{
  display:inline-flex !important; align-items:center; justify-content:center;
  width:30px !important; height:28px !important; padding:0 !important;
  background:transparent; border:1px solid var(--rule); border-radius:0;
  transition:background .12s ease, border-color .12s ease;
}
[data-testid="stSidebarCollapseButton"] button:hover{
  background:var(--ink); border-color:var(--ink);
}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]{
  color:var(--ink-soft) !important; font-size:16px !important; line-height:1;
}
[data-testid="stSidebarCollapseButton"] button:hover [data-testid="stIconMaterial"]{
  color:var(--paper) !important;
}

/* ---------- GENERIC BUTTONS ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  border-radius:0; font-family:'IBM Plex Mono', monospace;
  font-weight:600; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase;
  border:1px solid var(--ink); background:transparent; color:var(--ink);
  padding:.52rem .9rem; transition:background .12s ease, color .12s ease, border-color .12s ease;
  box-shadow:none;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  background:var(--ink); border-color:var(--ink); color:var(--paper);
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"], .stFormSubmitButton > button{
  background:var(--ink); border:1px solid var(--ink); color:var(--paper);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover{
  background:var(--red); border-color:var(--red); color:var(--paper_2, #f7f5ec);
}
.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span,
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] span,
.stFormSubmitButton > button p, .stFormSubmitButton > button span{
  color:inherit !important; font-weight:600; letter-spacing:inherit;
}
.stButton > button p, .stDownloadButton > button p{color:inherit; font-size:.68rem; margin:0;}
.stDownloadButton > button{margin-top:.7rem;}

/* ---------- SECTION HEADINGS ---------- */
.sec{
  display:flex; flex-wrap:wrap; align-items:baseline; gap:.7rem;
  border-top:2px solid var(--ink); margin:.9rem 0 0; padding-top:.55rem;
}
.sec-num{
  font-size:.68rem; font-weight:600; letter-spacing:.1em; color:var(--paper);
  background:var(--ink); padding:.14rem .4rem; line-height:1.4;
}
.sec-title{
  font-size:.95rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:var(--ink);
}
.sec-sub{
  flex-basis:100%; font-size:.72rem; color:var(--muted); letter-spacing:.02em;
  line-height:1.7; margin:0;
}

/* ---------- KPI CARDS ---------- */
/* A grid row sizes itself to the tallest card, so cards can never spill
   past the row and collide with whatever follows. */
.kpi-grid{
  display:grid; gap:.75rem; align-items:stretch;
  grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));
}
/* Rows nested in a Streamlit column (the Predict model votes) pick their own
   track count, so a half-width column stacks them instead of squeezing the
   value into a one-character-per-line column. */
[data-testid="stColumn"] .kpi-grid{
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
}
/* Streamlit trims 1rem off every markdown container. Our blocks have no
   paragraph margins to trim, so the wrapper ends up 16px shorter than the
   block and the next element starts inside it. Neutralise it for every
   custom block in the main column; the sidebar keeps its own rhythm. */
[data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(> div[class]){
  margin-bottom:0 !important;
}
.kpi{
  position:relative; overflow:hidden;
  background:var(--paper2);
  border:1px solid var(--ink);
  padding:.8rem .85rem .75rem;
  height:100%;
  transition:background .14s ease, color .14s ease;
}
.kpi:hover{background:var(--ink); color:var(--paper);}
.kpi::after{
  content:""; position:absolute; top:0; right:0;
  border-width:0 9px 9px 0; border-style:solid;
  border-color:transparent var(--paper) transparent transparent;
}
.kpi-top{display:flex; align-items:center; justify-content:space-between; gap:.5rem;}
.kpi-label{font-size:.57rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:500;}
.kpi:hover .kpi-label{color:#b8b2a2;}
.kpi-ico{font-size:.72rem; color:var(--red); line-height:1;}
.kpi-value{
  font-size:1.7rem; font-weight:500; color:var(--ink); letter-spacing:-.045em;
  margin-top:.5rem; line-height:1; font-variant-numeric:tabular-nums;
  white-space:nowrap;
}
.kpi:hover .kpi-value{color:var(--paper);}
.kpi-delta{font-size:.6rem; color:var(--muted); margin-top:.45rem; letter-spacing:.05em; text-transform:uppercase;}
.kpi:hover .kpi-delta{color:#b8b2a2;}

/* ---------- PANELS (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--paper2);
  border:1px solid var(--ink) !important;
  border-radius:0;
  box-shadow:none;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{padding:.55rem .7rem .35rem;}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid var(--rule) !important;
}

/* ---------- MISC NATIVE OVERRIDES ---------- */
p, span, label, div{color:inherit;}
[data-testid="stMarkdownContainer"] p:not([class]){font-size:.78rem; color:var(--ink-soft); line-height:1.75;}
[data-testid="stWidgetLabel"] p{font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-soft); font-weight:500;}
[data-testid="stCaptionContainer"], .stCaption{color:var(--muted) !important; font-size:.62rem !important; letter-spacing:.06em; text-transform:uppercase;}


[data-testid="stSlider"] [role="slider"]{background:var(--ink); border:1px solid var(--ink); box-shadow:none; border-radius:0;}
[data-testid="stSlider"] [data-testid="stTickBar"]{background:transparent;}
[data-testid="stSlider"] [data-testid="stThumbValue"]{color:var(--ink); font-size:.62rem;}

[data-testid="stFileUploaderDropzone"]{
  background:var(--paper); border:1px dashed var(--ink); border-radius:0;
}
[data-testid="stFileUploaderDropzone"]:hover{border-style:solid; background:rgba(18,17,15,.04);}

[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input, .stTextInput input{
  background:var(--paper) !important;
  border:1px solid var(--ink) !important;
  border-radius:0 !important;
  color:var(--ink) !important;
  font-family:'IBM Plex Mono', monospace !important;
  font-size:.74rem;
}
[data-baseweb="select"] > div:hover, [data-baseweb="input"] > div:hover{border-color:var(--red) !important;}

[data-baseweb="tab-list"]{gap:0; border-bottom:1px solid var(--ink);}
[data-baseweb="tab"]{border-radius:0; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);}
[aria-selected="true"][data-baseweb="tab"]{color:var(--ink);}

[data-testid="stDataFrame"]{border:1px solid var(--ink); border-radius:0; overflow:hidden; background:var(--paper2);}
[data-testid="stDataFrame"] *{font-family:'IBM Plex Mono', monospace !important;}
/* Streamlit's global rule paints every scrollbar transparent until hover, which
   leaves the grid's own horizontal bar invisible on the paper track. It is the
   only cue that more columns wait off the right edge, so keep it drawn. */
[data-testid="stDataFrame"] .dvn-scroller{
  scrollbar-width:auto; scrollbar-color:var(--ink) var(--paper);
}
[data-testid="stDataFrame"] .dvn-scroller::-webkit-scrollbar{height:12px; width:12px;}
[data-testid="stDataFrame"] .dvn-scroller::-webkit-scrollbar-track{
  background:var(--paper); border-top:1px solid var(--rule);
}
[data-testid="stDataFrame"] .dvn-scroller::-webkit-scrollbar-thumb{
  background:var(--ink); border:3px solid var(--paper);
}
[data-testid="stDataFrame"] .dvn-scroller::-webkit-scrollbar-thumb:hover{background:var(--red);}
/* The note is switched on by the overflow watcher only while the grid really
   scrolls sideways; see the OVERFLOW WATCHER script near the tables. */
.df-more{
  display:none; margin:-.15rem 0 .45rem; font-family:'IBM Plex Mono', monospace;
  font-size:.58rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted);
}
.df-more[data-overflow="1"]{display:block;}
.df-more b{color:var(--ink); font-weight:600;}
/* Zero-height component that carries the measuring script; must not take space. */
[data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]){
  position:absolute; height:0; overflow:hidden; pointer-events:none;
}

[data-testid="stAlert"]{border-radius:0; border:1px solid var(--ink); background:var(--paper2);}
[data-testid="stAlert"] p{color:var(--ink-soft) !important;}

hr{border:none; border-top:1px solid var(--rule) !important;}

/* ---------- TABLES ---------- */
.sheet-table{border:1px solid var(--ink); background:var(--paper2); overflow-x:auto;}
.sheet-table table{width:100%; min-width:640px; border-collapse:collapse;}
.sheet-table th{
  text-align:left; padding:.5rem .7rem; font-size:.56rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted); font-weight:500;
  border-bottom:1px solid var(--ink); background:var(--paper);
}
.sheet-table td{padding:.5rem .7rem; font-size:.7rem; color:var(--ink-soft); border-bottom:1px dotted var(--rule); white-space:nowrap; font-variant-numeric:tabular-nums;}
.sheet-table tr:last-child td{border-bottom:none;}
.sheet-table tr:hover td{background:rgba(18,17,15,.035);}
.sheet-table td.strong{color:var(--ink); font-weight:600;}

/* ---------- VERDICT / SIGNAL BLOCKS ---------- */
.verdict{position:relative; border:1px solid; padding:1rem 1.05rem; margin-top:0;}
.verdict.fraud{background:rgba(200,52,31,.06); border-color:var(--red);}
.verdict.safe{background:rgba(28,107,74,.06); border-color:var(--green);}
.verdict-title{
  font-size:.86rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink); display:flex; align-items:center; gap:.45rem; padding-right:6.2rem;
}
.verdict-sub{font-size:.68rem; color:var(--ink-soft); margin-top:.4rem; line-height:1.7;}
.stamp{
  position:absolute; top:.85rem; right:.85rem; transform:rotate(-7deg);
  border:1.5px solid currentColor; padding:.18rem .4rem;
  font-size:.55rem; letter-spacing:.2em; text-transform:uppercase; font-weight:600;
}
.stamp.fraud{color:var(--red);}
.stamp.safe{color:var(--green);}

.sig{
  display:inline-block; font-size:.62rem; letter-spacing:.05em; color:var(--ink-soft);
  background:var(--paper); border:1px solid var(--rule);
  padding:.28rem .5rem; margin:.22rem .3rem .22rem 0; text-transform:uppercase;
}
.sig::before{content:"▲ "; color:var(--red);}
.sig.ok::before{content:"✓ "; color:var(--green);}

.pill{
  display:inline-block; font-size:.55rem; font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; padding:.16rem .4rem; border:1px solid;
}
.pill.ok{background:rgba(28,107,74,.08); color:var(--green); border-color:var(--green);}
.pill.bad{background:rgba(200,52,31,.08); color:var(--red); border-color:var(--red);}

/* ---------- CHECKLIST STRIPS ---------- */
.step{border:1px solid var(--ink); background:var(--paper2); padding:.75rem .8rem; height:100%;}
.step-no{font-size:.58rem; letter-spacing:.16em; color:var(--red); font-weight:600;}
.step-title{font-size:.74rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--ink); margin-top:.45rem;}
.step-body{font-size:.64rem; color:var(--ink-soft); line-height:1.75; margin-top:.4rem;}

/* ---------- FOOTER ---------- */
.footer{
  display:flex; flex-wrap:wrap; justify-content:space-between; gap:.6rem;
  border-top:1px solid var(--ink); margin-top:2rem; padding-top:.6rem;
  font-size:.58rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted);
}
.footer b{color:var(--ink); font-weight:600;}
.footer a{color:var(--ink); text-decoration:none; border-bottom:1px solid var(--ink);}

/* ---------- PAGE HEADER (non-overview sheets) ---------- */
.page-head{
  border:1px solid var(--ink); border-top:2px solid var(--ink);
  background:var(--paper2); padding:1rem 1.15rem 1.05rem; margin-bottom:0;
}
.ph-eyebrow{
  display:flex; flex-wrap:wrap; justify-content:space-between; gap:.5rem;
  font-size:.58rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted);
  border-bottom:1px dashed var(--rule); padding-bottom:.45rem; margin-bottom:.8rem;
}
.ph-body{
  display:grid; grid-template-columns:minmax(0,.8fr) minmax(0,1.2fr);
  gap:.35rem 2rem; align-items:baseline;
}
.ph-title{
  font-size:1.65rem; font-weight:600; letter-spacing:-.03em; line-height:1.15;
  text-transform:uppercase; color:var(--ink);
}
.ph-desc{font-size:.76rem; line-height:1.75; color:var(--ink-soft); max-width:72ch;}

/* ---------- NARROW VIEWPORTS ---------- */
@media(max-width:1100px){
  .kpi-grid{grid-template-columns:repeat(3,minmax(0,1fr));}
  .masthead h1{font-size:2.1rem;}
  .mast-meta{grid-template-columns:repeat(2,minmax(0,1fr)); margin:0 -1.6rem;}
  .mast-cell{border-bottom:1px solid var(--rule); padding-bottom:.45rem;}
  .mast-cell:nth-child(2n){border-right:none;}
  .verdict-title{padding-right:0;}
}
@media(max-width:800px){
  [data-testid="stExpandSidebarButton"]{top:8px; right:8px; width:38px; height:30px;}
}
@media(max-width:760px){
  .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
  .masthead h1{font-size:1.5rem;}
  .mast-meta{grid-template-columns:1fr; margin:0 -1.6rem;}
  .mast-cell{border-right:none; padding-right:1.6rem;}
  .topbar{font-size:.55rem;}
  .sec-title{font-size:.8rem;}
  .ph-body{grid-template-columns:1fr; gap:.3rem;}
  .ph-title{font-size:1.35rem;}
}
@media(max-width:520px){
  .kpi-grid{grid-template-columns:1fr;}
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# =====================================================
# HELPERS
# =====================================================
def theme(fig, height=300, title=None, legend=True):
    """Apply the shared print-sheet chart theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=44 if title else 14, b=46 if legend else 10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=10, color=MUTED),
        title=dict(
            text=title.upper() if title else None, x=0, xanchor="left", y=0.98,
            font=dict(family="IBM Plex Mono, monospace", size=11, color=INK),
        ) if title else None,
        hoverlabel=dict(bgcolor=INK, bordercolor=INK,
                        font=dict(color=PAPER, size=10, family="IBM Plex Mono, monospace")),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        colorway=[INK, RED, GREEN, AMBER, MUTED],
        bargap=0.42,
        transition_duration=250,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=RULE, linewidth=1,
                     tickfont=dict(size=9), title_font=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, griddash="dot", zeroline=False,
                     linecolor=RULE, tickfont=dict(size=9), title_font=dict(size=9))
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, **WIDE, config={"displayModeBar": False}, key=key)


def confusion_figure(data):
    """Print-style confusion-matrix heatmap for labelled data."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(data["is_fraud"], data["fraud_prediction"])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Legit", "Fraud"], y=["Legit", "Fraud"],
        colorscale=[[0, "#f7f5ec"], [0.45, "#ded8c6"], [1, "#8d8676"]],
        showscale=False, text=cm, texttemplate="%{text:,}",
        textfont=dict(size=17, color=INK, family="IBM Plex Mono"),
        hovertemplate="Actual %{y} · Predicted %{x}: %{z:,}<extra></extra>",
        xgap=2, ygap=2))
    theme(fig, 320, "Confusion matrix", legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Predicted")
    return fig


def roc_figure(data):
    """Print-style ROC curve for labelled data."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(data["is_fraud"], data["fraud_probability"])
    auc_v = roc_auc_score(data["is_fraud"], data["fraud_probability"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=RULE, dash="dash", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=RED, width=2.2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(200,52,31,.08)",
        name=f"AUC {auc_v:.3f}",
        hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
    theme(fig, 320, "ROC curve", legend=True)
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return fig


def kpi_row(items):
    """Render one grid row of KPI cards: (label, value, delta, icon) tuples.

    A grid row is sized by its tallest card, so the cards always fit the row
    instead of relying on percentage heights inside Streamlit's columns.
    """
    cards = "".join(
        f'<div class="kpi">'
        f'<div class="kpi-top"><span class="kpi-label">{label}</span>'
        f'<span class="kpi-ico">{icon}</span></div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta">{delta}</div>'
        f'</div>'
        for label, value, delta, icon in items
    )
    st.markdown(
        f'<div class="kpi-grid" style="--cols:{len(items)}">{cards}</div>',
        unsafe_allow_html=True,
    )


def section(num, title, sub=""):
    st.markdown(
        f'<div class="sec"><span class="sec-num">{num}</span>'
        f'<span class="sec-title">{title}</span>'
        + (f'<p class="sec-sub">{sub}</p>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def page_head(page, rows, threshold):
    """Compact sheet header used on every page except the Overview masthead."""
    st.markdown(
        f'<div class="page-head" id="top">'
        f'<div class="ph-eyebrow"><span>Sheet {PAGE_NUM[page]} / 06</span>'
        f'<span>{PAGE_TITLE[page]}</span>'
        f'<span>{rows:,} records · cutoff {threshold:.0%}</span></div>'
        f'<div class="ph-body">'
        f'<div class="ph-title">{PAGE_TITLE[page]}</div>'
        f'<div class="ph-desc">{PAGE_DESC[page]}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def topbar(caption):
    st.markdown(
        f'<div class="topbar">'
        f'<span><span class="blink"></span>LIVE · {caption}</span>'
        f'<span>FRAUD—2026 · CONTROL SHEET № 0001</span>'
        f'<span>SENTINEL / E-COMMERCE RISK</span>'
        f'</div>',
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
        '<div class="brand-mark">S—26</div>'
        '<div><div class="brand-name">Sentinel</div>'
        '<div class="brand-sub">Fraud control sheet</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Index <span>06 sheets</span></div>',
                unsafe_allow_html=True)
    for key, num in NAV:
        if st.button(f"{num} / {PAGE_TITLE[key]}", key=f"nav_{key}", **WIDE,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="side-label">Decision engine</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Risk threshold",
        min_value=0.10, max_value=0.95, value=0.50, step=0.05,
        help="Transactions scoring at or above this probability are flagged as fraud.",
    )
    st.markdown(
        f'<div class="cutoff-box"><span>Cutoff</span><b>{threshold:.0%}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipe-row"><span>01 Scale + encode</span><em>Std · OHE</em></div>'
        '<div class="pipe-row"><span>02 Balance</span><em>SMOTE</em></div>'
        '<div class="pipe-row"><span>03 Project</span><em>LDA</em></div>'
        '<div class="pipe-row"><span>04 Ensemble</span><em>ANN + DNN</em></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-foot">Models served from cached artifacts. '
        'Retrain from sheet 04 to refresh metrics and plots.</div>',
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
# MASTHEAD
# =====================================================
topbar("Hybrid ensemble engine")

if page == "Overview":
    st.markdown(
        f'<div class="masthead" id="top">'
        f'<div class="mast-eyebrow"><span>RS—2026 · Staff pass № 0001</span>'
        f'<span>Section 01 — Overview</span><span>Access: public</span></div>'
        f'<h1>Fraud intelligence<br><span class="slash">/</span> marking sheet</h1>'
        f'<div class="mast-role">Sentinel — <b>e-commerce transaction risk</b> · ANN + DNN ensemble</div>'
        f'<p class="mast-lede">A SMOTE-balanced, LDA-projected ensemble scores every transaction '
        f'against a behavioural risk profile. Six sheets: overview, dataset, analytics, model lab, '
        f'monitor and a single-transaction simulator — all driven by one tunable cutoff.</p>'
        f'<div class="mast-meta">'
        f'<div class="mast-cell"><span class="mast-k">Dataset</span><span class="mast-v">{len(df):,} rows</span></div>'
        f'<div class="mast-cell"><span class="mast-k">Flagged</span><span class="mast-v red">{len(fraud_df):,}</span></div>'
        f'<div class="mast-cell"><span class="mast-k">Fraud rate</span><span class="mast-v">{fraud_rate:.2f}%</span></div>'
        f'<div class="mast-cell"><span class="mast-k">ROC-AUC</span><span class="mast-v">{metrics.get("auc", 0):.3f}</span></div>'
        f'<div class="mast-cell"><span class="mast-k">Cutoff</span><span class="mast-v">{threshold:.0%}</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    ticker_items = [
        "◍ Tune the cutoff in the index",
        "✳ Every transaction gets a mark",
        "SMOTE · LDA · ANN 128·64 · DNN 256·128·64",
        "Metrics refresh on every training run",
        "Sheet 01 / 06 — scroll to inspect",
    ]
    tick = "".join(f"<span>{item}</span><span>✦</span>" for item in ticker_items)
    st.markdown(f'<div class="ticker"><div class="ticker-track">{tick}{tick}</div></div>',
                unsafe_allow_html=True)
else:
    page_head(page, len(df), threshold)

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
    section("01", "Ensemble performance",
            "Validation metrics from the latest full training run.")

    kpi_row([
        ("Accuracy", f"{metrics.get('accuracy', 0):.1%}", "Validation split", "01"),
        ("Precision", f"{metrics.get('precision', 0):.1%}", "Alert quality", "02"),
        ("Recall", f"{metrics.get('recall', 0):.1%}", "Fraud captured", "03"),
        ("F1 score", f"{metrics.get('f1_score', 0):.1%}", "Harmonic mean", "04"),
        ("ROC-AUC", f"{metrics.get('auc', 0):.3f}", "Discrimination", "05"),
    ])

    section("02", "How a transaction gets scored",
            "Four stages, from raw attributes to a fused risk probability.")

    steps = [
        ("01", "SMOTE balancing", "Synthesises minority fraud samples on the training split only, so the imbalance is fixed without leaking validation rows."),
        ("02", "LDA projection", "Linear Discriminant Analysis collapses scaled numeric and one-hot categorical features into a maximally separable space."),
        ("03", "ANN + DNN ensemble", "A 128·64 shallow net learns direct boundaries; a 256·128·64 deep net captures non-linear interactions."),
        ("04", "Probability fusion", "Both networks vote; the averaged score is compared to your threshold to produce the final alert."),
    ]
    cols = st.columns(4, gap="small")
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="step"><div class="step-no">{num}</div>'
                f'<div class="step-title">{title}</div>'
                f'<div class="step-body">{body}</div></div>',
                unsafe_allow_html=True,
            )

    section("03", "Live snapshot", "Traffic and risk distribution on the active dataset.")
    left, right = st.columns([1.45, 1], gap="medium")

    with left:
        with st.container(border=True):
            if not hourly.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly["Transaction Hour"], y=hourly["Fraud Rate (%)"],
                    mode="lines", line=dict(color=INK, width=1.6, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(18,17,15,.06)",
                    hovertemplate="%{x}:00 · %{y:.1f}%<extra></extra>", name="Fraud rate"))
                fig.add_trace(go.Scatter(
                    x=hourly["Transaction Hour"], y=hourly["Fraud Rate (%)"],
                    mode="markers", marker=dict(color=RED, size=4, symbol="square"),
                    hoverinfo="skip", showlegend=False))
                theme(fig, 320, "Fraud rate by hour of day", legend=False)
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
                    hole=0.64, sort=False,
                    marker=dict(colors=[INK, RED], line=dict(color=PAPER_2, width=4)),
                    textinfo="none",
                    hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
                theme(fig, 320, "Flagged vs legitimate", legend=True)
                fig.update_layout(
                    annotations=[dict(text=f"<b>{fraud_rate:.1f}%</b><br>"
                                          f"<span style='font-size:9px;letter-spacing:.14em'>FRAUD</span>",
                                      x=0.5, y=0.5, showarrow=False,
                                      font=dict(size=24, color=INK, family="IBM Plex Mono"))])
                fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
                show(fig, "ov_pie")
            else:
                st.caption("No scored data available.")

    section("04", "Priority queue", "The five records closest to or above your current cutoff.")
    if scored_ok:
        top = df.sort_values("fraud_probability", ascending=False).head(5)
        rows = ""
        for _, r in top.iterrows():
            is_fraud = int(r["fraud_prediction"]) == 1
            pill = ('<span class="pill bad">Fraud</span>' if is_fraud
                    else '<span class="pill ok">Legit</span>')
            rows += (
                f'<tr><td>{r.get("Transaction ID", "—")}</td>'
                f'<td class="strong" style="text-align:right">₹{r.get("Transaction Amount", 0):,.0f}</td>'
                f'<td>{r.get("Device Used", "—")}</td>'
                f'<td>{r.get("Product Category", "—")}</td>'
                f'<td>{r.get("Customer Location", "—")}</td>'
                f'<td style="text-align:right;color:{RED if is_fraud else GREEN};font-weight:600;'
                f'letter-spacing:.04em">{r["fraud_probability"]:.1%}</td>'
                f'<td style="text-align:right">{pill}</td></tr>'
            )
        st.markdown(
            f'<div class="sheet-table"><table>'
            f'<thead><tr>'
            + "".join(
                f'<th style="text-align:{"right" if i in (1, 5, 6) else "left"}">{h}</th>'
                for i, h in enumerate(["Transaction", "Amount", "Device", "Category", "Location", "Risk", "Verdict"]))
            + f'</tr></thead><tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇  Export priority queue (CSV)",
            data=top.to_csv(index=False).encode("utf-8"),
            file_name="priority_queue.csv",
            mime="text/csv",
        )


# =====================================================
# DATASET EXPLORER
# =====================================================
elif page == "Dataset":
    section("01", "Dataset intake",
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

        kpi_row([
            ("Total rows", f"{total_rows:,}", f"{df.shape[1]} columns", "▤"),
            ("Flagged fraud", f"{fraud_cnt:,}", f"{(fraud_cnt/total_rows*100):.2f}% of total", "▲"),
            ("Legitimate", f"{legit_cnt:,}", f"{(legit_cnt/total_rows*100):.2f}% of total", "▼"),
            ("Avg fraud ticket", f"₹{avg_fraud:,.0f}" if avg_fraud else "—",
             "Mean fraud amount", "₹"),
        ])

        section("02", "Distributions", "Class balance and ticket size on the active dataset.")
        g1, g2 = st.columns(2, gap="medium")

        with g1:
            with st.container(border=True):
                cc = df["fraud_prediction"].value_counts().reset_index()
                cc.columns = ["Class", "Count"]
                cc["Label"] = cc["Class"].map({0: "Legitimate", 1: "Fraud"})
                fig = px.bar(cc, x="Label", y="Count", color="Label",
                             color_discrete_map={"Legitimate": INK, "Fraud": RED})
                theme(fig, 300, "Class distribution", legend=False)
                fig.update_traces(marker_line_width=0)
                fig.update_yaxes(title=None)
                show(fig, "ds_bar")

        with g2:
            with st.container(border=True):
                if "Transaction Amount" in df.columns:
                    p = df.copy()
                    p["Status"] = p["fraud_prediction"].map({0: "Legitimate", 1: "Fraud"})
                    fig = px.histogram(p, x="Transaction Amount", color="Status", nbins=45,
                                       barmode="overlay",
                                       color_discrete_map={"Legitimate": INK, "Fraud": RED})
                    theme(fig, 300, "Transaction amount distribution")
                    fig.update_traces(opacity=0.72, marker_line_width=0)
                    fig.update_yaxes(title=None)
                    fig.update_xaxes(title=None)
                    show(fig, "ds_hist")

        section("03", "Feature preview", "Raw records with ensemble scores.")
        preview_cols = [c for c in [
            "Transaction ID", "Transaction Amount", "Account Age Days",
            "Transaction Hour", "Device Used", "Product Category",
            "Customer Location", "fraud_probability", "fraud_prediction",
        ] if c in df.columns]

        view = df[preview_cols].head(200).copy()
        if "fraud_probability" in view.columns:
            view["Risk %"] = (view.pop("fraud_probability") * 100).round(1)
        if "fraud_prediction" in view.columns:
            view["Verdict"] = view.pop("fraud_prediction").map({0: "Legitimate", 1: "Fraud"})

        overflow_note()
        st.dataframe(
            view, **WIDE, hide_index=True, height=420,
            column_config={
                "Transaction ID": st.column_config.TextColumn("ID", width=96),
                "Transaction Amount": st.column_config.NumberColumn("Amount", format="₹%.0f", width=80),
                "Account Age Days": st.column_config.NumberColumn("Acct Age (d)", width=112),
                "Transaction Hour": st.column_config.NumberColumn("Hour", width=56),
                "Device Used": st.column_config.TextColumn("Device", width=84),
                "Product Category": st.column_config.TextColumn("Category", width=112),
                "Customer Location": st.column_config.TextColumn("Location", width=96),
                "Risk %": st.column_config.ProgressColumn(
                    "Risk", min_value=0, max_value=100, format="%.1f%%", width=100),
                "Verdict": st.column_config.TextColumn("Verdict", width=92),
            },
        )
        watch_table_overflow()


# =====================================================
# FRAUD ANALYTICS
# =====================================================
elif page == "Analytics":
    section("01", "Where fraud concentrates",
            "Breakdowns computed on the active dataset at your current threshold.")

    if scored_ok and not df.empty:
        r1, r2, r3 = st.columns(3, gap="medium")

        with r1:
            with st.container(border=True):
                if "Device Used" in df.columns and not fraud_df.empty:
                    dev = fraud_df["Device Used"].value_counts().reset_index()
                    dev.columns = ["Device", "Fraud Count"]
                    fig = px.bar(dev, x="Device", y="Fraud Count", color="Device",
                                 color_discrete_sequence=[RED, AMBER, INK])
                    theme(fig, 290, "Fraud by device", legend=False)
                    fig.update_traces(marker_line_width=0)
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
                                 color="Fraud Count",
                                 color_continuous_scale=["#e7e2d3", "#c8341f"])
                    theme(fig, 290, "Top high-risk locations", legend=False)
                    fig.update_traces(marker_line_width=0)
                    fig.update_layout(coloraxis_showscale=False)
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
                        line=dict(color=INK, width=1.5, shape="spline"),
                        marker=dict(size=4, color=RED, symbol="square", line=dict(width=0)),
                        hovertemplate="%{x}:00 · %{y:.1f}%<extra></extra>"))
                    theme(fig, 290, "Hourly fraud trend", legend=False)
                    fig.update_xaxes(dtick=4, title=None)
                    fig.update_yaxes(ticksuffix="%")
                    show(fig, "an_hour")

        if "Payment Method" in df.columns and "Product Category" in df.columns:
            section("02", "Interaction map", "Fraud rate by category and payment method.")
            with st.container(border=True):
                pivot = (df.pivot_table(index="Product Category", columns="Payment Method",
                                        values="fraud_prediction", aggfunc="mean") * 100)
                fig = px.imshow(pivot.round(1), text_auto=".1f", aspect="auto",
                                color_continuous_scale=[[0, "#f7f5ec"], [0.4, "#e3c4b8"],
                                                        [0.75, "#d4735f"], [1, "#8f1d0c"]])
                theme(fig, 340, None, legend=False)
                fig.update_layout(coloraxis_colorbar=dict(
                    title="Rate %", thickness=10, len=0.75,
                    tickfont=dict(size=10), title_font=dict(size=10)))
                fig.update_xaxes(side="bottom")
                show(fig, "an_heat")

    section("03", "Evaluation", "Confusion matrix and ROC, computed live from ground-truth labels.")

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
    section("01", "Train and evaluate the ensemble",
            "Re-fit StandardScaler → OneHot → SMOTE → LDA → ANN + DNN and refresh every artifact.")

    kpi_row([
        ("Balanced train rows", f"{metrics.get('train_samples_after_smote', 0):,}",
         "After SMOTE", "◈"),
        ("Validation rows", f"{metrics.get('val_samples', 0):,}",
         "Held out before SMOTE", "◇"),
        ("Ensemble AUC", f"{metrics.get('auc', 0):.4f}",
         "Validation ROC-AUC", "◉"),
    ])

    section("02", "Run the pipeline")
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
                kpi_row([
                    ("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}", "", ""),
                    ("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}", "", ""),
                    ("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2%}", "", ""),
                    ("F1", f"{f1_score(y_true, y_pred, zero_division=0):.2%}", "", ""),
                    ("ROC-AUC", f"{roc_auc_score(y_true, y_prob):.2%}", "", ""),
                ])
            else:
                st.info("The active dataset has no ground-truth `is_fraud` labels.")

    section("03", "Latest evaluation plots")
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
    section("01", "Transaction risk monitor",
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

    kpi_row([
        ("Matching", f"{len(filtered):,}", f"of {len(df):,} records", "☰"),
        ("Flagged in view", f"{int(filtered['fraud_prediction'].sum()) if scored_ok else 0:,}",
         f"cutoff {threshold:.0%}", "▲"),
        ("Risk exposure", f"₹{filtered[filtered['fraud_prediction']==1]['Transaction Amount'].sum():,.0f}"
         if scored_ok and "Transaction Amount" in filtered.columns else "—",
         "Sum of flagged amounts", "₹"),
    ])

    csv_cols = [c for c in [
        "Transaction ID", "Transaction Amount", "Quantity", "Account Age Days",
        "Transaction Hour", "Device Used", "Payment Method", "Product Category",
        "Customer Location", "fraud_probability", "fraud_prediction",
    ] if c in filtered.columns]

    # The on-screen set is trimmed to what fits the sheet; the CSV keeps every column.
    display_cols = [c for c in [
        "Transaction ID", "Transaction Amount", "Account Age Days",
        "Transaction Hour", "Device Used", "Payment Method", "Product Category",
        "Customer Location", "fraud_probability", "fraud_prediction",
    ] if c in filtered.columns]

    view = filtered[display_cols].head(300).copy()
    if "fraud_probability" in view.columns:
        view["Risk %"] = (view.pop("fraud_probability") * 100).round(1)
    if "fraud_prediction" in view.columns:
        view["Verdict"] = view.pop("fraud_prediction").map({0: "Legitimate", 1: "Fraud"})

    st.caption(f"Showing {len(view):,} of {len(filtered):,} matching transactions, sorted by risk.")
    overflow_note()
    st.dataframe(
        view, **WIDE, hide_index=True, height=460,
        column_config={
            "Transaction ID": st.column_config.TextColumn("ID", width=96),
            "Transaction Amount": st.column_config.NumberColumn("Amount", format="₹%.0f", width=80),
            "Account Age Days": st.column_config.NumberColumn("Acct Age (d)", width=112),
            "Transaction Hour": st.column_config.NumberColumn("Hour", width=56),
            "Device Used": st.column_config.TextColumn("Device", width=84),
            "Payment Method": st.column_config.TextColumn("Payment", width=104),
            "Product Category": st.column_config.TextColumn("Category", width=112),
            "Customer Location": st.column_config.TextColumn("Location", width=96),
            "Risk %": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=100, format="%.1f%%", width=100),
            "Verdict": st.column_config.TextColumn("Verdict", width=92),
        },
    )
    watch_table_overflow()

    st.download_button(
        "⬇  Download filtered transactions (CSV)",
        data=filtered[csv_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_fraud_transactions.csv",
        mime="text/csv",
    )


# =====================================================
# RISK SIMULATOR
# =====================================================
elif page == "Predict":
    section("01", "Score a single transaction",
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

        section("02", "Ensemble decision")
        left, right = st.columns([1.05, 1], gap="medium")

        with left:
            with st.container(border=True):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    number={"suffix": "%",
                            "font": {"size": 40, "color": INK, "family": "IBM Plex Mono"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": MUTED,
                                 "tickfont": {"size": 9, "family": "IBM Plex Mono"},
                                 "tickwidth": 1},
                        "bar": {"color": RED if is_fraud else INK, "thickness": 0.26},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 1,
                        "bordercolor": RULE,
                        "steps": [
                            {"range": [0, threshold * 100], "color": "rgba(28,107,74,.10)"},
                            {"range": [threshold * 100, 100], "color": "rgba(200,52,31,.10)"},
                        ],
                        "threshold": {"line": {"color": RED, "width": 2},
                                      "thickness": 0.85, "value": threshold * 100},
                    },
                ))
                theme(fig, 280, None, legend=False)
                fig.update_layout(margin=dict(l=28, r=28, t=18, b=6))
                show(fig, "pr_gauge")
                st.markdown(
                    f'<div style="text-align:center;font-size:.6rem;letter-spacing:.14em;'
                    f'text-transform:uppercase;color:{MUTED};margin-top:-.4rem;">'
                    f'Threshold marker at {threshold:.0%}</div>',
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
                f'<div class="stamp {cls}">{"Flagged" if is_fraud else "Cleared"}</div>'
                f'<div class="verdict-title">{icon}&nbsp; {headline}</div>'
                f'<div class="verdict-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            kpi_row([
                ("ANN", f"{ann_prob:.1%}", "128·64", ""),
                ("DNN", f"{dnn_prob:.1%}", "256·128·64", ""),
                ("Ensemble", f"{score:.1%}", f"cutoff {threshold:.0%}", ""),
            ])

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
                chips = "".join(f'<span class="sig">{s}</span>' for s in signals)
            else:
                chips = '<span class="sig ok">Normal behavioural profile detected</span>'
            st.markdown(chips, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    f'<div class="footer">'
    f'<span>Sheet <b>{PAGE_NUM[page]} / 06</b> · {PAGE_TITLE[page]}</span>'
    f'<span>QC pass · <b>{len(df):,}</b> records scored at <b>{threshold:.0%}</b></span>'
    f'<span>Sentinel — control sheet · 2026</span>'
    f'<a href="#top">Top ↑</a>'
    f'</div>',
    unsafe_allow_html=True,
)
