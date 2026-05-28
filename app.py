import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import importlib
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'autods-ai')))

for mod in ['src.visualization.charts','src.modeling.hyperparameter_tuning','src.reports.pdf_generator']:
    if mod in sys.modules:
        del sys.modules[mod]

from cleaning import clean_data
from analysis import dataset_summary
from charts import plot_histogram, get_matplotlib_seaborn_plot
from train_model import train_model
from kmeans_clustering import perform_kmeans_clustering
from dbscan_clustering import perform_dbscan_clustering
from helpers import sanitize_col_names
from pdf_generator import generate_pdf_report

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="AutoDS AI", page_icon="🤖", layout="wide")

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;500&display=swap');

    /* ══ Reset ══ */
    html, body, [data-testid="stAppViewContainer"] {
        background: #050714 !important;
        color: #e2e8f8 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding: 0 2.5rem 4rem !important; max-width: 1200px !important; }
    * { box-sizing: border-box !important; }

    /* ══ Animated grid background ══ */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(99,179,237,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,179,237,0.04) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
        z-index: 0;
    }

    /* ══ Sidebar ══ */
    [data-testid="stSidebar"] { min-width: 230px !important; max-width: 250px !important; }
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #080c1f 0%, #050714 100%) !important;
        border-right: 1px solid rgba(99,179,237,0.12) !important;
        padding: 0 !important;
    }
    [data-testid="stSidebarContent"] * { color: #94a3b8 !important; }
    [data-testid="stSidebarContent"] .stButton > button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        border-radius: 0 !important;
        text-align: left !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.03em !important;
        text-transform: none !important;
        padding: 0.65rem 1.4rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: none !important;
        transform: none !important;
        border-left: 2px solid transparent !important;
    }
    [data-testid="stSidebarContent"] .stButton > button:hover {
        background: rgba(99,179,237,0.06) !important;
        color: #e2e8f8 !important;
        border-left-color: #63b3ed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ══ Typography ══ */
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

    /* ══ Landing hero ══ */
    .landing-wrap {
        min-height: 90vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 4rem 2rem;
        position: relative;
        z-index: 1;
    }
    .landing-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99,179,237,0.08);
        border: 1px solid rgba(99,179,237,0.25);
        border-radius: 99px;
        padding: 0.35rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        color: #63b3ed;
        margin-bottom: 2rem;
        animation: fadeSlideDown 0.8s ease both;
    }
    .landing-badge::before {
        content: '';
        width: 6px; height: 6px;
        background: #63b3ed;
        border-radius: 99px;
        box-shadow: 0 0 8px #63b3ed;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:0.5; transform:scale(1.4); }
    }
    .landing-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(3rem, 7vw, 6.5rem);
        font-weight: 800;
        line-height: 0.95;
        letter-spacing: -0.02em;
        margin: 0 0 1.5rem;
        background: linear-gradient(135deg, #e2e8f8 0%, #63b3ed 40%, #a78bfa 80%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeSlideUp 0.9s ease 0.2s both;
    }
    .landing-sub {
        font-size: clamp(1rem, 2vw, 1.2rem);
        color: #64748b;
        font-weight: 300;
        max-width: 580px;
        line-height: 1.7;
        margin: 0 auto 3rem;
        animation: fadeSlideUp 0.9s ease 0.4s both;
    }
    .landing-sub strong { color: #94a3b8; font-weight: 500; }
    @keyframes fadeSlideUp {
        from { opacity:0; transform:translateY(24px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes fadeSlideDown {
        from { opacity:0; transform:translateY(-16px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* ══ Feature cards ══ */
    .feat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        max-width: 900px;
        margin: 0 auto 3rem;
        animation: fadeSlideUp 0.9s ease 0.6s both;
    }
    .feat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(99,179,237,0.04) 100%);
        border: 1px solid rgba(99,179,237,0.12);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: left;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .feat-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(99,179,237,0.06) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .feat-card:hover::before { opacity: 1; }
    .feat-card:hover {
        border-color: rgba(99,179,237,0.35);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(99,179,237,0.1);
    }
    .feat-icon {
        font-size: 1.6rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    .feat-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #e2e8f8;
        margin-bottom: 0.4rem;
    }
    .feat-desc {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.6;
    }

    /* ══ Stats strip ══ */
    .stats-strip {
        display: flex;
        gap: 2.5rem;
        justify-content: center;
        margin-bottom: 3rem;
        animation: fadeSlideUp 0.9s ease 0.8s both;
    }
    .stat-item { text-align: center; }
    .stat-num {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #63b3ed, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .stat-lbl {
        font-size: 0.72rem;
        color: #475569;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ══ CTA button ══ */
    .cta-wrap { animation: fadeSlideUp 0.9s ease 1s both; }
    .cta-wrap .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.9rem 3rem !important;
        box-shadow: 0 0 32px rgba(99,179,237,0.25), 0 0 0 1px rgba(139,92,246,0.3) !important;
        transition: all 0.3s !important;
    }
    .cta-wrap .stButton > button:hover {
        box-shadow: 0 0 48px rgba(99,179,237,0.4), 0 0 0 1px rgba(139,92,246,0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* ══ Glow orbs ══ */
    .orb1 {
        position: fixed; top: -200px; left: -200px;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
        pointer-events: none; z-index: 0;
        animation: orbFloat1 12s ease-in-out infinite;
    }
    .orb2 {
        position: fixed; bottom: -200px; right: -200px;
        width: 700px; height: 700px;
        background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
        pointer-events: none; z-index: 0;
        animation: orbFloat2 15s ease-in-out infinite;
    }
    @keyframes orbFloat1 {
        0%,100% { transform: translate(0,0); }
        50%      { transform: translate(60px, 40px); }
    }
    @keyframes orbFloat2 {
        0%,100% { transform: translate(0,0); }
        50%      { transform: translate(-50px,-60px); }
    }

    /* ══ Page header ══ */
    .page-header {
        padding: 2.5rem 0 1.5rem;
        border-bottom: 1px solid rgba(99,179,237,0.08);
        margin-bottom: 2rem;
    }
    .page-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.4rem;
    }
    .page-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(1.8rem, 4vw, 2.8rem);
        font-weight: 800;
        color: #e2e8f8;
        line-height: 1;
        letter-spacing: -0.01em;
    }
    .page-title span {
        background: linear-gradient(90deg, #63b3ed, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ══ Glassy section card ══ */
    .g-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(99,179,237,0.03) 100%);
        border: 1px solid rgba(99,179,237,0.1);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    .g-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,179,237,0.4), transparent);
    }
    .g-card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .g-card-title::before {
        content: '';
        width: 16px; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 1px;
    }

    /* ══ Inputs ══ */
    input, textarea, [data-baseweb="input"] input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(99,179,237,0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f8 !important;
        font-family: 'Outfit', sans-serif !important;
        caret-color: #63b3ed !important;
    }
    input:focus, textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
        outline: none !important;
    }
    [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.04) !important;
        border-color: rgba(99,179,237,0.15) !important;
        color: #e2e8f8 !important;
        border-radius: 10px !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background: #0d1225 !important;
        border: 1px solid rgba(99,179,237,0.15) !important;
        border-radius: 10px !important;
    }
    [data-baseweb="option"] { background: transparent !important; color: #94a3b8 !important; }
    [data-baseweb="option"]:hover { background: rgba(99,179,237,0.08) !important; color: #e2e8f8 !important; }
    label { color: #64748b !important; font-size: 0.8rem !important; letter-spacing: 0.04em !important; font-family: 'Outfit', sans-serif !important; }

    /* ══ Slider ══ */
    [data-baseweb="slider"] [data-testid="stSliderThumb"] { background: #3b82f6 !important; box-shadow: 0 0 8px rgba(59,130,246,0.6) !important; }
    [data-baseweb="slider"] div[role="progressbar"] { background: linear-gradient(90deg,#3b82f6,#8b5cf6) !important; }
    [data-baseweb="slider"] div[role="slider"] { background: #0d1225 !important; border-color: rgba(99,179,237,0.2) !important; }

    /* ══ Buttons (global) ══ */
    .stButton > button {
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15)) !important;
        color: #93c5fd !important;
        border: 1px solid rgba(59,130,246,0.3) !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.65rem 1.6rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(59,130,246,0.3), rgba(139,92,246,0.3)) !important;
        border-color: rgba(59,130,246,0.6) !important;
        color: #e2e8f8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(59,130,246,0.2) !important;
    }

    /* ══ Download button ══ */
    .stDownloadButton > button {
        background: transparent !important;
        color: #63b3ed !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 10px !important;
        font-size: 0.78rem !important;
        padding: 0.55rem 1.4rem !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(99,179,237,0.08) !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ══ File uploader ══ */
    [data-testid="stFileUploader"] {
        background: rgba(59,130,246,0.04) !important;
        border: 1.5px dashed rgba(99,179,237,0.25) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        transition: all 0.3s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99,179,237,0.5) !important;
        background: rgba(59,130,246,0.07) !important;
    }

    /* ══ Dataframe ══ */
    [data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; }
    [data-testid="stDataFrame"] * { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; }

    /* ══ Progress & spinner ══ */
    .stProgress > div > div { background: linear-gradient(90deg,#3b82f6,#8b5cf6) !important; border-radius: 99px !important; }
    .stProgress > div { background: rgba(255,255,255,0.06) !important; border-radius: 99px !important; }
    .stSpinner > div { border-top-color: #3b82f6 !important; }

    /* ══ Alerts ══ */
    [data-testid="stAlert"] { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; border-color: rgba(99,179,237,0.2) !important; }

    /* ══ Success / info / warning ══ */
    .stSuccess { background: rgba(52,211,153,0.08) !important; border-color: rgba(52,211,153,0.25) !important; }
    .stInfo    { background: rgba(99,179,237,0.08) !important; border-color: rgba(99,179,237,0.25) !important; }
    .stWarning { background: rgba(251,191,36,0.08) !important; border-color: rgba(251,191,36,0.25) !important; }

    /* ══ Result badge ══ */
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.8rem 1.4rem;
        border-radius: 12px;
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .result-good { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: #34d399; }
    .result-warn { background: rgba(251,191,36,0.1);  border: 1px solid rgba(251,191,36,0.3);  color: #fbbf24; }

    /* ══ Metric cards ══ */
    .metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
    .metric-box {
        flex:1; min-width:140px;
        background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(99,179,237,0.05));
        border: 1px solid rgba(99,179,237,0.12);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg,#63b3ed,#a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .metric-key { font-size: 0.65rem; letter-spacing: 0.16em; text-transform: uppercase; color: #475569; margin-top: 0.4rem; }

    /* ══ Section divider ══ */
    hr { border-color: rgba(99,179,237,0.08) !important; margin: 2rem 0 !important; }

    /* ══ Mono labels ══ */
    .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #475569; }

    /* ══ Footer ══ */
    .footer { text-align:center; padding:3rem 0 1rem; font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#1e293b; letter-spacing:0.14em; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK STYLE
# ─────────────────────────────────────────────
def set_plot_style():
    mpl.rcParams.update({
        "figure.facecolor":  "#0d1225",
        "axes.facecolor":    "#080c1f",
        "axes.edgecolor":    "#1e293b",
        "axes.labelcolor":   "#64748b",
        "axes.titlecolor":   "#e2e8f8",
        "xtick.color":       "#475569",
        "ytick.color":       "#475569",
        "grid.color":        "#0f172a",
        "text.color":        "#e2e8f8",
        "figure.edgecolor":  "#0d1225",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

# ─────────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────────
inject_css()
set_plot_style()

for key, default in [
    ('page', 'landing'),
    ('generated_plots', []),
    ('model_results', None),
    ('clustering_results', {}),
    ('df_summary', {}),
    ('df_loaded', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def nav_btn(label, icon, key):
    if st.sidebar.button(f"{icon}  {label}", key=key, use_container_width=True):
        st.session_state.page = label
        st.rerun()

with st.sidebar:
    st.markdown("""
    <div style='padding:2rem 1.4rem 1.5rem;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.58rem;letter-spacing:0.28em;text-transform:uppercase;color:#3b82f6;margin-bottom:0.6rem;'>System v2.0</div>
        <div style='font-family:"Syne",sans-serif;font-size:1.9rem;font-weight:800;color:#e2e8f8;line-height:1;letter-spacing:-0.02em;'>Auto<span style="background:linear-gradient(135deg,#63b3ed,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">DS</span></div>
        <div style='height:1.5px;width:100%;background:linear-gradient(90deg,#3b82f6,transparent);margin:1rem 0 1.8rem;'></div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.58rem;letter-spacing:0.2em;text-transform:uppercase;color:#1e293b;margin-bottom:0.8rem;'>Navigation</div>
    </div>
    """, unsafe_allow_html=True)
    nav_btn("Home",              "◎", "nav_home")
    nav_btn("Upload & Clean",    "⬆", "nav_upload")
    nav_btn("Visualisation",     "◈", "nav_viz")
    nav_btn("Machine Learning",  "⬡", "nav_ml")
    nav_btn("Clustering",        "◉", "nav_cluster")
    nav_btn("Report",            "▤", "nav_report")
    st.markdown("""
    <div style='padding:2rem 1.4rem 1rem;margin-top:auto;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.58rem;letter-spacing:0.1em;color:#1e293b;'>AutoDS AI · Built with Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ─────────────────────────────────────────────
#  ORBS (always visible)
# ─────────────────────────────────────────────
st.markdown('<div class="orb1"></div><div class="orb2"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: LANDING
# ═══════════════════════════════════════════════════════
if page == "landing" or page == "Home":
    st.markdown("""
    <div class="landing-wrap">
        <div class="landing-badge">● Autonomous AI · Data Science Platform</div>
        <div class="landing-title">Autonomous<br>Data Scientist</div>
        <div class="landing-sub">
            Upload your data and let <strong>AutoDS AI</strong> handle everything —
            cleaning, visualisation, machine learning, clustering, and PDF reporting —
            all in one intelligent platform.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <span class="feat-icon">🧹</span>
            <div class="feat-title">Smart Cleaning</div>
            <div class="feat-desc">Auto-detects missing values, sanitizes column names, and normalises data types in seconds.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">📊</span>
            <div class="feat-title">Rich Visualisations</div>
            <div class="feat-desc">Histograms, scatter plots, heatmaps, pair plots, QQ plots, and more via Matplotlib & Plotly.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🤖</span>
            <div class="feat-title">AutoML</div>
            <div class="feat-desc">Random Forest, XGBoost, CatBoost, LightGBM, SVM with auto problem-type detection.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">⚡</span>
            <div class="feat-title">Hyperparameter Tuning</div>
            <div class="feat-desc">Optuna, GridSearchCV, and RandomizedSearchCV with configurable cross-validation.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🔵</span>
            <div class="feat-title">Clustering</div>
            <div class="feat-desc">K-Means and DBSCAN unsupervised learning with silhouette scoring and interactive plots.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">📄</span>
            <div class="feat-title">PDF Reports</div>
            <div class="feat-desc">One-click export of your full analysis — charts, metrics, and model results — as a polished PDF.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats strip
    st.markdown("""
    <div class="stats-strip">
        <div class="stat-item"><div class="stat-num">10+</div><div class="stat-lbl">ML Models</div></div>
        <div class="stat-item"><div class="stat-num">9</div><div class="stat-lbl">Chart Types</div></div>
        <div class="stat-item"><div class="stat-num">2</div><div class="stat-lbl">Cluster Algos</div></div>
        <div class="stat-item"><div class="stat-num">3</div><div class="stat-lbl">Tuning Methods</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_c = st.columns([2, 1, 2])[1]
    with col_c:
        st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
        if st.button("Get Started →", key="cta_start"):
            st.session_state.page = "Upload & Clean"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;margin-top:2rem;font-family:"JetBrains Mono",monospace;font-size:0.65rem;color:#1e293b;letter-spacing:0.12em;'>
        CSV · XLSX · Auto-detect · Export PDF
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: UPLOAD & CLEAN
# ═══════════════════════════════════════════════════════
elif page == "Upload & Clean":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Step 01 · Data Ingestion</div>
        <div class="page-title">Upload <span>&amp; Clean</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="g-card"><div class="g-card-title">Dataset Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv","xlsx"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state['loaded_df_name'] = uploaded_file.name
        df = clean_data(df)
        df = sanitize_col_names(df)
        st.session_state['df'] = df
        st.session_state.df_loaded = True
        st.session_state.df_summary['Raw Data Head'] = df.head().to_html(index=False)
        st.session_state.df_summary.update({
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Missing Values": int(df.isnull().sum().sum()),
        })
        st.session_state.df_summary['Statistical Summary'] = df.describe().to_html()

        numeric_cols = df.select_dtypes(include='number').columns

        st.success(f"✓ Dataset loaded and cleaned — {df.shape[0]:,} rows × {df.shape[1]} columns")

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">{df.shape[0]:,}</div><div class="metric-key">Rows</div></div>
            <div class="metric-box"><div class="metric-val">{df.shape[1]}</div><div class="metric-key">Columns</div></div>
            <div class="metric-box"><div class="metric-val">{len(numeric_cols)}</div><div class="metric-key">Numeric</div></div>
            <div class="metric-box"><div class="metric-val">{int(df.isnull().sum().sum())}</div><div class="metric-key">Nulls</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="g-card-title">Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="g-card-title">Data Types</div>', unsafe_allow_html=True)
        st.dataframe(df.dtypes.reset_index().rename(columns={'index':'Column', 0:'Type'}), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        dataset_summary(df)

# ═══════════════════════════════════════════════════════
#  PAGE: VISUALISATION
# ═══════════════════════════════════════════════════════
elif page == "Visualisation":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Step 02 · Exploratory Analysis</div>
        <div class="page-title">Data <span>Visualisation</span></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.df_loaded:
        st.warning("Please upload a dataset first on the Upload & Clean page.")
    else:
        df = st.session_state['df']
        numeric_cols = df.select_dtypes(include='number').columns
        all_columns  = df.columns.tolist()

        if len(numeric_cols) > 0:
            st.markdown('<div class="g-card"><div class="g-card-title">Quick Plotly Histogram</div>', unsafe_allow_html=True)
            selected_col_plotly = st.selectbox("Select Column", numeric_cols, key="plotly_hist_select")
            if selected_col_plotly:
                fig_plotly = plot_histogram(df, selected_col_plotly)
                st.plotly_chart(fig_plotly, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="g-card-title">Advanced Chart Builder</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            plot_name_input = st.text_input("Chart Name", placeholder="e.g. Age Distribution", key="plot_name_input")
            x_axis_col = st.selectbox("X-axis Column", ['None'] + all_columns, key="x_axis_select")
            y_axis_col = st.selectbox("Y-axis Column", ['None'] + all_columns, key="y_axis_select")
        with c2:
            plot_type_advanced = st.selectbox("Chart Type",
                ('Histogram','Box Plot','Scatter Plot','Pair Plot','Correlation Heatmap','Pie Chart','Density Plot','QQ Plot','Overlaid Histograms'),
                key="plot_type_type")
            selected_cols_general = st.multiselect("Additional Columns",
                all_columns, default=all_columns[:3] if all_columns else [], key="selected_cols_general_viz")

        if st.button("Generate Chart"):
            if not plot_name_input:
                st.warning("Please enter a chart name.")
            else:
                generated_fig = get_matplotlib_seaborn_plot(df, plot_type_advanced,
                    selected_cols=selected_cols_general, x_col=x_axis_col, y_col=y_axis_col)
                if generated_fig:
                    st.session_state.generated_plots.append({'name': plot_name_input, 'figure': generated_fig, 'type': 'matplotlib'})
                    st.success(f"Chart '{plot_name_input}' added.")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.generated_plots:
            st.markdown('<div class="g-card"><div class="g-card-title">Generated Charts</div>', unsafe_allow_html=True)
            for i, plot_data in enumerate(st.session_state.generated_plots):
                st.markdown(f"**{i+1}. {plot_data['name']}**")
                if plot_data['type'] == 'matplotlib':
                    st.pyplot(plot_data['figure'])
                    buf = io.BytesIO()
                    plot_data['figure'].savefig(buf, format="png", bbox_inches="tight")
                    st.download_button(f"↓ Download {plot_data['name']}", data=buf.getvalue(),
                        file_name=f"{plot_data['name'].replace(' ','_').lower()}.png",
                        mime="image/png", key=f"dl_mpl_{i}")
                    plt.close(plot_data['figure'])
                else:
                    st.plotly_chart(plot_data['figure'])

            if st.button("Clear All Charts"):
                st.session_state.generated_plots = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: MACHINE LEARNING
# ═══════════════════════════════════════════════════════
elif page == "Machine Learning":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Step 03 · Supervised Learning</div>
        <div class="page-title">Machine <span>Learning</span></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.df_loaded:
        st.warning("Please upload a dataset first on the Upload & Clean page.")
    else:
        df = st.session_state['df']

        st.markdown('<div class="g-card"><div class="g-card-title">Model Configuration</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox("Target Column", df.columns)
        with c2:
            problem_type = ""
            if target_col and target_col in df.columns:
                if is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
                    problem_type = "regression"
                    st.info(f"Detected: **Regression**")
                else:
                    problem_type = "classification"
                    st.info(f"Detected: **Classification**")

        classification_models = ("Random Forest","Logistic Regression","XGBoost","CatBoost","LightGBM","Naive Bayes","SVM")
        regression_models     = ("Random Forest","Linear Regression","Ridge Regression","Lasso Regression","XGBoost","CatBoost","LightGBM","SVM")
        available_models = regression_models if problem_type == "regression" else classification_models if problem_type == "classification" else ()
        model_selection  = st.selectbox("Model", available_models)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Train Model"):
            if model_selection:
                with st.spinner("Training…"):
                    metric_value, metric_name = train_model(df, target_col, model_selection)
                if metric_value is not None:
                    st.session_state.model_results = {
                        "model_type": problem_type, "selected_model_name": model_selection,
                        "target_col": target_col, "metric_name": metric_name,
                        "metric_value": metric_value, "tuning_enabled": False
                    }
                    st.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-box"><div class="metric-val">{metric_value:.3f}</div><div class="metric-key">{metric_name}</div></div>
                        <div class="metric-box"><div class="metric-val">{model_selection.split()[0]}</div><div class="metric-key">Model</div></div>
                        <div class="metric-box"><div class="metric-val">{problem_type[:5].upper()}</div><div class="metric-key">Type</div></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Model training failed.")
            else:
                st.warning("Select a model type first.")

        st.markdown('<div class="g-card"><div class="g-card-title">Hyperparameter Tuning</div>', unsafe_allow_html=True)
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning")
        if enable_tuning:
            c3, c4 = st.columns(2)
            with c3:
                tuning_method = st.selectbox("Tuning Method", ("Optuna","GridSearchCV","RandomizedSearchCV"))
            with c4:
                cv_folds = st.slider("CV Folds", 2, 10, 5)

            n_tuning_trials = None; n_iter_random = None
            if tuning_method == "Optuna":
                n_tuning_trials = st.slider("Optuna Trials", 5, 100, 20, 5)
            elif tuning_method == "RandomizedSearchCV":
                n_iter_random = st.slider("Random Iterations", 5, 100, 10)

            if st.button("Train with Tuning"):
                if model_selection:
                    from hyperparameter_tuning import optimize_hyperparameters
                    with st.spinner(f"Running {tuning_method}…"):
                        metric_value, metric_name, best_params, best_model = optimize_hyperparameters(
                            df, target_col, model_selection, problem_type,
                            tuning_method=tuning_method, n_trials=n_tuning_trials,
                            n_iter_random=n_iter_random, cv_folds=cv_folds)
                    if metric_value is not None:
                        st.success(f"Best {metric_name}: {metric_value:.3f}")
                        st.json(best_params)
                        st.session_state.model_results = {
                            "model_type": problem_type, "selected_model_name": model_selection,
                            "target_col": target_col, "metric_name": metric_name,
                            "metric_value": metric_value, "tuning_enabled": True,
                            "tuning_method": tuning_method, "best_params": best_params
                        }
                    else:
                        st.error(f"Tuning with {tuning_method} failed.")
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: CLUSTERING
# ═══════════════════════════════════════════════════════
elif page == "Clustering":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Step 04 · Unsupervised Learning</div>
        <div class="page-title">Cluster <span>Analysis</span></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.df_loaded:
        st.warning("Please upload a dataset first on the Upload & Clean page.")
    else:
        df = st.session_state['df']
        numeric_cols = df.select_dtypes(include='number').columns

        # K-Means
        st.markdown('<div class="g-card"><div class="g-card-title">K-Means Clustering</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            clustering_cols = st.multiselect("Features", numeric_cols,
                default=numeric_cols.tolist() if not numeric_cols.empty else [], key="clustering_cols_select")
        with c2:
            n_clusters = st.slider("Number of Clusters K", 2, 10, 3, key="n_clusters_slider")

        if st.button("Run K-Means", key="kmeans_button"):
            if not clustering_cols:
                st.warning("Select at least one feature.")
            else:
                clustered_df, silhouette_val, _ = perform_kmeans_clustering(df[clustering_cols], n_clusters)
                if clustered_df is not None:
                    st.success(f"K-Means complete — {n_clusters} clusters")
                    if silhouette_val:
                        st.markdown(f'<div class="metric-row"><div class="metric-box"><div class="metric-val">{silhouette_val:.3f}</div><div class="metric-key">Silhouette</div></div><div class="metric-box"><div class="metric-val">{n_clusters}</div><div class="metric-key">Clusters</div></div></div>', unsafe_allow_html=True)
                    st.session_state.clustering_results['K-Means'] = {'features': clustering_cols, 'n_clusters': n_clusters, 'silhouette_score': silhouette_val}
                    if len(clustering_cols) >= 2:
                        fig_c = px.scatter(clustered_df, x=clustering_cols[0], y=clustering_cols[1], color='Cluster',
                            title=f'{clustering_cols[0]} vs {clustering_cols[1]}',
                            template='plotly_dark', color_continuous_scale='Viridis')
                        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # DBSCAN
        st.markdown('<div class="g-card"><div class="g-card-title">DBSCAN Clustering</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            dbscan_cols = st.multiselect("Features", numeric_cols,
                default=numeric_cols.tolist() if not numeric_cols.empty else [], key="dbscan_clustering_cols_select")
            eps_value = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1, key="eps_slider")
        with c4:
            min_samples_value = st.slider("Min Samples", 2, 20, 5, 1, key="min_samples_slider")

        if st.button("Run DBSCAN", key="dbscan_button"):
            if not dbscan_cols:
                st.warning("Select at least one feature.")
            else:
                dbscan_df, dbscan_sil, _ = perform_dbscan_clustering(df[dbscan_cols], eps=eps_value, min_samples=min_samples_value)
                if dbscan_df is not None:
                    n_found   = dbscan_df['Cluster'].nunique() - (1 if -1 in dbscan_df['Cluster'].unique() else 0)
                    noise_pts = (dbscan_df['Cluster'] == -1).sum()
                    st.success(f"DBSCAN complete — {n_found} clusters, {noise_pts} noise points")
                    if dbscan_sil:
                        st.markdown(f'<div class="metric-row"><div class="metric-box"><div class="metric-val">{dbscan_sil:.3f}</div><div class="metric-key">Silhouette</div></div><div class="metric-box"><div class="metric-val">{n_found}</div><div class="metric-key">Clusters</div></div><div class="metric-box"><div class="metric-val">{noise_pts}</div><div class="metric-key">Noise</div></div></div>', unsafe_allow_html=True)
                    st.session_state.clustering_results['DBSCAN'] = {'features': dbscan_cols, 'eps': eps_value, 'min_samples': min_samples_value, 'silhouette_score': dbscan_sil, 'noise_points': noise_pts}
                    if len(dbscan_cols) >= 2:
                        fig_db = px.scatter(dbscan_df, x=dbscan_cols[0], y=dbscan_cols[1], color='Cluster',
                            title=f'DBSCAN: {dbscan_cols[0]} vs {dbscan_cols[1]}',
                            template='plotly_dark', color_continuous_scale='Plasma')
                        st.plotly_chart(fig_db, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE: REPORT
# ═══════════════════════════════════════════════════════
elif page == "Report":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Step 05 · Export</div>
        <div class="page-title">Generate <span>Report</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="g-card"><div class="g-card-title">Report Summary</div>', unsafe_allow_html=True)

    has_data    = st.session_state.df_loaded
    has_plots   = len(st.session_state.generated_plots) > 0
    has_model   = st.session_state.model_results is not None
    has_cluster = len(st.session_state.clustering_results) > 0

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{'✓' if has_data else '✗'}</div><div class="metric-key">Dataset</div></div>
        <div class="metric-box"><div class="metric-val">{len(st.session_state.generated_plots)}</div><div class="metric-key">Charts</div></div>
        <div class="metric-box"><div class="metric-val">{'✓' if has_model else '✗'}</div><div class="metric-key">ML Model</div></div>
        <div class="metric-box"><div class="metric-val">{len(st.session_state.clustering_results)}</div><div class="metric-key">Clusters</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not has_data:
        st.warning("Upload a dataset first to generate a report.")
    else:
        st.markdown('<div class="g-card"><div class="g-card-title">Export PDF</div>', unsafe_allow_html=True)
        if st.button("Generate PDF Report"):
            with st.spinner("Building report…"):
                pdf_output = generate_pdf_report(
                    st.session_state.df_summary,
                    st.session_state.generated_plots,
                    st.session_state.model_results,
                    st.session_state.clustering_results)
            st.download_button("↓ Download PDF Report", data=pdf_output,
                file_name="autods_ai_report.pdf", mime="application/pdf")
            st.success("Report ready!")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown('<div class="footer">AutoDS AI · Autonomous Data Scientist · Built with Streamlit & Python</div>', unsafe_allow_html=True)
ENDOFFILE
echo "Done"

Done

You are out of free messa
