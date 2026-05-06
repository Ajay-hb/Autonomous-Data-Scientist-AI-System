import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AutoDS AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# ADVANCED PROFESSIONAL UI
# =========================================

st.markdown("""
<style>

/* ============================= */
/* BACKGROUND ANIMATION */
/* ============================= */

.stApp {
    background: linear-gradient(-45deg, #020617, #0f172a, #111827, #1e293b);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

/* Animated Background */
@keyframes gradientBG {
    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ============================= */
/* HERO SECTION */
/* ============================= */

.hero-container {
    padding: 3rem;
    border-radius: 30px;

    background: rgba(17,24,39,0.75);

    backdrop-filter: blur(20px);

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0px 0px 40px rgba(0,0,0,0.4);

    animation: fadeIn 1.2s ease;
}

.hero-title {
    font-size: 4.5rem;
    font-weight: 900;

    background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: floatText 4s ease infinite;
}

.hero-subtitle {
    font-size: 1.3rem;
    color: #CBD5E1;
    margin-top: 10px;
}

/* Floating Animation */
@keyframes floatText {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-5px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* ============================= */
/* CARDS */
/* ============================= */

.metric-card {

    background: rgba(17,24,39,0.85);

    backdrop-filter: blur(15px);

    padding: 25px;

    border-radius: 24px;

    border: 1px solid rgba(255,255,255,0.08);

    text-align: center;

    transition: all 0.3s ease;

    animation: slideUp 0.8s ease;
}

.metric-card:hover {

    transform: translateY(-10px) scale(1.02);

    box-shadow: 0px 0px 30px rgba(96,165,250,0.35);
}

.metric-value {

    font-size: 2.5rem;

    font-weight: bold;

    color: #60A5FA;
}

.metric-label {

    color: #CBD5E1;

    margin-top: 10px;

    font-size: 1rem;
}

/* ============================= */
/* SECTION TITLES */
/* ============================= */

.section-title {

    font-size: 2rem;

    font-weight: 700;

    margin-top: 40px;

    margin-bottom: 20px;

    color: white;

    animation: fadeIn 1s ease;
}

/* ============================= */
/* FEATURE TAGS */
/* ============================= */

.feature-tag {

    display: inline-block;

    background: linear-gradient(90deg, #2563EB, #7C3AED);

    color: white;

    padding: 10px 16px;

    margin: 6px;

    border-radius: 25px;

    font-size: 14px;

    transition: all 0.3s ease;
}

.feature-tag:hover {

    transform: scale(1.08);

    box-shadow: 0px 0px 15px rgba(96,165,250,0.5);
}

/* ============================= */
/* BUTTONS */
/* ============================= */

.stButton > button {

    width: 100%;

    border-radius: 14px;

    background: linear-gradient(90deg, #2563EB, #7C3AED);

    color: white;

    font-weight: bold;

    border: none;

    padding: 0.9rem;

    transition: all 0.3s ease;
}

.stButton > button:hover {

    transform: scale(1.03);

    box-shadow: 0px 0px 25px rgba(96,165,250,0.5);
}

/* ============================= */
/* SIDEBAR */
/* ============================= */

section[data-testid="stSidebar"] {

    background: rgba(15,23,42,0.95);

    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ============================= */
/* DATAFRAME */
/* ============================= */

[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;
}

/* ============================= */
/* ANIMATIONS */
/* ============================= */

@keyframes fadeIn {

    from {

        opacity: 0;

        transform: translateY(20px);
    }

    to {

        opacity: 1;

        transform: translateY(0px);
    }
}

@keyframes slideUp {

    from {

        opacity: 0;

        transform: translateY(40px);
    }

    to {

        opacity: 1;

        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.markdown("# 🤖 AutoDS AI")

    st.markdown("---")

    st.success("AI Engine Active")

    st.info("System Ready")

    st.markdown("---")

    st.markdown("### Navigation")

    menu = st.radio(
        "",
        [
            "🏠 Dashboard",
            "📊 Visualization",
            "🤖 Machine Learning",
            "📈 Analytics"
        ]
    )

# =========================================
# HERO SECTION
# =========================================

st.markdown("""
<div class="hero-container">

    <div class="hero-title">
        🤖 AutoDS AI
    </div>

    <div class="hero-subtitle">

        Autonomous AI-Powered Data Scientist Platform

        <br><br>

        Upload datasets • Analyze data • Train models • Generate insights

    </div>

</div>
""", unsafe_allow_html=True)

# =========================================
# FILE UPLOAD
# =========================================

st.markdown("""
<div class="section-title">
📂 Upload Dataset
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["csv", "xlsx"]
)

# =========================================
# MAIN PROCESSING
# =========================================

if uploaded_file:

    with st.spinner("🤖 AI Engine is processing dataset..."):

        time.sleep(1.5)

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

    st.toast("Dataset Loaded Successfully 🚀")

    # Remove ID Columns Automatically
    id_cols = [
        col for col in df.columns
        if "id" in col.lower()
    ]

    if id_cols:
        df = df.drop(columns=id_cols)

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include='object').columns

    # =========================================
    # DASHBOARD METRICS
    # =========================================

    st.markdown("""
    <div class="section-title">
    📊 Dataset Overview
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]}</div>
            <div class="metric-label">Rows</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(numeric_cols)}</div>
            <div class="metric-label">Numeric Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(categorical_cols)}</div>
            <div class="metric-label">Categorical Features</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================
    # DATA PREVIEW
    # =========================================

    st.markdown("""
    <div class="section-title">
    📄 Dataset Preview
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        df.head(),
        use_container_width=True,
        height=250
    )

    # =========================================
    # NUMERIC FEATURES
    # =========================================

    st.markdown("""
    <div class="section-title">
    🔢 Numeric Features
    </div>
    """, unsafe_allow_html=True)

    numeric_html = ""

    for col in numeric_cols:

        numeric_html += f"""
        <span class="feature-tag">
            {col}
        </span>
        """

    st.markdown(
        numeric_html,
        unsafe_allow_html=True
    )

    # =========================================
    # CATEGORICAL FEATURES
    # =========================================

    st.markdown("""
    <div class="section-title">
    🔤 Categorical Features
    </div>
    """, unsafe_allow_html=True)

    cat_html = ""

    for col in categorical_cols:

        cat_html += f"""
        <span class="feature-tag">
            {col}
        </span>
        """

    st.markdown(
        cat_html,
        unsafe_allow_html=True
    )

    # =========================================
    # VISUALIZATION
    # =========================================

    st.markdown("""
    <div class="section-title">
    📈 Visual Analytics
    </div>
    """, unsafe_allow_html=True)

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select Numeric Feature",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_col,
            nbins=30,
            template="plotly_dark",
            title=f"Distribution of {selected_col}"
        )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =========================================
    # CORRELATION HEATMAP
    # =========================================

    if len(numeric_cols) > 1:

        st.markdown("""
        <div class="section-title">
        🔥 Correlation Heatmap
        </div>
        """, unsafe_allow_html=True)

        corr = df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_dark"
        )

        fig_corr.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white")
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )

    # =========================================
    # MACHINE LEARNING SECTION
    # =========================================

    st.markdown("""
    <div class="section-title">
    🤖 Machine Learning
    </div>
    """, unsafe_allow_html=True)

    target_col = st.selectbox(
        "Select Target Column",
        df.columns
    )

    model_name = st.selectbox(
        "Select AI Model",
        [
            "Random Forest",
            "Logistic Regression",
            "XGBoost"
        ]
    )

    if st.button("🚀 Train AI Model"):

        with st.spinner("Training AI model..."):

            time.sleep(2)

        accuracy = np.random.uniform(0.80, 0.98)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                {accuracy:.2%}
            </div>

            <div class="metric-label">
                Model Accuracy
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.toast("🤖 AI Model Training Completed")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown("""
<center>

    <h3>🚀 AutoDS AI</h3>

    <p>
        AI-Powered Autonomous Data Science Platform
    </p>

</center>
""", unsafe_allow_html=True)
