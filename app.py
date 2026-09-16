"""
Streamlit App - Breast Cancer Diagnosis Prediction
Redesigned UI inspired by a minimal editorial / clinical workbench style.
ML logic and saved .pkl artifacts are kept unchanged.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title=" Breast Cancer Diagnosis Prediction",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

    :root {
        --paper: #f5f2ea;
        --paper-2: #eeeadf;
        --ink: #252522;
        --muted: #77736a;
        --line: #d8d2c4;
        --soft: #e7e1d4;
        --accent: #6f6a5e;
        --dark: #30332f;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', Arial, sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: var(--paper);
    }

    [data-testid="stHeader"] {
        background: var(--paper);
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .main .block-container {
        max-width: 1500px;
        padding: 3.2rem 4rem 4rem 4rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #f8f5ee;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] > div {
        padding: 2.2rem 1.5rem 1.5rem 1.5rem;
    }

    .brand {
        padding: 0.3rem 0.6rem 2.4rem 0.6rem;
    }

    .brand-name {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2rem;
        letter-spacing: -0.03em;
        color: var(--ink);
        margin: 0;
    }

    .brand-sub {
        margin-top: 0.45rem;
        color: #8b867b;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        line-height: 1.6;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .side-stat {
        color: #817c71;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        line-height: 1.8;
        margin: 2.3rem 0.6rem 1.3rem 0.6rem;
    }

    /* Hide the radio circle but keep navigation usable */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.25rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 16px;
        padding: 0.72rem 0.75rem !important;
        margin: 0 !important;
        transition: 0.15s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #eee9dc;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: #e4dece;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 0.9rem;
        color: #555249;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: var(--ink);
        font-weight: 600;
    }

    .nav-number {
        color: #918b80;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        margin-right: 0.7rem;
    }

    /* ---------- Typography ---------- */
    .eyebrow {
        color: #898378;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.7rem;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--ink) !important;
        font-weight: 500 !important;
        letter-spacing: -0.025em !important;
    }

    h1 {
        font-size: clamp(2.4rem, 4vw, 4.2rem) !important;
        line-height: 1.03 !important;
    }

    h2 {
        font-size: 2rem !important;
    }

    h3 {
        font-size: 1.35rem !important;
    }

    p, li {
        color: #59564f;
    }

    .mono {
        font-family: 'DM Mono', monospace;
    }

    /* ---------- Cards ---------- */
    .card {
        background: rgba(255, 253, 248, 0.72);
        border: 1px solid var(--line);
        border-radius: 26px;
        padding: 1.55rem 1.7rem;
        box-shadow: 0 7px 25px rgba(70, 65, 55, 0.035);
    }

    .stat-card {
        background: #eee9dc;
        border: 1px solid #d9d1c0;
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        min-height: 105px;
    }

    .stat-label {
        color: #827d72;
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .stat-value {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.65rem;
        margin-top: 0.3rem;
    }

    .divider {
        height: 1px;
        background: var(--line);
        margin: 1.4rem 0;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: var(--dark) !important;
        color: #f8f5ee !important;
        border: 1px solid var(--dark) !important;
        border-radius: 14px !important;
        min-height: 3rem;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background: #454842 !important;
        border-color: #454842 !important;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="input"] {
        background: #fbf9f3 !important;
        border: 1px solid #d7d1c4 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div {
        background: #fbf9f3 !important;
        border-color: #d7d1c4 !important;
        border-radius: 10px !important;
    }

    label, .stNumberInput label, .stSelectbox label {
        color: #68635a !important;
        font-size: 0.78rem !important;
    }

    /* ---------- Tables ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
    }

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid var(--line);
    }

    /* ---------- Hide Streamlit branding ---------- */
    #MainMenu, footer {
        visibility: hidden;
    }

    /* ---------- Mobile ---------- */
    @media (max-width: 900px) {
        .main .block-container {
            padding: 2rem 1.2rem;
        }
        h1 {
            font-size: 2.5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    return df


model, scaler, feature_names = load_artifacts()
df = load_data()


# =========================================================
# HELPERS
# =========================================================
def section_title(kicker, title, description=None):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if description:
        st.markdown(
            f'<p style="max-width:760px; font-size:1rem; line-height:1.7;">{description}</p>',
            unsafe_allow_html=True,
        )


def stat(label, value):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_clean_bar_chart(values, labels, title):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#fffdf8")
    ax.set_facecolor("#fffdf8")
    bars = ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_title(title, loc="left", fontsize=14, pad=15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#d8d2c4")
    ax.tick_params(axis="y", length=0, colors="#59564f")
    ax.tick_params(axis="x", colors="#817c71")
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontsize=9,
            color="#59564f",
        )
    plt.tight_layout()
    return fig


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-name"> Breast Cancer Diagnosis Prediction</div>
      
    </div>
    """,
    unsafe_allow_html=True,
)

nav_options = [
    "01  Overview",
    "02  Data",
    "03  Models",
    "04  Cluster",
    "05  Predict",
    "06  About",
]

page = st.sidebar.radio(
    "Navigation",
    nav_options,
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f"""
    <div class="side-stat">
        {len(df)} cases<br>
        {len(feature_names)} features
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 01 OVERVIEW
# =========================================================
if page.startswith("01"):
    section_title(
        "01 / Overview",
        "A quieter way to read the model.",
        "An interactive machine-learning workbench for the Breast Cancer Wisconsin (Diagnostic) dataset.",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat("Cases", f"{len(df):,}")
    with c2:
        stat("Features", f"{len(feature_names)}")
    with c3:
        stat("Best model", "RBF SVM")
    with c4:
        stat("ROC-AUC", "99.6%")

    st.write("")

    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">Project pipeline</div>
                <h2>From raw measurements to a model decision.</h2>
                <p>
                    
  1. Data Cleaning & EDA
  2. Feature Simplification (Scaling + PCA)
  3. Unsupervised Learning (Clustering)
  4. Supervised Learning (Multiple Classifiers)
  5. Model Selection & Evaluation
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        counts = df["diagnosis"].value_counts()
        fig = make_clean_bar_chart(
            counts.values,
            counts.index.tolist(),
            "Diagnosis distribution",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.write("")

    st.markdown(
        """
        <div class="card">
            <div class="eyebrow">Current champion</div>
            <h2>Tuned SVM with RBF kernel</h2>
            <p>
                Selected after comparing multiple classification models and tuning
                their hyperparameters. The saved model is loaded directly from
                <span class="mono">models/best_model.pkl</span>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 02 DATA
# =========================================================
elif page.startswith("02"):
    section_title(
        "02 / Data",
        "The dataset, without the noise.",
        "A quick view of the 569-case Wisconsin Diagnostic dataset and its 30 numeric features.",
    )

    counts = df["diagnosis"].value_counts()

    c1, c2, c3 = st.columns(3)
    with c1:
        stat("Total samples", f"{len(df):,}")
    with c2:
        stat("Benign", f"{counts.get('B', 0):,}")
    with c3:
        stat("Malignant", f"{counts.get('M', 0):,}")

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Sample records")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    left, right = st.columns(2)

    with left:
        fig = make_clean_bar_chart(
            counts.values,
            counts.index.tolist(),
            "Diagnosis count",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">Feature groups</div>
                <h3>30 measurements</h3>
                <p>Each feature belongs to one of three groups:</p>
                <ul>
                    <li><b>Mean</b> — average cell measurements</li>
                    <li><b>SE</b> — standard error measurements</li>
                    <li><b>Worst</b> — largest / most severe measurements</li>
                </ul>
                <div class="divider"></div>
                <p class="mono">Target: diagnosis → B / M</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 03 MODELS
# =========================================================
elif page.startswith("03"):
    section_title(
        "03 / Models",
        "Which model earned the lead?",
        "Comparison of the trained classifiers and the tuned results saved by the analysis pipeline.",
    )

    results_path = os.path.join(BASE_DIR, "figures", "model_results.csv")
    tuned_path = os.path.join(BASE_DIR, "figures", "tuned_results.csv")

    if os.path.exists(results_path):
        res = pd.read_csv(results_path)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Base models")
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if os.path.exists(tuned_path):
        tuned = pd.read_csv(tuned_path)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Tuned models")
        st.dataframe(tuned, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    for img_name, title in [
        ("model_comparison.png", "Performance comparison"),
        ("roc_curves.png", "ROC curves"),
        ("confusion_matrix.png", "Confusion matrix"),
    ]:
        img_path = os.path.join(BASE_DIR, "figures", img_name)
        if os.path.exists(img_path):
            st.markdown(
                f'<div class="eyebrow">{title}</div>',
                unsafe_allow_html=True,
            )
            st.image(img_path, use_container_width=True)


# =========================================================
# 04 CLUSTER
# =========================================================
elif page.startswith("04"):
    section_title(
        "04 / Cluster",
        "Looking for structure without labels.",
        "KMeans clustering was applied after scaling, while PCA was used to simplify the feature space for analysis.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        stat("Clusters", "K = 2")
    with c2:
        stat("Adjusted Rand", "≈ 0.65")
    with c3:
        stat("Silhouette", "≈ 0.34")

    st.write("")

    for img_name, title in [
        ("clustering_pca2d.png", "Clusters vs true labels · PCA 2D"),
        ("clustering_elbow.png", "Elbow & silhouette"),
        ("pca_variance.png", "PCA explained variance"),
    ]:
        img_path = os.path.join(BASE_DIR, "figures", img_name)
        if os.path.exists(img_path):
            st.markdown(
                f'<div class="card"><h3>{title}</h3>',
                unsafe_allow_html=True,
            )
            st.image(img_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")


# =========================================================
# 05 PREDICT
# =========================================================
elif page.startswith("05"):
    section_title(
        "05 / Predict",
        "Run a patient profile through the model.",
        "Enter measurements manually or load a real sample from the dataset. This interface is for educational demonstration only.",
    )

    mode = st.radio(
        "Input mode",
        ["Manual entry", "Sample from dataset"],
        horizontal=True,
    )

    input_data = {}
    true_label = None

    if mode == "Sample from dataset":
        idx = st.number_input(
            "Dataset row",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
        )
        sample = df.iloc[idx]
        true_label = sample["diagnosis"]

        st.markdown(
            f"""
            <div class="card">
                <div class="eyebrow">Selected sample</div>
                <h3>Row {idx}</h3>
                <p class="mono">True label: {true_label}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for feat in feature_names:
            input_data[feat] = float(sample[feat])

    else:
        mean_feats = [f for f in feature_names if f.endswith("_mean")]
        se_feats = [f for f in feature_names if f.endswith("_se")]
        worst_feats = [f for f in feature_names if f.endswith("_worst")]

        for group_name, group_features in [
            ("Mean measurements", mean_feats),
            ("Standard error measurements", se_feats),
            ("Worst measurements", worst_feats),
        ]:
            st.markdown(
                f'<div class="eyebrow">{group_name}</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(3)
            for i, feat in enumerate(group_features):
                with cols[i % 3]:
                    default = float(df[feat].median())
                    input_data[feat] = st.number_input(
                        feat,
                        value=default,
                        format="%.5f",
                        key=f"predict_{feat}",
                    )

            st.write("")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("Run prediction", type="primary", use_container_width=True):
        X_input = pd.DataFrame([input_data])[feature_names]
        X_scaled = scaler.transform(X_input)

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        label = "Malignant" if pred == 1 else "Benign"

        st.write("")
        st.markdown(
            f"""
            <div class="card">
                <div class="eyebrow">Model output</div>
                <h1 style="font-size:3rem !important;">{label}</h1>
                <p>
                    The result is generated by the saved tuned SVM model.
                    It should not be interpreted as a medical diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        c1, c2, c3 = st.columns(3)
        with c1:
            stat("Prediction", label)
        with c2:
            stat("Benign probability", f"{proba[0] * 100:.1f}%")
        with c3:
            stat("Malignant probability", f"{proba[1] * 100:.1f}%")

        st.write("")

        fig, ax = plt.subplots(figsize=(9, 2.4))
        fig.patch.set_facecolor("#fffdf8")
        ax.set_facecolor("#fffdf8")

        values = [proba[0] * 100, proba[1] * 100]
        labels = ["Benign", "Malignant"]

        bars = ax.barh(labels, values, height=0.42)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#d8d2c4")
        ax.tick_params(axis="y", length=0)

        for bar, value in zip(bars, values):
            ax.text(
                min(value + 1.2, 96),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if mode == "Sample from dataset":
            correct = (
                (pred == 1 and true_label == "M")
                or (pred == 0 and true_label == "B")
            )

            if correct:
                st.success("Prediction matches the dataset label.")
            else:
                st.error("Prediction does not match the dataset label.")


# =========================================================
# 06 ABOUT
# =========================================================
elif page.startswith("06"):
    section_title(
        "06 / About",
        "Inside the workbench.",
        "Project notes, technology stack and the reproducible pipeline behind the interface.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">Dataset</div>
                <h2>Wisconsin Diagnostic</h2>
                <p>
                    569 samples with 30 numeric features used for binary
                    classification of benign and malignant cases.
                </p>

                <div class="divider"></div>

                <div class="eyebrow">Pipeline</div>
                <p>
                    Cleaning → EDA → Scaling → PCA → Clustering →
                    Multiple Classifiers → Hyperparameter Tuning
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="eyebrow">Best model</div>
                <h2>SVM · RBF kernel</h2>
                <p class="mono">C = 10 · gamma = 0.01</p>

                <div class="divider"></div>

                <div class="eyebrow">Tech stack</div>
                <p>
                    Python · Pandas · NumPy · scikit-learn ·
                    Matplotlib · Streamlit · Joblib
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        """
        <div class="card">
            <div class="eyebrow">Saved artifacts</div>
            <p class="mono">
                models/best_model.pkl<br>
                models/scaler.pkl<br>
                models/feature_names.pkl
            </p>

            <div class="divider"></div>

            <div class="eyebrow">Run locally</div>
            <p class="mono">
                pip install -r requirements.txt<br>
                streamlit run app.py
            </p>

            <div class="divider"></div>

            <p>
                <b>Educational disclaimer:</b>
                This application is a machine-learning project demonstration
                and is not a medical diagnosis system.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
