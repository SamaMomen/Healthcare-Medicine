"""
Streamlit App - Breast Cancer Diagnosis Prediction
Uses the best trained SVM model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Breast Cancer Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

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

def main():
    st.title("🩺 Breast Cancer Diagnosis Prediction System")
    st.markdown("""
    **Dataset**: Breast Cancer Wisconsin (Diagnostic)  
    **Best Model**: Tuned SVM (RBF) — Accuracy ≈ 98.2%, F1 ≈ 97.6%, ROC-AUC ≈ 99.6%  
    Use the sidebar to enter feature values or try sample patients.
    """)

    model, scaler, feature_names = load_artifacts()
    df = load_data()

    # Sidebar
    st.sidebar.header("⚙️ Input Features")
    mode = st.sidebar.radio("Input mode", ["Manual entry", "Sample from dataset"])

    input_data = {}

    if mode == "Sample from dataset":
        idx = st.sidebar.number_input("Row index (0-568)", min_value=0, max_value=len(df)-1, value=0)
        sample = df.iloc[idx]
        true_label = sample["diagnosis"]
        st.sidebar.info(f"True diagnosis: **{true_label}**")
        for feat in feature_names:
            input_data[feat] = float(sample[feat])
            st.sidebar.write(f"{feat}: {input_data[feat]:.4f}")
    else:
        # Group features for better UX
        mean_feats = [f for f in feature_names if f.endswith("_mean")]
        se_feats = [f for f in feature_names if f.endswith("_se")]
        worst_feats = [f for f in feature_names if f.endswith("_worst")]

        st.sidebar.subheader("Mean features")
        for feat in mean_feats:
            default = float(df[feat].median())
            input_data[feat] = st.sidebar.number_input(feat, value=default, format="%.5f", key=feat)

        st.sidebar.subheader("SE features")
        for feat in se_feats:
            default = float(df[feat].median())
            input_data[feat] = st.sidebar.number_input(feat, value=default, format="%.5f", key=feat)

        st.sidebar.subheader("Worst features")
        for feat in worst_feats:
            default = float(df[feat].median())
            input_data[feat] = st.sidebar.number_input(feat, value=default, format="%.5f", key=feat)

    # Predict button
    if st.sidebar.button("🔮 Predict", type="primary"):
        X_input = pd.DataFrame([input_data])[feature_names]
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            label = "Malignant 🔴" if pred == 1 else "Benign 🟢"
            st.metric("Prediction", label)
        with col2:
            st.metric("Probability Benign", f"{proba[0]*100:.1f}%")
        with col3:
            st.metric("Probability Malignant", f"{proba[1]*100:.1f}%")

        # Probability bar
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(["Benign", "Malignant"], [proba[0], proba[1]], color=["#2ecc71", "#e74c3c"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Confidence")
        for i, v in enumerate([proba[0], proba[1]]):
            ax.text(v + 0.02, i, f"{v*100:.1f}%", va="center")
        st.pyplot(fig)
        plt.close()

        if mode == "Sample from dataset":
            correct = (pred == 1 and true_label == "M") or (pred == 0 and true_label == "B")
            if correct:
                st.success("✅ Prediction matches the true label!")
            else:
                st.error("❌ Prediction does not match the true label.")

    # Tabs for extra info
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📈 Model Performance", "🔬 Clustering", "ℹ️ About"])

    with tab1:
        st.subheader("Dataset Summary")
        st.write(f"Total samples: **{len(df)}**")
        st.write(df["diagnosis"].value_counts())
        st.dataframe(df.head(10))

        # Simple distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        df["diagnosis"].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"], ax=ax)
        ax.set_title("Diagnosis Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Model Comparison (from training)")
        results_path = os.path.join(BASE_DIR, "figures", "model_results.csv")
        tuned_path = os.path.join(BASE_DIR, "figures", "tuned_results.csv")
        if os.path.exists(results_path):
            res = pd.read_csv(results_path)
            st.write("**Base Models**")
            st.dataframe(res.style.highlight_max(axis=0, subset=["Accuracy", "F1-Score", "ROC-AUC"]))
        if os.path.exists(tuned_path):
            tuned = pd.read_csv(tuned_path)
            st.write("**Tuned Models**")
            st.dataframe(tuned)

        # Show images if available
        for img_name, title in [
            ("model_comparison.png", "Performance Comparison"),
            ("roc_curves.png", "ROC Curves"),
            ("confusion_matrix.png", "Confusion Matrix (best base model)"),
        ]:
            img_path = os.path.join(BASE_DIR, "figures", img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption=title, use_container_width=True)

    with tab3:
        st.subheader("Unsupervised Clustering Results")
        st.markdown("""
        - **KMeans (K=2)** was applied after StandardScaler.
        - Adjusted Rand Index vs true labels ≈ **0.65**
        - Silhouette Score ≈ **0.34**
        - PCA reduced 30 features → 10 components for 95% variance.
        """)
        for img_name, title in [
            ("clustering_pca2d.png", "Clusters vs True Labels (PCA 2D)"),
            ("clustering_elbow.png", "Elbow & Silhouette"),
            ("pca_variance.png", "PCA Explained Variance"),
        ]:
            img_path = os.path.join(BASE_DIR, "figures", img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption=title, use_container_width=True)

    with tab4:
        st.markdown("""
        ### About this project
        - **Data**: UCI Breast Cancer Wisconsin (Diagnostic) – 569 samples, 30 numeric features.
        - **Pipeline**: Cleaning → EDA → Scaling → PCA → Clustering → Multiple Classifiers → Hyperparameter Tuning.
        - **Best model**: SVM with RBF kernel (C=10, gamma=0.01).
        - **Tech stack**: Python, scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn.
        
        ### How to run locally
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```
        
        ### Disclaimer
        This tool is for educational purposes only and is **not** a medical diagnosis system.
        """)

if __name__ == "__main__":
    main()
