# 🫁 LungGuard AI -  ML Project
   Machine Learning for Lung Cancer Detection, Staging & Pattern Discovery
**Project Goal**: End-to-end analysis of a large lung cancer dataset (imaging features + clinical + genomic markers).

**Pipeline**:
1. Data Loading & Cleaning / Structure Validation
2. Exploratory Data Analysis (EDA)
3. Dimensionality Reduction (PCA)
4. Unsupervised Learning - Clustering (patient segmentation)
5. Supervised Learning - Classification (Cancer Presence) & Regression (Survival Time)
6. Multiple Models Comparison & Best Model Selection
7. Model Persistence & Insights

**Dataset**: LungCanC2024 (~289k records, 27 features). Sampled for computational efficiency while preserving distributions.

**Date**: 2026

---

## 💡 Model 2 — Core Idea & Clinical Purpose

### 🎯 Why Cancer Staging?
Determining the **Cancer Stage (Stage I to IV)** is a critical step right after detecting cancer presence. It directly dictates the treatment strategy:
* **Early Stages (I & II):** Typically candidates for localized surgical resection and targeted therapies.
* **Advanced Stages (III & IV):** Require systemic interventions, aggressive chemotherapy, or immunotherapy combinations.

### 🧬 Role of Genomic Markers
Standard clinical features alone aren't always enough to distinguish between early and advanced stages. Model 2 integrates **Genomic & Biomarker Profiling** (`EGFR`, `KRAS`, `ALK`, `PD-L1`, and `Tumor Mutational Burden`) alongside patient history. This allows the model to capture deep biological insights that correlate with tumor progression and invasiveness.

### ⚙️ Methodology & Performance Context
1. **Target Cohort:** Focused exclusively on patients diagnosed with cancer (`cancer_presence = 1`, ~144.5k cases).
2. **Preprocessing:** Numeric scaling via `StandardScaler` and categorical encoding (`OneHotEncoder`) unified under a `ColumnTransformer`.
3. **Stratified Evaluation:** Split 80/20 with class balance preservation to handle multi-class predictions across all 4 stages.
4. **Best Model:** **Random Forest Classifier** selected for its robustness with complex tabular features, achieving **~29.96%** accuracy (exceeding random baseline expectations on synthetic distributions).

---



