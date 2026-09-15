# Breast Cancer Wisconsin Diagnostic – Machine Learning Project Report

**Date**: September 2026  
**Tools**: Python, scikit-learn, XGBoost, Streamlit, Jupyter/Notebook equivalent

---

## 1. Project Goal

Classify breast tumors as **Malignant (M)** or **Benign (B)** using the Breast Cancer Wisconsin (Diagnostic) dataset.

The full pipeline includes:
1. Data cleaning & structural fixing
2. Exploratory Data Analysis (EDA)
3. Feature simplification (StandardScaler + PCA)
4. Unsupervised learning (Clustering)
5. Supervised learning (Classification) with multiple models
6. Hyperparameter tuning and selection of the best model
7. Interactive Streamlit application

---

## 2. Dataset Overview

| Item | Value |
|------|-------|
| Source | UCI / Kaggle – Breast Cancer Wisconsin Diagnostic |
| Samples | 569 |
| Features | 30 numeric (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension – each with mean, se, worst) |
| Target | `diagnosis` → M (Malignant) or B (Benign) |
| Class balance | Benign: 357 (62.7%) · Malignant: 212 (37.3%) |
| Missing values | 1 empty column (`Unnamed: 32`) + `id` column (dropped) |

After cleaning: **569 rows × 31 columns** (30 features + encoded target).

---

## 3. Data Cleaning

- Dropped `id` (identifier, not useful for prediction)
- Dropped `Unnamed: 32` (completely empty)
- Encoded target with `LabelEncoder`: **B → 0**, **M → 1**
- No remaining missing values
- All features are continuous numeric → suitable for scaling and distance-based methods

---

## 4. Exploratory Data Analysis (Key Findings)

- Strong positive correlation between many size-related features (`radius`, `perimeter`, `area`) and the malignant class.
- Top features by absolute correlation with diagnosis:
  1. `concave points_worst` (0.794)
  2. `perimeter_worst` (0.783)
  3. `concave points_mean` (0.777)
  4. `radius_worst` (0.776)
  5. `perimeter_mean` (0.743)

- Distributions of the top features show clear separation between Benign and Malignant groups (Malignant tends to have higher values).

Figures generated: `eda_target.png`, `eda_corr_mean.png`, `eda_feature_importance_corr.png`, `eda_distributions.png`.

---

## 5. Feature Simplification (Scaling + PCA)

- **StandardScaler** applied to all 30 features (zero mean, unit variance).
- **PCA**:
  - 10 principal components explain **95%** of variance
  - 17 components explain **99%** of variance
- Dimensionality reduced from 30 → 10 while retaining most information (useful for visualization and potential speed-up).

---

## 6. Unsupervised Learning – Clustering

### Method
- K-Means and Agglomerative Clustering on scaled data.
- Elbow method + Silhouette score used to choose K.

### Results (K = 2)

| Metric | K-Means | Agglomerative |
|--------|---------|---------------|
| Adjusted Rand Index (vs true labels) | **0.654** | ~0.65 |
| Silhouette Score | **0.343** | similar |

- Clusters recover a large part of the true Benign/Malignant structure (ARI ≈ 0.65 is reasonably good for this dataset).
- 2-D PCA visualization clearly shows two groups that largely align with the true diagnosis.

Figures: `clustering_elbow.png`, `clustering_pca2d.png`.

---

## 7. Supervised Learning – Classification

### Setup
- Stratified train/test split (80% / 20%), `random_state=42`
- Models evaluated: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM (RBF), KNN, Naive Bayes, XGBoost
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC + 5-fold stratified CV F1

### Base Models Ranking (by F1-Score on test set)

Most models achieved **> 95%** accuracy. Top performers were typically SVM, Random Forest, XGBoost and Logistic Regression.

### Hyperparameter Tuning (GridSearchCV, scoring = F1)

| Model | Best Params | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|-------------|----------|-----------|--------|----------|---------|
| **SVM (RBF)** | C=10, gamma=0.01 | **0.982** | **1.000** | 0.952 | **0.976** | **0.996** |
| Random Forest | n_estimators=100, max_depth=None, min_samples_split=2 | 0.974 | 1.000 | 0.929 | 0.963 | 0.993 |
| XGBoost | lr=0.1, max_depth=7, n_estimators=200 | 0.974 | 1.000 | 0.929 | 0.963 | 0.994 |
| Logistic Regression | C=1, penalty=l2 | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |

**Final selected model: Tuned SVM (RBF)**  
- Highest F1-Score and perfect Precision on the test set.  
- Excellent ROC-AUC (0.996).

Confusion matrix and ROC curves are saved in the `figures/` folder.

---

## 8. Deliverables

| File / Folder | Description |
|---------------|-------------|
| `breast_cancer_analysis.ipynb` | Full analysis notebook (source code visible) |
| `run_analysis.py` | Executable script that reproduces all results & figures |
| `app.py` | Streamlit interactive web application |
| `models/` | `best_model.pkl`, `scaler.pkl`, `feature_names.pkl` |
| `figures/` | All EDA, clustering, model comparison, ROC, confusion matrix plots |
| `data.csv` | Original cleaned dataset |
| `requirements.txt` | Python dependencies |
| `REPORT.md` | This report |

---

## 9. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce analysis (generates figures + models)
python run_analysis.py

# Launch Streamlit app
streamlit run app.py
```

Or open `breast_cancer_analysis.ipynb` in Jupyter / VS Code / Colab.

---

## 10. Conclusions

1. The dataset is clean after removing two useless columns and is well-suited for classification.
2. PCA shows that 10 components capture 95% of the variance → good candidate for dimensionality reduction.
3. Unsupervised clustering (K=2) recovers a substantial portion of the true diagnosis structure (ARI ≈ 0.65).
4. Supervised models achieve very high performance; **tuned SVM (RBF)** is the best with:
   - Accuracy ≈ **98.2%**
   - F1-Score ≈ **97.6%**
   - ROC-AUC ≈ **99.6%**
5. The Streamlit app allows interactive prediction and exploration of the results.

**Disclaimer**: This project is strictly educational. It must **not** be used as a medical diagnostic tool.

---

*End of Report*
