# Healthcare-Medicine
# 🫁 LungGuard AI - Machine Learning Pipeline

End-to-end Machine Learning project for Lung Cancer Detection, Staging, and Pattern Discovery using clinical, demographic, and genomic dataset (**LungCanc2024**).

---

## 📌 Project Architecture & Pipeline

1. **Data Loading & Cleaning / Structure Validation**
2. **Exploratory Data Analysis (EDA)**
3. **Dimensionality Reduction (PCA)**
4. **Unsupervised Learning** - Clustering & Patient Segmentation
5. **Supervised Learning**:
   * **Model 1**: Cancer Presence Classification (`0` vs `1`)
   * **Model 2**: Cancer Staging Classification (`Stage I`, `Stage II`, `Stage III`, `Stage IV`)
6. **Model Evaluation & Persistence**

---

## 🎯 Model 2 — Cancer Staging (`cancer_stage`)

> **Target Population:** Filtered strictly for diagnosed patients where `cancer_presence = 1` (~144,500 records out of ~289k dataset).

### 🧬 Features Used
In addition to general clinical and demographic features, **Model 2** incorporates key genomic markers:
* `EGFR_mutation_status`
* `KRAS_mutation_status`
* `ALK_fusion_status`
* `PD-L1_expression_level`
* `tumor_mutational_burden`

### ⚙️ Preprocessing & Engineering
* **Numerical Scaling:** `StandardScaler` applied to continuous variables.
* **Categorical Encoding:** `OneHotEncoder` applied to features (`patient_gender`, `smoking_history`, `tumor_location`).
* **Pipeline Integration:** Combined using `ColumnTransformer`.
* **Data Splitting:** 80/20 Train/Test split with `stratification` to preserve target class proportions across all stages.

### 📊 Model Selection & Performance
Tested multiple multi-class algorithms to predict cancer stage:
* Decision Tree Classifier
* Random Forest Classifier *(Selected Best Model)*
* Extra Trees Classifier
* XGBoost Classifier

* **Optimal Model:** **Random Forest Classifier** achieved the highest relative accuracy (~29.96%).
* **Performance Context:** The performance aligns with expected boundaries for the synthetic nature of the dataset while maintaining superior predictive power over random selection (25%).

---

## 🛠️ File Structure

```text
├── 04_cancer_staging_model.ipynb    # Model 2 Notebook (Omar)
├── cancer_staging_model.joblib      # Saved Random Forest Model (Local)
├── staging_scaler.joblib            # Fitted Scaler Pipeline (Local)
├── staging_label_encoder.joblib     # Label Encoder for Target Stages (Local)
└── README.md                        # Documentation
