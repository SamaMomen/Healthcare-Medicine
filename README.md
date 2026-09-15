# Breast Cancer Diagnosis – Full ML Project

## Contents of this package

```
breast_cancer_project/
├── data.csv                          # Original dataset
├── breast_cancer_analysis.ipynb      # Main Jupyter Notebook (source code)
├── run_analysis.py                   # Script that runs the full pipeline
├── app.py                            # Streamlit interactive app
├── requirements.txt                  # Dependencies
├── REPORT.md                         # Project report
├── README.md                         # This file
├── models/
│   ├── best_model.pkl                # Tuned SVM model
│   ├── scaler.pkl                    # StandardScaler
│   └── feature_names.pkl             # Feature list
└── figures/                          # All generated plots & result CSVs
```

## Quick Start

```bash
pip install -r requirements.txt
python run_analysis.py          # (re)generate models & figures
streamlit run app.py            # launch the web app
```

Open `breast_cancer_analysis.ipynb` in any Jupyter environment to see the full analysis step by step.

## Results Summary

- **Best model**: SVM (RBF) tuned with C=10, gamma=0.01
- **Test Accuracy**: ~98.2%
- **Test F1-Score**: ~97.6%
- **Test ROC-AUC**: ~99.6%
- **Clustering ARI** (KMeans K=2 vs true labels): ~0.65

## Notes

- All source code is plain text / Python (no binary notebooks with hidden cells).
- Models are saved with `joblib` for the Streamlit app.
- The notebook and the Python script contain the complete reproducible pipeline.
- ....
