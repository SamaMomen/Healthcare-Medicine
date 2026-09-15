#!/usr/bin/env python3
"""
Breast Cancer Wisconsin Diagnostic - Full Analysis Script
Generates all plots, trains models, saves best model.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    silhouette_score, adjusted_rand_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import joblib
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv('data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:\n{df['diagnosis'].value_counts()}")

print("\n" + "=" * 60)
print("2. CLEANING")
print("=" * 60)

df_clean = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
le = LabelEncoder()
df_clean['diagnosis'] = le.fit_transform(df_clean['diagnosis'])  # B=0, M=1
print(f"After cleaning: {df_clean.shape}")
print(f"Missing: {df_clean.isnull().sum().sum()}")
print(f"Diagnosis (0=B, 1=M):\n{df_clean['diagnosis'].value_counts()}")

print("\n" + "=" * 60)
print("3. EDA")
print("=" * 60)

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df_clean['diagnosis'].value_counts()
axes[0].bar(['Benign (0)', 'Malignant (1)'], counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Diagnosis Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha='center')
axes[1].pie(counts.values, labels=['Benign', 'Malignant'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Diagnosis Percentage')
plt.tight_layout()
plt.savefig('figures/eda_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eda_target.png")

# Correlation mean features
mean_cols = [c for c in df_clean.columns if 'mean' in c]
plt.figure(figsize=(12, 10))
corr = df_clean[mean_cols + ['diagnosis']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
plt.title('Correlation Heatmap - Mean Features + Diagnosis')
plt.tight_layout()
plt.savefig('figures/eda_corr_mean.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eda_corr_mean.png")

# Top correlations
corr_with_target = df_clean.corr()['diagnosis'].drop('diagnosis').abs().sort_values(ascending=False)
print("Top 10 features correlated with diagnosis:")
print(corr_with_target.head(10))

plt.figure(figsize=(10, 8))
corr_with_target.head(15).plot(kind='barh', color='steelblue')
plt.xlabel('Absolute Correlation with Diagnosis')
plt.title('Top 15 Features by Correlation with Target')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/eda_feature_importance_corr.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eda_feature_importance_corr.png")

# Distributions
top_features = corr_with_target.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for i, feat in enumerate(top_features):
    for label, color, name in [(0, '#2ecc71', 'Benign'), (1, '#e74c3c', 'Malignant')]:
        subset = df_clean[df_clean['diagnosis'] == label][feat]
        axes[i].hist(subset, bins=25, alpha=0.6, color=color, label=name)
    axes[i].set_title(feat)
    axes[i].legend()
plt.suptitle('Distribution of Top Features by Diagnosis', fontsize=14)
plt.tight_layout()
plt.savefig('figures/eda_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/eda_distributions.png")

print("\n" + "=" * 60)
print("4. SCALING + PCA")
print("=" * 60)

X = df_clean.drop('diagnosis', axis=1)
y = df_clean['diagnosis']
feature_names = list(X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Individual Explained Variance')
axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, marker='o', color='darkorange')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
plt.tight_layout()
plt.savefig('figures/pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/pca_variance.png")

n_95 = int(np.argmax(cumsum_var >= 0.95) + 1)
n_99 = int(np.argmax(cumsum_var >= 0.99) + 1)
print(f"Components for 95% variance: {n_95}")
print(f"Components for 99% variance: {n_99}")

pca_final = PCA(n_components=n_95)
X_pca_reduced = pca_final.fit_transform(X_scaled)
print(f"Original features: {X.shape[1]} -> PCA 95%: {X_pca_reduced.shape[1]}")

print("\n" + "=" * 60)
print("5. CLUSTERING")
print("=" * 60)

inertias = []
silhouettes = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[1].plot(K_range, silhouettes, 'go-')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K')
plt.tight_layout()
plt.savefig('figures/clustering_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/clustering_elbow.png")
print("Silhouette scores:", dict(zip(K_range, np.round(silhouettes, 3))))

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
ari = adjusted_rand_score(y, cluster_labels)
sil = silhouette_score(X_scaled, cluster_labels)
print(f"KMeans K=2 | ARI: {ari:.4f} | Silhouette: {sil:.4f}")
print("Cluster vs True:")
print(pd.crosstab(pd.Series(y, name='True'), pd.Series(cluster_labels, name='Cluster')))

# PCA 2D viz
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='RdYlGn_r', alpha=0.7, edgecolors='k', linewidth=0.3)
axes[0].set_title('True Diagnosis (PCA 2D)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
plt.colorbar(scatter1, ax=axes[0], label='Diagnosis (0=B, 1=M)')
scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.3)
axes[1].set_title('KMeans Clusters (K=2)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')
plt.tight_layout()
plt.savefig('figures/clustering_pca2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/clustering_pca2d.png")
print(f"PCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.3f}")

agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(X_scaled)
print(f"Agglomerative | ARI: {adjusted_rand_score(y, agg_labels):.4f} | Sil: {silhouette_score(X_scaled, agg_labels):.4f}")

print("\n" + "=" * 60)
print("6. SUPERVISED LEARNING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
trained_models = {}

print("\nTraining base models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    cv_scores = cross_val_score(model, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1')
    results.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC-AUC': auc, 'CV F1 (mean)': cv_scores.mean(), 'CV F1 (std)': cv_scores.std()
    })
    trained_models[name] = model
    print(f"{name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | CV-F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)
print("\n=== Ranked Results ===")
print(results_df.to_string())
results_df.to_csv('figures/model_results.csv', index=False)

# Comparison plot
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(results_df))
width = 0.15
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, results_df[metric], width, label=metric)
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0.85, 1.02)
plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/model_comparison.png")

best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\nBest base model: {best_model_name}")

y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_best, target_names=['Benign', 'Malignant']))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/confusion_matrix.png")

# ROC curves
plt.figure(figsize=(10, 7))
for name, model in trained_models.items():
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/roc_curves.png")

print("\n" + "=" * 60)
print("7. HYPERPARAMETER TUNING")
print("=" * 60)

param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=2000, random_state=42),
        'params': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
    },
    'SVM (RBF)': {
        'model': SVC(probability=True, random_state=42),
        'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]}
    }
}

tuned_results = []
best_tuned_models = {}

for name, config in param_grids.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(
        config['model'], config['params'],
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    tuned_results.append({
        'Model': name + ' (Tuned)',
        'Best Params': str(grid.best_params_),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })
    best_tuned_models[name] = best
    print(f"  Best params: {grid.best_params_}")
    print(f"  Test F1: {f1_score(y_test, y_pred):.4f}")

tuned_df = pd.DataFrame(tuned_results).sort_values('F1-Score', ascending=False).reset_index(drop=True)
print("\n=== Tuned Models Ranking ===")
print(tuned_df.to_string())
tuned_df.to_csv('figures/tuned_results.csv', index=False)

final_best_name = tuned_df.iloc[0]['Model'].replace(' (Tuned)', '')
final_best_model = best_tuned_models[final_best_name]
print(f"\n=== FINAL BEST MODEL: {final_best_name} ===")
print(tuned_df.iloc[0].to_string())

# Save
joblib.dump(final_best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')
print("\nSaved models/best_model.pkl, scaler.pkl, feature_names.pkl")

# Feature importance
if hasattr(final_best_model, 'feature_importances_'):
    importances = final_best_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    feat_imp.head(15).plot(kind='barh', color='teal')
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importances - {final_best_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/feature_importance.png")
    print(feat_imp.head(10))
elif hasattr(final_best_model, 'coef_'):
    coefs = np.abs(final_best_model.coef_[0])
    feat_imp = pd.Series(coefs, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    feat_imp.head(15).plot(kind='barh', color='teal')
    plt.xlabel('|Coefficient|')
    plt.title(f'Top 15 Feature Coefficients - {final_best_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/feature_importance.png")
    print(feat_imp.head(10))

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"Dataset: Breast Cancer Wisconsin Diagnostic")
print(f"Samples: {len(df_clean)} | Features: {X.shape[1]}")
print(f"Classes: Benign={sum(y==0)}, Malignant={sum(y==1)}")
print(f"PCA: {n_95} components explain 95% variance")
print(f"Clustering ARI: {ari:.4f} | Silhouette: {sil:.4f}")
print(f"Best Model: {final_best_name}")
print(f"  Accuracy : {tuned_df.iloc[0]['Accuracy']:.4f}")
print(f"  Precision: {tuned_df.iloc[0]['Precision']:.4f}")
print(f"  Recall   : {tuned_df.iloc[0]['Recall']:.4f}")
print(f"  F1-Score : {tuned_df.iloc[0]['F1-Score']:.4f}")
print(f"  ROC-AUC  : {tuned_df.iloc[0]['ROC-AUC']:.4f}")
print("=" * 60)
print("DONE!")
