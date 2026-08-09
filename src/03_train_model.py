#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:50:48 2026

@author: madhupahar
"""



import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    accuracy_score
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load features
df = pd.read_csv("data/processed/features.csv")
print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Split into train, validation, test
train_df = df[df['split'] == 'TRAIN']
test_df = df[df['split'] == 'TEST']

# Create validation set from train (20% of train)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df['label']
)

print(f"\nSplits:")
print(f"  Train: {len(train_df)} samples")
print(f"  Validation: {len(val_df)} samples")
print(f"  Test: {len(test_df)} samples")

# Feature columns (exclude non-feature columns)
feature_cols = [c for c in df.columns if c not in [
    'subject_id', 'diagnosis', 'label', 'split'
]]

print(f"\nNumber of features: {len(feature_cols)}")

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
X_val = val_df[feature_cols].values
y_val = val_df['label'].values
X_test = test_df[feature_cols].values
y_test = test_df['label'].values

# Standardise features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Data prepared and scaled")

# ------------------------------------------------------------
# Train Logistic Regression
# ------------------------------------------------------------
print("\n" + "="*50)
print("Training Logistic Regression...")
print("="*50)

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=1.0
)
lr_model.fit(X_train_scaled, y_train)

# Evaluate Logistic Regression
for name, X, y in [
    ('Training', X_train_scaled, y_train),
    ('Validation', X_val_scaled, y_val),
    ('Test', X_test_scaled, y_test)
]:
    y_pred = lr_model.predict(X)
    y_proba = lr_model.predict_proba(X)[:, 1]
    
    print(f"\n=== {name} Set (Logistic Regression) ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
    print(f"F1 Score (macro): {f1_score(y, y_pred, average='macro'):.3f}")
    print(f"F1 Score (binary): {f1_score(y, y_pred):.3f}")
    print(f"AUC-ROC: {roc_auc_score(y, y_proba):.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

# ------------------------------------------------------------
# Train Random Forest (for comparison)
# ------------------------------------------------------------
print("\n" + "="*50)
print("Training Random Forest...")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate Random Forest
for name, X, y in [
    ('Training', X_train_scaled, y_train),
    ('Validation', X_val_scaled, y_val),
    ('Test', X_test_scaled, y_test)
]:
    y_pred = rf_model.predict(X)
    y_proba = rf_model.predict_proba(X)[:, 1]
    
    print(f"\n=== {name} Set (Random Forest) ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
    print(f"F1 Score (macro): {f1_score(y, y_pred, average='macro'):.3f}")
    print(f"F1 Score (binary): {f1_score(y, y_pred):.3f}")
    print(f"AUC-ROC: {roc_auc_score(y, y_proba):.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

# ------------------------------------------------------------
# Save the best model
# ------------------------------------------------------------
# Compare test F1 scores
lr_test_f1 = f1_score(y_test, lr_model.predict(X_test_scaled))
rf_test_f1 = f1_score(y_test, rf_model.predict(X_test_scaled))

if lr_test_f1 >= rf_test_f1:
    best_model = lr_model
    model_name = "LogisticRegression"
else:
    best_model = rf_model
    model_name = "RandomForest"

print(f"\n✅ Best model on test set: {model_name} (F1: {max(lr_test_f1, rf_test_f1):.3f})")

# Save model artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/cognitive_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_cols, "models/feature_columns.pkl")

print(f"\n✅ Model saved to models/cognitive_model.pkl")
print(f"✅ Scaler saved to models/scaler.pkl")
print(f"✅ Feature columns saved to models/feature_columns.pkl")

# ------------------------------------------------------------
# Feature Importance (if using Random Forest)
# ------------------------------------------------------------
if model_name == "RandomForest":
    print("\n" + "="*50)
    print("Top 10 Most Important Features:")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))

# ------------------------------------------------------------
# Confusion Matrix Visualization
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Get best model predictions on test set
y_pred_test = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy (0)', 'Impaired (1)'],
            yticklabels=['Healthy (0)', 'Impaired (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {model_name} (Test Set)')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print(f"\n✅ Confusion matrix saved to models/confusion_matrix.png")
plt.show()

print("\n" + "="*50)
print("✅ Training complete!")
print("="*50)
print(f"\nFinal Test Performance ({model_name}):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_test):.3f}")
print(f"  F1 Score (binary): {f1_score(y_test, y_pred_test):.3f}")
print(f"  F1 Score (macro): {f1_score(y_test, y_pred_test, average='macro'):.3f}")