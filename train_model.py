"""
Air Quality Prediction - Model Training Script
Trains two models:
  1. Random Forest Regressor  -> predicts AQI value
  2. Random Forest Classifier -> predicts AQI Category
Saves models, scaler, and evaluation plots to /models and /static
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score)
import joblib

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "air_quality.csv")
MODELS = os.path.join(BASE, "models")
PLOTS  = os.path.join(BASE, "static", "plots")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(PLOTS,  exist_ok=True)

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
            "Temperature", "Humidity", "WindSpeed"]

# ── 1. Load / generate data ────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA):
        print("Dataset not found – generating …")
        sys.path.insert(0, os.path.join(BASE, "data"))
        from generate_dataset import generate_data
        df = generate_data()
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")
    return df

# ── 2. Pre-process ─────────────────────────────────────────────────────────────
def preprocess(df):
    df = df.dropna()
    X = df[FEATURES].copy()
    y_reg = df["AQI"].copy()
    y_cls = df["Category"].copy()

    le = LabelEncoder()
    y_cls_enc = le.fit_transform(y_cls)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_reg, y_cls_enc, le, scaler

# ── 3. Train regressor ─────────────────────────────────────────────────────────
def train_regressor(X_tr, X_te, y_tr, y_te):
    print("\n── Training AQI Regressor ──")
    model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)

    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    return model, preds, metrics

# ── 4. Train classifier ────────────────────────────────────────────────────────
def train_classifier(X_tr, X_te, y_tr, y_te, le):
    print("\n── Training AQI Category Classifier ──")
    model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    acc = accuracy_score(y_te, preds)
    print(f"  Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_te, preds, target_names=le.classes_))

    metrics = {"Accuracy": round(acc * 100, 2)}
    return model, preds, metrics

# ── 5. Plots ───────────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_te, preds):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_te, preds, alpha=0.3, color="#4C72B0", s=10)
    lims = [min(y_te.min(), preds.min()), max(y_te.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Actual vs Predicted AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "actual_vs_predicted.png"), dpi=120)
    plt.close(fig)

def plot_feature_importance(model, features):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Feature Importance (AQI Regressor)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "feature_importance.png"), dpi=120)
    plt.close(fig)

def plot_confusion_matrix(y_te, preds, classes):
    cm = confusion_matrix(y_te, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix – AQI Category")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

def plot_aqi_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # histogram
    axes[0].hist(df["AQI"], bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_title("AQI Distribution")
    axes[0].set_xlabel("AQI")
    axes[0].set_ylabel("Count")
    # category pie
    cat_counts = df["Category"].value_counts()
    colors = ["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad","#2c3e50"]
    axes[1].pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%",
                colors=colors[:len(cat_counts)], startangle=140)
    axes[1].set_title("AQI Category Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "aqi_distribution.png"), dpi=120)
    plt.close(fig)

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURES + ["AQI"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)

# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    X, y_reg, y_cls_enc, le, scaler = preprocess(df)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_cls_enc, test_size=0.2, random_state=42)

    reg_model,  reg_preds,  reg_metrics  = train_regressor(X_tr, X_te, yr_tr, yr_te)
    cls_model,  cls_preds,  cls_metrics  = train_classifier(X_tr, X_te, yc_tr, yc_te, le)

    # Save models & artefacts
    joblib.dump(reg_model, os.path.join(MODELS, "aqi_regressor.pkl"))
    joblib.dump(cls_model, os.path.join(MODELS, "aqi_classifier.pkl"))
    joblib.dump(scaler,    os.path.join(MODELS, "scaler.pkl"))
    joblib.dump(le,        os.path.join(MODELS, "label_encoder.pkl"))

    all_metrics = {k: float(v) for k, v in {**reg_metrics, **cls_metrics}.items()}
    with open(os.path.join(MODELS, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n── Generating plots ──")
    plot_actual_vs_predicted(yr_te, reg_preds)
    plot_feature_importance(reg_model, FEATURES)
    plot_confusion_matrix(yc_te, cls_preds, le.classes_)
    plot_aqi_distribution(df)
    plot_correlation_heatmap(df)

    print("\n✓ All models and plots saved successfully.")
    print(f"  Models  → {MODELS}")
    print(f"  Plots   → {PLOTS}")
    print(f"\n  Metrics: {all_metrics}")


if __name__ == "__main__":
    main()
