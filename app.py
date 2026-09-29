"""
Air Quality Prediction - Flask Web Application
Routes:
  GET  /            -> Home / prediction form
  POST /predict     -> Run prediction, return result
  GET  /dashboard   -> Model metrics + visualisation charts
  GET  /about       -> About page
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify


# ── app setup ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__)

MODELS_DIR = os.path.join(BASE, "models")
FEATURES   = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
              "Temperature", "Humidity", "WindSpeed"]

# ── load artefacts (lazy – only once) ─────────────────────────────────────────
_regressor    = None
_classifier   = None
_scaler       = None
_label_encoder= None
_metrics      = None

def load_models():
    global _regressor, _classifier, _scaler, _label_encoder, _metrics

    reg_path = os.path.join(MODELS_DIR, "aqi_regressor.pkl")
    if not os.path.exists(reg_path):
        return False, "Models not found. Please run train_model.py first."

    _regressor     = joblib.load(os.path.join(MODELS_DIR, "aqi_regressor.pkl"))
    _classifier    = joblib.load(os.path.join(MODELS_DIR, "aqi_classifier.pkl"))
    _scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    _label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics = json.load(f)
    else:
        _metrics = {}

    return True, "OK"

# ── AQI helpers ────────────────────────────────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "Good",                          "#2ecc71", "Air quality is satisfactory and poses little or no risk."),
    (51,  100, "Moderate",                      "#f1c40f", "Acceptable quality; some pollutants may be a concern for sensitive people."),
    (101, 150, "Unhealthy for Sensitive Groups","#e67e22", "Members of sensitive groups may experience health effects."),
    (151, 200, "Unhealthy",                     "#e74c3c", "Everyone may begin to experience health effects."),
    (201, 300, "Very Unhealthy",                "#8e44ad", "Health alert: everyone may experience more serious effects."),
    (301, 500, "Hazardous",                     "#2c3e50", "Health warning of emergency conditions; entire population is affected."),
]

def aqi_info(aqi_value):
    for lo, hi, label, color, msg in AQI_LEVELS:
        if lo <= aqi_value <= hi:
            return label, color, msg
    return "Hazardous", "#2c3e50", "Extreme pollution level."

def health_recommendations(category):
    recs = {
        "Good": [
            "Enjoy outdoor activities freely.",
            "No special precautions needed.",
            "Great day for exercise outdoors.",
        ],
        "Moderate": [
            "Sensitive individuals should limit prolonged outdoor exertion.",
            "Keep windows open for ventilation.",
            "Monitor air quality updates.",
        ],
        "Unhealthy for Sensitive Groups": [
            "People with respiratory or heart disease should limit outdoor activity.",
            "Children and elderly should reduce prolonged exertion.",
            "Consider wearing an N95 mask outdoors.",
        ],
        "Unhealthy": [
            "Everyone should reduce prolonged outdoor exertion.",
            "Avoid outdoor exercise; exercise indoors instead.",
            "Keep windows closed; use air purifiers indoors.",
        ],
        "Very Unhealthy": [
            "Avoid all outdoor activities.",
            "Stay indoors with air purifiers running.",
            "Wear N95/KN95 mask if going outside is unavoidable.",
            "Seek medical attention if experiencing symptoms.",
        ],
        "Hazardous": [
            "Emergency conditions – stay indoors.",
            "Seal windows and doors if possible.",
            "Avoid any outdoor exposure.",
            "Follow local health authority advisories.",
        ],
    }
    return recs.get(category, ["Monitor air quality closely."])


# ── routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    ok, msg = load_models()
    return render_template("index.html", models_ready=ok, error_msg=msg if not ok else None)


@app.route("/predict", methods=["POST"])
def predict():
    ok, msg = load_models()
    if not ok:
        return jsonify({"error": msg}), 500

    try:
        values = [float(request.form.get(f, 0)) for f in FEATURES]
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X = pd.DataFrame([dict(zip(FEATURES, values))])
    X_scaled = _scaler.transform(X)

    aqi_pred      = int(round(float(_regressor.predict(X_scaled)[0])))
    aqi_pred      = max(0, min(500, aqi_pred))
    cat_enc       = _classifier.predict(X_scaled)[0]
    category      = _label_encoder.inverse_transform([cat_enc])[0]
    label, color, desc = aqi_info(aqi_pred)
    recommendations    = health_recommendations(category)

    # confidence / probability for the predicted class
    proba = _classifier.predict_proba(X_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)

    result = {
        "aqi":             aqi_pred,
        "category":        category,
        "color":           color,
        "description":     desc,
        "recommendations": recommendations,
        "confidence":      confidence,
        "inputs": dict(zip(FEATURES, values)),
    }
    return jsonify(result)


@app.route("/dashboard")
def dashboard():
    ok, msg = load_models()
    plots_dir = os.path.join(BASE, "static", "plots")
    plots_exist = os.path.isdir(plots_dir) and len(os.listdir(plots_dir)) > 0
    metrics = _metrics if ok else {}
    return render_template("dashboard.html",
                           metrics=metrics,
                           plots_exist=plots_exist,
                           models_ready=ok)


@app.route("/about")
def about():
    return render_template("about.html")


# ── run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Air Quality Prediction App …")
    print("Visit http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
