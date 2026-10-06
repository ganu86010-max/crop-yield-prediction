"""Crop Yield Analytics — Flask API and web application."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "crop_yield_dataset.csv"
FEATURES = ["Rainfall", "Temperature", "Fertilizer", "Soil_Quality"]

app = Flask(__name__)
# In production, restrict browser access with ALLOWED_ORIGINS=https://your-frontend.example.
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": allowed_origins.split(",") if allowed_origins != "*" else "*"}})
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["Crop_Yield"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# A small model registry makes the analytics page useful beyond a single prediction.
models = {
    "Polynomial regression": Pipeline([
        ("features", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Random forest": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2),
}
for model in models.values():
    model.fit(X_train, y_train)

best_model_name = max(models, key=lambda name: r2_score(y_test, models[name].predict(X_test)))
best_model = models[best_model_name]


def metrics_for(model):
    prediction = model.predict(X_test)
    return {
        "mae": round(float(mean_absolute_error(y_test, prediction)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 2),
        "r2": round(float(r2_score(y_test, prediction)), 3),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "crop-yield-analytics", "model": best_model_name})


@app.get("/api/summary")
def summary():
    return jsonify({
        "rows": int(len(df)),
        "average_yield": round(float(y.mean()), 1),
        "max_yield": round(float(y.max()), 1),
        "min_yield": round(float(y.min()), 1),
        "average_rainfall": round(float(df.Rainfall.mean()), 1),
        "average_temperature": round(float(df.Temperature.mean()), 1),
        "best_model": best_model_name,
        "metrics": {name: metrics_for(model) for name, model in models.items()},
        "distribution": {
            "labels": ["< 400", "400–699", "700–999", "1,000+"],
            "values": [int((y < 400).sum()), int(((y >= 400) & (y < 700)).sum()), int(((y >= 700) & (y < 1000)).sum()), int((y >= 1000).sum())],
        },
        "trend": [{"rainfall": int(row.Rainfall), "yield": int(row.Crop_Yield)} for row in df.sort_values("Rainfall").itertuples()],
    })


@app.get("/api/records")
def records():
    limit = min(max(request.args.get("limit", 100, type=int), 1), 500)
    return jsonify(df.head(limit).to_dict(orient="records"))


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = {
            "Rainfall": float(payload["rainfall"]),
            "Temperature": float(payload["temperature"]),
            "Fertilizer": float(payload["fertilizer"]),
            "Soil_Quality": float(payload["soil_quality"]),
        }
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide rainfall, temperature, fertilizer, and soil quality values."}), 400
    if values["Rainfall"] < 0 or not -20 <= values["Temperature"] <= 60 or values["Fertilizer"] < 0 or not 0 <= values["Soil_Quality"] <= 10:
        return jsonify({"error": "Use realistic values: temperature -20–60°C, soil quality 0–10, and non-negative inputs."}), 400
    result = float(best_model.predict(pd.DataFrame([values]))[0])
    return jsonify({"prediction": round(max(0, result), 1), "model": best_model_name, "inputs": values})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
