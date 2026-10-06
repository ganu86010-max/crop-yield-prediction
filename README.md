# AI Crop Yield Prediction System

A production-ready Flask web application for agricultural analytics and crop **yield** estimation. It uses the existing project dataset (`crop_yield_dataset.csv`) and preserves its original prediction target: `Crop_Yield` in kg/ha.

> **Important model note:** the original repository does not contain a crop-classification model or a crop-name label column. Its ML logic predicts numeric crop yield from `Rainfall`, `Temperature`, `Fertilizer`, and `Soil_Quality`. The application therefore provides yield prediction and does not invent a “most suitable crop” recommendation. A crop recommender requires a trained model and a dataset with a crop-label target.

## Features

- Responsive agriculture-themed dashboard
- Live KPI, yield distribution, and rainfall/yield analytics
- Prediction form with validation and error handling
- Polynomial regression and random forest comparison
- JSON API with CORS support and a health endpoint
- Production server configuration with Gunicorn and platform-provided `PORT`
- Render deployment blueprint included

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env           # optional; export variables in your shell
python app.py
```

Open **[http://localhost:5000](http://localhost:5000)**. The local server binds to `0.0.0.0` and uses `PORT` when provided.

For a production-like local run:

```bash
PORT=5000 gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 --timeout 120 app:app
```

## API

- `GET /health` — deployment health check
- `GET /api/summary` — KPIs, model metrics, and chart data
- `GET /api/records?limit=10` — dataset records
- `POST /api/predict` — JSON body:

```json
{"rainfall": 900, "temperature": 25, "fertilizer": 120, "soil_quality": 7}
```

Example response:

```json
{"prediction": 535.3, "model": "Polynomial regression", "inputs": {"Rainfall": 900, "Temperature": 25, "Fertilizer": 120, "Soil_Quality": 7}}
```

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and select the repository; Render reads `render.yaml`.
3. Set `ALLOWED_ORIGINS` to the deployed application URL, for example `https://ai-crop-yield-prediction.onrender.com`.
4. Deploy. Render supplies `PORT`; Gunicorn serves the app and Render checks `/health`.

The same commands work on Railway or another Python host:

- Build: `pip install -r requirements.txt`
- Start: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app`
- Health check: `/health`

## Existing ML logic

`cropyieldprediction.py` remains available as the original standalone polynomial regression analysis. `app.py` loads the same CSV, uses the same four features, trains the model registry at startup, evaluates on a reproducible holdout split, and exposes the prediction API. No trained `.pkl`/`.joblib` file existed in the original repository, so there is no model artifact to load or preserve.
