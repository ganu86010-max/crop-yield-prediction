# Fieldwise — Crop Yield Analytics

A full-stack agricultural data analytics application built around the crop yield dataset. The project now includes a Flask API, a responsive analytics dashboard, interactive charts, live field records, model comparison, and a prediction workflow.

## What is included

- **Analytics dashboard** — yield KPIs, rainfall/yield relationship chart, yield distribution, and recent records.
- **Prediction API and UI** — enter rainfall, temperature, fertilizer, and soil quality to estimate output in kg/ha.
- **Model evaluation** — compares polynomial regression and random forest on a reproducible 80/20 holdout split using R², MAE, and RMSE.
- **Data endpoints** — JSON endpoints for dashboard summary, records, and predictions.
- **Responsive frontend** — works on desktop, tablet, and mobile without a separate build step.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in a browser.

## API

- `GET /api/summary` — KPIs, model metrics, and chart data
- `GET /api/records?limit=10` — dataset records
- `POST /api/predict` — JSON body: `{"rainfall": 900, "temperature": 25, "fertilizer": 120, "soil_quality": 7}`

## Original analysis script

`cropyieldprediction.py` is kept as a standalone notebook-style polynomial regression analysis. The production application in `app.py` reuses the same CSV data and adds a model registry and web API.
