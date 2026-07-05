# ⚙️ Predictive Maintenance System

An end-to-end, production-quality **Predictive Maintenance** platform that predicts machine
failures before they happen, estimates **Remaining Useful Life (RUL)**, flags abnormal
sensor behavior with a **4-model anomaly detection ensemble**, explains every prediction
with **SHAP**, and turns all of it into concrete **maintenance recommendations** — all
wrapped in an interactive **Streamlit control-room dashboard** and a **FastAPI REST API**.

Built with scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow/Keras (LSTM + Autoencoder),
SHAP, Streamlit, FastAPI, and MLflow.

---

## Table of Contents
1. [What this project does](#what-this-project-does)
2. [Architecture](#architecture)
3. [Folder structure](#folder-structure)
4. [Quickstart](#quickstart)
5. [Data](#data)
6. [Module reference](#module-reference)
7. [Model performance](#model-performance)
8. [Dashboard](#dashboard)
9. [REST API](#rest-api)
10. [Testing](#testing)
11. [Deployment guide](#deployment-guide)
12. [Known limitations & next steps](#known-limitations--next-steps)

---

## What this project does

Given raw, noisy, multi-sensor time-series data from a fleet of machines, the system:

1. **Cleans and normalizes** sensor data (missing values, duplicates, outliers).
2. **Engineers 200+ features** per sensor: rolling statistics, EMA, lags, FFT spectral
   features, skew/kurtosis, trend slopes, and cross-sensor correlations.
3. **Trains and compares 5 failure-prediction models** (Random Forest, XGBoost, LightGBM,
   CatBoost, LSTM) and **automatically deploys the best one** by ROC-AUC.
4. **Estimates Remaining Useful Life** (cycles + hours), **failure probability**, and a
   **0-100% health score** per machine.
5. **Detects anomalies** with an ensemble of Isolation Forest, One-Class SVM, Local Outlier
   Factor, and a Keras Autoencoder.
6. **Explains predictions** with SHAP (global feature importance + per-machine breakdowns).
7. **Generates maintenance recommendations**: likely failing component, root cause,
   priority, suggested parts, estimated cost, and estimated downtime.
8. Surfaces all of this in a **dark-themed, industrial-style Streamlit dashboard** and a
   **FastAPI REST API** for programmatic integration.

---

## Architecture

```
Raw Sensor Data (CSV)
        |
        v
+--------------------+   Module 1
|  preprocessing.py   |   missing values . duplicates . outliers . normalization
+---------+-----------+
          v
+-------------------------+  Module 3
| feature_engineering.py  |  rolling stats . EMA . lags . FFT . trend . correlation
+----------+---------------+
           |
    +------+----------------+----------------------+
    v      v                v                       v
+------------------+ +-------------------+ +----------------------+
|failure_prediction | | rul_prediction.py | | anomaly_detection.py |
|  Module 4         | |  Module 5         | |  Module 6            |
| RF/XGB/LGBM/CB/   | | LightGBM regressor| | IsoForest/OCSVM/     |
| LSTM + auto-select| | -> RUL/health score| | LOF/Autoencoder      |
+---------+----------+ +---------+---------+ +-----------+----------+
          |                      |                       |
          +----------+-----------+-----------------------+
                     v
          +-------------------------+   Module 8
          |   explainability.py     |   SHAP global + per-prediction
          +-----------+--------------+
                      v
          +-------------------------+   Module 7
          |  maintenance_engine.py  |   root cause . priority . cost . downtime
          +-----------+--------------+
                      v
         +------------+--------------+
         v                           v
+----------------+          +-------------------+   Module 9
|  app.py         |          |  api/main.py      |
|  Streamlit      |          |  FastAPI REST     |
|  dashboard      |          |  API              |
+----------------+          +-------------------+
```

`train.py` orchestrates modules 1-8 for training (with MLflow tracking).
`predict.py` orchestrates inference using the artifacts `train.py` saved.

---

## Folder structure

```
PredictiveMaintenance/
|-- data/
|   |-- generate_synthetic_data.py      # synthetic CMAPSS/AI4I-schema data generator
|   |-- generate_live_fleet_sample.py   # realistic "machines mid-life" demo snapshot
|   |-- sensor_timeseries.csv           # run-to-failure training data (CMAPSS-style)
|   |-- discrete_failures.csv           # discrete-record failure data (AI4I-style)
|   `-- live_fleet_snapshot.csv         # dashboard demo data (mixed health states)
|-- models/                             # trained model artifacts (created by train.py)
|-- notebooks/                          # (empty scaffold for exploratory analysis)
|-- reports/                            # training summaries, SHAP CSVs, curve data
|-- tests/                              # pytest unit tests (27 tests)
|-- api/
|   `-- main.py                         # FastAPI REST service
|-- app.py                              # Streamlit dashboard (Module 9)
|-- train.py                            # end-to-end training pipeline (CLI)
|-- predict.py                          # end-to-end inference pipeline (CLI + import)
|-- preprocessing.py                    # Module 1: Data Ingestion
|-- feature_engineering.py              # Module 3: Feature Engineering
|-- failure_prediction.py               # Module 4: Failure Prediction (multi-model)
|-- rul_prediction.py                   # Module 5: RUL Prediction
|-- anomaly_detection.py                # Module 6: Anomaly Detection
|-- maintenance_engine.py               # Module 7: Maintenance Recommendation Engine
|-- explainability.py                   # Module 8: Explainable AI (SHAP)
|-- utils.py                            # shared helpers, health-score formula
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- pytest.ini
`-- README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic demo datasets (schema-compatible with real
#    NASA C-MAPSS / AI4I data -- see "Data" section below)
python data/generate_synthetic_data.py
python data/generate_live_fleet_sample.py

# 3. Train all models (takes ~2-3 minutes on CPU)
python train.py

# 4. Run inference and generate a report
python predict.py --data data/live_fleet_snapshot.csv --output reports/latest_predictions.csv

# 5. Launch the dashboard
streamlit run app.py

# 6. (Optional) Launch the REST API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8501` for the dashboard or `http://localhost:8000/docs`
for interactive API docs.

---

## Data

This environment doesn't have live internet access to the original NASA C-MAPSS or
UCI AI4I datasets, so `data/generate_synthetic_data.py` generates a **schema-compatible
synthetic dataset**: run-to-failure multivariate degradation trajectories with realistic
non-linear wear curves, sensor drift, noise, injected missing values/duplicates, and
transient spikes.

**To use real data instead**, drop in a CSV with the same column names
(`unit_id, cycle, op_setting_1..3, <sensor columns>, RUL, failure_within_30_cycles`) --
no code changes required in any downstream module. Recommended real sources:
- NASA C-MAPSS Turbofan Engine Degradation Simulation (NASA Prognostics Data Repository)
- AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)

Two generated files matter:
- **`sensor_timeseries.csv`** -- full run-to-failure trajectories, used for **training**
  (models need to see machines run all the way to failure to learn what "approaching
  failure" looks like).
- **`live_fleet_snapshot.csv`** -- each machine's trajectory **truncated at a random point**
  in its life (30-99%), used for the **dashboard demo**, simulating a realistic fleet of
  machines caught at arbitrary points in their operating life (a healthy mix of risk
  levels, not every machine one cycle from failure).

---

## Module reference

| Module | File | What it does |
|---|---|---|
| 1. Data Ingestion | `preprocessing.py` | Missing-value handling (interpolate/median/drop), duplicate removal, z-score/IQR outlier flagging, StandardScaler normalization (scaler is persisted and reused at inference time to avoid train/serve skew) |
| 3. Feature Engineering | `feature_engineering.py` | Rolling mean/std, EMA, lags, **vectorized** FFT spectral energy/entropy, rolling skew/kurtosis/min/max, rolling trend slope (closed-form OLS), rolling cross-sensor correlation |
| 4. Failure Prediction | `failure_prediction.py` | Trains RF, XGBoost, LightGBM, CatBoost, and an LSTM (sliding-window sequences); evaluates all on a **group-disjoint** test split (no machine's cycles leak between train/test); auto-selects best by ROC-AUC |
| 5. RUL Prediction | `rul_prediction.py` | LightGBM regressor -> remaining cycles/hours, failure probability, 0-100% health score |
| 6. Anomaly Detection | `anomaly_detection.py` | Isolation Forest, One-Class SVM, Local Outlier Factor, Keras Autoencoder; ensemble anomaly score + per-detector flags |
| 7. Maintenance Engine | `maintenance_engine.py` | Transparent, auditable rule-based system mapping the most extreme sensor to a component/failure-mode knowledge base -> root cause, priority, parts, cost/downtime estimate |
| 8. Explainable AI | `explainability.py` | SHAP TreeExplainer wrapper: global feature importance, per-machine contribution breakdown, beeswarm-ready data |
| 9. Dashboard | `app.py` | 7-tab Streamlit control room (see below) |

---

## Model performance

From the most recent `python train.py` run (synthetic data, 60 machines, 221 engineered features):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.884 | 0.952 | 0.054 | 0.102 | 0.885 |
| XGBoost | 0.900 | 0.672 | 0.343 | 0.454 | 0.893 |
| LightGBM | 0.910 | 0.816 | 0.335 | 0.475 | 0.913 |
| CatBoost | 0.907 | 0.663 | 0.484 | 0.559 | 0.928 |
| **LSTM (deployed)** | **0.985** | **0.900** | **0.995** | **0.945** | **0.998** |

RUL regression (LightGBM): **MAE approx 48 cycles, RMSE approx 62 cycles, R-squared approx 0.44**
on synthetic data with injected noise/spikes -- real sensor data with cleaner degradation
signal (and/or hyperparameter tuning) would improve this further.

Note: Random Forest's low recall here is a genuine characteristic of that run (its
default decision threshold is conservative on this feature set) -- exactly why the
pipeline **trains multiple models and auto-selects the best**, rather than committing
to one algorithm. Re-run `train.py` to see this comparison table regenerate from scratch.

SHAP explainability always runs against the best **tree-based** model even when the LSTM
wins overall (SHAP's fast `TreeExplainer` doesn't apply to recurrent nets) -- so the
dashboard's Explainable AI tab has data regardless of which model is deployed.

---

## Dashboard

Run `streamlit run app.py`. Seven tabs:

1. **Fleet Overview** -- health-score gauge, failure-probability gauge, fleet health
   distribution, sortable status table, RUL failure timeline.
2. **Sensor Analysis** -- multi-sensor trend lines per machine, correlation heatmap.
3. **Anomaly Detection** -- anomaly-score timeline with flagged points, per-detector
   flag table.
4. **Explainable AI** -- global SHAP feature importance, top sensors driving the
   selected machine's risk score.
5. **Maintenance** -- root cause, priority, cost/downtime estimate for the selected
   machine; fleet-wide priority queue.
6. **Model Performance** -- model comparison table, ROC curve, Precision-Recall curve,
   confusion matrix, RUL regression metrics.
7. **Compare Machines** -- side-by-side comparison across any subset of the fleet.

Sidebar: switch between the bundled live-fleet demo or **upload your own CSV**, pick a
machine, and download the full report as CSV or JSON.

Design: dark industrial theme (graphite background, amber accent), Oswald headers,
JetBrains Mono numeric readouts, and a custom semicircular gauge -- built to read like a
plant control-room panel rather than a generic SaaS dashboard.

---

## REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/models/info` | GET | Deployed model name + full comparison metrics |
| `/predict` | POST | JSON body of sensor readings -> predictions |
| `/predict/file` | POST | Upload a CSV file -> predictions |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"readings": [{"unit_id": 1, "cycle": 1, "temperature": 22, "vibration": 5}]}'
```

Interactive Swagger docs at `http://localhost:8000/docs`.

---

## Testing

```bash
pytest tests/ -v
```

27 unit tests covering preprocessing, feature engineering, RUL/health-score math,
maintenance recommendations, and the anomaly detection ensemble -- including a regression
test for a real index-alignment bug caught during development (see
`test_handle_missing_values_preserves_index_alignment`).

---

## Deployment guide

### Docker
```bash
docker build -t predictive-maintenance .
docker run -p 8501:8501 -v $(pwd)/models:/app/models predictive-maintenance
# API instead:
docker run -p 8000:8000 -v $(pwd)/models:/app/models predictive-maintenance \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```
or `docker compose up` to run both the dashboard and API together (see `docker-compose.yml`).

### Streamlit Community Cloud
Push this repo to GitHub, connect it at share.streamlit.io, set the main file to
`app.py`. Make sure `models/` and `data/` are either committed (small enough here) or
regenerated via a startup script, since Streamlit Cloud containers are ephemeral.

### Render
Create a **Web Service** pointing at this repo:
- Build command: `pip install -r requirements.txt && python data/generate_synthetic_data.py && python data/generate_live_fleet_sample.py && python train.py`
- Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### FastAPI on any cloud VM / container platform
Use the `api/main.py` entrypoint with `uvicorn` behind a reverse proxy (nginx/Caddy) and
a process manager (systemd/supervisord), or deploy the same Docker image with the
`uvicorn api.main:app` command override.

---

## Known limitations & next steps

- **Synthetic data**: ships with a schema-compatible synthetic dataset rather than the
  real NASA C-MAPSS/AI4I files (no live internet access in this environment). Swap in
  real CSVs with matching column names for production use -- no code changes needed.
- **LSTM real-time inference**: the deployed LSTM needs a sliding window of history to
  score a machine; `predict.py`'s single-snapshot API path falls back to the best
  tree-based runner-up (LightGBM) when a full sequence isn't available. For true
  streaming/real-time scoring, maintain a rolling buffer of the last N cycles per
  machine and feed it to the LSTM directly (see `failure_prediction._build_sequences`).
- **RUL regression accuracy** (R-squared approx 0.44) is modest on the noisy synthetic
  data; hyperparameter tuning (Optuna/GridSearch) and/or cleaner real sensor data would
  improve this.
- **Bonus features not implemented** (flagged as optional in the original spec): live
  MQTT/Kafka ingestion, email/WhatsApp alerting, Digital Twin visualization, and CI/CD
  pipeline config. The architecture (clean module boundaries, a single
  `PredictiveMaintenancePipeline` inference class, FastAPI service) is intentionally
  structured to make adding these straightforward -- e.g. a Kafka consumer could call
  `PredictiveMaintenancePipeline.run()` per message and push alerts when
  `risk_level == "Critical"`.
