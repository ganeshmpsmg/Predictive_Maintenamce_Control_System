"""
rul_prediction.py
==================
Module 5: Remaining Useful Life (RUL) Prediction.

Trains a regression model (default: LightGBM Regressor, gradient-boosted
trees generalize very well on tabular sensor-degradation data) to predict
the number of remaining operating cycles before failure, then derives:
- remaining operating hours
- failure probability (sigmoid-style transform of RUL relative to a
  machine-specific critical threshold, blended with the classifier output
  when available)
- health score (0-100%) via utils.compute_health_score
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from utils import MODELS_DIR, compute_health_score, get_logger, save_sklearn_model

logger = get_logger(__name__)

RUL_MODEL_NAME = "rul_lightgbm"


def train_rul_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "RUL",
    group_column: str = "unit_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a LightGBM regressor for RUL estimation, splitting by unit_id
    (GroupShuffleSplit) so that no machine's cycles leak between train/test —
    critical for a valid time-series evaluation.
    """
    X = df[feature_columns]
    y = df[target_column]
    groups = df[group_column] if group_column in df.columns else np.arange(len(df))

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }
    logger.info(f"RUL model metrics: {metrics}")

    save_sklearn_model(model, RUL_MODEL_NAME)
    save_sklearn_model(feature_columns, f"{RUL_MODEL_NAME}_features")

    return {"model": model, "metrics": metrics, "feature_columns": feature_columns}


def predict_rul(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
    failure_probability: Optional[np.ndarray] = None,
    anomaly_scores: Optional[np.ndarray] = None,
    max_rul_reference: float = 250.0,
    failure_threshold_cycles: float = 30.0,
    sigmoid_scale: float = 12.0,
) -> pd.DataFrame:
    """
    Predict RUL for each row and derive full health assessment fields.
    If failure_probability is not supplied (e.g. classifier not run yet),
    it's approximated with a logistic curve centered at `failure_threshold_cycles`
    (the same "failure imminent" horizon used to label the training data),
    so a machine well above the threshold reads as low-risk rather than
    the overly-aggressive exponential decay this used previously.
    """
    X = df[feature_columns]
    rul_pred = np.clip(model.predict(X), 0, None)

    if failure_probability is None:
        failure_probability = 1 / (1 + np.exp((rul_pred - failure_threshold_cycles) / sigmoid_scale))
        failure_probability = np.clip(failure_probability, 0, 1)
    if anomaly_scores is None:
        anomaly_scores = np.zeros(len(df))

    results = []
    for rul, fp, an in zip(rul_pred, failure_probability, anomaly_scores):
        assessment = compute_health_score(fp, rul, max_rul_reference, an)
        results.append(assessment.__dict__)

    result_df = pd.DataFrame(results)
    result_df["predicted_RUL_cycles"] = rul_pred
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


if __name__ == "__main__":
    from preprocessing import run_ingestion_pipeline
    from feature_engineering import engineer_features

    df, _ = run_ingestion_pipeline("data/sensor_timeseries.csv")
    df = engineer_features(df.sample(4000, random_state=1))
    exclude = {"unit_id", "cycle", "RUL", "failure_within_30_cycles", "is_outlier"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    result = train_rul_model(df, feature_cols)
    preview = predict_rul(result["model"], df.head(5), feature_cols)
    print(preview[["unit_id", "cycle", "RUL", "predicted_RUL_cycles", "health_score", "risk_level"]])
