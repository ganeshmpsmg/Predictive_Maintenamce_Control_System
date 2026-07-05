"""
predict.py
==========
Inference pipeline (CLI + importable API). Loads every artifact saved by
train.py and runs new sensor data through the full stack:

  ingestion -> feature engineering -> failure probability -> RUL/health -> anomaly
  score -> maintenance recommendation -> (optional) SHAP explanation

Produces one row per (unit_id, latest cycle) with a full report, and can
export to JSON for the dashboard's "download report" button.

Usage:
    python predict.py --data data/sensor_timeseries.csv --output reports/latest_predictions.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from preprocessing import handle_missing_values, remove_duplicates, detect_outliers
from feature_engineering import engineer_features
from rul_prediction import predict_rul
from anomaly_detection import score_anomalies
from maintenance_engine import generate_recommendations_batch
from utils import MODELS_DIR, REPORTS_DIR, get_logger, load_json, load_sklearn_model, save_json

logger = get_logger(__name__)

RAW_SENSOR_NAMES = [
    "temperature", "pressure", "rpm", "vibration", "voltage", "current",
    "humidity", "torque", "tool_wear", "oil_quality", "coolant_flow",
    "fan_speed", "bearing_temp", "exhaust_temp",
]


class PredictiveMaintenancePipeline:
    """Loads all trained artifacts once and exposes a single `.run()` call."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self._load_artifacts()

    def _load_artifacts(self):
        logger.info("Loading trained artifacts...")
        self.scaler = load_sklearn_model("sensor_scaler")
        self.feature_columns = load_sklearn_model("engineered_feature_columns")
        self.best_failure_model_name = load_sklearn_model("failure_best_model_name")

        # Failure model (handle keras vs sklearn transparently)
        if self.best_failure_model_name == "LSTM":
            from tensorflow import keras
            self.failure_model = keras.models.load_model(self.models_dir / "failure_lstm.keras")
        else:
            self.failure_model = load_sklearn_model(f"failure_{self.best_failure_model_name.lower()}")

        self.rul_model = load_sklearn_model("rul_lightgbm")

        # Anomaly detectors (best-effort; skip any that failed to train)
        self.iso_forest = self._safe_load("anomaly_isolation_forest")
        self.ocsvm = self._safe_load("anomaly_one_class_svm")
        self.lof = self._safe_load("anomaly_lof")
        self.autoencoder = None
        self.ae_threshold = None
        ae_path = self.models_dir / "anomaly_autoencoder.keras"
        if ae_path.exists():
            from tensorflow import keras
            self.autoencoder = keras.models.load_model(ae_path)
            try:
                self.ae_threshold = load_json("anomaly_config").get("ae_threshold")
            except FileNotFoundError:
                self.ae_threshold = None

        logger.info(f"Artifacts loaded. Deployed failure model: {self.best_failure_model_name}")

    def _safe_load(self, name: str):
        try:
            return load_sklearn_model(name)
        except FileNotFoundError:
            logger.warning(f"Optional artifact '{name}' not found; skipping.")
            return None

    def _predict_failure_probability(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.feature_columns]
        if self.best_failure_model_name == "LSTM":
            # Fallback: LSTM needs a sequence window; for single-row/batch tabular
            # inference without full history we approximate with the tree runner-up
            # if available, else return a neutral 0.5. (See README for real-time
            # sequence-buffer implementation notes.)
            try:
                fallback = load_sklearn_model("failure_lightgbm")
                return fallback.predict_proba(X)[:, 1]
            except FileNotFoundError:
                return np.full(len(df_features), 0.5)
        return self.failure_model.predict_proba(X)[:, 1]

    def run(self, raw_df: pd.DataFrame, latest_cycle_only: bool = True) -> pd.DataFrame:
        """Full inference pipeline on a raw (uncleaned) sensor dataframe."""
        if raw_df.empty:
            logger.warning("Received an empty dataframe; returning an empty report.")
            return raw_df.copy()

        raw_df = raw_df.reset_index(drop=True)  # guard against non-contiguous index from upstream slicing
        df = handle_missing_values(raw_df)
        df = remove_duplicates(df)
        df = detect_outliers(df)

        # The scaler was fit on a fixed set of sensor columns during training.
        # If the caller's data is missing some of those columns (e.g. a partial
        # sensor upload), fill them with the training-time mean rather than
        # crashing -- a neutral "no signal" placeholder in the original scale.
        expected_cols = list(getattr(self.scaler, "feature_names_in_", []))
        if not expected_cols:
            expected_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                              if c not in {"unit_id", "cycle", "RUL", "failure_within_30_cycles",
                                            "machine_failure", "is_outlier"}]
        for i, col in enumerate(expected_cols):
            if col not in df.columns:
                fill_value = float(self.scaler.mean_[i]) if hasattr(self.scaler, "mean_") else 0.0
                df[col] = fill_value
                logger.warning(f"Column '{col}' missing from input data; filled with training-mean placeholder.")

        df[expected_cols] = self.scaler.transform(df[expected_cols])

        df = engineer_features(df)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        failure_proba = self._predict_failure_probability(df)
        df = score_anomalies(
            df, self.feature_columns, self.iso_forest, self.ocsvm, self.lof,
            self.autoencoder, self.ae_threshold,
        )
        df = predict_rul(
            self.rul_model, df, self.feature_columns,
            failure_probability=failure_proba, anomaly_scores=df["anomaly_score"].values,
        )

        if latest_cycle_only and "unit_id" in df.columns and "cycle" in df.columns:
            df = df.sort_values("cycle").groupby("unit_id").tail(1).reset_index(drop=True)

        available_sensors = [s for s in RAW_SENSOR_NAMES if s in df.columns]
        recommendations = generate_recommendations_batch(df, available_sensors)
        # health_score / risk_level / failure_probability are already present in
        # df (from predict_rul); drop the recommendation engine's copies to avoid
        # duplicate columns after concat.
        overlap = {"machine_id", "health_score", "risk_level", "failure_probability"}
        recommendations = recommendations.drop(columns=[c for c in overlap if c in recommendations.columns])
        report = pd.concat([df.reset_index(drop=True), recommendations], axis=1)
        return report


def run_prediction_cli(data_path: str, output_path: str):
    raw_df = pd.read_csv(data_path)
    pipeline = PredictiveMaintenancePipeline()
    report = pipeline.run(raw_df)

    output_path = Path(output_path)
    report.to_csv(output_path, index=False)
    logger.info(f"Prediction report saved to {output_path} ({len(report)} machines)")

    summary = {
        "n_machines_evaluated": len(report),
        "critical_machines": int((report["risk_level"] == "Critical").sum()),
        "high_risk_machines": int((report["risk_level"] == "High").sum()),
        "average_health_score": round(float(report["health_score"].mean()), 2),
    }
    save_json(summary, "latest_prediction_summary")
    logger.info(f"Summary: {summary}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictive maintenance inference on new sensor data.")
    parser.add_argument("--data", default="data/sensor_timeseries.csv")
    parser.add_argument("--output", default="reports/latest_predictions.csv")
    args = parser.parse_args()

    run_prediction_cli(args.data, args.output)
