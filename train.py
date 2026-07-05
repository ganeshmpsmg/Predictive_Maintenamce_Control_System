"""
train.py
========
End-to-end training pipeline (CLI entry point). Orchestrates:

  1. Data Ingestion            (preprocessing.py)
  2. Feature Engineering       (feature_engineering.py)
  3. Failure Prediction        (failure_prediction.py) — multi-model + auto-select
  4. RUL Prediction            (rul_prediction.py)
  5. Anomaly Detection         (anomaly_detection.py) — 4 detectors
  6. Explainability            (explainability.py) — SHAP on the best model
  7. MLflow experiment tracking (metrics + params + artifacts)

Usage:
    python train.py --data data/sensor_timeseries.csv --sample 6000
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from preprocessing import run_ingestion_pipeline
from feature_engineering import engineer_features
from failure_prediction import train_all_models, get_roc_pr_curve_data
from rul_prediction import train_rul_model
from anomaly_detection import (
    train_isolation_forest, train_one_class_svm, train_lof, train_autoencoder,
)
from explainability import ModelExplainer
from utils import MODELS_DIR, REPORTS_DIR, get_logger, save_json, save_sklearn_model

logger = get_logger(__name__)

EXCLUDE_FROM_FEATURES = {
    "unit_id", "cycle", "RUL", "failure_within_30_cycles",
    "machine_failure", "is_outlier",
}

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def run_training_pipeline(
    data_path: str = "data/sensor_timeseries.csv",
    sample_size: int | None = 6000,
    max_features: int | None = None,
    lstm_epochs: int = 10,
    ae_epochs: int = 15,
    use_mlflow: bool = True,
) -> dict:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PREDICTIVE MAINTENANCE — TRAINING PIPELINE START")
    logger.info("=" * 70)

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("predictive_maintenance")
        mlflow.start_run(run_name=f"train_{int(t0)}")

    # ---------- Module 1: Ingestion ----------
    df, scaler = run_ingestion_pipeline(data_path)
    if sample_size and len(df) > sample_size:
        # sample whole units (not random rows) to preserve time-series continuity
        unit_ids = df["unit_id"].unique()
        rng = np.random.default_rng(42)
        keep_units = rng.choice(unit_ids, size=min(len(unit_ids), max(10, sample_size // 100)), replace=False)
        df = df[df["unit_id"].isin(keep_units)].reset_index(drop=True)
    save_sklearn_model(scaler, "sensor_scaler")

    # ---------- Module 3: Feature Engineering ----------
    df = engineer_features(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_columns = [c for c in numeric_cols if c not in EXCLUDE_FROM_FEATURES]
    if max_features:
        feature_columns = feature_columns[:max_features]
    logger.info(f"Using {len(feature_columns)} engineered features for modeling")

    # ---------- Module 4: Failure Prediction ----------
    failure_results = train_all_models(
        df, feature_columns, target_column="failure_within_30_cycles",
        lstm_epochs=lstm_epochs,
    )
    best_name = failure_results["best_model_name"]
    best_metrics = failure_results["models"][best_name]["metrics"]

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_param("best_failure_model", best_name)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("n_units", df["unit_id"].nunique())
        for metric_name, value in best_metrics.items():
            if not np.isnan(value):
                mlflow.log_metric(f"failure_{metric_name}", value)

    # curve data for dashboard (skip for LSTM's differently-shaped X_test)
    curve_data = None
    if best_name != "LSTM":
        curve_data = get_roc_pr_curve_data(
            failure_results["best_model"], failure_results["X_test"], failure_results["y_test"], best_name
        )
        np.savez(REPORTS_DIR / "roc_pr_curve_data.npz",
                 fpr=curve_data["fpr"], tpr=curve_data["tpr"],
                 precision=curve_data["precision"], recall=curve_data["recall"],
                 confusion_matrix=curve_data["confusion_matrix"])

    # ---------- Module 5: RUL Prediction ----------
    rul_results = train_rul_model(df, feature_columns, target_column="RUL")
    if use_mlflow and MLFLOW_AVAILABLE:
        for metric_name, value in rul_results["metrics"].items():
            mlflow.log_metric(f"rul_{metric_name}", value)

    # ---------- Module 6: Anomaly Detection ----------
    iso = train_isolation_forest(df, feature_columns)
    svm = train_one_class_svm(df, feature_columns)
    lof = train_lof(df, feature_columns)
    ae, ae_threshold = train_autoencoder(df, feature_columns, epochs=ae_epochs)
    save_json({"ae_threshold": ae_threshold}, "anomaly_config")

    # ---------- Module 8: Explainability ----------
    # SHAP requires a tree-based model. If the LSTM wins the failure-prediction
    # contest on ROC-AUC, we still compute SHAP on the best *tree-based*
    # contender so the dashboard's Explainable AI panel always has data —
    # the LSTM remains the deployed predictor, SHAP explains the tree runner-up.
    shap_summary = None
    tree_candidates = {k: v for k, v in failure_results["models"].items() if k != "LSTM"}
    explain_model_name, explain_model = None, None
    if tree_candidates:
        explain_model_name = max(tree_candidates, key=lambda k: tree_candidates[k]["metrics"]["ROC_AUC"])
        explain_model = tree_candidates[explain_model_name]["model"]

    if explain_model is not None:
        try:
            explainer = ModelExplainer(explain_model, feature_columns)
            shap_summary = explainer.global_feature_importance(df[feature_columns].sample(
                min(300, len(df)), random_state=42))
            shap_summary.to_csv(REPORTS_DIR / "global_feature_importance.csv", index=False)
            save_json({"explained_model": explain_model_name}, "explainability_config")
            logger.info(f"SHAP explainability computed using '{explain_model_name}' "
                        f"(best tree-based model; deployed predictor is '{best_name}')")
        except Exception as e:
            logger.warning(f"SHAP explanation skipped: {e}")

    # ---------- Persist run summary ----------
    summary = {
        "best_failure_model": best_name,
        "failure_model_metrics": {k: v["metrics"] for k, v in failure_results["models"].items()},
        "rul_model_metrics": rul_results["metrics"],
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "n_units_trained": int(df["unit_id"].nunique()) if "unit_id" in df.columns else None,
        "training_time_seconds": round(time.time() - t0, 1),
    }
    save_json(summary, "training_summary")
    save_sklearn_model(feature_columns, "engineered_feature_columns")

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_artifact(str(REPORTS_DIR / "training_summary.json"))
        mlflow.end_run()

    logger.info(f"Training pipeline complete in {summary['training_time_seconds']}s. "
                f"Best failure model: {best_name}")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the predictive maintenance model suite.")
    parser.add_argument("--data", default="data/sensor_timeseries.csv")
    parser.add_argument("--sample", type=int, default=6000, help="Approx. row budget (sampled by whole units)")
    parser.add_argument("--max-features", type=int, default=None, help="Optional cap on engineered feature count")
    parser.add_argument("--lstm-epochs", type=int, default=10)
    parser.add_argument("--ae-epochs", type=int, default=15)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        sample_size=args.sample,
        max_features=args.max_features,
        lstm_epochs=args.lstm_epochs,
        ae_epochs=args.ae_epochs,
        use_mlflow=not args.no_mlflow,
    )
