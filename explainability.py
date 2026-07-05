"""
explainability.py
==================
Module 8: Explainable AI (SHAP).

Wraps SHAP's TreeExplainer (fast, exact for tree ensembles like XGBoost/
LightGBM/RandomForest/CatBoost) to produce:
- global feature importance (mean |SHAP value|)
- per-prediction contribution breakdowns (why THIS machine was flagged)
- ready-to-plot data for summary/bar/force-style charts (returned as
  dataframes/arrays so the Streamlit dashboard can render them with Plotly
  without depending on matplotlib's non-interactive SHAP plots)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import shap

from utils import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """Thin, model-agnostic wrapper around shap.TreeExplainer / shap.Explainer."""

    def __init__(self, model, feature_names: list[str], background_data: Optional[pd.DataFrame] = None):
        self.feature_names = feature_names
        try:
            self.explainer = shap.TreeExplainer(model)
            self.kind = "tree"
        except Exception as e:
            logger.warning(f"TreeExplainer unavailable ({e}); falling back to generic Explainer.")
            sample = background_data.sample(min(100, len(background_data)), random_state=42) \
                if background_data is not None else None
            self.explainer = shap.Explainer(model, sample)
            self.kind = "generic"

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return a (n_samples, n_features) SHAP value matrix for the positive class
        where applicable (binary classification)."""
        shap_values = self.explainer.shap_values(X) if self.kind == "tree" else self.explainer(X).values

        # Normalize output shape across sklearn/xgboost/lightgbm binary-classifier variants
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        return np.asarray(shap_values)

    def global_feature_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """Mean |SHAP value| per feature, sorted descending -> dashboard bar chart data."""
        shap_values = self.compute_shap_values(X)
        importance = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": importance})
        return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    def explain_instance(self, X_row: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Per-prediction contribution breakdown for a single row: which features
        pushed the prediction up (toward failure) vs down (toward healthy).
        """
        shap_values = self.compute_shap_values(X_row)[0]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": shap_values,
            "feature_value": X_row.iloc[0].values,
        })
        df["direction"] = np.where(df["shap_value"] >= 0, "increases failure risk", "decreases failure risk")
        df["abs_shap"] = df["shap_value"].abs()
        return df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap").reset_index(drop=True)

    def summary_plot_data(self, X: pd.DataFrame, sample_size: int = 300) -> dict:
        """
        Data payload sized for a Plotly SHAP-summary-style beeswarm chart:
        one point per (sample, feature) with its SHAP value and normalized
        feature value (for color-coding high/low).
        """
        if len(X) > sample_size:
            X = X.sample(sample_size, random_state=42)
        shap_values = self.compute_shap_values(X)
        top_idx = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:15]

        records = []
        for feat_idx in top_idx:
            feat_name = self.feature_names[feat_idx]
            vals = X.iloc[:, feat_idx].values
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
            for shap_val, norm_val in zip(shap_values[:, feat_idx], norm_vals):
                records.append({"feature": feat_name, "shap_value": shap_val, "normalized_feature_value": norm_val})
        return {"records": pd.DataFrame(records)}


if __name__ == "__main__":
    from preprocessing import run_ingestion_pipeline
    from feature_engineering import engineer_features
    from failure_prediction import train_all_models

    df, _ = run_ingestion_pipeline("data/sensor_timeseries.csv")
    df = engineer_features(df.sample(2500, random_state=1))
    exclude = {"unit_id", "cycle", "RUL", "failure_within_30_cycles", "is_outlier"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude][:25]

    results = train_all_models(df, feature_cols, "failure_within_30_cycles")
    best_name = results["best_model_name"]
    best_model = results["models"][best_name]["model"]
    print(f"Explaining best model: {best_name}")

    explainer = ModelExplainer(best_model, feature_cols)
    print(explainer.global_feature_importance(df[feature_cols].head(300)))
