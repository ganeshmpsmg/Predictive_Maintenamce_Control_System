"""
anomaly_detection.py
=====================
Module 6: Anomaly Detection.

Implements four complementary anomaly detectors:
- Isolation Forest      (fast, tree-based, handles high dimensions well)
- One-Class SVM         (good for tight, non-linear normal-behavior boundaries)
- Local Outlier Factor  (density-based, good for local/contextual anomalies)
- Autoencoder (Keras)   (reconstruction-error based, captures complex
                          nonlinear multivariate sensor interactions)

All models are trained on "normal" operating data (early-life cycles, low
failure probability) and score how anomalous each new reading is, so that
degradation signatures can be flagged before the failure classifier's
prediction threshold is crossed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from utils import get_logger, save_sklearn_model

logger = get_logger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _normal_subset(df: pd.DataFrame, label_col: str = "failure_within_30_cycles") -> pd.DataFrame:
    """Select rows representing normal (non-imminent-failure) operation for training detectors."""
    if label_col in df.columns:
        return df[df[label_col] == 0]
    return df


def train_isolation_forest(
    df: pd.DataFrame, feature_columns: list[str], contamination: float = 0.03, random_state: int = 42
) -> IsolationForest:
    normal = _normal_subset(df)
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1
    )
    model.fit(normal[feature_columns])
    save_sklearn_model(model, "anomaly_isolation_forest")
    logger.info("Isolation Forest trained on normal-operation subset")
    return model


def train_one_class_svm(
    df: pd.DataFrame, feature_columns: list[str], nu: float = 0.03, sample_size: int = 3000
) -> OneClassSVM:
    normal = _normal_subset(df)
    if len(normal) > sample_size:
        normal = normal.sample(sample_size, random_state=42)  # SVM doesn't scale well; subsample
    model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    model.fit(normal[feature_columns])
    save_sklearn_model(model, "anomaly_one_class_svm")
    logger.info("One-Class SVM trained on normal-operation subset")
    return model


def train_lof(
    df: pd.DataFrame, feature_columns: list[str], n_neighbors: int = 20, contamination: float = 0.03
) -> LocalOutlierFactor:
    """LOF has no separate predict(); we fit with novelty=True so it can score new data."""
    normal = _normal_subset(df)
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, novelty=True, n_jobs=-1
    )
    model.fit(normal[feature_columns])
    save_sklearn_model(model, "anomaly_lof")
    logger.info("Local Outlier Factor trained on normal-operation subset")
    return model


def build_autoencoder(input_dim: int, encoding_dim: int = 8):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available; cannot build autoencoder.")
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(encoding_dim, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(
    df: pd.DataFrame, feature_columns: list[str], epochs: int = 20, batch_size: int = 64
):
    if not TF_AVAILABLE:
        logger.warning("TensorFlow unavailable; skipping autoencoder training.")
        return None, None

    normal = _normal_subset(df)
    X = normal[feature_columns].values.astype("float32")

    autoencoder = build_autoencoder(input_dim=X.shape[1])
    history = autoencoder.fit(
        X, X, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )
    autoencoder.save("models/anomaly_autoencoder.keras")

    reconstructions = autoencoder.predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructions), axis=1)
    threshold = float(np.percentile(errors, 97))  # top 3% treated as anomalous
    logger.info(f"Autoencoder trained. Reconstruction-error threshold (97th pct): {threshold:.4f}")
    return autoencoder, threshold


def score_anomalies(
    df: pd.DataFrame,
    feature_columns: list[str],
    iso_forest: Optional[IsolationForest] = None,
    ocsvm: Optional[OneClassSVM] = None,
    lof: Optional[LocalOutlierFactor] = None,
    autoencoder=None,
    ae_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Score every row with all available detectors and combine into a single
    normalized ensemble anomaly_score in [0, 1] (mean of available detector
    votes), plus per-detector boolean flags.
    """
    df = df.copy()
    X = df[feature_columns]
    votes = []

    if iso_forest is not None:
        raw = -iso_forest.score_samples(X)  # higher = more anomalous
        df["anomaly_isoforest"] = iso_forest.predict(X) == -1
        votes.append((raw - raw.min()) / (raw.max() - raw.min() + 1e-9))

    if ocsvm is not None:
        raw = -ocsvm.decision_function(X)
        df["anomaly_ocsvm"] = ocsvm.predict(X) == -1
        votes.append((raw - raw.min()) / (raw.max() - raw.min() + 1e-9))

    if lof is not None:
        raw = -lof.decision_function(X)
        df["anomaly_lof"] = lof.predict(X) == -1
        votes.append((raw - raw.min()) / (raw.max() - raw.min() + 1e-9))

    if autoencoder is not None:
        X_arr = X.values.astype("float32")
        recon = autoencoder.predict(X_arr, verbose=0)
        errors = np.mean(np.square(X_arr - recon), axis=1)
        df["anomaly_autoencoder"] = errors > (ae_threshold or np.percentile(errors, 97))
        votes.append((errors - errors.min()) / (errors.max() - errors.min() + 1e-9))

    if votes:
        df["anomaly_score"] = np.mean(votes, axis=0)
    else:
        df["anomaly_score"] = 0.0

    flag_cols = [c for c in ["anomaly_isoforest", "anomaly_ocsvm", "anomaly_lof", "anomaly_autoencoder"]
                 if c in df.columns]
    df["is_anomalous_consensus"] = df[flag_cols].sum(axis=1) >= max(1, len(flag_cols) // 2 + 1) if flag_cols else False

    return df


if __name__ == "__main__":
    from preprocessing import run_ingestion_pipeline
    from feature_engineering import engineer_features

    df, _ = run_ingestion_pipeline("data/sensor_timeseries.csv")
    df = engineer_features(df.sample(3000, random_state=1))
    exclude = {"unit_id", "cycle", "RUL", "failure_within_30_cycles", "is_outlier"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude][:30]

    iso = train_isolation_forest(df, feature_cols)
    svm = train_one_class_svm(df, feature_cols)
    lof = train_lof(df, feature_cols)
    ae, ae_thr = train_autoencoder(df, feature_cols, epochs=5)

    scored = score_anomalies(df.head(20), feature_cols, iso, svm, lof, ae, ae_thr)
    print(scored[["unit_id", "cycle", "anomaly_score", "is_anomalous_consensus"]])
