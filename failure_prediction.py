"""
failure_prediction.py
======================
Module 4: Failure Prediction.

Trains multiple classifiers on the binary "failure within N cycles" target:
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- LSTM Neural Network (sequence model over each machine's cycle history)

Compares all models on Accuracy / Precision / Recall / F1 / ROC-AUC and
automatically selects the best model by ROC-AUC (most robust to the class
imbalance typical of failure data).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit

import lightgbm as lgb
import xgboost as xgb

from utils import get_logger, save_sklearn_model

logger = get_logger(__name__)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _group_train_test_split(df, feature_columns, target_column, group_column, test_size, random_state):
    X, y = df[feature_columns], df[target_column]
    groups = df[group_column] if group_column in df.columns else np.arange(len(df))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], train_idx, test_idx


def _evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def _build_sequences(df: pd.DataFrame, feature_columns: list[str], target_column: str,
                      group_column: str, seq_len: int = 15):
    """Build fixed-length sliding-window sequences per machine for the LSTM."""
    X_seqs, y_seqs, idx_map = [], [], []
    for unit, g in df.groupby(group_column):
        g = g.sort_values("cycle") if "cycle" in g.columns else g
        values = g[feature_columns].values
        targets = g[target_column].values
        orig_idx = g.index.values
        if len(values) < seq_len:
            padded = np.vstack([np.zeros((seq_len - len(values), values.shape[1])), values])
            X_seqs.append(padded)
            y_seqs.append(targets[-1])
            idx_map.append(orig_idx[-1])
            continue
        for i in range(seq_len, len(values) + 1):
            X_seqs.append(values[i - seq_len:i])
            y_seqs.append(targets[i - 1])
            idx_map.append(orig_idx[i - 1])
    return np.array(X_seqs), np.array(y_seqs), np.array(idx_map)


def train_lstm_model(df, feature_columns, target_column, group_column="unit_id",
                      seq_len=15, test_size=0.2, random_state=42, epochs=15):
    if not TF_AVAILABLE:
        logger.warning("TensorFlow unavailable; skipping LSTM model.")
        return None

    X_seq, y_seq, _ = _build_sequences(df, feature_columns, target_column, group_column, seq_len)
    n = len(X_seq)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = perm[:split], perm[split:]

    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]

    model = keras.Sequential([
        keras.Input(shape=(seq_len, X_seq.shape[2])),
        layers.LSTM(48, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(24),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    class_weight = None
    pos_rate = y_train.mean()
    if 0 < pos_rate < 1:
        class_weight = {0: 1.0, 1: (1 - pos_rate) / pos_rate}

    model.fit(
        X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1,
        class_weight=class_weight, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )

    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = _evaluate(y_test, y_pred, y_proba)
    model.save("models/failure_lstm.keras")
    logger.info(f"LSTM metrics: {metrics}")
    return {"model": model, "metrics": metrics, "seq_len": seq_len}


def train_all_models(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "failure_within_30_cycles",
    group_column: str = "unit_id",
    test_size: float = 0.2,
    random_state: int = 42,
    include_lstm: bool = True,
    lstm_epochs: int = 10,
) -> dict:
    """
    Train Random Forest, XGBoost, LightGBM, CatBoost (if available), and
    optionally an LSTM; evaluate all on a held-out, group-disjoint test set;
    auto-select the best by ROC-AUC.
    """
    X_train, X_test, y_train, y_test, train_idx, test_idx = _group_train_test_split(
        df, feature_columns, target_column, group_column, test_size, random_state
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    models["RandomForest"] = {"model": rf, "metrics": _evaluate(y_test, (proba >= 0.5).astype(int), proba)}

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=random_state, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    proba = xgb_model.predict_proba(X_test)[:, 1]
    models["XGBoost"] = {"model": xgb_model, "metrics": _evaluate(y_test, (proba >= 0.5).astype(int), proba)}

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=random_state, verbosity=-1,
    )
    lgb_model.fit(X_train, y_train)
    proba = lgb_model.predict_proba(X_test)[:, 1]
    models["LightGBM"] = {"model": lgb_model, "metrics": _evaluate(y_test, (proba >= 0.5).astype(int), proba)}

    # --- CatBoost ---
    if CATBOOST_AVAILABLE:
        cb_model = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=random_state, verbose=False,
        )
        cb_model.fit(X_train, y_train)
        proba = cb_model.predict_proba(X_test)[:, 1]
        models["CatBoost"] = {"model": cb_model, "metrics": _evaluate(y_test, (proba >= 0.5).astype(int), proba)}

    # --- LSTM ---
    if include_lstm and TF_AVAILABLE:
        lstm_result = train_lstm_model(
            df, feature_columns, target_column, group_column,
            test_size=test_size, random_state=random_state, epochs=lstm_epochs,
        )
        if lstm_result is not None:
            models["LSTM"] = lstm_result

    # --- Select best by ROC-AUC ---
    valid_models = {k: v for k, v in models.items() if not np.isnan(v["metrics"]["ROC_AUC"])}
    best_name = max(valid_models, key=lambda k: valid_models[k]["metrics"]["ROC_AUC"])
    logger.info("Model comparison:\n" + pd.DataFrame(
        {k: v["metrics"] for k, v in models.items()}).T.to_string())
    logger.info(f"Best model selected: {best_name} (ROC-AUC={models[best_name]['metrics']['ROC_AUC']:.4f})")

    # persist tree-based models with joblib; keras models save themselves
    for name, result in models.items():
        if name != "LSTM":
            save_sklearn_model(result["model"], f"failure_{name.lower()}")
    save_sklearn_model(feature_columns, "failure_model_features")
    save_sklearn_model(best_name, "failure_best_model_name")

    return {
        "models": models,
        "best_model_name": best_name,
        "best_model": models[best_name]["model"],
        "X_test": X_test,
        "y_test": y_test,
        "test_idx": test_idx,
        "feature_columns": feature_columns,
    }


def get_roc_pr_curve_data(model, X_test, y_test, model_name: str = "") -> dict:
    """Return ROC and Precision-Recall curve coordinates for dashboard plotting."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:  # keras model
        proba = model.predict(X_test, verbose=0).flatten()

    fpr, tpr, _ = roc_curve(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    cm = confusion_matrix(y_test, (proba >= 0.5).astype(int))
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
            "confusion_matrix": cm, "model_name": model_name}


if __name__ == "__main__":
    from preprocessing import run_ingestion_pipeline
    from feature_engineering import engineer_features

    df, _ = run_ingestion_pipeline("data/sensor_timeseries.csv")
    df = engineer_features(df.sample(3000, random_state=1))
    exclude = {"unit_id", "cycle", "RUL", "failure_within_30_cycles", "is_outlier"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude][:25]

    results = train_all_models(df, feature_cols, lstm_epochs=5)
    print(f"\nBest model: {results['best_model_name']}")
    print(pd.DataFrame({k: v["metrics"] for k, v in results["models"].items()}).T)
