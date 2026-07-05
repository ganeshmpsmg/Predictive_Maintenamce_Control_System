"""
preprocessing.py
================
Module 1: Data Ingestion.

Responsibilities:
- Load sensor datasets (CMAPSS-style time series or AI4I-style discrete records)
- Handle missing values
- Remove duplicates
- Normalize sensor readings
- Detect outliers (z-score + IQR based)

All functions are pure and side-effect-free (except load_dataset which reads
from disk) so they can be unit tested independently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ID_COLUMNS = {"unit_id", "cycle", "failure_type"}
LABEL_COLUMNS = {"RUL", "failure_within_30_cycles", "machine_failure"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV sensor dataset from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset '{path.name}' with shape {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "interpolate") -> pd.DataFrame:
    """
    Handle missing values in sensor columns.

    strategy:
        'interpolate' - linear interpolation (best for time series), falls back
                        to median fill for any remaining NaNs at series edges.
        'median'      - simple median imputation.
        'drop'        - drop rows with any missing values.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_missing_before = df[numeric_cols].isna().sum().sum()

    if strategy == "drop":
        df = df.dropna()
    elif strategy == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    else:  # interpolate
        if "unit_id" in df.columns:
            # transform() (unlike apply()+reset_index) preserves the original,
            # possibly non-contiguous index -- critical when df is a slice of
            # a larger frame (e.g. one machine's history), where a naive
            # reset_index would misalign the result and corrupt every column.
            df[numeric_cols] = (
                df.groupby("unit_id")[numeric_cols]
                .transform(lambda s: s.interpolate(limit_direction="both"))
            )
        else:
            df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
        # safety net for any columns that were entirely NaN
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    n_missing_after = df.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"Missing values: {n_missing_before} -> {n_missing_after} (strategy='{strategy}')")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} duplicate rows")
    return df


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "zscore",
    threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Flag outliers per numeric sensor column without dropping them
    (predictive maintenance signals are often legitimate extreme readings,
    so we flag rather than silently remove).

    Returns the original dataframe with an added boolean column
    'is_outlier' (True if ANY monitored sensor is an outlier on that row).
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ID_COLUMNS | LABEL_COLUMNS]

    outlier_flags = pd.DataFrame(False, index=df.index, columns=columns)

    if method == "zscore":
        for col in columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma == 0 or np.isnan(sigma):
                continue
            z = (df[col] - mu) / sigma
            outlier_flags[col] = z.abs() > threshold
    elif method == "iqr":
        for col in columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_flags[col] = (df[col] < lower) | (df[col] > upper)
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    df["is_outlier"] = outlier_flags.any(axis=1)
    logger.info(f"Detected {df['is_outlier'].sum()} outlier rows out of {len(df)} "
                f"(method='{method}', threshold={threshold})")
    return df


def normalize_sensors(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "standard",
    scaler=None,
):
    """
    Normalize sensor readings. Returns (df_normalized, fitted_scaler) so the
    same scaler can be reused at inference time (critical to avoid train/serve skew).
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ID_COLUMNS | LABEL_COLUMNS | {"is_outlier"}]

    if scaler is None:
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])

    logger.info(f"Normalized {len(columns)} sensor columns using {type(scaler).__name__}")
    return df, scaler


def run_ingestion_pipeline(
    path: str | Path,
    missing_strategy: str = "interpolate",
    outlier_method: str = "zscore",
    normalize: bool = True,
) -> tuple[pd.DataFrame, Optional[object]]:
    """Convenience wrapper chaining the full Module 1 pipeline."""
    df = load_dataset(path)
    df = handle_missing_values(df, strategy=missing_strategy)
    df = remove_duplicates(df)
    df = detect_outliers(df, method=outlier_method)
    scaler = None
    if normalize:
        df, scaler = normalize_sensors(df)
    return df, scaler


if __name__ == "__main__":
    df, scaler = run_ingestion_pipeline("data/sensor_timeseries.csv")
    print(df.head())
    print(df.shape)
