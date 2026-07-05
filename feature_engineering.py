"""
feature_engineering.py
=======================
Module 3: Feature Engineering for time-series sensor data.

Generates, per unit_id (machine), grouped in time order by cycle:
- Rolling mean / std
- Exponential moving average (EMA)
- Lag features
- FFT-based spectral features (dominant frequency energy, spectral entropy)
- Statistical features (skew, kurtosis, min, max, range)
- Trend features (linear regression slope over a rolling window)
- Cross-sensor correlation features (rolling correlation between key sensor pairs)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

logger = logging.getLogger(__name__)

NON_SENSOR_COLUMNS = {
    "unit_id", "cycle", "RUL", "failure_within_30_cycles",
    "machine_failure", "is_outlier", "op_setting_1", "op_setting_2", "op_setting_3",
}


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in NON_SENSOR_COLUMNS]


def add_rolling_features(df: pd.DataFrame, columns: list[str], window: int = 5) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby("unit_id") if "unit_id" in df.columns else [(None, df)]
    for col in columns:
        if "unit_id" in df.columns:
            df[f"{col}_roll_mean_{window}"] = (
                df.groupby("unit_id")[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
            )
            df[f"{col}_roll_std_{window}"] = (
                df.groupby("unit_id")[col].transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
            )
        else:
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
            df[f"{col}_roll_std_{window}"] = df[col].rolling(window, min_periods=1).std().fillna(0)
    return df


def add_ema_features(df: pd.DataFrame, columns: list[str], span: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if "unit_id" in df.columns:
            df[f"{col}_ema_{span}"] = (
                df.groupby("unit_id")[col].transform(lambda s: s.ewm(span=span, adjust=False).mean())
            )
        else:
            df[f"{col}_ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    return df


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int] = (1, 3, 5)) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for lag in lags:
            if "unit_id" in df.columns:
                df[f"{col}_lag_{lag}"] = df.groupby("unit_id")[col].shift(lag)
            else:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    # backfill lag NaNs at the start of each series
    lag_cols = [c for c in df.columns if "_lag_" in c]
    df[lag_cols] = df[lag_cols].bfill().ffill()
    return df


def _fft_energy_and_entropy(values: np.ndarray) -> tuple[float, float]:
    """Return (dominant-band spectral energy, spectral entropy) for a 1D signal."""
    if len(values) < 4 or np.all(values == values[0]):
        return 0.0, 0.0
    spectrum = np.abs(rfft(values - np.mean(values)))
    power = spectrum ** 2
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0
    # dominant band energy: top-3 frequency bins (excluding DC) as fraction of total
    order = np.argsort(power[1:])[::-1][:3]
    dominant_energy = power[1:][order].sum() / total
    # spectral entropy
    p = power / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(len(p)) if len(p) > 1 else 0.0
    return float(dominant_energy), float(entropy)


def _sliding_windows(vals: np.ndarray, window: int) -> np.ndarray:
    """
    Build a (n, window) matrix where row i is the trailing window ending at i,
    left-padded by repeating the first value for i < window-1 (approximates
    pandas' min_periods=1 behavior while staying fully vectorized).
    """
    n = len(vals)
    if n == 0:
        return np.empty((0, window))
    pad = np.full(window - 1, vals[0])
    padded = np.concatenate([pad, vals])
    return np.lib.stride_tricks.sliding_window_view(padded, window)


def add_fft_features(df: pd.DataFrame, columns: list[str], window: int = 15) -> pd.DataFrame:
    """
    Vectorized rolling-window FFT features per sensor (dominant energy +
    spectral entropy). Uses numpy's batched rfft across all windows at once
    instead of a per-row Python loop -- orders of magnitude faster on real
    fleet-sized data.
    """
    df = df.copy()

    def compute(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = _sliding_windows(vals.astype(float), window)
        w_centered = w - w.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(w_centered, axis=1)
        power = np.abs(spectrum) ** 2
        total = power.sum(axis=1)
        total_safe = np.where(total <= 0, 1.0, total)

        power_no_dc = power[:, 1:]
        k = min(3, power_no_dc.shape[1]) if power_no_dc.shape[1] > 0 else 0
        if k > 0:
            top_idx = np.argpartition(-power_no_dc, k - 1, axis=1)[:, :k]
            dominant = np.take_along_axis(power_no_dc, top_idx, axis=1).sum(axis=1)
        else:
            dominant = np.zeros(len(vals))
        dominant_energy = np.where(total > 0, dominant / total_safe, 0.0)

        p = power / total_safe[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log2(p), 0.0)
        entropy_raw = -np.sum(p * logp, axis=1)
        denom = np.log2(power.shape[1]) if power.shape[1] > 1 else 1.0
        entropy = np.where(total > 0, entropy_raw / denom, 0.0)
        return dominant_energy, entropy

    for col in columns:
        if "unit_id" in df.columns:
            energies = np.zeros(len(df))
            entropies = np.zeros(len(df))
            for _, g in df.groupby("unit_id"):
                e, h = compute(g[col].values)
                energies[df.index.get_indexer(g.index)] = e
                entropies[df.index.get_indexer(g.index)] = h
            df[f"{col}_fft_energy"] = energies
            df[f"{col}_fft_entropy"] = entropies
        else:
            e, h = compute(df[col].values)
            df[f"{col}_fft_energy"] = e
            df[f"{col}_fft_entropy"] = h
    return df


def add_statistical_features(df: pd.DataFrame, columns: list[str], window: int = 10) -> pd.DataFrame:
    """Vectorized rolling min/max/range/skew/kurtosis using batched sliding windows."""
    df = df.copy()

    def compute(vals: np.ndarray):
        w = _sliding_windows(vals.astype(float), window)
        roll_min = w.min(axis=1)
        roll_max = w.max(axis=1)
        roll_skew = skew(w, axis=1, bias=True)
        roll_kurt = kurtosis(w, axis=1, bias=True)
        return roll_min, roll_max, np.nan_to_num(roll_skew), np.nan_to_num(roll_kurt)

    for col in columns:
        if "unit_id" in df.columns:
            n = len(df)
            mins, maxs, skews, kurts = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            for _, g in df.groupby("unit_id"):
                mn, mx, sk, ku = compute(g[col].values)
                positions = df.index.get_indexer(g.index)
                mins[positions], maxs[positions] = mn, mx
                skews[positions], kurts[positions] = sk, ku
        else:
            mins, maxs, skews, kurts = compute(df[col].values)

        df[f"{col}_roll_min_{window}"] = mins
        df[f"{col}_roll_max_{window}"] = maxs
        df[f"{col}_roll_range_{window}"] = maxs - mins
        df[f"{col}_roll_skew_{window}"] = skews
        df[f"{col}_roll_kurt_{window}"] = kurts
    return df


def add_trend_features(df: pd.DataFrame, columns: list[str], window: int = 10) -> pd.DataFrame:
    """
    Vectorized rolling linear-regression slope via the closed-form OLS formula
    (x is the fixed 0..window-1 index for every window, so x_mean/x_var are
    computed once and reused): slope = cov(x, y) / var(x).
    """
    df = df.copy()
    x = np.arange(window)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)

    def compute(vals: np.ndarray) -> np.ndarray:
        w = _sliding_windows(vals.astype(float), window)
        y_mean = w.mean(axis=1, keepdims=True)
        cov = np.sum((w - y_mean) * (x - x_mean), axis=1)
        return cov / x_var if x_var > 0 else np.zeros(len(vals))

    for col in columns:
        if "unit_id" in df.columns:
            slopes = np.zeros(len(df))
            for _, g in df.groupby("unit_id"):
                s = compute(g[col].values)
                slopes[df.index.get_indexer(g.index)] = s
            df[f"{col}_trend_slope_{window}"] = slopes
        else:
            df[f"{col}_trend_slope_{window}"] = compute(df[col].values)
    return df


def add_correlation_features(
    df: pd.DataFrame, sensor_pairs: list[tuple[str, str]], window: int = 15
) -> pd.DataFrame:
    """Rolling correlation between sensor pairs whose joint drift is a known failure indicator."""
    df = df.copy()
    for s1, s2 in sensor_pairs:
        if s1 not in df.columns or s2 not in df.columns:
            continue
        colname = f"corr_{s1}_{s2}_{window}"
        if "unit_id" in df.columns:
            corr = np.zeros(len(df))
            for _, g in df.groupby("unit_id"):
                c = g[s1].rolling(window, min_periods=2).corr(g[s2]).fillna(0)
                corr[df.index.get_indexer(g.index)] = c.values
            df[colname] = corr
        else:
            df[colname] = df[s1].rolling(window, min_periods=2).corr(df[s2]).fillna(0)
    return df


def engineer_features(
    df: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
    rolling_window: int = 5,
    ema_span: int = 5,
    lags: list[int] = (1, 3, 5),
    fft_window: int = 15,
    stat_window: int = 10,
    trend_window: int = 10,
) -> pd.DataFrame:
    """Full Module 3 pipeline: applies every feature family in sequence."""
    if df.empty:
        logger.warning("engineer_features received an empty dataframe; returning as-is.")
        return df

    if sensor_columns is None:
        sensor_columns = _sensor_columns(df)

    logger.info(f"Engineering features for {len(sensor_columns)} sensor columns")
    df = add_rolling_features(df, sensor_columns, rolling_window)
    df = add_ema_features(df, sensor_columns, ema_span)
    df = add_lag_features(df, sensor_columns, list(lags))
    df = add_fft_features(df, sensor_columns, fft_window)
    df = add_statistical_features(df, sensor_columns, stat_window)
    df = add_trend_features(df, sensor_columns, trend_window)

    # correlate the top-variance sensor pairs automatically
    variances = df[sensor_columns].var().sort_values(ascending=False)
    top_sensors = variances.index[:6].tolist()
    pairs = [(top_sensors[i], top_sensors[j])
             for i in range(len(top_sensors)) for j in range(i + 1, len(top_sensors))][:8]
    df = add_correlation_features(df, pairs)

    df = df.replace([np.inf, -np.inf], np.nan)
    engineered_cols = [c for c in df.columns if c not in df.columns[:0]]
    df[engineered_cols] = df[engineered_cols].fillna(0) if df.select_dtypes(include=[np.number]).isna().any().any() else df[engineered_cols]

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    from preprocessing import run_ingestion_pipeline

    df, _ = run_ingestion_pipeline("data/sensor_timeseries.csv")
    df = engineer_features(df.head(2000))  # quick smoke test on a slice
    print(df.shape)
    print([c for c in df.columns if "temperature" in c or "_lag_" in c][:10])
