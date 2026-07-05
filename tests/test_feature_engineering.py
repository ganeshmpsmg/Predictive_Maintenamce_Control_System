"""Unit tests for feature_engineering.py (Module 3: Feature Engineering)."""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    add_ema_features,
    add_fft_features,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
    engineer_features,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(1)
    n = 60
    return pd.DataFrame({
        "unit_id": [1] * n,
        "cycle": range(1, n + 1),
        "temperature": 50 + np.cumsum(rng.normal(0, 0.5, n)),
        "vibration": 5 + np.cumsum(rng.normal(0, 0.1, n)),
    })


def test_rolling_features_add_expected_columns(sample_df):
    result = add_rolling_features(sample_df, ["temperature"], window=5)
    assert "temperature_roll_mean_5" in result.columns
    assert "temperature_roll_std_5" in result.columns
    assert len(result) == len(sample_df)


def test_ema_features(sample_df):
    result = add_ema_features(sample_df, ["temperature"], span=5)
    assert "temperature_ema_5" in result.columns
    assert result["temperature_ema_5"].isna().sum() == 0


def test_lag_features_no_nans(sample_df):
    result = add_lag_features(sample_df, ["temperature"], lags=[1, 3])
    assert result["temperature_lag_1"].isna().sum() == 0
    assert result["temperature_lag_3"].isna().sum() == 0


def test_fft_features_bounded(sample_df):
    result = add_fft_features(sample_df, ["temperature"], window=10)
    assert "temperature_fft_energy" in result.columns
    assert "temperature_fft_entropy" in result.columns
    # dominant-band energy and entropy are both normalized fractions in [0, 1]
    assert result["temperature_fft_energy"].between(0, 1.0001).all()
    assert result["temperature_fft_entropy"].between(0, 1.0001).all()


def test_trend_features_direction(sample_df):
    """A strictly increasing series should have a positive trend slope."""
    df = sample_df.copy()
    df["ramp"] = np.arange(len(df), dtype=float)
    result = add_trend_features(df, ["ramp"], window=10)
    # after the initial padded window, slope should be ~1.0 (unit ramp)
    assert result["ramp_trend_slope_10"].iloc[-1] == pytest.approx(1.0, abs=0.05)


def test_engineer_features_end_to_end(sample_df):
    result = engineer_features(sample_df)
    assert len(result) == len(sample_df)
    assert result.shape[1] > sample_df.shape[1]
    # no NaNs or infinities should remain in the engineered output
    numeric = result.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any()


def test_engineer_features_handles_empty_dataframe():
    empty = pd.DataFrame(columns=["unit_id", "cycle", "temperature"])
    result = engineer_features(empty)
    assert result.empty


def test_engineer_features_handles_single_row():
    single = pd.DataFrame({"unit_id": [1], "cycle": [1], "temperature": [50.0]})
    result = engineer_features(single)
    assert len(result) == 1
