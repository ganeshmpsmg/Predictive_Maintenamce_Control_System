"""Unit tests for preprocessing.py (Module 1: Data Ingestion)."""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    detect_outliers,
    handle_missing_values,
    normalize_sensors,
    remove_duplicates,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "unit_id": [1] * 20 + [2] * 20,
        "cycle": list(range(1, 21)) * 2,
        "temperature": rng.normal(50, 2, 40),
        "vibration": rng.normal(5, 0.5, 40),
    })


def test_handle_missing_values_interpolate(sample_df):
    df = sample_df.copy()
    df.loc[3, "temperature"] = np.nan
    df.loc[10, "vibration"] = np.nan
    result = handle_missing_values(df, strategy="interpolate")
    assert result["temperature"].isna().sum() == 0
    assert result["vibration"].isna().sum() == 0
    assert len(result) == len(df)


def test_handle_missing_values_median(sample_df):
    df = sample_df.copy()
    df.loc[0, "temperature"] = np.nan
    result = handle_missing_values(df, strategy="median")
    assert result["temperature"].isna().sum() == 0


def test_handle_missing_values_preserves_index_alignment():
    """Regression test: a non-contiguous index (e.g. a sliced dataframe) must
    not corrupt values -- this was a real bug caught during development."""
    df = pd.DataFrame({
        "unit_id": [5, 5, 5, 5, 5],
        "cycle": [1, 2, 3, 4, 5],
        "temperature": [10.0, 11.0, 12.0, 13.0, 14.0],
    })
    sliced = df.iloc[[1, 2, 3]]  # non-contiguous index: 1, 2, 3
    result = handle_missing_values(sliced, strategy="interpolate")
    # values must be unchanged (no missing values existed) and correctly aligned
    assert list(result["temperature"]) == [11.0, 12.0, 13.0]


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [1, 1, 2, 3]})
    result = remove_duplicates(df)
    assert len(result) == 3


def test_detect_outliers_flags_extreme_values():
    df = pd.DataFrame({
        "unit_id": [1] * 10,
        "cycle": range(1, 11),
        "temperature": [50, 51, 49, 50, 52, 48, 50, 51, 500, 49],  # 500 is an outlier
    })
    result = detect_outliers(df, columns=["temperature"], method="zscore", threshold=2.5)
    assert result["is_outlier"].sum() >= 1
    assert result.loc[result["temperature"] == 500, "is_outlier"].all()


def test_normalize_sensors_returns_reusable_scaler(sample_df):
    df1, scaler = normalize_sensors(sample_df, columns=["temperature", "vibration"])
    assert abs(df1["temperature"].mean()) < 1e-6  # standardized: mean ~0

    # reusing the fitted scaler on new data should not refit
    df2, scaler2 = normalize_sensors(sample_df.copy(), columns=["temperature", "vibration"], scaler=scaler)
    assert scaler2 is scaler
