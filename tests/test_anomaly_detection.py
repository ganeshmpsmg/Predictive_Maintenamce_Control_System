"""Unit tests for anomaly_detection.py (Module 6)."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection import score_anomalies, train_isolation_forest, train_lof, train_one_class_svm


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(3)
    n = 300
    normal = rng.normal(0, 1, (n, 3))
    df = pd.DataFrame(normal, columns=["f1", "f2", "f3"])
    df["unit_id"] = 1
    df["failure_within_30_cycles"] = 0
    # inject a handful of obvious anomalies
    df.loc[295:, ["f1", "f2", "f3"]] = rng.normal(15, 1, (5, 3))
    return df


def test_isolation_forest_flags_injected_anomalies(sample_df):
    model = train_isolation_forest(sample_df, ["f1", "f2", "f3"], contamination=0.02)
    scored = score_anomalies(sample_df, ["f1", "f2", "f3"], iso_forest=model)
    assert scored["anomaly_isoforest"].iloc[295:].sum() >= 3  # most injected anomalies caught


def test_one_class_svm_runs(sample_df):
    model = train_one_class_svm(sample_df, ["f1", "f2", "f3"], nu=0.05)
    scored = score_anomalies(sample_df, ["f1", "f2", "f3"], ocsvm=model)
    assert "anomaly_ocsvm" in scored.columns


def test_lof_novelty_scoring(sample_df):
    model = train_lof(sample_df, ["f1", "f2", "f3"], n_neighbors=10, contamination=0.02)
    scored = score_anomalies(sample_df, ["f1", "f2", "f3"], lof=model)
    assert "anomaly_lof" in scored.columns


def test_ensemble_anomaly_score_bounded(sample_df):
    iso = train_isolation_forest(sample_df, ["f1", "f2", "f3"])
    scored = score_anomalies(sample_df, ["f1", "f2", "f3"], iso_forest=iso)
    assert scored["anomaly_score"].between(0, 1).all()


def test_score_anomalies_with_no_detectors_returns_zero_score(sample_df):
    scored = score_anomalies(sample_df, ["f1", "f2", "f3"])
    assert (scored["anomaly_score"] == 0.0).all()
