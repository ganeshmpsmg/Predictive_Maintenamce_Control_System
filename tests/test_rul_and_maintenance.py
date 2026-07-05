"""Unit tests for rul_prediction.py (Module 5) and maintenance_engine.py (Module 7)."""

import numpy as np
import pandas as pd
import pytest

from utils import compute_health_score
from maintenance_engine import generate_recommendation, _identify_driving_sensors


def test_compute_health_score_healthy_machine():
    assessment = compute_health_score(failure_probability=0.02, rul_cycles=200, max_rul_reference=250)
    assert assessment.risk_level == "Low"
    assert assessment.health_score > 70


def test_compute_health_score_critical_machine():
    assessment = compute_health_score(failure_probability=0.95, rul_cycles=3, max_rul_reference=250)
    assert assessment.risk_level == "Critical"
    assert assessment.health_score < 30


def test_compute_health_score_monotonic_in_rul():
    """Higher RUL (all else equal) should never produce a lower health score."""
    low_rul = compute_health_score(0.3, rul_cycles=10, max_rul_reference=250)
    high_rul = compute_health_score(0.3, rul_cycles=200, max_rul_reference=250)
    assert high_rul.health_score >= low_rul.health_score


def test_compute_health_score_bounded():
    for fp in [0.0, 0.5, 1.0]:
        for rul in [0, 100, 1000]:
            assessment = compute_health_score(fp, rul)
            assert 0 <= assessment.health_score <= 100


def test_identify_driving_sensors_picks_largest_magnitude():
    row = pd.Series({"temperature": 0.2, "vibration": 3.5, "bearing_temp": -2.9})
    drivers = _identify_driving_sensors(row, ["temperature", "vibration", "bearing_temp"], top_n=2)
    assert drivers[0] == "vibration"
    assert drivers[1] == "bearing_temp"


def test_generate_recommendation_critical_machine():
    row = pd.Series({
        "unit_id": 7, "risk_level": "Critical", "health_score": 12.0,
        "failure_probability": 0.9, "bearing_temp": 3.1, "vibration": 2.0,
    })
    rec = generate_recommendation(row, ["bearing_temp", "vibration"])
    assert rec.risk_level == "Critical"
    assert "Immediate" in rec.maintenance_priority
    assert rec.likely_component == "Bearing Assembly"
    assert rec.estimated_cost_usd[0] < rec.estimated_cost_usd[1]


def test_generate_recommendation_low_risk_machine():
    row = pd.Series({
        "unit_id": 3, "risk_level": "Low", "health_score": 92.0,
        "failure_probability": 0.01, "temperature": 0.1,
    })
    rec = generate_recommendation(row, ["temperature"])
    assert rec.risk_level == "Low"
    assert "Routine" in rec.maintenance_priority


def test_generate_recommendation_falls_back_to_default_for_unknown_sensor():
    row = pd.Series({"unit_id": 1, "risk_level": "Medium", "health_score": 60.0, "failure_probability": 0.4})
    rec = generate_recommendation(row, [])  # no sensors available
    assert rec.likely_component == "General Machinery"
