"""
utils.py
========
Shared helpers used across the pipeline: paths, logging setup, model
persistence (joblib/keras aware), and the machine health-score formula.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


# ----------------------------------------------------------------------
# Model persistence
# ----------------------------------------------------------------------
def save_sklearn_model(model: Any, name: str) -> Path:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_sklearn_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)


def save_json(obj: dict, name: str, directory: Path = REPORTS_DIR) -> Path:
    path = directory / f"{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def load_json(name: str, directory: Path = REPORTS_DIR) -> dict:
    path = directory / f"{name}.json"
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Domain formulas
# ----------------------------------------------------------------------
@dataclass
class HealthAssessment:
    health_score: float          # 0-100, 100 = perfect health
    failure_probability: float   # 0-1
    estimated_rul_cycles: float
    estimated_rul_hours: float
    risk_level: str              # Low / Medium / High / Critical


def compute_health_score(
    failure_probability: float,
    rul_cycles: float,
    max_rul_reference: float = 200.0,
    anomaly_score: float = 0.0,
) -> HealthAssessment:
    """
    Combine failure probability, RUL, and anomaly score into a single
    0-100 health score and a categorical risk level.

    health_score = 100 * (1 - failure_probability) * rul_ratio * (1 - anomaly_weight)
    where rul_ratio is RUL capped/normalized against a reference max RUL,
    and anomaly_weight softly penalizes abnormal sensor behavior even if the
    classifier hasn't caught up yet.
    """
    rul_ratio = float(np.clip(rul_cycles / max_rul_reference, 0, 1))
    anomaly_weight = float(np.clip(anomaly_score, 0, 1)) * 0.3  # cap anomaly influence at 30%

    raw_score = 100 * (1 - failure_probability) * (0.4 + 0.6 * rul_ratio) * (1 - anomaly_weight)
    health_score = float(np.clip(raw_score, 0, 100))

    if health_score >= 80:
        risk = "Low"
    elif health_score >= 55:
        risk = "Medium"
    elif health_score >= 30:
        risk = "High"
    else:
        risk = "Critical"

    hours_per_cycle = 24.0  # assume 1 operating cycle == 1 day; adjust per real asset
    return HealthAssessment(
        health_score=round(health_score, 2),
        failure_probability=round(float(failure_probability), 4),
        estimated_rul_cycles=round(float(rul_cycles), 1),
        estimated_rul_hours=round(float(rul_cycles) * hours_per_cycle, 1),
        risk_level=risk,
    )
