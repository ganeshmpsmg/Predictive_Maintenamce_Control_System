"""
maintenance_engine.py
======================
Module 7: Maintenance Recommendation Engine.

A transparent, rule-based expert system (deliberately NOT a black box here —
maintenance recommendations need to be auditable) that turns model outputs
(health score, risk level, failure probability, anomaly flags, and which
sensors are driving the anomaly) into actionable guidance:
- likely failure reason / root cause
- maintenance priority
- suggested replacement parts
- estimated repair cost (range, in USD)
- estimated downtime (hours)

The sensor -> component -> failure-mode mapping below is illustrative
(typical of rotating/industrial equipment); in a real deployment this
knowledge base would be built with the plant's reliability engineers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Sensor -> (component, failure mode, replacement part, cost range USD, downtime hours)
SENSOR_KNOWLEDGE_BASE = {
    "temperature":   ("Cooling System", "Overheating / coolant degradation", "Coolant + Thermostat", (150, 500), (2, 6)),
    "bearing_temp":  ("Bearing Assembly", "Bearing wear / lubrication failure", "Bearing Set", (300, 1200), (4, 12)),
    "exhaust_temp":  ("Combustion/Exhaust System", "Exhaust restriction or seal failure", "Exhaust Gasket Kit", (200, 800), (3, 8)),
    "pressure":      ("Hydraulic/Pneumatic System", "Seal degradation / pressure loss", "Seal Kit + Pump Check", (250, 900), (3, 10)),
    "rpm":           ("Drive/Motor Assembly", "Motor imbalance or drive slippage", "Drive Belt / Motor Bearing", (200, 1500), (4, 16)),
    "vibration":     ("Rotating Assembly", "Misalignment / imbalance / bearing wear", "Alignment Service + Bearings", (400, 2000), (6, 20)),
    "voltage":       ("Electrical System", "Voltage instability / winding degradation", "Voltage Regulator", (150, 700), (2, 8)),
    "current":       ("Electrical/Motor System", "Overcurrent / winding short risk", "Motor Windings Inspection", (300, 2500), (6, 24)),
    "humidity":      ("Enclosure/Sealing", "Moisture ingress risk", "Enclosure Seal / Desiccant", (50, 300), (1, 4)),
    "torque":        ("Drivetrain", "Overload / gear wear", "Gearbox Service", (400, 1800), (6, 18)),
    "tool_wear":     ("Cutting Tool", "Tool wear beyond tolerance", "Cutting Tool / Insert Replacement", (30, 250), (0.5, 2)),
    "oil_quality":   ("Lubrication System", "Oil degradation / contamination", "Oil + Filter Change", (80, 300), (1, 3)),
    "coolant_flow":  ("Cooling System", "Coolant flow restriction / pump wear", "Coolant Pump", (200, 900), (3, 8)),
    "fan_speed":     ("Cooling Fan Assembly", "Fan motor degradation", "Fan Motor Assembly", (100, 500), (1, 4)),
}

DEFAULT_ENTRY = ("General Machinery", "Unspecified degradation pattern", "General Inspection", (100, 500), (2, 8))


@dataclass
class MaintenanceRecommendation:
    machine_id: str
    risk_level: str
    health_score: float
    failure_probability: float
    likely_component: str
    root_cause: str
    maintenance_priority: str
    suggested_parts: list[str] = field(default_factory=list)
    estimated_cost_usd: tuple = (0, 0)
    estimated_downtime_hours: tuple = (0, 0)
    recommended_action: str = ""
    driving_sensors: list[str] = field(default_factory=list)


def _priority_from_risk(risk_level: str) -> str:
    return {
        "Critical": "Immediate (schedule within 24 hours)",
        "High": "Urgent (schedule within 3-5 days)",
        "Medium": "Planned (schedule within 2-3 weeks)",
        "Low": "Routine (monitor at next scheduled service)",
    }.get(risk_level, "Routine")


def _identify_driving_sensors(row: pd.Series, sensor_names: list[str], top_n: int = 3) -> list[str]:
    """
    Identify which raw sensors show the most extreme (standardized) values
    for this row -- used as a proxy for "what's driving the anomaly" when
    SHAP explanations aren't available/needed for a quick recommendation.
    """
    available = [s for s in sensor_names if s in row.index]
    if not available:
        return []
    magnitudes = row[available].abs()
    return magnitudes.sort_values(ascending=False).index[:top_n].tolist()


def generate_recommendation(
    row: pd.Series,
    sensor_names: list[str],
    machine_id_col: str = "unit_id",
) -> MaintenanceRecommendation:
    """Build a single maintenance recommendation from one prediction row."""
    
    # --- FIXED: Handle duplicate risk_level columns safely ---
    risk_level = row.get("risk_level", "Medium")
    if isinstance(risk_level, pd.Series):
        risk_level = risk_level.iloc[0]
    risk_level = str(risk_level)

    # --- FIXED: Handle duplicate health_score columns safely ---
    health_score = row.get("health_score", 50.0)
    if isinstance(health_score, pd.Series):
        health_score = health_score.iloc[0]
    health_score = float(health_score)

    # --- FIXED: Handle duplicate failure_probability columns safely ---
    failure_probability = row.get("failure_probability", 0.5)
    if isinstance(failure_probability, pd.Series):
        failure_probability = failure_probability.iloc[0]
    failure_probability = float(failure_probability)

    driving_sensors = _identify_driving_sensors(row, sensor_names)
    primary_sensor = driving_sensors[0] if driving_sensors else None
    component, root_cause, part, cost_range, downtime_range = SENSOR_KNOWLEDGE_BASE.get(
        primary_sensor, DEFAULT_ENTRY
    )

    # scale cost/downtime estimate by severity within the risk band
    severity_multiplier = 1.0 + (1.0 - health_score / 100.0)  # 1.0 (healthy) -> 2.0 (critical)
    cost = (round(cost_range[0] * severity_multiplier), round(cost_range[1] * severity_multiplier))
    downtime = (round(downtime_range[0] * severity_multiplier, 1), round(downtime_range[1] * severity_multiplier, 1))

    action = {
        "Critical": f"Stop operation and inspect {component.lower()} immediately. Failure risk is imminent.",
        "High": f"Schedule urgent maintenance on {component.lower()} within the week.",
        "Medium": f"Plan a maintenance window for {component.lower()} in the coming weeks; monitor trend.",
        "Low": f"No action required beyond routine monitoring of {component.lower()}.",
    }.get(risk_level, f"Monitor {component.lower()}.")

    # --- FIXED: Defensive machine_id handling in case it's also a Series ---
    machine_id_val = row.get(machine_id_col, "unknown")
    if isinstance(machine_id_val, pd.Series):
        machine_id_val = machine_id_val.iloc[0]

    return MaintenanceRecommendation(
        machine_id=str(machine_id_val),
        risk_level=risk_level,
        health_score=health_score,
        failure_probability=failure_probability,
        likely_component=component,
        root_cause=root_cause,
        maintenance_priority=_priority_from_risk(risk_level),
        suggested_parts=[part],
        estimated_cost_usd=cost,
        estimated_downtime_hours=downtime,
        recommended_action=action,
        driving_sensors=driving_sensors,
    )


def generate_recommendations_batch(
    df: pd.DataFrame, sensor_names: list[str], machine_id_col: str = "unit_id"
) -> pd.DataFrame:
    """Vectorized-friendly wrapper: applies generate_recommendation across all rows."""
    records = []
    for _, row in df.iterrows():
        rec = generate_recommendation(row, sensor_names, machine_id_col)
        records.append(rec.__dict__)
    return pd.DataFrame(records)


if __name__ == "__main__":
    demo_row = pd.Series({
        "unit_id": 12, "risk_level": "Critical", "health_score": 18.5,
        "failure_probability": 0.87, "bearing_temp": 3.4, "vibration": 2.9, "temperature": 1.1,
    })
    rec = generate_recommendation(demo_row, ["bearing_temp", "vibration", "temperature"])
    print(rec)