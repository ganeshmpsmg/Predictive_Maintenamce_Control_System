"""
generate_synthetic_data.py
==========================
Generates a physically-plausible, schema-compatible synthetic dataset that mimics:

1. NASA C-MAPSS turbofan degradation format:
   unit_id, cycle, op_setting_1..3, sensor_1..14 (temperature, pressure, RPM,
   vibration, voltage, current, humidity-analog, etc.), RUL (derived)

2. AI4I-style discrete failure record format:
   air_temp, process_temp, rotational_speed, torque, tool_wear, machine_failure,
   failure_type

This exists ONLY because live internet access to the original NASA/UCI datasets
is not available in this sandboxed environment. The code in every downstream
module (preprocessing, feature engineering, training, etc.) reads a generic
column schema, so swapping this file's CSV output for the real CMAPSS/AI4I
CSVs (same column names) requires no code changes elsewhere.

Run:
    python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).parent


def generate_cmapss_like(n_units: int = 60, max_cycle_range=(150, 350)) -> pd.DataFrame:
    """Simulate multivariate run-to-failure degradation trajectories."""
    all_rows = []

    sensor_names = [
        "temperature", "pressure", "rpm", "vibration", "voltage",
        "current", "humidity", "torque", "tool_wear", "oil_quality",
        "coolant_flow", "fan_speed", "bearing_temp", "exhaust_temp",
    ]

    for unit_id in range(1, n_units + 1):
        max_cycle = RNG.integers(*max_cycle_range)
        t = np.arange(1, max_cycle + 1)
        degradation = (t / max_cycle) ** 1.7  # nonlinear wear curve, 0 -> 1

        # operational settings (regimes)
        op1 = RNG.normal(0, 1, size=max_cycle)
        op2 = RNG.normal(0, 1, size=max_cycle)
        op3 = RNG.choice([0, 1, 2], size=max_cycle)

        unit_df = pd.DataFrame({
            "unit_id": unit_id,
            "cycle": t,
            "op_setting_1": op1,
            "op_setting_2": op2,
            "op_setting_3": op3,
        })

        # Each sensor: baseline + drift with degradation + noise; some sensors
        # trend up, some down, to mimic real CMAPSS sensor behavior diversity.
        baselines = RNG.uniform(20, 100, size=len(sensor_names))
        directions = RNG.choice([-1, 1], size=len(sensor_names))
        drift_scale = RNG.uniform(5, 40, size=len(sensor_names))
        noise_scale = RNG.uniform(0.3, 2.0, size=len(sensor_names))

        for i, name in enumerate(sensor_names):
            trend = directions[i] * drift_scale[i] * degradation
            noise = RNG.normal(0, noise_scale[i], size=max_cycle)
            # small random spikes to emulate transient anomalies
            spike_mask = RNG.random(max_cycle) < 0.01
            spikes = spike_mask * RNG.normal(0, drift_scale[i] * 2, size=max_cycle)
            unit_df[name] = baselines[i] + trend + noise + spikes

        unit_df["RUL"] = max_cycle - t
        # Binary failure-imminent label: failure within next 30 cycles
        unit_df["failure_within_30_cycles"] = (unit_df["RUL"] <= 30).astype(int)
        all_rows.append(unit_df)

    df = pd.concat(all_rows, ignore_index=True)

    # inject missing values and duplicates to exercise cleaning code
    mask = RNG.random(df.shape) < 0.003
    df_values = df.values.astype(object)
    df_values[mask] = np.nan
    df = pd.DataFrame(df_values, columns=df.columns)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def generate_ai4i_like(n_samples: int = 8000) -> pd.DataFrame:
    """Simulate discrete-record industrial failure dataset (AI4I-style)."""
    air_temp = RNG.normal(300, 2, n_samples)
    process_temp = air_temp + RNG.normal(10, 1, n_samples)
    rot_speed = RNG.normal(1500, 180, n_samples).clip(800, 2800)
    torque = RNG.normal(40, 10, n_samples).clip(3, 80)
    tool_wear = RNG.integers(0, 260, n_samples)

    power = torque * rot_speed
    overstrain = tool_wear * torque

    failure_prob = (
        0.00025 * tool_wear
        + 0.00001 * np.maximum(0, overstrain - 11000)
        + 0.02 * (process_temp - air_temp < 8.5).astype(float)
        + 0.015 * (power < 3500).astype(float)
        + RNG.normal(0, 0.02, n_samples)
    )
    failure_prob = np.clip(failure_prob, 0, 1)
    machine_failure = (RNG.random(n_samples) < failure_prob).astype(int)

    failure_types = np.where(
        machine_failure == 0, "None",
        RNG.choice(
            ["Tool Wear Failure", "Heat Dissipation Failure",
             "Power Failure", "Overstrain Failure", "Random Failure"],
            size=n_samples,
        ),
    )

    df = pd.DataFrame({
        "air_temperature_K": air_temp,
        "process_temperature_K": process_temp,
        "rotational_speed_rpm": rot_speed,
        "torque_Nm": torque,
        "tool_wear_min": tool_wear,
        "machine_failure": machine_failure,
        "failure_type": failure_types,
    })
    return df


if __name__ == "__main__":
    cmapss_df = generate_cmapss_like()
    ai4i_df = generate_ai4i_like()

    cmapss_df.to_csv(DATA_DIR / "sensor_timeseries.csv", index=False)
    ai4i_df.to_csv(DATA_DIR / "discrete_failures.csv", index=False)

    print(f"sensor_timeseries.csv -> {cmapss_df.shape}")
    print(f"discrete_failures.csv -> {ai4i_df.shape}")
