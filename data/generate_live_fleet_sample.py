"""
generate_live_fleet_sample.py
==============================
The full run-to-failure dataset (sensor_timeseries.csv) is exactly that --
every unit's trajectory runs all the way to its failure point, because that's
how degradation models are trained (they need to see the whole failure curve,
cycle-by-cycle, to learn what "approaching failure" looks like).

But a live monitoring dashboard should show a *fleet of machines currently in
operation* -- a realistic mix of healthy, aging, and at-risk machines, not a
list where every single one is one cycle from failure.

This script simulates that: for each unit, it truncates the trajectory at a
random point in its life (anywhere from 30% to 99% of its total lifespan),
mimicking "here's the sensor history up to right now" for a fleet of machines
caught at arbitrary points in their operating life.

Run:
    python data/generate_live_fleet_sample.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
RNG = np.random.default_rng(7)


def build_live_fleet_snapshot(source_path: str = "sensor_timeseries.csv") -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / source_path)
    df = df.dropna(subset=["unit_id"])

    truncated_rows = []
    for unit_id, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        max_cycle = g["cycle"].max()
        # pick a random "current" point in this machine's life
        cutoff_fraction = RNG.uniform(0.30, 0.99)
        cutoff_cycle = max(1, int(max_cycle * cutoff_fraction))
        truncated_rows.append(g[g["cycle"] <= cutoff_cycle])

    live_df = pd.concat(truncated_rows, ignore_index=True)
    return live_df


if __name__ == "__main__":
    live_df = build_live_fleet_snapshot()
    out_path = DATA_DIR / "live_fleet_snapshot.csv"
    live_df.to_csv(out_path, index=False)
    print(f"live_fleet_snapshot.csv -> {live_df.shape} "
          f"({live_df['unit_id'].nunique()} machines, mixed health states)")
