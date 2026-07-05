"""
api/main.py
===========
FastAPI REST API for the Predictive Maintenance System.

Endpoints:
    GET  /health                      - liveness check
    GET  /models/info                 - which models are deployed + their metrics
    POST /predict                     - run inference on uploaded sensor JSON/CSV
    POST /predict/file                - run inference on an uploaded CSV file

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow importing sibling modules

from predict import PredictiveMaintenancePipeline
from utils import get_logger, load_json

logger = get_logger(__name__)

app = FastAPI(
    title="Predictive Maintenance API",
    description="Failure prediction, RUL estimation, anomaly detection, and "
                "maintenance recommendations for industrial sensor data.",
    version="1.0.0",
)

_pipeline: Optional[PredictiveMaintenancePipeline] = None


def get_pipeline() -> PredictiveMaintenancePipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = PredictiveMaintenancePipeline()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Models not trained yet. Run `python train.py` first. ({e})",
            )
    return _pipeline


class SensorReading(BaseModel):
    unit_id: int
    cycle: int
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    rpm: Optional[float] = None
    vibration: Optional[float] = None
    voltage: Optional[float] = None
    current: Optional[float] = None
    humidity: Optional[float] = None
    torque: Optional[float] = None
    tool_wear: Optional[float] = None
    oil_quality: Optional[float] = None
    coolant_flow: Optional[float] = None
    fan_speed: Optional[float] = None
    bearing_temp: Optional[float] = None
    exhaust_temp: Optional[float] = None


class PredictRequest(BaseModel):
    readings: list[SensorReading]
    latest_cycle_only: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models/info")
def models_info():
    try:
        summary = load_json("training_summary")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No training summary found. Run train.py first.")
    return summary


@app.post("/predict")
def predict(request: PredictRequest):
    pipeline = get_pipeline()
    records = [r.model_dump(exclude_none=True) for r in request.readings]
    if not records:
        raise HTTPException(status_code=400, detail="No sensor readings provided.")

    df = pd.DataFrame(records)
    try:
        report = pipeline.run(df, latest_cycle_only=request.latest_cycle_only)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {"n_machines": len(report), "predictions": report.to_dict(orient="records")}


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...), latest_cycle_only: bool = True):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    pipeline = get_pipeline()
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        report = pipeline.run(df, latest_cycle_only=latest_cycle_only)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {"n_machines": len(report), "predictions": report.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
