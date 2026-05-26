
from __future__ import annotations

import json
import time
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from utils import (
    MODEL_DIR, OUTPUT_DIR, DATE_COL, UCI_COLUMNS, FEATURE_COLUMNS, HORIZON_MINUTES,
    make_supervised_frame, fill_missing_for_api, risk_from_prediction, recommendation_from_risk,
    reason_from_risk
)

MODEL_BUNDLE_PATH = MODEL_DIR / "forecast_model_bundle_v1.joblib"
METRICS_PATH = OUTPUT_DIR / "forecast_metrics.json"

app = FastAPI(
    title="LAB 4 AIoT Forecasting API",
    description="Demo deploy forecasting model: telemetry history -> predicted_value -> risk_level -> recommendation",
    version="1.0.0"
)

model_bundle = None
if MODEL_BUNDLE_PATH.exists():
    model_bundle = joblib.load(MODEL_BUNDLE_PATH)


class TelemetryPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str = Field(..., examples=["2016-01-21 12:00:00"])
    Appliances: float = Field(..., examples=[80.0])
    lights: float | None = None
    T1: float | None = None
    RH_1: float | None = None
    T2: float | None = None
    RH_2: float | None = None
    T3: float | None = None
    RH_3: float | None = None
    T4: float | None = None
    RH_4: float | None = None
    T5: float | None = None
    RH_5: float | None = None
    T6: float | None = None
    RH_6: float | None = None
    T7: float | None = None
    RH_7: float | None = None
    T8: float | None = None
    RH_8: float | None = None
    T9: float | None = None
    RH_9: float | None = None
    T_out: float | None = None
    Press_mm_hg: float | None = None
    RH_out: float | None = None
    Windspeed: float | None = None
    Visibility: float | None = None
    Tdewpoint: float | None = None


class ForecastRequest(BaseModel):
    history: list[TelemetryPoint] = Field(
        ...,
        description="Recent telemetry history. Send at least 24 points for stable lag/rolling features."
    )


def _dump_model(point: TelemetryPoint) -> dict[str, Any]:
    if hasattr(point, "model_dump"):
        return point.model_dump()
    return point.dict()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
        "model_bundle_path": str(MODEL_BUNDLE_PATH),
    }


@app.get("/model-info")
def model_info():
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    if model_bundle is None:
        return {
            "model_loaded": False,
            "message": "Chưa có model. Hãy chạy: python src/download_data.py && python src/train_forecast.py"
        }

    return {
        "model_loaded": True,
        "model_name": type(model_bundle["model"]).__name__,
        "model_version": model_bundle.get("model_version", "unknown"),
        "target": model_bundle.get("target", "Appliances"),
        "forecast_horizon_minutes": model_bundle.get("forecast_horizon_minutes", HORIZON_MINUTES),
        "input": "history of UCI Appliances telemetry rows",
        "output": "predicted_value, risk_level, recommendation, safety_note",
        "feature_count": len(model_bundle.get("feature_columns", FEATURE_COLUMNS)),
        "risk_thresholds": model_bundle.get("risk_thresholds", {}),
        "metrics": metrics,
    }


@app.post("/forecast")
def forecast(payload: ForecastRequest):
    if model_bundle is None:
        return {"error": "Model chưa được train. Hãy chạy: python src/train_forecast.py"}

    start = time.time()
    warnings = []
    if len(payload.history) < 24:
        warnings.append("history có ít hơn 24 điểm; lag/rolling feature có thể chưa ổn định.")

    rows = [_dump_model(p) for p in payload.history]
    df = pd.DataFrame(rows)
    if DATE_COL not in df.columns:
        return {"error": "Payload cần có cột date trong từng telemetry point."}

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Ensure all expected raw columns are present and fill missing optional values using training medians.
    df = fill_missing_for_api(df, model_bundle.get("raw_medians", {}))
    features_df = make_supervised_frame(df, horizon_steps=model_bundle.get("forecast_horizon_steps", 1), include_target=False)
    latest = features_df.iloc[[-1]].copy()

    feature_columns = model_bundle.get("feature_columns", FEATURE_COLUMNS)
    for col in feature_columns:
        if col not in latest.columns:
            latest[col] = float(model_bundle.get("feature_medians", {}).get(col, 0.0))
    X = latest[feature_columns].replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(model_bundle.get("feature_medians", {})).fillna(0.0)

    model = model_bundle["model"]
    predicted_value = float(model.predict(X)[0])
    predicted_value = max(predicted_value, 0.0)
    thresholds = model_bundle.get("risk_thresholds", {"warning": 80, "high": 140, "critical": 220})
    risk_level = risk_from_prediction(predicted_value, thresholds)
    recommendation = recommendation_from_risk(risk_level)

    return {
        "model_output": {
            "target": model_bundle.get("target", "Appliances"),
            "forecast_horizon_minutes": model_bundle.get("forecast_horizon_minutes", HORIZON_MINUTES),
            "predicted_value": round(predicted_value, 4),
            "unit": "Wh per 10-minute interval",
            "model_version": model_bundle.get("model_version", "forecast_v1"),
        },
        "evaluation_hint": {
            "metrics_file": "outputs/forecast_metrics.json",
            "best_model_mae": model_bundle.get("metrics_by_model", {}).get(model_bundle.get("model_version", ""), {}).get("mae"),
            "best_model_rmse": model_bundle.get("metrics_by_model", {}).get(model_bundle.get("model_version", ""), {}).get("rmse"),
        },
        "decision": {
            "risk_level": risk_level,
            "recommendation": recommendation,
            "reason": reason_from_risk(predicted_value, thresholds),
            "safety_note": "Forecast output is a recommendation signal, not an automatic actuator command. Apply safety rules and human confirmation before control.",
        },
        "api_check": {
            "latency_ms": round((time.time() - start) * 1000, 2),
            "input_points": len(payload.history),
            "warnings": warnings,
        }
    }
