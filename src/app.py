
from __future__ import annotations

import json
import socket
import time
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

try:
    from .interface.dashboard import INTERFACE_STATIC_DIR, render_dashboard_html
    from .utils import (
        MODEL_DIR, OUTPUT_DIR, FIGURE_DIR, DATE_COL, UCI_COLUMNS, FEATURE_COLUMNS, HORIZON_MINUTES,
        load_dataset, make_supervised_frame, fill_missing_for_api, risk_from_prediction, recommendation_from_risk,
        reason_from_risk
    )
except ImportError:
    from interface.dashboard import INTERFACE_STATIC_DIR, render_dashboard_html
    from utils import (
        MODEL_DIR, OUTPUT_DIR, FIGURE_DIR, DATE_COL, UCI_COLUMNS, FEATURE_COLUMNS, HORIZON_MINUTES,
        load_dataset, make_supervised_frame, fill_missing_for_api, risk_from_prediction, recommendation_from_risk,
        reason_from_risk
    )

MODEL_BUNDLE_PATH = MODEL_DIR / "forecast_model_bundle_v1.joblib"
METRICS_PATH = OUTPUT_DIR / "forecast_metrics.json"
FORECAST_LOG_PATH = OUTPUT_DIR / "forecast_log.csv"

app = FastAPI(
    title="LAB 4 AIoT Forecasting API",
    description="Demo deploy forecasting model: telemetry history -> predicted_value -> risk_level -> recommendation",
    version="1.0.0"
)
app.mount("/figures", StaticFiles(directory=FIGURE_DIR), name="figures")
app.mount("/interface-static", StaticFiles(directory=INTERFACE_STATIC_DIR), name="interface-static")

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


class ManualForecastRequest(BaseModel):
    model_name: str | None = "random_forest_v1"
    Appliances: float = Field(80.0, ge=0)
    lights: float | None = Field(5.0, ge=0)
    T1: float | None = 21.0
    RH_1: float | None = 40.0
    T_out: float | None = 8.0
    RH_out: float | None = 80.0
    Windspeed: float | None = Field(4.0, ge=0)
    Visibility: float | None = Field(40.0, ge=0)
    Press_mm_hg: float | None = Field(755.0, ge=0)
    Tdewpoint: float | None = 5.0
    horizon_minutes: int | None = Field(10, ge=10)
    use_case: str | None = "energy_load_warning"


def _dump_model(point: TelemetryPoint) -> dict[str, Any]:
    if hasattr(point, "model_dump"):
        return point.model_dump()
    return point.dict()


def _predict_from_rows(rows: list[dict[str, Any]], warnings: list[str] | None = None, model_name: str | None = None) -> dict[str, Any]:
    if model_bundle is None:
        return {"error": "Model chưa được train. Hãy chạy: python src/train_forecast.py"}

    start = time.time()
    warnings = warnings or []
    if len(rows) < 24:
        warnings.append("history có ít hơn 24 điểm; lag/rolling feature có thể chưa ổn định.")

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

    selected_model_name = model_name or model_bundle.get("model_version", "forecast_v1")
    trained_models = model_bundle.get("trained_models", {})
    model = trained_models.get(selected_model_name, model_bundle["model"])
    if selected_model_name not in trained_models:
        selected_model_name = model_bundle.get("model_version", "forecast_v1")

    predicted_value = float(model.predict(X)[0])
    predicted_value = max(predicted_value, 0.0)
    interval_widths = model_bundle.get("prediction_interval_widths", {})
    interval_90 = float(interval_widths.get("q90", 0.0))
    interval_95 = float(interval_widths.get("q95", 0.0))
    thresholds = model_bundle.get("risk_thresholds", {"warning": 80, "high": 140, "critical": 220})
    risk_level = risk_from_prediction(predicted_value, thresholds)
    recommendation = recommendation_from_risk(risk_level)

    return {
        "model_output": {
            "target": model_bundle.get("target", "Appliances"),
            "forecast_horizon_minutes": model_bundle.get("forecast_horizon_minutes", HORIZON_MINUTES),
            "predicted_value": round(predicted_value, 4),
            "unit": "Wh per 10-minute interval",
            "model_version": selected_model_name,
            "prediction_interval_90": {
                "lower": round(max(predicted_value - interval_90, 0.0), 4),
                "upper": round(predicted_value + interval_90, 4),
            },
            "prediction_interval_95": {
                "lower": round(max(predicted_value - interval_95, 0.0), 4),
                "upper": round(predicted_value + interval_95, 4),
            },
        },
        "evaluation_hint": {
            "metrics_file": "outputs/forecast_metrics.json",
            "selected_model_mae": model_bundle.get("metrics_by_model", {}).get(selected_model_name, {}).get("mae"),
            "selected_model_rmse": model_bundle.get("metrics_by_model", {}).get(selected_model_name, {}).get("rmse"),
        },
        "decision": {
            "risk_level": risk_level,
            "recommendation": recommendation,
            "reason": reason_from_risk(predicted_value, thresholds),
            "safety_note": "Forecast output is a recommendation signal, not an automatic actuator command. Apply safety rules and human confirmation before control.",
        },
        "api_check": {
            "latency_ms": round((time.time() - start) * 1000, 2),
            "input_points": len(rows),
            "warnings": warnings,
        }
    }


def _manual_payload_to_history(payload: ManualForecastRequest, points: int = 24) -> list[dict[str, Any]]:
    values = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    values.pop("model_name", None)
    values.pop("horizon_minutes", None)
    values.pop("use_case", None)

    try:
        history_df = load_dataset().tail(points).copy()
        for col, value in values.items():
            if col in history_df.columns and value is not None:
                history_df.loc[history_df.index[-1], col] = value
        history_df[DATE_COL] = pd.to_datetime(history_df[DATE_COL]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return history_df.to_dict(orient="records")
    except Exception:
        end_time = pd.Timestamp("2016-01-27 18:00:00")
        rows = []
        for idx in range(points):
            timestamp = end_time - pd.Timedelta(minutes=10 * (points - idx - 1))
            row = {DATE_COL: timestamp.strftime("%Y-%m-%d %H:%M:%S")}
            row.update(values)
            rows.append(row)
        return rows


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(render_dashboard_html(model_bundle is not None))


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
    rows = [_dump_model(p) for p in payload.history]
    return _predict_from_rows(rows)


@app.post("/forecast-manual")
def forecast_manual(payload: ManualForecastRequest):
    rows = _manual_payload_to_history(payload)
    result = _predict_from_rows(rows, model_name=payload.model_name)
    if "error" not in result:
        result["manual_input"] = {
            "selected_model": payload.model_name,
            "requested_horizon_minutes": payload.horizon_minutes,
            "use_case": payload.use_case,
            "note": "Model hiện tại được train cho horizon lưu trong model bundle; muốn đổi horizon thật sự cần train lại model.",
        }
    return result


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    while port < 8010:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) != 0:
                break
        port += 1

    print(f"Starting AIoT Forecasting Dashboard: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)
