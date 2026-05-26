
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

TARGET_COL = "Appliances"
DATE_COL = "date"
HORIZON_STEPS = 1
HORIZON_MINUTES = 10
MODEL_VERSION = "forecast_v1"

UCI_COLUMNS = [
    "date", "Appliances", "lights",
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
    "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2"
]

# rv1/rv2 are random variables in the UCI dataset; they are intentionally excluded.
EXOGENOUS_COLUMNS = [c for c in UCI_COLUMNS if c not in [DATE_COL, TARGET_COL, "rv1", "rv2"]]
TIME_FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]
LAG_FEATURE_COLUMNS = [
    "appliances_lag_1", "appliances_lag_2", "appliances_lag_3", "appliances_lag_6",
    "appliances_lag_12", "appliances_lag_24",
    "appliances_rolling_mean_3", "appliances_rolling_mean_6", "appliances_rolling_mean_12",
    "appliances_rolling_mean_24", "appliances_rolling_std_6", "appliances_rolling_std_12",
    "appliances_delta_1", "appliances_delta_3", "appliances_delta_6"
]

FEATURE_COLUMNS = EXOGENOUS_COLUMNS + TIME_FEATURE_COLUMNS + LAG_FEATURE_COLUMNS + [TARGET_COL]


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_candidates() -> list[Path]:
    return [
        DATA_DIR / "energydata_complete.csv",         # official UCI if downloaded or copied by user
        DATA_DIR / "sample_energydata_complete.csv",  # offline fallback
    ]


def find_dataset_path() -> Path:
    for path in dataset_candidates():
        if path.exists():
            return path
    raise FileNotFoundError(
        "Không tìm thấy dataset. Hãy chạy `python src/download_data.py` hoặc đặt UCI `energydata_complete.csv` vào thư mục data/."
    )


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = find_dataset_path()
    path = Path(path)
    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError("Dataset phải có cột `date`.")
    if TARGET_COL not in df.columns:
        raise ValueError("Dataset phải có cột target `Appliances`.")

    # Normalize expected columns. Missing optional telemetry columns are filled later.
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL).reset_index(drop=True)

    for col in UCI_COLUMNS:
        if col == DATE_COL:
            continue
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[UCI_COLUMNS]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    out["hour"] = out[DATE_COL].dt.hour + out[DATE_COL].dt.minute / 60.0
    out["dayofweek"] = out[DATE_COL].dt.dayofweek
    out["month"] = out[DATE_COL].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    return out


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y = out[TARGET_COL]
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"appliances_lag_{lag}"] = y.shift(lag)
    for win in [3, 6, 12, 24]:
        out[f"appliances_rolling_mean_{win}"] = y.rolling(window=win, min_periods=win).mean()
    out["appliances_rolling_std_6"] = y.rolling(window=6, min_periods=6).std()
    out["appliances_rolling_std_12"] = y.rolling(window=12, min_periods=12).std()
    for lag in [1, 3, 6]:
        out[f"appliances_delta_{lag}"] = y.diff(lag)
    return out


def make_supervised_frame(df: pd.DataFrame, horizon_steps: int = HORIZON_STEPS, include_target: bool = True) -> pd.DataFrame:
    """Create a supervised learning table for forecasting.

    At row t, features use current and past values; target is Appliances at t+horizon.
    This is the main difference from anomaly detection: we explicitly create a future target.
    """
    out = add_time_features(df)
    out = add_lag_rolling_features(out)
    if include_target:
        out["target_future"] = out[TARGET_COL].shift(-horizon_steps)
    return out


def clean_supervised_frame(df: pd.DataFrame, feature_columns: list[str] | None = None, require_target: bool = True) -> pd.DataFrame:
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    needed = feature_columns + (["target_future"] if require_target else [])
    out = df.copy()
    for col in needed:
        if col not in out.columns:
            out[col] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=needed).reset_index(drop=True)
    return out


def time_split(df: pd.DataFrame, train_ratio: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_ratio <= 0.9:
        raise ValueError("train_ratio nên nằm trong khoảng 0.5 đến 0.9")
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = np.mean(np.abs(err) / denom) * 100.0
    bias = np.mean(err)
    return {
        "mae": float(round(mae, 4)),
        "rmse": float(round(rmse, 4)),
        "mape_percent": float(round(mape, 4)),
        "forecast_bias": float(round(bias, 4)),
    }


def risk_from_prediction(predicted_value: float, thresholds: dict) -> str:
    if predicted_value >= thresholds["critical"]:
        return "CRITICAL"
    if predicted_value >= thresholds["high"]:
        return "HIGH"
    if predicted_value >= thresholds["warning"]:
        return "WARNING"
    return "NORMAL"


def recommendation_from_risk(risk_level: str) -> str:
    if risk_level == "CRITICAL":
        return "HUMAN_CHECK_BEFORE_ACTUATOR_CONTROL"
    if risk_level == "HIGH":
        return "REDUCE_NON_CRITICAL_LOAD_OR_CHECK_HVAC"
    if risk_level == "WARNING":
        return "MONITOR_AND_PREPARE_ENERGY_SAVING_ACTION"
    return "CONTINUE_MONITORING"


def reason_from_risk(predicted_value: float, thresholds: dict) -> str:
    return (
        f"Predicted appliance energy is {predicted_value:.2f} Wh; "
        f"warning/high/critical thresholds are "
        f"{thresholds['warning']:.2f}/{thresholds['high']:.2f}/{thresholds['critical']:.2f} Wh."
    )


def build_forecast_log(test_df: pd.DataFrame, predicted_values, thresholds: dict, model_version: str = MODEL_VERSION) -> pd.DataFrame:
    out = test_df[[DATE_COL, "target_future"]].copy()
    out = out.rename(columns={DATE_COL: "timestamp", "target_future": "actual_value"})
    out["predicted_value"] = np.asarray(predicted_values, dtype=float)
    out["forecast_error"] = out["predicted_value"] - out["actual_value"]
    out["abs_error"] = out["forecast_error"].abs()
    out["risk_level"] = [risk_from_prediction(v, thresholds) for v in out["predicted_value"]]
    out["recommendation"] = [recommendation_from_risk(r) for r in out["risk_level"]]
    out["reason"] = [reason_from_risk(v, thresholds) for v in out["predicted_value"]]
    out["model_version"] = model_version
    return out


def fill_missing_for_api(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    out = df.copy()
    for col in UCI_COLUMNS:
        if col == DATE_COL:
            continue
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col, value in medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(float(value))
    return out
