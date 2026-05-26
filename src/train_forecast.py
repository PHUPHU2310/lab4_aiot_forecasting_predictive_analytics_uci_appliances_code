
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils import (
    MODEL_DIR, OUTPUT_DIR, TARGET_COL, DATE_COL, FEATURE_COLUMNS, MODEL_VERSION,
    HORIZON_STEPS, HORIZON_MINUTES, load_dataset, make_supervised_frame, clean_supervised_frame,
    time_split, regression_metrics, build_forecast_log, save_json
)

MODEL_BUNDLE_PATH = MODEL_DIR / "forecast_model_bundle_v1.joblib"
MULTI_STEP_HORIZONS = [1, 3, 6]


def _optional_advanced_models() -> tuple[dict, dict]:
    models = {}
    availability = {
        "xgboost_v1": {"available": False, "status": "not_installed", "package": "xgboost"},
        "lightgbm_v1": {"available": False, "status": "not_installed", "package": "lightgbm"},
        "lstm_sequence_v1": {
            "available": False,
            "status": "not_installed",
            "package": "tensorflow",
            "note": "LSTM needs TensorFlow/Keras. This lab records availability but keeps deployment in joblib-compatible sklearn models.",
        },
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost_v1"] = XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )
        availability["xgboost_v1"] = {"available": True, "status": "enabled", "package": "xgboost"}
    except Exception as exc:
        availability["xgboost_v1"]["error"] = str(exc)

    try:
        from lightgbm import LGBMRegressor

        models["lightgbm_v1"] = LGBMRegressor(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        availability["lightgbm_v1"] = {"available": True, "status": "enabled", "package": "lightgbm"}
    except Exception as exc:
        availability["lightgbm_v1"]["error"] = str(exc)

    try:
        import tensorflow  # noqa: F401

        availability["lstm_sequence_v1"] = {
            "available": True,
            "status": "available_not_trained",
            "package": "tensorflow",
            "note": "TensorFlow is installed. Add a Keras sequence trainer if the lab requires neural sequence modeling.",
        }
    except Exception as exc:
        availability["lstm_sequence_v1"]["error"] = str(exc)

    return models, availability


def _prediction_interval_summary(y_true, y_pred) -> tuple[dict, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_residual = np.abs(y_pred - y_true)
    q90 = float(np.quantile(abs_residual, 0.90))
    q95 = float(np.quantile(abs_residual, 0.95))
    lower_90 = np.maximum(y_pred - q90, 0.0)
    upper_90 = y_pred + q90
    lower_95 = np.maximum(y_pred - q95, 0.0)
    upper_95 = y_pred + q95

    summary = {
        "method": "conformal_abs_residual_quantile_on_test_split",
        "interval_90_half_width": round(q90, 4),
        "interval_95_half_width": round(q95, 4),
        "coverage_90_percent": round(float(np.mean((y_true >= lower_90) & (y_true <= upper_90)) * 100), 4),
        "coverage_95_percent": round(float(np.mean((y_true >= lower_95) & (y_true <= upper_95)) * 100), 4),
    }
    widths = {"q90": q90, "q95": q95}
    return summary, widths


def _psi(expected, actual, bins: int = 10) -> float:
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if expected.empty or actual.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = pd.cut(expected, breakpoints, include_lowest=True).value_counts(sort=False)
    actual_counts = pd.cut(actual, breakpoints, include_lowest=True).value_counts(sort=False)
    expected_pct = np.maximum(expected_counts.to_numpy(dtype=float) / len(expected), 1e-6)
    actual_pct = np.maximum(actual_counts.to_numpy(dtype=float) / len(actual), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _drift_monitoring_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    target_psi = _psi(train_df[TARGET_COL], test_df[TARGET_COL])
    feature_drift = []
    for col in FEATURE_COLUMNS:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_col = pd.to_numeric(train_df[col], errors="coerce")
        test_col = pd.to_numeric(test_df[col], errors="coerce")
        train_std = float(train_col.std()) or 1.0
        standardized_mean_diff = abs(float(test_col.mean()) - float(train_col.mean())) / train_std
        feature_drift.append({
            "feature": col,
            "standardized_mean_diff": round(float(standardized_mean_diff), 4),
        })

    feature_drift = sorted(feature_drift, key=lambda item: item["standardized_mean_diff"], reverse=True)[:10]
    if target_psi >= 0.25:
        risk_level = "HIGH"
    elif target_psi >= 0.10:
        risk_level = "WARNING"
    else:
        risk_level = "LOW"

    return {
        "method": "train_vs_test_population_stability_index_and_standardized_mean_diff",
        "target_psi": round(target_psi, 4),
        "drift_risk_level": risk_level,
        "top_feature_drift": feature_drift,
        "interpretation": "PSI < 0.10 low drift, 0.10-0.25 warning, >= 0.25 high drift.",
    }


def _evaluate_multi_step(df: pd.DataFrame) -> dict:
    results = {}
    for steps in MULTI_STEP_HORIZONS:
        supervised = make_supervised_frame(df, horizon_steps=steps, include_target=True)
        supervised = clean_supervised_frame(supervised, FEATURE_COLUMNS, require_target=True)
        train_df, test_df = time_split(supervised, train_ratio=0.75)
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["target_future"]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["target_future"]

        candidates = {
            "linear_regression_v1": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "random_forest_v1": RandomForestRegressor(
                n_estimators=80,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=1,
            ),
        }
        horizon_metrics = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            horizon_metrics[name] = regression_metrics(y_test, pred)

        best_name = min(horizon_metrics, key=lambda n: horizon_metrics[n]["mae"])
        results[f"{steps * HORIZON_MINUTES}_minutes"] = {
            "horizon_steps": steps,
            "horizon_minutes": steps * HORIZON_MINUTES,
            "best_model_name": best_name,
            "metrics_by_model": horizon_metrics,
        }
    return results


def train_forecasting_models() -> dict:
    df = load_dataset()
    supervised = make_supervised_frame(df, horizon_steps=HORIZON_STEPS, include_target=True)
    supervised = clean_supervised_frame(supervised, FEATURE_COLUMNS, require_target=True)

    train_df, test_df = time_split(supervised, train_ratio=0.75)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_future"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target_future"]

    # Baselines are not optional. A forecasting model is not meaningful until it beats simple reference rules.
    baseline_predictions = {
        "last_value_baseline": test_df[TARGET_COL].to_numpy(dtype=float),
        "moving_average_6_baseline": test_df["appliances_rolling_mean_6"].to_numpy(dtype=float),
    }

    optional_models, advanced_model_availability = _optional_advanced_models()
    models = {
        "linear_regression_v1": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "random_forest_v1": RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1
        ),
        # Advanced-but-still-lightweight model. It gives students a bridge toward XGBoost/LightGBM without extra packages.
        "gradient_boosting_advanced_v1": GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            random_state=42
        ),
    }
    models.update(optional_models)

    all_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}

    for name, pred in baseline_predictions.items():
        all_predictions[name] = np.asarray(pred, dtype=float)
        metrics[name] = regression_metrics(y_test, pred)
        metrics[name]["model_type"] = "baseline"

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        all_predictions[name] = np.asarray(pred, dtype=float)
        metrics[name] = regression_metrics(y_test, pred)
        metrics[name]["model_type"] = type(model).__name__
        trained_models[name] = model

    # Select the best deployable ML model by MAE. Baselines remain in the report but are not deployed.
    deployable_names = list(trained_models.keys())
    best_model_name = min(deployable_names, key=lambda n: metrics[n]["mae"])
    best_predictions = all_predictions[best_model_name]
    prediction_interval, interval_widths = _prediction_interval_summary(y_test, best_predictions)
    drift_monitoring = _drift_monitoring_summary(train_df, test_df)
    multi_step_forecasting = _evaluate_multi_step(df)

    risk_thresholds = {
        "warning": float(np.quantile(y_train, 0.70)),
        "high": float(np.quantile(y_train, 0.90)),
        "critical": float(np.quantile(y_train, 0.97)),
    }

    forecast_log = build_forecast_log(
        test_df=test_df,
        predicted_values=best_predictions,
        thresholds=risk_thresholds,
        model_version=best_model_name
    )
    forecast_log["interval_lower_90"] = np.maximum(best_predictions - interval_widths["q90"], 0.0)
    forecast_log["interval_upper_90"] = best_predictions + interval_widths["q90"]
    forecast_log["interval_lower_95"] = np.maximum(best_predictions - interval_widths["q95"], 0.0)
    forecast_log["interval_upper_95"] = best_predictions + interval_widths["q95"]

    # Save prediction table with columns for all models so students can inspect errors.
    pred_table = test_df[[DATE_COL, TARGET_COL, "target_future"]].copy()
    pred_table = pred_table.rename(columns={"target_future": "actual_future_value"})
    for name, pred in all_predictions.items():
        pred_table[f"pred_{name}"] = pred
        pred_table[f"abs_error_{name}"] = np.abs(pred - y_test.to_numpy(dtype=float))

    metrics_summary = {
        "dataset_rows_after_feature_engineering": int(len(supervised)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target": TARGET_COL,
        "forecast_horizon_steps": int(HORIZON_STEPS),
        "forecast_horizon_minutes": int(HORIZON_MINUTES),
        "split_policy": "Chronological split: first 75% train, last 25% test. No random split.",
        "feature_policy": "Use current and historical lag/rolling features only; target_future is shifted after feature creation.",
        "best_model_name": best_model_name,
        "risk_thresholds_from_training_target": {k: round(v, 4) for k, v in risk_thresholds.items()},
        "metrics_by_model": metrics,
        "advanced_model_availability": advanced_model_availability,
        "multi_step_forecasting": multi_step_forecasting,
        "prediction_interval": prediction_interval,
        "drift_monitoring": drift_monitoring,
        "interpretation_note": "MAE/RMSE/MAPE measure numerical forecast error. They are not classification metrics like Precision/Recall/F1 used in anomaly detection.",
        "safety_note": "Forecast output is not an actuator command. It must pass risk rule, recommendation and human/safety confirmation before device control."
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    save_json(metrics_summary, OUTPUT_DIR / "forecast_metrics.json")
    pred_table.to_csv(OUTPUT_DIR / "forecast_test_predictions.csv", index=False)
    forecast_log.to_csv(OUTPUT_DIR / "forecast_log.csv", index=False)

    feature_medians = train_df[FEATURE_COLUMNS].median(numeric_only=True).to_dict()
    raw_medians = df.drop(columns=[DATE_COL], errors="ignore").median(numeric_only=True).to_dict()

    model_bundle = {
        "model": trained_models[best_model_name],
        "trained_models": trained_models,
        "feature_columns": FEATURE_COLUMNS,
        "feature_medians": {k: float(v) for k, v in feature_medians.items()},
        "raw_medians": {k: float(v) for k, v in raw_medians.items()},
        "risk_thresholds": risk_thresholds,
        "target": TARGET_COL,
        "forecast_horizon_steps": HORIZON_STEPS,
        "forecast_horizon_minutes": HORIZON_MINUTES,
        "model_version": best_model_name,
        "lab_version": MODEL_VERSION,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics_by_model": metrics,
        "prediction_interval": prediction_interval,
        "prediction_interval_widths": interval_widths,
        "drift_monitoring": drift_monitoring,
    }
    joblib.dump(model_bundle, MODEL_BUNDLE_PATH)

    print("=== Forecasting metrics ===")
    print(json.dumps(metrics_summary, indent=2, ensure_ascii=False))
    print(f"Saved model bundle: {MODEL_BUNDLE_PATH}")
    print(f"Saved forecast log: {OUTPUT_DIR / 'forecast_log.csv'}")
    return metrics_summary


if __name__ == "__main__":
    train_forecasting_models()
