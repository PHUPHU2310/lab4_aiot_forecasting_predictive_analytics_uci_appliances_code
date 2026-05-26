
from __future__ import annotations

import json
from pathlib import Path
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
    }
    joblib.dump(model_bundle, MODEL_BUNDLE_PATH)

    print("=== Forecasting metrics ===")
    print(json.dumps(metrics_summary, indent=2, ensure_ascii=False))
    print(f"Saved model bundle: {MODEL_BUNDLE_PATH}")
    print(f"Saved forecast log: {OUTPUT_DIR / 'forecast_log.csv'}")
    return metrics_summary


if __name__ == "__main__":
    train_forecasting_models()
