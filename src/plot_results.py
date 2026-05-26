
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils import OUTPUT_DIR, FIGURE_DIR

PRED_PATH = OUTPUT_DIR / "forecast_test_predictions.csv"
LOG_PATH = OUTPUT_DIR / "forecast_log.csv"
METRICS_PATH = OUTPUT_DIR / "forecast_metrics.json"


def main():
    if not PRED_PATH.exists() or not LOG_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("Hãy chạy `python src/train_forecast.py` trước khi vẽ biểu đồ.")

    FIGURE_DIR.mkdir(exist_ok=True)
    pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
    log = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = metrics["best_model_name"]

    # 1. Forecast vs actual
    sample = pred.tail(min(350, len(pred)))
    plt.figure(figsize=(12, 5))
    plt.plot(sample["date"], sample["actual_future_value"], label="actual future value")
    plt.plot(sample["date"], sample[f"pred_{best}"], label=f"prediction: {best}")
    plt.title("Lab 4 - Forecast vs Actual on test time-series")
    plt.xlabel("time")
    plt.ylabel("Appliances energy use (Wh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "forecast_vs_actual.png", dpi=160)
    plt.close()

    # 2. Forecast error over time
    sample_log = log.tail(min(350, len(log)))
    plt.figure(figsize=(12, 4))
    plt.plot(sample_log["timestamp"], sample_log["forecast_error"], label="forecast error")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Lab 4 - Forecast error over time")
    plt.xlabel("time")
    plt.ylabel("predicted - actual (Wh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "forecast_error_over_time.png", dpi=160)
    plt.close()

    # 3. MAE comparison
    names = list(metrics["metrics_by_model"].keys())
    maes = [metrics["metrics_by_model"][n]["mae"] for n in names]
    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(names)), maes)
    plt.xticks(range(len(names)), names, rotation=25, ha="right")
    plt.title("Lab 4 - Model comparison by MAE")
    plt.ylabel("MAE (Wh)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_mae.png", dpi=160)
    plt.close()

    print("Saved figures:")
    for fn in ["forecast_vs_actual.png", "forecast_error_over_time.png", "model_comparison_mae.png"]:
        print(FIGURE_DIR / fn)


if __name__ == "__main__":
    main()
