
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from utils import DATA_DIR, OUTPUT_DIR, load_dataset
from app import ForecastRequest, TelemetryPoint, forecast, health, model_info

OUTPUT_DIR.mkdir(exist_ok=True)

df = load_dataset().tail(36)
history = []
for _, row in df.iterrows():
    item = row.to_dict()
    item["date"] = str(item["date"])
    history.append(TelemetryPoint(**item))

print("Kiểm tra /health")
health_result = health()
print(health_result)

print("\nKiểm tra /model-info")
info_result = model_info()
print(info_result)

print("\nKiểm tra /forecast")
response = forecast(ForecastRequest(history=history))
print(response)

assert "model_output" in response
assert "decision" in response
assert "predicted_value" in response["model_output"]
assert "risk_level" in response["decision"]
assert "recommendation" in response["decision"]

(OUTPUT_DIR / "api_test_result.json").write_text(
    json.dumps({"health": health_result, "model_info": info_result, "forecast": response}, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("\nPASS: API local response schema hợp lệ.")
print("Đã lưu outputs/api_test_result.json")
