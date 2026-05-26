
from __future__ import annotations

import json
import requests
import pandas as pd
from pathlib import Path

from utils import OUTPUT_DIR, load_dataset

OUTPUT_DIR.mkdir(exist_ok=True)

df = load_dataset().tail(36)
history = []
for _, row in df.iterrows():
    item = row.to_dict()
    item["date"] = str(item["date"])
    history.append(item)

print("Kiểm tra /health")
health = requests.get("http://127.0.0.1:8000/health", timeout=10).json()
print(health)

print("\nKiểm tra /model-info")
model_info = requests.get("http://127.0.0.1:8000/model-info", timeout=10).json()
print(model_info)

print("\nKiểm tra /forecast")
resp = requests.post("http://127.0.0.1:8000/forecast", json={"history": history}, timeout=10)
data = resp.json()
print(data)

assert resp.status_code == 200
assert "model_output" in data
assert "decision" in data
assert "predicted_value" in data["model_output"]
assert "risk_level" in data["decision"]
assert "recommendation" in data["decision"]

(OUTPUT_DIR / "api_test_result.json").write_text(
    json.dumps({"health": health, "model_info": model_info, "forecast": data}, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("\nPASS: API response schema hợp lệ.")
