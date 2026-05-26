# LAB 4: Forecasting & Predictive Analytics cho dữ liệu IoT new

## 1. Lab này khác Lab 3 ở đâu?

- **Lab 3 = Event Pipeline**: hỏi *hệ thống có đang bất thường không?* Output chính là `anomaly_score`, `severity`, `anomaly_event_log.csv`, API `/detect-anomaly`.
- **Lab 4 = Forecasting Pipeline**: hỏi *giá trị sắp tới có thể là bao nhiêu và rủi ro vận hành là gì?* Output chính là `predicted_value`, `forecast_error`, `risk_level`, `recommendation`, `forecast_log.csv`, API `/forecast`.

Luồng chính của project:

```text
UCI Appliances telemetry
→ time-series feature engineering
→ lag / rolling features
→ chronological train/test split
→ baseline forecasting
→ machine-learning forecasting
→ advanced boosting model demo
→ forecast_value
→ risk_level
→ recommendation
→ forecast_log.csv
→ API /forecast
```

## 2. Dataset

Bài mẫu dùng **UCI Appliances Energy Prediction**.

- Dataset page: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- Target chính: `Appliances`, năng lượng tiêu thụ của thiết bị gia dụng theo Wh.
- Chu kỳ đo: 10 phút/lần.
- Schema có timestamp, năng lượng, đèn, nhiệt độ/độ ẩm các phòng và dữ liệu thời tiết.

Nếu máy có Internet, chạy `python src/download_data.py` để tải file UCI `energydata_complete.csv`.
Nếu không có Internet, project dùng `data/sample_energydata_complete.csv` để sinh viên vẫn chạy được toàn bộ pipeline. File sample này chỉ dùng cho kiểm thử lớp học, không phải dữ liệu UCI gốc.

## 3. Cấu trúc project

```text
lab4_aiot_forecasting_predictive_analytics_uci_appliances/
├─ data/
│  └─ sample_energydata_complete.csv        # fallback để chạy offline
├─ notebooks/
│  └─ 01_forecasting_predictive_analytics.ipynb
├─ src/
│  ├─ download_data.py                      # tải UCI dataset hoặc dùng fallback
│  ├─ train_forecast.py                     # train baseline + ML + advanced model
│  ├─ plot_results.py                       # vẽ forecast vs actual và metric chart
│  ├─ app.py                                # FastAPI deploy endpoint /forecast
│  ├─ test_api.py                           # test API khi uvicorn đang chạy
│  ├─ test_api_local.py                     # test logic API không cần mở port
│  └─ utils.py                              # hàm dùng chung
├─ models/                                  # model bundle .joblib
├─ outputs/                                 # metrics, prediction, forecast_log, api_test_result
├─ figures/                                 # biểu đồ kết quả
├─ diagrams/                                # hình minh họa pipeline
└─ requirements.txt
```

## 4. Cài môi trường

```bash
cd lab4_aiot_forecasting_predictive_analytics_uci_appliances
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Tuỳ chọn cho phần nâng cao XGBoost/LightGBM/LSTM:

```bash
pip install -r requirements-advanced.txt
```

## 5. Chạy bài mẫu bằng script

Tải dataset public hoặc dùng fallback sample:

```bash
python src/download_data.py
```

Train, test, đánh giá model:

```bash
python src/train_forecast.py
```

Vẽ biểu đồ:

```bash
python src/plot_results.py
```

Test API logic không cần mở port:

```bash
python src/test_api_local.py
```

Kết quả cần thấy:

```text
models/forecast_model_bundle_v1.joblib
outputs/forecast_metrics.json
outputs/forecast_test_predictions.csv
outputs/forecast_log.csv
outputs/api_test_result.json
figures/forecast_vs_actual.png
figures/forecast_error_over_time.png
figures/model_comparison_mae.png
```

## 6. Chạy notebook

```bash
jupyter notebook notebooks/01_forecasting_predictive_analytics.ipynb
```

Chạy từng cell từ trên xuống. Sau mỗi phần, đọc kỹ câu hỏi phân tích.

## 7. Deploy model bằng FastAPI

Sau khi train model xong:

```bash
python src/app.py
```

Hoặc chạy bằng uvicorn:

```bash
uvicorn src.app:app --reload
```

Mở trình duyệt:

```text
http://127.0.0.1:8000/       # giao diện dashboard hiển thị dữ liệu
http://127.0.0.1:8000/docs
```

Test API bằng script ở terminal khác:

```bash
python src/test_api.py
```

Nếu máy không mở được local port, test logic API bằng:

```bash
python src/test_api_local.py
```

## 8. Kiểm tra hoàn thành

Bạn hoàn thành bài mẫu khi có đủ:

- Notebook chạy hết không lỗi.
- Có `forecast_metrics.json` và đọc được MAE/RMSE/MAPE.
- Có `forecast_log.csv` với `actual_value`, `predicted_value`, `forecast_error`, `risk_level`, `recommendation`.
- Có ít nhất 3 biểu đồ trong `figures/`.
- API `/health` trả `model_loaded: true`.
- API `/forecast` trả `predicted_value`, `risk_level`, `recommendation`, `safety_note`.
- Bạn giải thích được vì sao Lab 4 không dùng Precision/Recall/F1 như Lab 3.
- Bạn giải thích được vì sao dự báo cao chưa được phép tự động cắt/bật thiết bị.

## 9. Phần báo cáo bổ sung

Xem `BAOCAO.md` để có phần phân tích:

- Điều chỉnh forecasting horizon.
- Chọn target khác hoặc ánh xạ sang use-case nhóm.
- So sánh Linear Regression với Random Forest bằng MAE/RMSE/MAPE.
- XGBoost/LightGBM/LSTM optional, multi-step forecasting, prediction interval và drift monitoring.
