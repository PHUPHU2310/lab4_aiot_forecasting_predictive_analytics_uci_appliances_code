# BÁO CÁO BỔ SUNG: Horizon, Target/Use-case và So sánh mô hình

## 1. Điều chỉnh forecasting horizon

Trong bài Lab 4, dữ liệu UCI Appliances được ghi nhận theo chu kỳ 10 phút/lần. Vì vậy:

- `forecast_horizon_steps = 1` tương ứng dự báo trước 10 phút.
- Nếu muốn dự báo trước 30 phút, đặt `forecast_horizon_steps = 3`.
- Nếu muốn dự báo trước 60 phút, đặt `forecast_horizon_steps = 6`.

Trong project hiện tại, horizon đang được cấu hình trong `src/utils.py`:

```python
HORIZON_STEPS = 1
HORIZON_MINUTES = 10
```

Hàm `make_supervised_frame()` tạo nhãn dự báo bằng cách dịch target về tương lai:

```python
out["target_future"] = out[TARGET_COL].shift(-horizon_steps)
```

Điều này đảm bảo tại thời điểm `t`, model chỉ dùng dữ liệu hiện tại và quá khứ để dự báo giá trị ở thời điểm `t + horizon`.

### Nhận xét khi tăng horizon

Khi horizon tăng từ 10 phút lên 30 hoặc 60 phút, bài toán khó hơn vì giá trị cần dự báo nằm xa hơn trong tương lai. Thông thường:

- MAE/RMSE có xu hướng tăng.
- Model tuyến tính có thể suy giảm nếu quan hệ dữ liệu thay đổi nhanh.
- Random Forest có thể bắt được quan hệ phi tuyến tốt hơn, nhưng cũng dễ bị kém ổn định nếu dữ liệu test khác nhiều so với train.

Với use-case vận hành IoT, horizon 10 phút phù hợp cho cảnh báo sớm ngắn hạn. Horizon 30-60 phút phù hợp hơn cho lập kế hoạch năng lượng, tối ưu tải hoặc chuẩn bị hành động tiết kiệm điện.

## 2. Chọn target khác hoặc ánh xạ sang use-case nhóm

Target mặc định của project là:

```python
TARGET_COL = "Appliances"
```

Target này biểu diễn năng lượng tiêu thụ của thiết bị gia dụng theo Wh trong mỗi khoảng 10 phút. Đây là target phù hợp nếu nhóm xây dựng use-case:

**Dự báo tiêu thụ năng lượng để cảnh báo tải cao và đề xuất hành động tiết kiệm điện.**

### Ánh xạ sang use-case nhóm

Trong bối cảnh hệ thống AIoT cho nhà thông minh hoặc tòa nhà nhỏ:

- Input: nhiệt độ, độ ẩm trong phòng, thời tiết ngoài trời, ánh sáng, lịch thời gian và lịch sử tiêu thụ điện.
- Output: `predicted_value`, `risk_level`, `recommendation`.
- Mục tiêu: dự báo mức tiêu thụ sắp tới, phát hiện nguy cơ tải cao và đưa ra khuyến nghị vận hành.

Ví dụ ánh xạ:

| Thành phần dữ liệu | Ý nghĩa trong use-case |
|---|---|
| `Appliances` | Điện năng tiêu thụ của thiết bị, dùng làm target dự báo |
| `lights` | Mức tiêu thụ đèn, phản ánh hành vi sử dụng trong nhà |
| `T1`-`T9` | Nhiệt độ các khu vực, liên quan đến nhu cầu HVAC/làm mát |
| `RH_1`-`RH_9` | Độ ẩm các khu vực, hỗ trợ đánh giá môi trường |
| `T_out`, `RH_out`, `Windspeed` | Điều kiện thời tiết ngoài trời |
| `hour`, `dayofweek`, `is_weekend` | Đặc trưng thời gian, phản ánh thói quen sử dụng |

Nếu nhóm muốn chọn target khác, có thể thay `TARGET_COL` sang một biến khác như `lights` để dự báo tiêu thụ đèn. Tuy nhiên, khi đổi target cần cập nhật lại các đặc trưng lag/rolling trong `src/utils.py`, vì hiện tại các feature như `appliances_lag_1`, `appliances_rolling_mean_6` đang được thiết kế riêng cho target `Appliances`.

## 3. So sánh Linear Regression với Random Forest

Kết quả hiện tại được lấy từ `outputs/forecast_metrics.json`.

| Mô hình | MAE | RMSE | MAPE (%) | Forecast bias | Nhận xét |
|---|---:|---:|---:|---:|---|
| Linear Regression | 13.2916 | 26.4976 | 15.7168 | -0.0958 | Tốt nhất theo MAE/RMSE, sai lệch trung bình gần 0 |
| Random Forest | 14.4224 | 27.1028 | 17.2894 | 1.0093 | Bắt quan hệ phi tuyến tốt, nhưng kém Linear Regression trên tập test hiện tại |

### Phân tích

Linear Regression đạt MAE thấp hơn Random Forest:

```text
13.2916 < 14.4224
```

Điều này cho thấy trên tập dữ liệu và horizon hiện tại, quan hệ giữa các feature lịch sử/thời gian/thời tiết với target `Appliances` đủ tuyến tính để Linear Regression hoạt động tốt. Ngoài ra, Linear Regression có forecast bias gần 0, nghĩa là mô hình không có xu hướng dự báo cao hoặc thấp quá rõ rệt.

Random Forest có thể học quan hệ phi tuyến và tương tác phức tạp giữa các biến, nhưng trong kết quả này MAE và MAPE cao hơn. Nguyên nhân có thể là:

- Dữ liệu sau feature engineering không quá lớn.
- Horizon hiện tại chỉ 10 phút, nên baseline/feature tuyến tính đã rất mạnh.
- Random Forest có thể chưa được tối ưu hyperparameter.
- Dữ liệu time-series được chia theo thời gian, nên phân phối test có thể khác train.

### Kết luận lựa chọn mô hình

Với kết quả hiện tại, chọn `linear_regression_v1` làm model deploy là hợp lý vì:

- MAE thấp nhất trong các model ML được so sánh.
- RMSE thấp nhất, nghĩa là ít lỗi lớn hơn Random Forest.
- Bias gần 0, phù hợp cho hệ thống khuyến nghị vận hành.
- Mô hình đơn giản, dễ giải thích hơn Random Forest.

Random Forest vẫn là lựa chọn đáng thử nếu nhóm mở rộng bài toán sang horizon dài hơn, target khác hoặc dữ liệu có quan hệ phi tuyến mạnh hơn.

## 4. Mở rộng nâng cao: XGBoost/LightGBM/LSTM, multi-step, interval và drift

Project đã bổ sung các phần nâng cao trong `src/train_forecast.py` và hiển thị tại tab **Nâng cao** trên web.

### XGBoost, LightGBM, LSTM

XGBoost và LightGBM được thêm theo cơ chế optional:

- Nếu package `xgboost` đã cài, training sẽ tự thêm `xgboost_v1`.
- Nếu package `lightgbm` đã cài, training sẽ tự thêm `lightgbm_v1`.
- Nếu chưa cài, hệ thống không lỗi mà ghi trạng thái `not_installed` vào `forecast_metrics.json`.

LSTM cần TensorFlow/Keras và sequence pipeline riêng, nên project hiện ghi nhận trạng thái package trong metrics. Có thể cài phần nâng cao bằng:

```bash
pip install -r requirements-advanced.txt
```

### Multi-step forecasting

Project đánh giá thêm các horizon:

| Horizon | Ý nghĩa |
|---|---|
| 10 phút | Dự báo 1 bước tiếp theo |
| 30 phút | Dự báo 3 bước tiếp theo |
| 60 phút | Dự báo 6 bước tiếp theo |

Kết quả được lưu trong trường `multi_step_forecasting` của `outputs/forecast_metrics.json`. Khi horizon càng dài, MAE/RMSE thường tăng vì bài toán dự báo xa hơn và bất định hơn.

### Prediction interval

Ngoài `predicted_value`, project bổ sung khoảng dự báo:

- `interval_lower_90`, `interval_upper_90`
- `interval_lower_95`, `interval_upper_95`

Khoảng này được tính bằng residual quantile trên test split. Ý nghĩa: thay vì chỉ nói “dự báo 100 Wh”, hệ thống có thể nói “dự báo khoảng 50-150 Wh với mức 90% theo sai số lịch sử”.

### Drift monitoring

Project bổ sung drift monitoring bằng:

- PSI cho target `Appliances`.
- Standardized mean difference cho các feature.
- Danh sách top feature drift mạnh nhất.

Quy ước đọc PSI:

| PSI | Mức drift |
|---:|---|
| `< 0.10` | Thấp |
| `0.10 - 0.25` | Cảnh báo |
| `>= 0.25` | Cao |

Drift monitoring giúp biết khi dữ liệu test/production khác dữ liệu train. Nếu drift cao, cần kiểm tra lại model, cập nhật dữ liệu hoặc train lại.

## 5. Trả lời câu hỏi ôn tập

### 1. Lab 4 đang dự báo giá trị tương lai hay phát hiện bất thường?

Lab 4 đang làm **dự báo giá trị tương lai** (forecasting), không phải phát hiện bất thường.

- Input: chuỗi telemetry theo thời gian gồm `Appliances`, `lights`, nhiệt độ, độ ẩm, thời tiết, đặc trưng thời gian và các lag/rolling feature.
- Output chính: `predicted_value`, tức giá trị `Appliances` dự báo cho tương lai sau `forecast_horizon_minutes = 10` phút.
- Output phụ cho vận hành: `risk_level`, `recommendation`, `prediction_interval_90/95`.
- Metric: dùng metric hồi quy như MAE, RMSE, MAPE và forecast bias.

Nếu là anomaly detection thì output thường là `anomaly_score`, nhãn bất thường/bình thường hoặc severity, và metric thường là Precision, Recall, F1. Trong Lab 4, model học `target_future`, nên bản chất là dự báo số liên tục.

### 2. Vì sao forecasting cần lag feature?

Forecasting cần lag feature vì dữ liệu time-series có phụ thuộc thời gian: giá trị hiện tại và quá khứ gần thường chứa thông tin mạnh để dự báo tương lai gần. Ví dụ `appliances_lag_1`, `appliances_lag_6`, `appliances_rolling_mean_24` giúp model biết mức tiêu thụ 10 phút trước, 1 giờ trước, xu hướng trung bình và độ dao động gần đây.

Nếu không có lag feature, model chỉ nhìn được các biến hiện tại như nhiệt độ, độ ẩm, giờ trong ngày, nhưng mất thông tin về quán tính và thói quen tiêu thụ trước đó.

### 3. Vì sao không random split dữ liệu time-series?

Không random split vì random split làm trộn dữ liệu quá khứ và tương lai. Khi đó model có thể được train trên các mẫu xảy ra sau mẫu test, gây rò rỉ thông tin thời gian và làm metric đẹp giả tạo.

Project dùng chronological split: 75% đầu làm train, 25% cuối làm test. Cách này mô phỏng đúng triển khai thực tế: dùng dữ liệu quá khứ để dự báo dữ liệu tương lai.

### 4. MAE và RMSE khác nhau thế nào?

MAE là trung bình trị tuyệt đối của lỗi:

```text
MAE = mean(|predicted - actual|)
```

RMSE là căn bậc hai của trung bình bình phương lỗi:

```text
RMSE = sqrt(mean((predicted - actual)^2))
```

MAE dễ hiểu vì biểu diễn lỗi trung bình theo cùng đơn vị Wh. RMSE phạt lỗi lớn mạnh hơn vì lỗi được bình phương trước khi lấy trung bình.

### 5. Nếu RMSE cao hơn MAE nhiều, điều đó gợi ý gì về lỗi dự báo?

Nếu RMSE cao hơn MAE nhiều, điều đó gợi ý model có một số lỗi rất lớn/outlier. Nói cách khác, phần lớn thời điểm có thể dự báo ổn, nhưng ở một số thời điểm tiêu thụ tăng vọt hoặc thay đổi đột ngột, model dự báo sai mạnh.

Trong kết quả hiện tại, Linear Regression có MAE = 13.2916 và RMSE = 26.4976. RMSE gần gấp đôi MAE, cho thấy lỗi lớn xuất hiện tại một số giai đoạn, đặc biệt khi `actual_value` tăng cao đột biến.

### 6. Model dự báo sai nhiều ở giai đoạn nào trong biểu đồ `forecast_error_over_time`?

Dựa trên `outputs/forecast_log.csv`, lỗi lớn thường xuất hiện tại các giai đoạn tiêu thụ tăng vọt, khi `actual_value` cao nhưng `predicted_value` không tăng kịp. Theo chia tập test thành 4 phần theo thời gian, 25% cuối tập test có MAE trung bình cao nhất, khoảng 30.86 Wh.

Một số lỗi lớn nhất là các điểm model dự báo thấp hơn thực tế rất nhiều, ví dụ:

- `2016-05-06 07:20:00`: actual 670 Wh, predicted khoảng 66.66 Wh, lỗi khoảng -603.34 Wh.
- `2016-04-29 11:30:00`: actual 780 Wh, predicted khoảng 259.23 Wh, lỗi khoảng -520.77 Wh.
- `2016-05-27 09:30:00`: actual 580 Wh, predicted khoảng 75.51 Wh, lỗi khoảng -504.49 Wh.

Vì vậy có thể kết luận model sai nhiều ở các đoạn có spike/tải cao đột ngột, và trung bình lỗi cao nhất nằm ở phần cuối của tập test.

### 7. Nếu `predicted_value` cao, hệ thống có nên tự động tắt thiết bị không? Vì sao?

Không nên tự động tắt thiết bị chỉ dựa trên `predicted_value`. Forecast là tín hiệu dự báo, không phải lệnh điều khiển chắc chắn. Model vẫn có sai số, đặc biệt ở các giai đoạn biến động mạnh; nếu tự động tắt thiết bị có thể ảnh hưởng an toàn, tiện nghi hoặc quy trình vận hành.

Cách hợp lý hơn là dùng `predicted_value` để tạo cảnh báo, gán `risk_level`, đưa ra `recommendation`, sau đó áp dụng rule an toàn, xác nhận của người dùng hoặc chính sách điều khiển riêng trước khi tác động lên thiết bị.

### 8. Forecast output khác decision như thế nào?

Forecast output là kết quả dự báo số học của model, ví dụ:

- `predicted_value`: dự báo điện năng tiêu thụ.
- `prediction_interval_90/95`: khoảng bất định của dự báo.
- `selected_model_mae`, `selected_model_rmse`: gợi ý chất lượng model.

Decision là quyết định vận hành được suy ra từ forecast và rule nghiệp vụ, ví dụ:

- `risk_level`: NORMAL, WARNING, HIGH, CRITICAL.
- `recommendation`: tiếp tục theo dõi, chuẩn bị tiết kiệm điện, giảm tải không quan trọng, hoặc yêu cầu kiểm tra thủ công.
- `safety_note`: nhắc rằng forecast không phải lệnh actuator tự động.

Nói ngắn gọn: forecast trả lời "sắp tới giá trị có thể là bao nhiêu", còn decision trả lời "hệ thống nên khuyến nghị hành động gì".

### 9. Nếu model dự báo sai liên tục trong dữ liệu mới, nhóm sẽ xử lý thế nào?

Nếu model sai liên tục trên dữ liệu mới, nhóm nên xử lý theo các bước:

1. Kiểm tra dữ liệu đầu vào: thiếu cột, sai đơn vị, timestamp lệch, sensor lỗi hoặc missing value.
2. Theo dõi lỗi production: tính MAE/RMSE mới theo thời gian và so với metric ban đầu.
3. Kiểm tra drift: so sánh phân phối dữ liệu mới với dữ liệu train bằng PSI hoặc feature drift.
4. Phân tích lỗi theo giai đoạn: xem model sai vào giờ nào, ngày nào, lúc tải cao hay lúc thời tiết thay đổi.
5. Cập nhật feature: thêm lag/rolling phù hợp hơn, feature lịch làm việc, mùa, occupancy nếu có.
6. Train lại model bằng dữ liệu mới và đánh giá bằng chronological split.
7. Chỉ deploy model mới nếu metric tốt hơn và hành vi khuyến nghị an toàn hơn.

### 10. Dashboard nên hiển thị gì ngoài `predicted_value`?

Dashboard không nên chỉ hiển thị `predicted_value`; cần thêm các thông tin giúp người vận hành hiểu độ tin cậy và hành động cần làm:

- `actual_value` gần đây để so sánh với dự báo.
- `forecast_error` và `abs_error`.
- MAE, RMSE, MAPE của model đang dùng.
- `risk_level` và `recommendation`.
- `prediction_interval_90/95` để thể hiện độ bất định.
- Biểu đồ forecast vs actual.
- Biểu đồ `forecast_error_over_time`.
- Thông tin horizon, target, model version.
- Số điểm input, cảnh báo thiếu history hoặc dữ liệu thiếu.
- Drift monitoring để biết dữ liệu mới có khác dữ liệu train hay không.
