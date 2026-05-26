from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ..utils import OUTPUT_DIR, HORIZON_MINUTES
except ImportError:
    from utils import OUTPUT_DIR, HORIZON_MINUTES

INTERFACE_DIR = Path(__file__).resolve().parent
INTERFACE_STATIC_DIR = INTERFACE_DIR / "static"
METRICS_PATH = OUTPUT_DIR / "forecast_metrics.json"
FORECAST_LOG_PATH = OUTPUT_DIR / "forecast_log.csv"


def _load_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def _load_forecast_log() -> pd.DataFrame:
    if FORECAST_LOG_PATH.exists():
        return pd.read_csv(FORECAST_LOG_PATH)
    return pd.DataFrame()


def _format_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _safe_text(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _risk_class(risk_level: Any) -> str:
    return {
        "NORMAL": "risk-normal",
        "WARNING": "risk-warning",
        "HIGH": "risk-high",
        "CRITICAL": "risk-critical",
    }.get(str(risk_level).upper(), "risk-normal")


def _latest_rows(df: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    if df.empty:
        return []
    columns = ["timestamp", "actual_value", "predicted_value", "abs_error", "risk_level", "recommendation"]
    rows = df.tail(limit).sort_values("timestamp", ascending=False)
    return rows[[c for c in columns if c in rows.columns]].to_dict(orient="records")


def _sparkline_svg(df: pd.DataFrame, y_col: str, color: str, width: int = 720, height: int = 190) -> str:
    if df.empty or y_col not in df.columns:
        return '<div class="empty-state">Chưa có dữ liệu để vẽ biểu đồ.</div>'

    values = pd.to_numeric(df[y_col], errors="coerce").dropna().tail(96)
    if values.empty:
        return '<div class="empty-state">Chưa có dữ liệu hợp lệ để vẽ biểu đồ.</div>'

    min_v = float(values.min())
    max_v = float(values.max())
    span = max(max_v - min_v, 1.0)
    padding = 18
    usable_w = width - padding * 2
    usable_h = height - padding * 2
    points = []

    for idx, value in enumerate(values):
        x = padding + (idx / max(len(values) - 1, 1)) * usable_w
        y = padding + (1 - ((float(value) - min_v) / span)) * usable_h
        points.append(f"{x:.1f},{y:.1f}")

    return f"""
    <svg class="line-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Biểu đồ {_safe_text(y_col)}">
        <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" />
        <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" />
        <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        <text x="{padding}" y="{padding - 4}">{_format_number(max_v)}</text>
        <text x="{padding}" y="{height - 4}">{_format_number(min_v)}</text>
    </svg>
    """


def _render_table_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<tr><td colspan="6">Chưa có forecast log. Hãy chạy python src/train_forecast.py trước.</td></tr>'

    return "\n".join(
        f"""
        <tr>
            <td>{_safe_text(row.get("timestamp", ""))}</td>
            <td>{_format_number(row.get("actual_value"))}</td>
            <td>{_format_number(row.get("predicted_value"))}</td>
            <td>{_format_number(row.get("abs_error"))}</td>
            <td><span class="pill {_risk_class(row.get("risk_level", ""))}">{_safe_text(row.get("risk_level", "N/A"))}</span></td>
            <td>{_safe_text(row.get("recommendation", ""))}</td>
        </tr>
        """
        for row in rows
    )


def _render_risk_cards(risk_counts: dict[str, int]) -> str:
    if not risk_counts:
        return '<div class="empty-state">Chưa có thống kê risk.</div>'

    return "\n".join(
        f'<div class="risk-item"><span class="pill {_risk_class(level)}">{_safe_text(level)}</span><strong>{count}</strong></div>'
        for level, count in risk_counts.items()
    )


def _metric(metrics: dict[str, Any], model_name: str, key: str) -> str:
    return _format_number(metrics.get("metrics_by_model", {}).get(model_name, {}).get(key), digits=4)


def _render_report_tab(metrics: dict[str, Any]) -> str:
    model_target = _safe_text(metrics.get("target", "Appliances"))
    model_horizon = _safe_text(metrics.get("forecast_horizon_minutes", HORIZON_MINUTES))

    return f"""
    <section class="tab-panel" id="tab-report" role="tabpanel" aria-labelledby="tab-button-report">
        <section class="grid two-col">
            <article class="card">
                <h2>Điều chỉnh forecasting horizon</h2>
                <div class="report-list">
                    <div class="risk-item"><span>Horizon hiện tại</span><strong>{model_horizon} phút</strong></div>
                    <div class="risk-item"><span>10 phút</span><strong>1 step</strong></div>
                    <div class="risk-item"><span>30 phút</span><strong>3 steps</strong></div>
                    <div class="risk-item"><span>60 phút</span><strong>6 steps</strong></div>
                </div>
                <p class="body-copy">Khi horizon dài hơn, model phải dự báo xa hơn trong tương lai nên MAE/RMSE thường tăng. Horizon 10 phút phù hợp cảnh báo ngắn hạn; 30-60 phút phù hợp lập kế hoạch tải và tối ưu năng lượng.</p>
            </article>
            <article class="card">
                <h2>Target và use-case nhóm</h2>
                <div class="risk-list">
                    <div class="risk-item"><span>Target hiện tại</span><strong>{model_target}</strong></div>
                    <div class="risk-item"><span>Use-case</span><strong>Cảnh báo tải cao</strong></div>
                    <div class="risk-item"><span>Output</span><strong>Risk + recommendation</strong></div>
                </div>
                <p class="body-copy">Target `Appliances` được ánh xạ thành bài toán dự báo tiêu thụ năng lượng thiết bị trong nhà thông minh hoặc tòa nhà nhỏ. Có thể đổi target sang `lights`, nhưng cần cập nhật lại feature lag/rolling theo target mới và train lại model.</p>
            </article>
        </section>

        <section class="card section">
            <h2>So sánh Linear Regression với Random Forest</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Mô hình</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>MAPE (%)</th>
                            <th>Forecast bias</th>
                            <th>Nhận xét</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Linear Regression</td>
                            <td>{_metric(metrics, "linear_regression_v1", "mae")}</td>
                            <td>{_metric(metrics, "linear_regression_v1", "rmse")}</td>
                            <td>{_metric(metrics, "linear_regression_v1", "mape_percent")}</td>
                            <td>{_metric(metrics, "linear_regression_v1", "forecast_bias")}</td>
                            <td>Tốt nhất theo kết quả hiện tại, bias gần 0 và dễ giải thích.</td>
                        </tr>
                        <tr>
                            <td>Random Forest</td>
                            <td>{_metric(metrics, "random_forest_v1", "mae")}</td>
                            <td>{_metric(metrics, "random_forest_v1", "rmse")}</td>
                            <td>{_metric(metrics, "random_forest_v1", "mape_percent")}</td>
                            <td>{_metric(metrics, "random_forest_v1", "forecast_bias")}</td>
                            <td>Bắt quan hệ phi tuyến tốt, nhưng kém Linear Regression trên test hiện tại.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <p class="body-copy">Kết luận: `linear_regression_v1` đang phù hợp để deploy vì MAE/RMSE thấp hơn và sai lệch trung bình nhỏ. Random Forest vẫn đáng thử khi đổi horizon, đổi target hoặc mở rộng dữ liệu.</p>
        </section>
    </section>
    """


def _availability_label(item: dict[str, Any]) -> str:
    return "Có sẵn" if item.get("available") else "Chưa cài"


def _render_advanced_tab(metrics: dict[str, Any]) -> str:
    availability = metrics.get("advanced_model_availability", {})
    multi_step = metrics.get("multi_step_forecasting", {})
    prediction_interval = metrics.get("prediction_interval", {})
    drift = metrics.get("drift_monitoring", {})

    availability_rows = "\n".join(
        f"""
        <tr>
            <td>{_safe_text(name)}</td>
            <td>{_safe_text(item.get("package", "N/A"))}</td>
            <td>{_availability_label(item)}</td>
            <td>{_safe_text(item.get("status", ""))}</td>
        </tr>
        """
        for name, item in availability.items()
    ) or '<tr><td colspan="4">Chưa có dữ liệu advanced model. Hãy chạy lại python src/train_forecast.py.</td></tr>'

    multi_step_rows = "\n".join(
        f"""
        <tr>
            <td>{_safe_text(info.get("horizon_minutes", label))} phút</td>
            <td>{_safe_text(info.get("best_model_name", "N/A"))}</td>
            <td>{_format_number(info.get("metrics_by_model", {}).get("linear_regression_v1", {}).get("mae"), 4)}</td>
            <td>{_format_number(info.get("metrics_by_model", {}).get("random_forest_v1", {}).get("mae"), 4)}</td>
        </tr>
        """
        for label, info in multi_step.items()
    ) or '<tr><td colspan="4">Chưa có dữ liệu multi-step. Hãy chạy lại python src/train_forecast.py.</td></tr>'

    drift_rows = "\n".join(
        f"""
        <tr>
            <td>{_safe_text(item.get("feature", ""))}</td>
            <td>{_format_number(item.get("standardized_mean_diff"), 4)}</td>
        </tr>
        """
        for item in drift.get("top_feature_drift", [])
    ) or '<tr><td colspan="2">Chưa có dữ liệu drift feature.</td></tr>'

    return f"""
    <section class="tab-panel" id="tab-advanced" role="tabpanel" aria-labelledby="tab-button-advanced">
        <section class="grid kpis">
            <article class="card">
                <div class="label">Prediction interval 90%</div>
                <div class="value">±{_format_number(prediction_interval.get("interval_90_half_width"))}</div>
                <div class="subvalue">Coverage: {_format_number(prediction_interval.get("coverage_90_percent"))}%</div>
            </article>
            <article class="card">
                <div class="label">Prediction interval 95%</div>
                <div class="value">±{_format_number(prediction_interval.get("interval_95_half_width"))}</div>
                <div class="subvalue">Coverage: {_format_number(prediction_interval.get("coverage_95_percent"))}%</div>
            </article>
            <article class="card">
                <div class="label">Drift PSI target</div>
                <div class="value">{_format_number(drift.get("target_psi"), 4)}</div>
                <div class="subvalue">Risk: {_safe_text(drift.get("drift_risk_level", "N/A"))}</div>
            </article>
            <article class="card">
                <div class="label">Advanced models</div>
                <div class="value">{sum(1 for item in availability.values() if item.get("available"))}/{len(availability) or 3}</div>
                <div class="subvalue">XGBoost / LightGBM / LSTM</div>
            </article>
        </section>

        <section class="grid two-col section">
            <article class="card">
                <h2>XGBoost, LightGBM, LSTM</h2>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Package</th>
                                <th>Availability</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>{availability_rows}</tbody>
                    </table>
                </div>
                <p class="body-copy">XGBoost và LightGBM được tự động thêm vào training nếu package đã cài. LSTM cần TensorFlow/Keras và thường nên train bằng sequence pipeline riêng.</p>
            </article>

            <article class="card">
                <h2>Multi-step forecasting</h2>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Horizon</th>
                                <th>Best model</th>
                                <th>Linear MAE</th>
                                <th>RF MAE</th>
                            </tr>
                        </thead>
                        <tbody>{multi_step_rows}</tbody>
                    </table>
                </div>
            </article>
        </section>

        <section class="grid two-col section">
            <article class="card">
                <h2>Drift monitoring</h2>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Std mean diff</th>
                            </tr>
                        </thead>
                        <tbody>{drift_rows}</tbody>
                    </table>
                </div>
            </article>
            <article class="card">
                <h2>Prediction interval</h2>
                <div class="risk-list">
                    <div class="risk-item"><span>Method</span><strong>{_safe_text(prediction_interval.get("method", "N/A"))}</strong></div>
                    <div class="risk-item"><span>90% half width</span><strong>{_format_number(prediction_interval.get("interval_90_half_width"))} Wh</strong></div>
                    <div class="risk-item"><span>95% half width</span><strong>{_format_number(prediction_interval.get("interval_95_half_width"))} Wh</strong></div>
                    <div class="risk-item"><span>Drift rule</span><strong>{_safe_text(drift.get("interpretation", "N/A"))}</strong></div>
                </div>
            </article>
        </section>
    </section>
    """


def _render_manual_tab() -> str:
    return """
    <section class="tab-panel" id="tab-manual" role="tabpanel" aria-labelledby="tab-button-manual">
        <section class="grid two-col">
            <article class="card">
                <h2>Nhập dữ liệu thủ công</h2>
                <form class="manual-form" id="manualForecastForm">
                    <label>
                        <span>Model</span>
                        <select name="model_name">
                            <option value="random_forest_v1">Random Forest</option>
                            <option value="linear_regression_v1">Linear Regression</option>
                            <option value="gradient_boosting_advanced_v1">Gradient Boosting</option>
                        </select>
                    </label>
                    <label>
                        <span>Use-case</span>
                        <select name="use_case">
                            <option value="energy_load_warning">Cảnh báo tải năng lượng</option>
                            <option value="smart_home_saving">Tối ưu nhà thông minh</option>
                            <option value="hvac_planning">Lập kế hoạch HVAC</option>
                        </select>
                    </label>
                    <label>
                        <span>Horizon mong muốn (phút)</span>
                        <input name="horizon_minutes" type="number" min="10" step="10" value="10" />
                    </label>
                    <label>
                        <span>Appliances (Wh)</span>
                        <input name="Appliances" type="number" min="0" step="1" value="80" />
                    </label>
                    <label>
                        <span>Lights (Wh)</span>
                        <input name="lights" type="number" min="0" step="1" value="5" />
                    </label>
                    <label>
                        <span>T1 nhiệt độ trong nhà</span>
                        <input name="T1" type="number" step="0.1" value="21" />
                    </label>
                    <label>
                        <span>RH_1 độ ẩm trong nhà</span>
                        <input name="RH_1" type="number" step="0.1" value="40" />
                    </label>
                    <label>
                        <span>T_out nhiệt độ ngoài trời</span>
                        <input name="T_out" type="number" step="0.1" value="8" />
                    </label>
                    <label>
                        <span>RH_out độ ẩm ngoài trời</span>
                        <input name="RH_out" type="number" step="0.1" value="80" />
                    </label>
                    <label>
                        <span>Windspeed</span>
                        <input name="Windspeed" type="number" min="0" step="0.1" value="4" />
                    </label>
                    <label>
                        <span>Visibility</span>
                        <input name="Visibility" type="number" min="0" step="0.1" value="40" />
                    </label>
                    <label>
                        <span>Press_mm_hg</span>
                        <input name="Press_mm_hg" type="number" min="0" step="0.1" value="755" />
                    </label>
                    <label>
                        <span>Tdewpoint</span>
                        <input name="Tdewpoint" type="number" step="0.1" value="5" />
                    </label>
                    <button type="submit">Chạy dự báo</button>
                </form>
            </article>

            <article class="card output-card">
                <h2>Output dự báo</h2>
                <div id="manualOutput" class="result-box">
                    <div class="empty-state">Chưa có kết quả. Nhập dữ liệu và chạy dự báo.</div>
                </div>
            </article>
        </section>
    </section>
    """


def render_dashboard_html(model_loaded: bool) -> str:
    metrics = _load_metrics()
    forecast_log = _load_forecast_log()
    best_model = metrics.get("best_model_name", "N/A")
    best_metrics = metrics.get("metrics_by_model", {}).get(best_model, {})
    latest_items = forecast_log.tail(1).to_dict(orient="records")
    latest = latest_items[0] if latest_items else {}
    risk_counts = forecast_log["risk_level"].value_counts().to_dict() if "risk_level" in forecast_log else {}
    actual_chart = _sparkline_svg(forecast_log, "actual_value", "#1f6feb")
    predicted_chart = _sparkline_svg(forecast_log, "predicted_value", "#d97706")
    table_rows = _render_table_rows(_latest_rows(forecast_log))
    risk_cards = _render_risk_cards(risk_counts)
    report_tab = _render_report_tab(metrics)
    advanced_tab = _render_advanced_tab(metrics)
    manual_tab = _render_manual_tab()

    return f"""
    <!doctype html>
    <html lang="vi">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AIoT Forecasting Dashboard</title>
        <link rel="stylesheet" href="/interface-static/dashboard.css" />
        <script defer src="/interface-static/dashboard.js"></script>
    </head>
    <body>
        <header>
            <h1>AIoT Forecasting Dashboard</h1>
            <p>Giao diện hiển thị dữ liệu dự báo năng lượng UCI Appliances: model, chỉ số đánh giá, risk level, recommendation và log dự báo mới nhất.</p>
        </header>
        <main>
            <nav class="tabs" aria-label="Dashboard tabs">
                <button class="tab-button active" id="tab-button-dashboard" type="button" data-tab="tab-dashboard">Dashboard</button>
                <button class="tab-button" id="tab-button-report" type="button" data-tab="tab-report">Báo cáo bổ sung</button>
                <button class="tab-button" id="tab-button-advanced" type="button" data-tab="tab-advanced">Nâng cao</button>
                <button class="tab-button" id="tab-button-manual" type="button" data-tab="tab-manual">Nhập tay dự báo</button>
            </nav>

            <section class="tab-panel active" id="tab-dashboard" role="tabpanel" aria-labelledby="tab-button-dashboard">
            <section class="grid kpis">
                <article class="card">
                    <div class="label">Trạng thái model</div>
                    <div class="value">{'Đã tải' if model_loaded else 'Chưa tải'}</div>
                    <div class="subvalue">{_safe_text(best_model)}</div>
                </article>
                <article class="card">
                    <div class="label">MAE tốt nhất</div>
                    <div class="value">{_format_number(best_metrics.get("mae"))}</div>
                    <div class="subvalue">Sai số tuyệt đối trung bình</div>
                </article>
                <article class="card">
                    <div class="label">Dự báo mới nhất</div>
                    <div class="value">{_format_number(latest.get("predicted_value"))} Wh</div>
                    <div class="subvalue">{_safe_text(latest.get("timestamp", "Chưa có log"))}</div>
                </article>
                <article class="card">
                    <div class="label">Risk mới nhất</div>
                    <div class="value"><span class="pill {_risk_class(latest.get("risk_level", ""))}">{_safe_text(latest.get("risk_level", "N/A"))}</span></div>
                    <div class="subvalue">{_safe_text(latest.get("recommendation", ""))}</div>
                </article>
            </section>

            <section class="grid two-col section">
                <article class="card">
                    <h2>Actual value gần đây</h2>
                    {actual_chart}
                </article>
                <article class="card">
                    <h2>Phân bố risk level</h2>
                    <div class="risk-list">{risk_cards}</div>
                </article>
            </section>

            <section class="grid two-col section">
                <article class="card">
                    <h2>Predicted value gần đây</h2>
                    {predicted_chart}
                </article>
                <article class="card">
                    <h2>Thông tin hệ thống</h2>
                    <div class="risk-list">
                        <div class="risk-item"><span>Forecast horizon</span><strong>{metrics.get("forecast_horizon_minutes", HORIZON_MINUTES)} phút</strong></div>
                        <div class="risk-item"><span>Dòng train</span><strong>{_safe_text(metrics.get("train_rows", "N/A"))}</strong></div>
                        <div class="risk-item"><span>Dòng test</span><strong>{_safe_text(metrics.get("test_rows", "N/A"))}</strong></div>
                        <div class="risk-item"><span>Target</span><strong>{_safe_text(metrics.get("target", "Appliances"))}</strong></div>
                    </div>
                </article>
            </section>

            <section class="card section">
                <h2>Forecast log mới nhất</h2>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Actual</th>
                                <th>Predicted</th>
                                <th>Abs error</th>
                                <th>Risk</th>
                                <th>Recommendation</th>
                            </tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </div>
            </section>

            <section class="grid figure-grid section">
                <article class="card">
                    <h2>Forecast vs Actual</h2>
                    <img src="/figures/forecast_vs_actual.png" alt="Forecast vs actual" />
                </article>
                <article class="card">
                    <h2>Forecast Error</h2>
                    <img src="/figures/forecast_error_over_time.png" alt="Forecast error over time" />
                </article>
                <article class="card">
                    <h2>Model MAE</h2>
                    <img src="/figures/model_comparison_mae.png" alt="Model comparison MAE" />
                </article>
            </section>
            </section>

            {report_tab}
            {advanced_tab}
            {manual_tab}
        </main>
    </body>
    </html>
    """
