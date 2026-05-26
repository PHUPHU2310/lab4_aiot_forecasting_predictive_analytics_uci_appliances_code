function riskClass(riskLevel) {
    const risk = String(riskLevel || "").toUpperCase();
    if (risk === "WARNING") return "risk-warning";
    if (risk === "HIGH") return "risk-high";
    if (risk === "CRITICAL") return "risk-critical";
    return "risk-normal";
}

function formatNumber(value) {
    const numberValue = Number(value);
    if (!Number.isFinite(numberValue)) return "N/A";
    return numberValue.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function renderManualOutput(data) {
    const output = document.getElementById("manualOutput");
    if (!output) return;

    if (data.error) {
        output.innerHTML = `<div class="error-box">${data.error}</div>`;
        return;
    }

    const modelOutput = data.model_output || {};
    const decision = data.decision || {};
    const apiCheck = data.api_check || {};
    const manualInput = data.manual_input || {};
    const risk = decision.risk_level || "N/A";

    output.innerHTML = `
        <div class="result-kpi">
            <span class="label">Predicted value</span>
            <strong>${formatNumber(modelOutput.predicted_value)} Wh</strong>
            <div class="subvalue">${modelOutput.target || "Appliances"} - ${modelOutput.forecast_horizon_minutes || "N/A"} phút</div>
        </div>
        <div class="result-detail">
            <div class="risk-item"><span>Risk level</span><strong><span class="pill ${riskClass(risk)}">${risk}</span></strong></div>
            <div class="risk-item"><span>Recommendation</span><strong>${decision.recommendation || "N/A"}</strong></div>
            <div class="risk-item"><span>Model version</span><strong>${modelOutput.model_version || "N/A"}</strong></div>
            <div class="risk-item"><span>Interval 90%</span><strong>${formatNumber(modelOutput.prediction_interval_90?.lower)} - ${formatNumber(modelOutput.prediction_interval_90?.upper)} Wh</strong></div>
            <div class="risk-item"><span>Interval 95%</span><strong>${formatNumber(modelOutput.prediction_interval_95?.lower)} - ${formatNumber(modelOutput.prediction_interval_95?.upper)} Wh</strong></div>
            <div class="risk-item"><span>Latency</span><strong>${formatNumber(apiCheck.latency_ms)} ms</strong></div>
            <div class="risk-item"><span>Use-case</span><strong>${manualInput.use_case || "N/A"}</strong></div>
            <div class="risk-item"><span>Horizon nhập tay</span><strong>${manualInput.requested_horizon_minutes || "N/A"} phút</strong></div>
        </div>
        <div class="empty-state">${decision.reason || ""}<br>${manualInput.note || ""}</div>
    `;
}

function setupTabs() {
    const buttons = document.querySelectorAll(".tab-button");
    const panels = document.querySelectorAll(".tab-panel");

    buttons.forEach((button) => {
        button.addEventListener("click", () => {
            const targetId = button.dataset.tab;
            buttons.forEach((item) => item.classList.remove("active"));
            panels.forEach((panel) => panel.classList.remove("active"));
            button.classList.add("active");
            document.getElementById(targetId)?.classList.add("active");
        });
    });
}

function setupManualForm() {
    const form = document.getElementById("manualForecastForm");
    const output = document.getElementById("manualOutput");
    if (!form || !output) return;

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = form.querySelector("button[type='submit']");
        const formData = new FormData(form);
        const payload = {};

        formData.forEach((value, key) => {
            const input = form.elements[key];
            payload[key] = input?.type === "number" ? Number(value) : value;
        });

        button.disabled = true;
        output.innerHTML = '<div class="empty-state">Đang chạy dự báo...</div>';

        try {
            const response = await fetch("/forecast-manual", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            renderManualOutput(data);
        } catch (error) {
            output.innerHTML = `<div class="error-box">Không gọi được API: ${error.message}</div>`;
        } finally {
            button.disabled = false;
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupTabs();
    setupManualForm();
});
