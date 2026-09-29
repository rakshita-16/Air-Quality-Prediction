/* ── Air Quality Predictor – main.js ─────────────────────────────────────── */

// Sample values for quick demo fill
const SAMPLE_VALUES = {
  "PM2.5":       35.5,
  "PM10":        75.0,
  "NO2":         40.0,
  "SO2":         18.5,
  "CO":           1.2,
  "O3":          62.0,
  "Temperature": 28.0,
  "Humidity":    65.0,
  "WindSpeed":    3.5,
};

// AQI color map
const AQI_COLORS = {
  "Good":                          "#2ecc71",
  "Moderate":                      "#f1c40f",
  "Unhealthy for Sensitive Groups":"#e67e22",
  "Unhealthy":                     "#e74c3c",
  "Very Unhealthy":                "#8e44ad",
  "Hazardous":                     "#2c3e50",
};

// ── DOM references ────────────────────────────────────────────────────────────
const form          = document.getElementById("predictForm");
const fillBtn       = document.getElementById("fillSample");
const resultPanel   = document.getElementById("resultPanel");
const spinnerOverlay= document.getElementById("spinnerOverlay");
const aqiGauge      = document.getElementById("aqiGauge");
const aqiValueEl    = document.getElementById("aqiValue");
const categoryEl    = document.getElementById("resultCategory");
const descEl        = document.getElementById("resultDesc");
const confidenceFill= document.getElementById("confidenceFill");
const confidenceLbl = document.getElementById("confidenceLabel");
const recList       = document.getElementById("recList");

// ── Fill sample values ────────────────────────────────────────────────────────
if (fillBtn) {
  fillBtn.addEventListener("click", () => {
    Object.entries(SAMPLE_VALUES).forEach(([name, val]) => {
      const el = document.querySelector(`[name="${name}"]`);
      if (el) el.value = val;
    });
    showToast("Sample values loaded!", "info");
  });
}

// ── Form submit → AJAX predict ────────────────────────────────────────────────
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Basic validation
    const inputs = form.querySelectorAll("input[required]");
    let valid = true;
    inputs.forEach(inp => {
      inp.classList.remove("input-error");
      if (!inp.value || isNaN(parseFloat(inp.value))) {
        inp.classList.add("input-error");
        valid = false;
      }
    });
    if (!valid) { showToast("Please fill all fields with valid numbers.", "error"); return; }

    showSpinner(true);

    try {
      const formData = new FormData(form);
      const resp = await fetch("/predict", { method: "POST", body: formData });
      const data = await resp.json();

      if (!resp.ok || data.error) {
        showToast(data.error || "Prediction failed.", "error");
        return;
      }

      displayResult(data);
    } catch (err) {
      showToast("Network error – is the server running?", "error");
      console.error(err);
    } finally {
      showSpinner(false);
    }
  });
}

// ── Display result ────────────────────────────────────────────────────────────
function displayResult(data) {
  const { aqi, category, color, description, recommendations, confidence } = data;

  // AQI gauge
  const gaugeColor = color || AQI_COLORS[category] || "#64748b";
  aqiGauge.style.background = `radial-gradient(circle at 30% 30%, ${lighten(gaugeColor, 20)}, ${gaugeColor})`;
  aqiGauge.style.boxShadow  = `0 8px 32px ${gaugeColor}60`;
  animateNumber(aqiValueEl, 0, aqi, 800);

  // Category & description
  categoryEl.textContent = category;
  categoryEl.style.color  = gaugeColor;
  descEl.textContent      = description;

  // Confidence bar
  setTimeout(() => {
    confidenceFill.style.width = confidence + "%";
  }, 300);
  confidenceLbl.textContent = confidence + "%";

  // Recommendations
  recList.innerHTML = "";
  recommendations.forEach(rec => {
    const li = document.createElement("li");
    li.textContent = rec;
    recList.appendChild(li);
  });

  // Show panel with scroll
  resultPanel.style.display = "block";
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  showToast(`AQI predicted: ${aqi} (${category})`, "success");
}

// ── Animate counter ───────────────────────────────────────────────────────────
function animateNumber(el, from, to, duration) {
  const start = performance.now();
  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    el.textContent = Math.round(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Lighten colour helper ─────────────────────────────────────────────────────
function lighten(hex, amount) {
  const num = parseInt(hex.replace("#", ""), 16);
  const r = Math.min(255, (num >> 16) + amount);
  const g = Math.min(255, ((num >> 8) & 0x00ff) + amount);
  const b = Math.min(255, (num & 0x0000ff) + amount);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}

// ── Spinner ───────────────────────────────────────────────────────────────────
function showSpinner(show) {
  if (spinnerOverlay) spinnerOverlay.style.display = show ? "flex" : "none";
}

// ── Toast notifications ───────────────────────────────────────────────────────
function showToast(message, type = "info") {
  // Remove existing toasts
  document.querySelectorAll(".toast").forEach(t => t.remove());

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  document.body.appendChild(toast);

  // Force reflow then animate in
  requestAnimationFrame(() => toast.classList.add("toast-show"));

  setTimeout(() => {
    toast.classList.remove("toast-show");
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}

// ── Toast styles (injected dynamically) ──────────────────────────────────────
const toastCSS = `
.toast {
  position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 9999;
  padding: .75rem 1.2rem; border-radius: 10px;
  font-size: .9rem; font-weight: 600; color: #fff;
  box-shadow: 0 8px 24px rgba(0,0,0,.2);
  opacity: 0; transform: translateY(20px);
  transition: opacity .35s, transform .35s;
  max-width: 340px;
}
.toast-show { opacity: 1; transform: translateY(0); }
.toast-success { background: #16a34a; }
.toast-error   { background: #dc2626; }
.toast-info    { background: #2563eb; }
.input-error   { border-color: #dc2626 !important; box-shadow: 0 0 0 3px rgba(220,38,38,.12) !important; }
`;
const styleEl = document.createElement("style");
styleEl.textContent = toastCSS;
document.head.appendChild(styleEl);
