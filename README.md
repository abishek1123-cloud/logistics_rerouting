# ⬡ Nexus Logistics AI
### AI-Powered Mid-Voyage Early Warning & Rerouting System

An end-to-end AI logistics intelligence system that predicts shipment delay risk **while the shipment is en route**, using live AIS/EDI tracking checkpoints — and recommends optimal rerouting strategies with full explainability.

---

## Table of Contents

- [Overview](#overview)
- [System Design — Mid-Voyage Prediction](#system-design--mid-voyage-prediction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Phase-wise Implementation](#phase-wise-implementation)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Dashboard](#running-the-dashboard)
- [Dashboard Tabs](#dashboard-tabs)
- [What to Add Next](#what-to-add-next)

---

## Overview

Supply chain teams face constant pressure from port congestion, labor strikes, extreme weather, and geopolitical instability. Nexus Logistics AI addresses this by:

- Scoring each in-transit shipment using AIS/EDI checkpoint data (8-hour update granularity)
- Predicting delay probability with a tuned HistGradientBoosting model (AUC-ROC: **0.7708**, Accuracy: **75.4%**)
- Triggering rerouting recommendations when `ai_delay_prob > 0.60`
- Running a LangGraph 3-node LLM pipeline (Llama-3.3-70b via Groq) for natural-language rerouting reasoning
- Delivering everything through a 7-tab cyberpunk-themed Streamlit dashboard

---

## System Design — Mid-Voyage Prediction

> When a shipment is already en route, port congestion, strikes, severe weather, and geopolitical events can cause significant delays. This system continuously monitors live AIS/EDI tracking data and predicts delay risk **mid-voyage**, enabling proactive rerouting decisions before delays materialise.

**Prediction window:** The model scores each shipment using its latest 8-hour AIS position checkpoint — information that is always available in a live logistics tracking system at the time of prediction.

A key design decision is `transit_delay_8h_bin` — the delay accumulated at the last AIS/EDI tracking checkpoint, quantized into 8-hour bins (the typical open-sea broadcast interval):

```python
transit_delay_8h_bin = round((actual_transit_days - scheduled_transit_days) × 3) / 3
```

The 8-hour bin size (480 min) is intentionally coarser than the delay classification threshold (121 min), so the model cannot trivially reconstruct the label from this feature alone. The remaining signal comes from disruption, congestion, and historical features.

| Feature | Source | Available at prediction time? |
|---|---|---|
| `transit_delay_8h_bin` | AIS/EDI 8h checkpoint | ✓ live tracking |
| `port_congestion_index` | Port authority feed | ✓ real-time |
| `weather_severity_score` | Met office API | ✓ forecast |
| `route_risk_score` | News/geopolitical API | ✓ real-time |
| `orig_delay_hist` | Historical database | ✓ pre-computed |

---

## Project Structure

```
nexus-logistics-ai/
│
├── data/
│   └── logistics_disruption_dataset.csv      # 52,034 shipment records, 26 raw features
│
├── models/
│   └── logistics_delay_model.pkl             # Tuned model + feature list + scaler + encoder
│
├── Logistics_Rerouting.ipynb                 # Full 4-phase development notebook
├── app.py                                    # Streamlit dashboard (7 tabs, 1,400+ lines)
├── requirements.txt                          # Python dependencies
├── .env                                      # GROQ_API_KEY (not committed)
└── README.md
```

---

## Dataset

**File:** `data/logistics_disruption_dataset.csv`
**Records:** 52,034 shipments | **Raw features:** 26 | **Missing values:** 0 | **Duplicates:** 0

> After lag/rolling feature engineering, `dropna()` reduces the working dataset to **51,695 samples** (339 rows dropped due to rolling window warm-up). All model training and evaluation uses this cleaned set.

| Category | Columns |
|---|---|
| Route | `shipment_id`, `origin_port`, `destination_port`, `carrier_id`, `transport_mode` |
| Schedule | `timestamp`, `scheduled_transit_days`, `actual_transit_days`, `delay_minutes` |
| Port Ops | `vessels_at_anchorage_count`, `berth_occupancy_pct`, `yard_utilization_pct`, `median_waiting_time_hours` |
| Weather | `wind_speed_knots`, `wave_height_meters`, `precipitation_mm`, `fog_density_index`, `visibility_km`, `air_temperature_c` |
| Disruption | `strike_alert_level`, `labor_strike_indicator`, `port_closure_flag`, `geopolitical_risk_index`, `news_sentiment_score` |
| Economic | `regional_fuel_price_index` |
| Target | `is_delayed` (binary) |

---

## Phase-wise Implementation

### Phase 1 — Discovery & EDA

- Loaded and cleaned 52,034 records (zero missing values, zero duplicates)
- Standardized disruption signals (`strike_alert_level`, `geopolitical_risk_index`, `regional_fuel_price_index`) to 0–1 scale
- Correlation analysis: `berth_occupancy_pct` and `yard_utilization_pct` are the strongest individual delay correlates
- Time-series trend analysis: Q1/Q2 seasonal delay spikes visible in 30-day rolling delay rate
- Port-level risk profiling: delay risk is port-specific — strike level 3+ nearly doubles delay probability
- Geospatial feature engineering: Haversine `distance_km`, `crossing_equator`, `transit_efficiency` for 20 real-world port coordinates (notebook); dashboard extends this to 21 with Cape Town

**Phase 1 conclusion:** Port congestion and labor disruptions are primary measurable delay drivers. Delay risk is port-specific, not globally uniform — route-level features are essential.

---

### Phase 2 — Feature Engineering & Modeling

Engineered **87 features** across 7 categories from 26 raw inputs:

| Category | Key Features |
|---|---|
| Congestion | `port_congestion_index`, `congestion_x_vessels`, `berth_over_yard`, `high_berth`, `high_yard`, `critical_congestion` |
| Weather | `weather_severity_score`, `wind_x_wave`, `fog_x_visibility`, `severe_weather` |
| Route Risk | `route_risk_score`*, `geopolit_x_strike`, `combined_disruption`, `fuel_x_risk` |
| Geospatial | `distance_km`, `crossing_equator`, `transit_efficiency` |
| Temporal | `hour`, `day_of_week`, `month`, `quarter`, `is_weekend`, `is_peak_season`, `is_monsoon` |
| Historical | `orig_delay_hist`, `dest_delay_hist`, `carrier_delay_hist`, `route_delay_hist`, `carrier_route_hist`, `port_delay_7d`, `port_delay_30d`, `carrier_delay_30d` |
| Lag/Rolling | lag1, lag3, lag7 + 7-day and 14-day rolling averages for 7 key operational signals |

*`route_risk_score` = `strike_alert_level × 0.30` + `geopolitical_risk_index × 0.30` + `(1 − news_sentiment) × 0.15` + `labor_strike_indicator × 0.15` + `port_closure_flag × 0.10`

**Anomaly Detection — Isolation Forest** (200 estimators, 5% contamination, 11 features):
- Anomalies detected: **2,585** (5.0% of shipments)
- Delay rate (anomaly): **76.7%** vs delay rate (normal): **56.4%**
- Risk lift factor: **1.36×** — confirming `anomaly_score` as a strong predictive feature

**Models trained:**
- `HistGradientBoostingClassifier` — primary (max_iter=600, learning_rate=0.03, max_depth=10, min_samples_leaf=8, l2=0.01)
- `RandomForestClassifier` — baseline (n_estimators=200, max_depth=18, min_samples_leaf=5, class_weight='balanced')

**Validation:** 80/20 chronological split (no shuffle) + 5-fold `TimeSeriesSplit` cross-validation.

**Phase 2 conclusion:** Lag and rolling features add temporal memory. HistGradientBoosting selected as primary model. Consistent CV accuracy across all 5 folds confirms generalisation across different time periods.

---

### Phase 3 — Hyperparameter Tuning, XAI & LLM Agent

**GridSearchCV** (3-fold TimeSeriesSplit, 16 combinations):
```
param_grid = {
    max_iter:         [500, 700]
    learning_rate:    [0.02, 0.04]
    max_depth:        [8, 10]
    min_samples_leaf: [6, 10]
}
Best params: learning_rate=0.02, max_depth=8, max_iter=500, min_samples_leaf=6
Best CV Accuracy: 0.7574  |  Test Accuracy: 75.4%  |  Test AUC-ROC: 0.7708
```

**Permutation Importance XAI** (n_repeats=20, model-agnostic):
- Measures accuracy drop when each feature is randomly shuffled
- Equivalent to SHAP's marginal contribution — works with any model
- Per-shipment explainability: `contribution = permutation_importance × |normalized_feature_value|`
- Congestion and lag/rolling features dominate importance, validating Phase 2 engineering decisions

**LangGraph Agent** — 3-node pipeline:
```
Port Risk Node  →  Route History Node  →  LLM Reroute Node (Llama-3.3-70b)
```
- **Port Risk Node:** Computes origin port delay rate, weather severity, and record count from the dataset
- **Route History Node:** Fetches historical delay rate and average delay for the specific origin → destination pair
- **LLM Reroute Node:** Sends structured context to Llama-3.3-70b via Groq; returns RISK / RECOMMENDATION / CONFIDENCE in structured format

Rerouting reasoning is accessible to non-technical operations teams through plain-English output.

**Phase 3 conclusion:** Every rerouting decision is now explainable. The 0.60 threshold balances recall against unnecessary rerouting cost.

---

### Phase 4 — Reporting & Visualization (Dashboard)

7-tab Streamlit dashboard (`app.py`). See [Dashboard Tabs](#dashboard-tabs) below.

---

## Model Performance

### Pre-Tuning (base HistGradientBoosting)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| On-Time | 0.66 | 0.99 | 0.80 |
| Delayed | **0.99** | 0.53 | 0.69 |
| **Weighted avg** | 0.83 | 0.75 | 0.74 |

**AUC-ROC: 0.7730** | **Accuracy: 75.4%**

### Post-Tuning (GridSearchCV — final model saved to pickle)

| Metric | HistGradientBoosting | Random Forest |
|---|---|---|
| Accuracy | **75.4%** | 75.2% |
| AUC-ROC | 0.7708 | 0.7795 |
| Precision (Delayed) | **99%** | 96% |
| Recall (Delayed) | 53% | 54% |

> **Why HGB over RF:** Random Forest achieved marginally higher AUC (0.7795 vs 0.7708) but HistGradientBoosting was selected as the primary model for better generalisation on the temporal holdout and consistent 5-fold CV behaviour. The 99% Delayed-class precision is the key business metric — when the model flags a shipment for rerouting, it is almost always genuinely at risk, minimising unnecessary rerouting costs.

---

## Tech Stack

| Area | Tools |
|---|---|
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| ML & Validation | `scikit-learn` — `HistGradientBoostingClassifier`, `RandomForestClassifier`, `IsolationForest`, `GridSearchCV`, `TimeSeriesSplit`, `cross_val_score`, `permutation_importance` |
| Optimization | `scipy.optimize.linprog` (LP-based route cost minimisation) |
| XAI | `sklearn.inspection.permutation_importance` (model-agnostic, SHAP-equivalent) |
| LLM Agent | `langchain-groq`, `langgraph`, `langchain-core`, Llama-3.3-70b-versatile via Groq |
| Dashboard | `streamlit`, `plotly` |
| Utilities | `python-dotenv`, `pickle` |

---

## Setup & Installation

**1. Clone the repo**
```bash
git clone https://github.com/your-username/nexus-logistics-ai.git
cd nexus-logistics-ai
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com). The LLM Agent tab requires this — all other tabs work without it.

**5. Place data and model files**
```
data/logistics_disruption_dataset.csv
models/logistics_delay_model.pkl
```

To regenerate the model pickle, run `Logistics_Rerouting.ipynb` end-to-end. The notebook saves: tuned model, feature list, scaler, and label encoder.

---

## Running the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar allows filtering by risk status, origin port, transport mode, date range, and minimum AI delay probability threshold.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| 🌐 Global Overview | World map of shipment network (Plotly Scattergeo), port delay rate bars, transport mode breakdown, monthly delay rate + risk trend |
| ⚡ Disruption Monitor | Real port risk heatmap (bubble size = shipment volume, colour = AI delay probability), disruption signal timeline, anomaly vs normal scatter plots |
| ⚠ Route Risk | AI delay probability distribution by risk class, risk status donut chart, origin × destination heatmap, top-N shipment detail cards |
| 🔀 AI Rerouting | LP-optimised route allocation with cost × risk comparison charts, global rerouting map with AI-optimised paths highlighted |
| 🧠 Explainable AI | Permutation importance chart, feature group contribution bars, per-shipment SHAP-style breakdown, top feature correlations with delay |
| 📊 Performance | Confusion matrix, ROC curve (HGB vs RF), 5-fold CV accuracy comparison, baseline vs AI-rerouted business impact metrics |
| 🤖 LLM Agent | LangGraph 3-node pipeline interface — select origin/destination, run Llama-3.3-70b rerouting analysis, view RISK / RECOMMENDATION / CONFIDENCE output with pipeline trace |

---
