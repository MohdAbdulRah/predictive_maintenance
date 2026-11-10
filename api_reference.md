# MTConnect Predictive Maintenance — API Reference

> **Base URL**: `http://localhost:8000`
> **API is live and running.** All responses are JSON. No authentication required.

---

## Quick Start (Frontend)

```js
const API = "http://localhost:8000";
const get = (path) => fetch(API + path).then(r => r.json());

// Poll these every 2–5 seconds for live updates:
get("/api/machine/live")                    // current sensor readings
get("/api/predictions/anomalies?rows=1")    // latest anomaly score
get("/api/maintenance/recommendations")     // maintenance alerts
```

---

## Endpoints

### `GET /health`
**Simple health check.**

```json
{ "ok": true }
```

---

### `GET /api/machine/live`
**Latest sensor reading for every signal on the machine.**

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `device` | string | auto | Device UUID (e.g. `OKUMA.123456`) |

**Response:**
```json
{
  "status": "ok",
  "machine_id": "OKUMA.123456",
  "observations": [
    {
      "data_item_id": "LX1load",
      "name": "X1load",
      "type": "LOAD",
      "category": "SAMPLE",
      "value": 43.0,
      "unavailable": false,
      "units": "PERCENT",
      "timestamp": "2026-11-02T12:08:53+00:00"
    }
    // ... one entry per active data item
  ],
  "generated_at": "2026-11-02T12:18:00+00:00"
}
```

**Key signals to display:**
| `data_item_id` | Description | Units |
|---|---|---|
| `LX1load` | X-axis motor load | `PERCENT` |
| `LX1actm` | X-axis machine position | `MILLIMETER` |
| `LZ1actm` | Z-axis machine position | `MILLIMETER` |
| `LS2speed` | Spindle speed | `REVOLUTION/MINUTE` |
| `LX1actw` | X-axis work position | `MILLIMETER` |
| `LpCuttingTime` | Cumulative cutting time | `SECOND` |

---

### `GET /api/machine/history`
**Time-series history for a single signal.**

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `data_item_id` | string | *required* | e.g. `LX1load` |
| `hours` | float | `2.0` | How many hours back to fetch |
| `device` | string | auto | Device UUID |

**Response:**
```json
{
  "status": "ok",
  "machine_id": "OKUMA.123456",
  "data_item_id": "LX1load",
  "series": [
    { "timestamp": "2026-11-02T12:00:00+00:00", "value": 43.0 },
    { "timestamp": "2026-11-02T12:00:05+00:00", "value": 44.0 }
  ],
  "generated_at": "2026-11-02T12:18:00+00:00"
}
```

**Usage in a chart:**
```js
const { series } = await get("/api/machine/history?data_item_id=LX1load&hours=1");
// series → [{ timestamp, value }, ...] ready for Chart.js / Recharts
```

---

### `GET /api/predictions/anomalies`
**Anomaly scores for the most recent N time steps. ✅ Model trained and active.**

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `rows` | int | `60` | Number of recent grid rows (each = 5 seconds) |
| `device` | string | auto | Device UUID |

**Response:**
```json
{
  "machine_id": "OKUMA.123456",
  "status": "prediction_available",
  "kind": "anomaly_not_failure_prediction",
  "disclaimer": "Anomaly = behaviour rare relative to this machine's own training data. It is not a confirmed fault and not a failure probability.",
  "results": [
    {
      "timestamp": "2026-11-02T12:18:55+00:00",
      "operating_state": "unknown",
      "anomaly_score": 0.4821,
      "status": "normal",
      "reason": "isolation_forest=ok, statistical=ok (max |z|=1.2, 0 features beyond threshold)",
      "top_features": [
        {
          "feature": "LX1load__60s_mean",
          "robust_z": 1.21,
          "value": 45.3,
          "training_median": 43.0
        }
      ]
    }
  ],
  "generated_at": "2026-11-02T12:18:59+00:00"
}
```

**`status` values per result row:**
| Value | Meaning | UI suggestion |
|---|---|---|
| `"normal"` | Within training distribution | 🟢 Green |
| `"watch"` | Slightly elevated — worth monitoring | 🟡 Yellow |
| `"anomalous"` | Rare pattern — alert | 🔴 Red |
| `"not_scored"` | State not modelled / warm-up | ⚪ Grey |

**`anomaly_score`**: 0.0–1.0 percentile of training distribution. `> 0.995` = alert threshold.

**Frontend polling pattern:**
```js
// Poll every 5 seconds; show latest score
setInterval(async () => {
  const { results, status, disclaimer } = await get("/api/predictions/anomalies?rows=1");
  if (status === "prediction_available" && results.length > 0) {
    const latest = results[results.length - 1];
    updateUI(latest.anomaly_score, latest.status, latest.top_features);
  }
}, 5000);
```

---

### `GET /api/predictions/failure-risk`
**Failure probability at 1/6/24/72-hour horizons.**

> ⚠️ Returns `status: "not_trained"` — requires verified failure event labels in DB.

**Response:**
```json
{
  "machine_id": "OKUMA.123456",
  "status": "not_trained",
  "horizons_hours": [1, 6, 24, 72],
  "probabilities": null,
  "message": "No verified-failure model exists. Supervised prediction requires verified failure events (see README).",
  "generated_at": "2026-11-02T12:18:59+00:00"
}
```

**When trained, `probabilities` will be:**
```json
{
  "p_fail_1h": 0.02,
  "p_fail_6h": 0.05,
  "p_fail_24h": 0.12,
  "p_fail_72h": 0.18
}
```

---

### `GET /api/predictions/rul`
**Remaining Useful Life estimate.**

> ⚠️ Returns `status: "not_trained"` — requires component lifecycle records.

**Response:**
```json
{
  "machine_id": "OKUMA.123456",
  "status": "not_trained",
  "rul_hours": null,
  "message": "No RUL model. Requires component-life records.",
  "generated_at": "2026-11-02T12:18:59+00:00"
}
```

---

### `GET /api/maintenance/recommendations`
**Rule-based maintenance suggestions triggered by active anomalies.**

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `device` | string | auto | Device UUID |

**Response (no anomaly):**
```json
{
  "machine_id": "OKUMA.123456",
  "status": "prediction_available",
  "recommendations": [],
  "note": "No active anomaly.",
  "generated_at": "2026-11-02T12:19:00+00:00"
}
```

**Response (when anomaly detected):**
```json
{
  "machine_id": "OKUMA.123456",
  "status": "prediction_available",
  "recommendations": [
    {
      "signal_type": "LOAD",
      "risk_level": "medium",
      "detected_issue": "Elevated axis load detected",
      "evidence": ["LX1load__60s_mean z=4.5"],
      "suggested_inspection": "Check axis lubrication and drive train",
      "urgency": "within_24h",
      "confidence": "anomaly_status_only",
      "limitations": "Without failure model, risk is based on anomaly status only"
    }
  ],
  "generated_at": "2026-11-02T12:19:00+00:00"
}
```

---

### `GET /api/model/status`
**Model training status for all three stages.**

**Response:**
```json
{
  "stages": {
    "anomaly": {
      "status": "trained",
      "artifact_present": true,
      "registry": {
        "status": "trained",
        "trained_at": "2026-11-02T12:18:18",
        "detail": {
          "signals": ["LX1load", "LX1actm", "..."],
          "train_rows": 1455,
          "states_modelled": ["unknown"],
          "states_skipped": { "active": "insufficient data: ...", "idle": "insufficient data: ..." }
        }
      }
    },
    "failure": { "status": "not_trained", "artifact_present": false },
    "rul":     { "status": "not_trained", "artifact_present": false }
  }
}
```

---

## Frontend Dashboard Recipe

```js
const API = "http://localhost:8000";
const get = (p) => fetch(API + p).then(r => r.json());

// 1. Live sensor panel — poll every 2s
async function updateLive() {
  const { observations } = await get("/api/machine/live");
  // observations is an array; find the signals you want by data_item_id
  const spindle = observations.find(o => o.data_item_id === "LS2speed");
  const xLoad   = observations.find(o => o.data_item_id === "LX1load");
  renderGauges({ spindle, xLoad });
}

// 2. Anomaly score card — poll every 5s
async function updateAnomaly() {
  const { results, status } = await get("/api/predictions/anomalies?rows=12");
  if (status !== "prediction_available") return;
  // results sorted oldest→newest; animate score bar
  const scores = results.map(r => ({ t: r.timestamp, score: r.anomaly_score, state: r.status }));
  renderAnomalyTimeline(scores);
  renderTopFeatures(results.at(-1)?.top_features ?? []);
}

// 3. Maintenance alerts — poll every 10s
async function updateRecommendations() {
  const { recommendations, note } = await get("/api/maintenance/recommendations");
  renderAlerts(recommendations, note);
}

// 4. Signal history chart — on demand or refresh every 30s
async function loadChart(signal = "LX1load", hours = 1) {
  const { series } = await get(`/api/machine/history?data_item_id=${signal}&hours=${hours}`);
  renderLineChart(series); // { timestamp, value }[]
}

// Wire up intervals
setInterval(updateLive,            2_000);
setInterval(updateAnomaly,         5_000);
setInterval(updateRecommendations, 10_000);
setInterval(() => loadChart(),     30_000);
```

---

## CORS
Allowed origins: `http://localhost:3000`, `http://localhost:5173` (Vite/CRA defaults).
To add more, edit `api/main.py` → `allow_origins`.

---

## Keep Collecting Data
Run the collector in a separate terminal to continuously stream from the machine:
```bash
python -m scripts.collect
```

Retrain periodically as more data accumulates:
```bash
python -m training.train_anomaly --days 7
```
