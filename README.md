# MTConnect Predictive Maintenance — Full Guide

> **Real-time CNC machine health monitoring** using MTConnect telemetry, ML anomaly detection,
> supervised failure prediction, and Remaining Useful Life (RUL) estimation.
> Built with FastAPI + React/Vite + SQLite/PostgreSQL.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Configuration](#4-configuration)
5. [Stage 0 — Discover Your Machine](#5-stage-0--discover-your-machine)
6. [Stage 1 — Collect Telemetry Data](#6-stage-1--collect-telemetry-data)
7. [Stage A — Train the Anomaly Detector](#7-stage-a--train-the-anomaly-detector)
8. [Stage B — Train the Failure Predictor (requires real labels)](#8-stage-b--train-the-failure-predictor-requires-real-labels)
9. [Stage C — Train the RUL Estimator (requires component-life data)](#9-stage-c--train-the-rul-estimator-requires-component-life-data)
10. [Run the FastAPI Backend](#10-run-the-fastapi-backend)
11. [Run the React Dashboard](#11-run-the-react-dashboard)
12. [Where to See the Data & Models](#12-where-to-see-the-data--models)
13. [API Reference Summary](#13-api-reference-summary)
14. [Diagnostics & Utilities](#14-diagnostics--utilities)
15. [Running Tests](#15-running-tests)
16. [Model Status at a Glance](#16-model-status-at-a-glance)

---

## 1. Project Overview

| Stage | What it does | Ready to run? |
|---|---|---|
| **Ingestion** | Polls MTConnect agent → stores to SQLite/Postgres | ✅ Yes |
| **A: Anomaly Detection** | IsolationForest + robust-z per operating state | ✅ After collecting ≥ 30 min of data |
| **B: Failure Prediction** | Calibrated binary classifiers at 1/6/24/72-hour horizons | ❌ Needs verified `FailureEvent` rows |
| **C: RUL Estimation** | Survival models (RSF/GBS/Cox) | ❌ Needs `ComponentLife` records |

---

## 2. Project Structure

```
predictive_maintenance/
├── config/
│   ├── machine_config.yaml        # Agent URL, DB URL, feature windows, thresholds
│   └── maintenance_rules.yaml     # Rule-based recommendation rules per signal type
│
├── data_ingestion/
│   ├── mtconnect_client.py        # HTTP client for /probe, /current, /sample
│   ├── observation_collector.py   # Continuous collector — writes to DB
│   └── probe_parser.py            # Parses MTConnect XML probe response
│
├── database/
│   ├── connection.py              # SQLAlchemy engine + session factory
│   └── models.py                  # ORM models: ObservationRow, DataItemRow,
│                                  #   FailureEvent, OperatingRun, ComponentLife,
│                                  #   CollectionGap, ModelRegistry
│
├── preprocessing/
│   ├── preprocessing.py           # load_observations, build_grid, select_signals
│   └── data_validation.py         # Validates data quality before training
│
├── feature_engineering/
│   └── rolling_features.py        # Builds 1/5/15/60-min rolling features per state
│
├── models/
│   ├── anomaly_detector.py        # IsolationForest + robust-z wrapper
│   ├── failure_predictor.py       # Calibrated multi-horizon classifier
│   └── rul_estimator.py           # Survival model wrapper (RSF/GBS/Cox)
│
├── training/
│   ├── train_anomaly.py           # ← Run this for Stage A
│   ├── train_failure.py           # ← Run this for Stage B
│   ├── train_rul.py               # ← Run this for Stage C
│   └── common.py                  # Shared utilities (pick_device, record_status)
│
├── evaluation/
│   ├── anomaly_metrics.py         # Alert rate, precision with labels
│   └── failure_metrics.py         # AUROC, Brier, calibration per horizon
│
├── api/
│   ├── main.py                    # FastAPI app entry point (CORS, router registration)
│   ├── service.py                 # Shared helpers (latest_features, ART path)
│   └── routes/
│       ├── machine.py             # GET /api/machine/live, /api/machine/history
│       ├── predictions.py         # GET /api/predictions/anomalies, /failure-risk, /rul
│       ├── maintenance.py         # GET /api/maintenance/recommendations
│       └── model.py               # GET /api/model/status
│
├── scripts/
│   ├── discover.py                # Step 0: inspect MTConnect agent signals
│   ├── collect.py                 # Step 1: continuous data collector (run in background)
│   ├── collect_bootstrap.py       # Alternative: multi-round snapshot collector
│   ├── collect_once.py            # Single-shot collection
│   ├── load_labels.py             # Load failure/run/life CSVs into DB
│   ├── check_db.py                # Quick DB row count & time-span check
│   └── diagnose.py                # Deep data-readiness diagnostic for training
│
├── dashboard/                     # React + Vite frontend
│   ├── src/
│   │   ├── App.tsx                # Root component
│   │   ├── api.ts                 # Fetch helpers — polls all API endpoints
│   │   └── components/            # LiveTelemetry, AnomalyScore, FailureRisk,
│   │                              #   RULGauge, MaintenanceAlerts, etc.
│   ├── .env                       # VITE_API=http://localhost:8003
│   └── package.json
│
├── artifacts/                     # ← ALL GENERATED DATA LIVES HERE
│   ├── pm.db                      # SQLite database (all telemetry + model registry)
│   ├── anomaly_detector.joblib    # Trained anomaly model (after Stage A)
│   ├── failure_predictor.joblib   # Trained failure model (after Stage B)
│   ├── rul_estimator.joblib       # Trained RUL model (after Stage C)
│   ├── discovery_report.json      # Machine signals report (after discover)
│   ├── probe.xml                  # Raw probe XML
│   ├── current.xml                # Raw current XML
│   └── sample.xml                 # Raw sample XML
│
├── requirements.txt
├── AI_Industrial_Maintenance_Model_Training.ipynb
└── api_reference.md               # Detailed API docs with example JSON
```

---

## 3. Prerequisites & Installation

### Requirements
- **Python 3.10+**
- **Node.js 18+** (for the dashboard)

### Python Environment

```powershell
# From the predictive_maintenance/ directory:

# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> **Stage C only** — install survival libraries when you have component-life data:
> ```powershell
> pip install scikit-survival>=0.22 lifelines>=0.28
> ```

> **PostgreSQL** — if using Postgres instead of SQLite:
> ```powershell
> pip install psycopg2-binary>=2.9
> # Then set in config/machine_config.yaml:
> # database.url: "postgresql+psycopg2://user:password@host/dbname"
> ```

### Dashboard (Node)

```powershell
cd dashboard
npm install
```

---

## 4. Configuration

Edit `config/machine_config.yaml` before running anything:

```yaml
agent:
  base_url: "https://demo.mtconnect.org"   # ← Change to your machine's MTConnect URL
  device: null                              # ← null = auto-detect; or set device name/UUID
  poll_interval_s: 1.0                      # Polling frequency

database:
  url: "sqlite:///artifacts/pm.db"          # ← SQLite default. Change for Postgres.

state:
  source_type: "EXECUTION"
  active_values: ["ACTIVE"]                 # ← Verify these match your machine's EXECUTION values
  idle_values: ["READY", "INTERRUPTED", "FEED_HOLD", "STOPPED"]
```

> Run `scripts/discover.py` first (Step 0) to see the **exact EXECUTION values** your machine reports.

---

## 5. Stage 0 — Discover Your Machine

**Purpose:** Inspect your MTConnect agent to find all available signals and their IDs.

```powershell
# Connect to live agent and save raw XML
python -m scripts.discover --url https://demo.mtconnect.org --save-raw

# Or use saved XML files
python -m scripts.discover --probe artifacts/probe.xml --current artifacts/current.xml
```

**What it does:**
- Prints all `DataItem` types, categories, units, and IDs
- Shows actual `EXECUTION` values your machine reports (needed for `state` config)
- Writes `artifacts/discovery_report.json`

**Where to find the output:**
```
artifacts/discovery_report.json   ← Full machine signal report
artifacts/probe.xml               ← Raw probe response (with --save-raw)
artifacts/current.xml             ← Raw current snapshot
artifacts/sample.xml              ← Sample of 500 observations
```

**After running:** Update `config/machine_config.yaml` with the correct `active_values` / `idle_values`.

---

## 6. Stage 1 — Collect Telemetry Data

**Purpose:** Stream sensor readings from the MTConnect agent into the local database.

### Option A — Continuous Collector (Recommended)
```powershell
# Runs indefinitely, Ctrl-C safe, automatically resumes from last sequence
python -m scripts.collect
```
Leave this running in a **separate terminal** for hours or days.

### Option B — Bootstrap Snapshot Collector (Demo / Demo Agent)
Use this if the demo agent's ring buffer overruns too fast:
```powershell
# 30 rounds, 3 seconds apart (captures ~90 seconds of data)
python -m scripts.collect_bootstrap --rounds 30 --delay 3

# More rounds for more data
python -m scripts.collect_bootstrap --rounds 200 --delay 2
```

### Option C — Single Snapshot
```powershell
python -m scripts.collect_once
```

### Where Data Is Stored
```
artifacts/pm.db                   ← SQLite database
  Tables:
    observation_row               ← All sensor readings (device_uuid, data_item_id, ts, value_num, ...)
    data_item_row                 ← Signal catalogue (id, name, type, units, category)
    collection_gap                ← Recorded gaps/outages (never interpolated across)
    model_registry                ← Training run history and status
    failure_event                 ← (Stage B) Verified failure timestamps
    operating_run                 ← (Stage B) Operating run boundaries
    component_life                ← (Stage C) Component install/replace records
```

### Check What Was Collected
```powershell
python -m scripts.check_db
```
Output example:
```
observations: 48320
data_items: 47
time span: 2026-11-02 10:00:00 to 2026-11-02 22:00:00
  LX1load: 43.0
  LS2speed: 1200.0
```

---

## 7. Stage A — Train the Anomaly Detector

**Prerequisite:** At least ~30–60 minutes of collected data (≥ 50 rows per operating state after resampling).

```powershell
# Train on last 7 days of data (default)
python -m training.train_anomaly

# Train on last 30 days
python -m training.train_anomaly --days 30

# Train on all available data
python -m training.train_anomaly --days 0

# Train for a specific device
python -m training.train_anomaly --device OKUMA.123456
```

**What it does:**
- Loads observations from DB → resamples to 5-second grid → builds rolling features (1/5/15/60 min windows)
- Chronological 70/30 split with an embargo equal to the longest feature window
- Fits one `IsolationForest` + robust-z baseline per operating state (`active`, `idle`, `unknown`)
- Prints train/test row counts and held-out alert rates
- Saves the trained model

**Successful output looks like:**
```json
{
  "status": "trained",
  "signals": ["LX1load", "LX1actm", "LS2speed", ...],
  "train_rows": 1455,
  "heldout_rows": 620,
  "states_modelled": ["unknown"],
  "heldout_alert_rates": { "per_active_day": 0.03 }
}
```

**Where the model is saved:**
```
artifacts/anomaly_detector.joblib   ← Trained anomaly model
```

**If you get `insufficient_data`:**
```powershell
# Run the diagnostic tool to understand why
python -m scripts.diagnose
```

---

## 8. Stage B — Train the Failure Predictor (requires real labels)

**Prerequisite:** Verified failure events and operating run records in the database.

### Step 8a — Prepare Label CSVs

**`runs.csv`** — Operating run boundaries:
```csv
run_id,machine_id,start_ts,observed_until
RUN001,OKUMA.123456,2026-03-01T08:00:00Z,2026-03-10T18:00:00Z
RUN002,OKUMA.123456,2026-03-11T08:00:00Z,2026-03-20T18:00:00Z
```

**`failures.csv`** — Verified failure events:
```csv
machine_id,run_id,component,failure_ts,failure_mode,verified_by,evidence_ref
OKUMA.123456,RUN001,spindle,2026-03-08T14:30:00Z,bearing_wear,J.Smith,WO-1234
```

### Step 8b — Load Labels into DB
```powershell
python -m scripts.load_labels runs     runs.csv
python -m scripts.load_labels failures failures.csv
```

### Step 8c — Train
```powershell
python -m training.train_failure
python -m training.train_failure --device OKUMA.123456   # specific device
```

**Training requires:**
- ≥ 20 verified failures per time horizon (1h, 6h, 24h, 72h)
- ≥ 3 distinct operating runs containing failures
- ≥ 5 operating runs total (for train/calibration/test split)

**Where the model is saved:**
```
artifacts/failure_predictor.joblib
```

---

## 9. Stage C — Train the RUL Estimator (requires component-life data)

**Prerequisite:** Component installation and replacement records.

### Step 9a — Prepare `lives.csv`
```csv
machine_id,component,installed_ts,ended_ts,end_reason,event_observed,verified_by
OKUMA.123456,spindle_bearing,2025-01-01T00:00:00Z,2026-03-08T00:00:00Z,bearing_failure,true,J.Smith
OKUMA.123456,spindle_bearing,2026-03-09T00:00:00Z,,still_in_service,false,J.Smith
```
- `event_observed = true` → real degradation endpoint (not a preventive swap)
- `ended_ts` empty → component still in service (censored)

### Step 9b — Load Lives
```powershell
python -m scripts.load_labels lives lives.csv
```

### Step 9c — Train
```powershell
# Default backend: Random Survival Forest
python -m training.train_rul

# Other backends
python -m training.train_rul --backend gbs
python -m training.train_rul --backend cox
```

**Training requires:**
- ≥ 15 component-life records
- ≥ 10 observed (non-censored) failure events
- ≥ 80% monitoring coverage per life

**Where the model is saved:**
```
artifacts/rul_estimator.joblib
```

---

## 10. Run the FastAPI Backend

```powershell
# From the predictive_maintenance/ directory (with .venv active):
uvicorn api.main:app --reload --port 8000
```

> **Note:** The dashboard's `.env` points to port `8003`. Either:
> - Start uvicorn on port 8003: `uvicorn api.main:app --reload --port 8003`
> - Or update `dashboard/.env` to `VITE_API=http://localhost:8000`

**Verify the API is running:**
```powershell
# Health check
curl http://localhost:8000/health
# Expected: {"ok": true}

# Check model training status
curl http://localhost:8000/api/model/status

# Get latest sensor readings
curl http://localhost:8000/api/machine/live

# Get anomaly predictions
curl http://localhost:8000/api/predictions/anomalies
```

Open **Swagger UI** in browser: `http://localhost:8000/docs`

---

## 11. Run the React Dashboard

```powershell
cd dashboard
npm run dev
```

Opens at: **`http://localhost:5173`**

The dashboard polls these endpoints automatically:
| Endpoint | Poll interval | What it shows |
|---|---|---|
| `/api/machine/live` | 2 seconds | Live sensor gauges |
| `/api/predictions/anomalies` | 5 seconds | Anomaly score + top flagged features |
| `/api/maintenance/recommendations` | 10 seconds | Inspection alerts |
| `/api/predictions/failure-risk` | 30 seconds | Failure probability (if model trained) |
| `/api/predictions/rul` | 30 seconds | Remaining useful life (if model trained) |

---

## 12. Where to See the Data & Models

### Database (SQLite)
```
artifacts/pm.db
```
Browse with any SQLite viewer (e.g. [DB Browser for SQLite](https://sqlitebrowser.org/)):

| Table | What it contains |
|---|---|
| `observation_row` | All raw sensor readings |
| `data_item_row` | Signal catalogue (names, units, types) |
| `collection_gap` | Recorded data gaps |
| `model_registry` | Training run history (status, trained_at, metrics) |
| `failure_event` | Verified failure timestamps (Stage B) |
| `operating_run` | Run boundaries (Stage B) |
| `component_life` | Component lifecycle records (Stage C) |

### Quick DB check (command line)
```powershell
python -m scripts.check_db
```

### Trained Models
```
artifacts/anomaly_detector.joblib     ← Stage A (exists after train_anomaly)
artifacts/failure_predictor.joblib    ← Stage B (exists after train_failure)
artifacts/rul_estimator.joblib        ← Stage C (exists after train_rul)
```

### Discovery & Raw XML
```
artifacts/discovery_report.json       ← All signals, EXECUTION values
artifacts/probe.xml                   ← Machine probe XML
artifacts/current.xml                 ← Latest current snapshot XML
artifacts/sample.xml                  ← Sample of 500 observations XML
```

### Model Training Status via API
```
GET http://localhost:8000/api/model/status
```
Returns training status (`trained` / `not_trained` / `insufficient_data`) and metrics for all three stages.

### Jupyter Notebook
```
AI_Industrial_Maintenance_Model_Training.ipynb
```
Interactive notebook for exploring data, training experiments, and visualisations.

---

## 13. API Reference Summary

> Full docs with request/response examples: [`api_reference.md`](api_reference.md)
> Interactive Swagger UI: `http://localhost:8000/docs`

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /api/machine/live` | Latest sensor values for all signals |
| `GET /api/machine/history?data_item_id=LX1load` | Time-series for one signal |
| `GET /api/predictions/anomalies?rows=60` | Anomaly scores (last N rows) |
| `GET /api/predictions/failure-risk` | Failure probability at 1/6/24/72h horizons |
| `GET /api/predictions/rul` | Remaining useful life estimate |
| `GET /api/maintenance/recommendations` | Rule-based inspection suggestions |
| `GET /api/model/status` | Training status for all 3 model stages |

**Key sensor IDs** (from OKUMA demo agent):
| `data_item_id` | Name | Units |
|---|---|---|
| `LX1load` | X-axis motor load | PERCENT |
| `LX1actm` | X-axis machine position | MILLIMETER |
| `LZ1actm` | Z-axis machine position | MILLIMETER |
| `LS2speed` | Spindle speed | REVOLUTION/MINUTE |
| `LpCuttingTime` | Cumulative cutting time | SECOND |

---

## 14. Diagnostics & Utilities

### Diagnose Training Readiness
```powershell
python -m scripts.diagnose
```
Shows: observation count, signal selection, grid shape, state distribution, usable features per state.

### Check Database Contents
```powershell
python -m scripts.check_db
```

### Retrain After More Data Arrives
```powershell
# Retrain anomaly detector on last 7 days (default)
python -m training.train_anomaly --days 7

# Retrain on all data
python -m training.train_anomaly --days 0
```

### CORS Configuration
Edit `api/main.py` to add allowed frontend origins:
```python
allow_origins=["http://localhost:3000", "http://localhost:5173", "http://yourhost:PORT"]
```

---

## 15. Running Tests

```powershell
# Run all tests
python -m pytest tests -q

# Run with verbose output
python -m pytest tests -v

# Run a specific test file
python -m pytest tests/test_preprocessing.py -v
```

Tests use **synthetic fixtures** (`tests/simulator.py`) — no real machine needed.
Tests cover: parsing, gap handling, deduplication, censoring, feature leakage, API gating.

---

## 16. Model Status at a Glance

After completing each stage, here is what you should see:

```
Stage A (Anomaly)  →  artifacts/anomaly_detector.joblib       ✅ Available
Stage B (Failure)  →  artifacts/failure_predictor.joblib      ❌ Until FailureEvent rows loaded
Stage C (RUL)      →  artifacts/rul_estimator.joblib          ❌ Until ComponentLife rows loaded
```

Check model status any time:
```powershell
curl http://localhost:8000/api/model/status
```

---

## Quick-Start Checklist

```
[ ] 1. pip install -r requirements.txt  (activate .venv first)
[ ] 2. Edit config/machine_config.yaml  (set agent base_url)
[ ] 3. python -m scripts.discover --url <your_agent_url> --save-raw
[ ] 4. Update config/machine_config.yaml with correct EXECUTION values
[ ] 5. python -m scripts.collect         (leave running in background)
      — OR —
      python -m scripts.collect_bootstrap --rounds 100 --delay 2
[ ] 6. python -m scripts.check_db        (verify data is being collected)
[ ] 7. python -m training.train_anomaly  (Stage A)
[ ] 8. uvicorn api.main:app --reload --port 8003
[ ] 9. cd dashboard && npm run dev       (open http://localhost:5173)
[ ] 10. (Optional) Load failure labels and train Stage B
[ ] 11. (Optional) Load component lives and train Stage C
```
