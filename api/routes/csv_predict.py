"""
FastAPI route: CSV upload → AI4I machine failure predictions
Endpoint: POST /api/csv-predict/predict
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/csv-predict", tags=["csv-predict"])

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
BUNDLE_PATH = ARTIFACTS_DIR / "nb_best_model.joblib"
META_PATH = ARTIFACTS_DIR / "nb_models_meta.json"

_bundle_cache = None


def _load_bundle():
    global _bundle_cache
    if _bundle_cache is None:
        if not BUNDLE_PATH.exists():
            return None
        _bundle_cache = joblib.load(BUNDLE_PATH)
    return _bundle_cache


@router.get("/status")
def csv_predict_status():
    """Check whether the notebook models have been trained."""
    if BUNDLE_PATH.exists() and META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        return {
            "trained": True,
            "best_model": meta.get("best_model_name"),
            "threshold": meta.get("threshold"),
            "feature_columns": meta.get("feature_columns"),
            "test_metrics": meta.get("test_metrics"),
            "all_models": list(meta.get("all_models", {}).keys()),
        }
    return {"trained": False, "message": "Models not trained. Run training/train_notebook_models.py first."}


@router.post("/predict")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and get machine failure predictions.

    Expected columns (same as AI4I dataset):
    - Type (L/M/H)
    - Air temperature [K]  (or 'Air temperature')
    - Process temperature [K]  (or 'Process temperature')
    - Rotational speed [rpm]  (or 'Rotational speed')
    - Torque [Nm]  (or 'Torque')
    - Tool wear [min]  (or 'Tool wear')

    Optional columns (ignored but allowed):
    - UDI, Product ID, Machine failure, TWF, HDF, PWF, OSF, RNF
    """
    bundle = _load_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Models not trained. Run: python -m training.train_notebook_models"
        )

    # Read CSV
    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df_raw.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    # Normalize column names: strip brackets/units, lowercase strip for matching
    col_map = {}
    for col in df_raw.columns:
        normalized = col.strip()
        # Remove unit suffixes like [K], [rpm], [Nm], [min]
        for suffix in [" [K]", " [rpm]", " [Nm]", " [min]"]:
            normalized = normalized.replace(suffix, "")
        col_map[col] = normalized
    df = df_raw.rename(columns=col_map)

    # Drop leakage/id columns
    leakage_cols = [
        c for c in df.columns
        if c.strip().lower() in {"udi", "product id", "machine failure", "twf", "hdf", "pwf", "osf", "rnf"}
    ]
    df_features = df.drop(columns=leakage_cols, errors="ignore")

    # Keep only expected feature columns
    expected_cols = bundle["feature_columns"]
    missing_cols = [c for c in expected_cols if c not in df_features.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_cols}. "
                   f"Available: {df_features.columns.tolist()}"
        )

    X = df_features[expected_cols].copy()
    n_rows = len(X)

    model = bundle["model"]
    threshold = bundle["threshold"]

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            predictions = (proba >= threshold).astype(int).tolist()
            probabilities = [round(float(p), 4) for p in proba]
        else:
            predictions = model.predict(X).tolist()
            probabilities = [None] * n_rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Build result rows with native Python types for JSON serialization
    rows = []
    for i in range(n_rows):
        row = {}
        # Include original input columns
        for col in expected_cols:
            val = X.iloc[i][col]
            if pd.isna(val):
                row[col] = None
            elif hasattr(val, "item"):
                row[col] = val.item()
            else:
                row[col] = val
        row["failure_probability"] = float(probabilities[i]) if probabilities[i] is not None else None
        row["predicted_failure"] = int(predictions[i])
        row["predicted_label"] = "⚠️ FAILURE" if predictions[i] == 1 else "✅ OK"
        rows.append(row)

    # Summary stats
    n_failures = int(sum(predictions))
    return {
        "model_used": str(bundle["best_model_name"]),
        "threshold": float(threshold),
        "total_rows": int(n_rows),
        "predicted_failures": n_failures,
        "failure_rate": round(float(n_failures / n_rows), 4) if n_rows > 0 else 0.0,
        "results": rows,
    }


@router.get("/models")
def list_models():
    """List all available trained models with their metrics."""
    if not META_PATH.exists():
        return {"trained": False}
    with open(META_PATH) as f:
        meta = json.load(f)
    return meta

