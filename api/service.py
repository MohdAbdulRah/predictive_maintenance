"""Shared inference helpers. Every function returns an explicit status; none fabricate output."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import func, select

from config import ROOT, load_config
from database.models import ModelRegistry, ObservationRow
from feature_engineering.rolling_features import build_features
from preprocessing.preprocessing import build_grid, load_catalogue, load_gaps, load_observations, select_signals

ART = ROOT / "artifacts"
NOT_TRAINED, INSUFFICIENT, AVAILABLE, UNAVAILABLE = "not_trained", "insufficient_data", "prediction_available", "model_unavailable"


def default_device(session) -> str | None:
    from database.models import DataItemRow
    return session.scalars(select(DataItemRow.device_uuid).distinct()).first()


def latest_features(session, device: str, cfg: dict | None = None, hours: float | None = None):
    """Features for the most recent `hours` of telemetry (default: longest window + 30 min warm-up)."""
    cfg = cfg or load_config()
    hours = hours or (max(cfg["features"]["windows_seconds"]) / 3600 + 0.5)
    t_last = session.scalar(select(func.max(ObservationRow.ts)).where(ObservationRow.device_uuid == device))
    if t_last is None:
        return None, None, "no observations stored"
    t_last = pd.Timestamp(t_last).tz_localize("UTC") if pd.Timestamp(t_last).tzinfo is None else pd.Timestamp(t_last)
    # include history before the window so ffilled state/values are known at window start
    obs = load_observations(session, device, t_last - timedelta(hours=hours) - timedelta(days=1), t_last)
    cat = load_catalogue(session, device)
    gaps = load_gaps(session, device, obs)
    sigs = select_signals(cat, obs, cfg)
    if not sigs:
        return None, cat, "no usable numeric signals in recent data"
    grid = build_grid(obs, cat, gaps, sigs, cfg)
    grid = grid[grid.index >= t_last - timedelta(hours=hours)]
    feats = build_features(grid, sigs, cfg)
    if feats.empty:
        return None, cat, "recent window is empty after gap handling"
    return feats, cat, None


def latest_registry(session, stage: str) -> dict | None:
    r = session.scalars(select(ModelRegistry).where(ModelRegistry.stage == stage).order_by(ModelRegistry.id.desc())).first()
    if r is None:
        return None
    return {"status": r.status, "trained_at": r.trained_at.isoformat() if r.trained_at else None,
            "artifact_path": r.artifact_path, "detail": json.loads(r.detail_json) if r.detail_json else None}


def now_iso():
    return datetime.now(timezone.utc).isoformat()
