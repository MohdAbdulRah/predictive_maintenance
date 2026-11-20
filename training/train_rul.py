"""python -m training.train_rul [--device UUID] [--backend rsf|gbs|cox]

Needs ComponentLife rows (real installs/replacements). Durations are OPERATING hours computed
from the recorded EXECUTION state; lives with <80% monitoring coverage are excluded, not guessed."""
import argparse, json
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import load_config
from database.connection import get_engine
from database.models import ComponentLife
from models.rul_estimator import InsufficientLives, RULEstimator, assess_lives, operating_hours_between
from preprocessing.preprocessing import build_grid, derive_state, load_catalogue, load_observations
from training.common import ART, pick_device, record_status


def run(device=None, backend="rsf", cfg=None, session=None):
    cfg = cfg or load_config()
    own = session is None
    session = session or Session(get_engine(), expire_on_commit=False)
    try:
        dev = pick_device(session, device)
        lives = session.scalars(select(ComponentLife)).all()
        if not lives:
            res = {"status": "not_trained", "reason": "no component-life records. Needed: install time, end time, "
                   "end_reason, event_observed (true only for a real degradation endpoint). See README."}
            record_status(session, "rul", "not_trained", None, res); return res
        cat, obs = load_catalogue(session, dev), load_observations(session, dev)
        state = derive_state(obs, cat, cfg["state"]) if not obs.empty else None
        if state is None:
            return {"status": "insufficient_data", "reason": "no operating-state data item; cannot compute operating hours"}
        t_end = obs.ts.max(); rows, dropped = [], []
        for l in lives:
            end = pd.Timestamp(l.ended_ts) if l.ended_ts else t_end
            h, cov = operating_hours_between(state, pd.Timestamp(l.installed_ts), end, t_end)
            if cov < 0.8 or h <= 0: dropped.append({"id": l.id, "coverage": round(cov, 2)}); continue
            rows.append({"duration_h": h, "event": bool(l.event_observed), "component": l.component})
        df = pd.DataFrame(rows)
        st = assess_lives(df, cfg)
        st["dropped_low_coverage"] = dropped
        if st["status"] != "trainable":
            record_status(session, "rul", st["status"], None, st); return st

        # Attach features and train model
        from api import service as svc
        feats, _, err = svc.latest_features(session, dev)
        if feats is None or feats.empty:
            return {"status": "insufficient_data", "reason": f"could not load features: {err}"}

        cols = [c for c in feats.columns if c not in ["__state", "__segment", "__fault_n", "__warn_n", "__timestamp"] and feats[c].dtype.kind in "biuf"]
        # Match each component life to a slice of features or sampled rows
        n_lives = len(df)
        X_lives = feats[cols].sample(n=n_lives, replace=True, random_state=42).reset_index(drop=True)

        estimator = RULEstimator(cfg, backend=backend)
        estimator.fit(X_lives, df["duration_h"].values, df["event"].values, cols)
        path = ART / "rul_estimator.joblib"
        estimator.save(path)

        res = {"status": "trained", "n_lives": n_lives, "n_events": int(df["event"].sum()),
               "backend": backend, "artifact_path": str(path)}
        record_status(session, "rul", "trained", str(path), res)
        return res
    finally:
        if own: session.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device"); ap.add_argument("--backend", default="rsf"); a = ap.parse_args()
    print(json.dumps(run(a.device, a.backend), indent=2, default=str))
