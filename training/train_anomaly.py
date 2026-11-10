"""python -m training.train_anomaly [--device UUID] [--days N]

Chronological split: first 70% fit, last 30% held out with an embargo equal to the longest
window so overlapping windows cannot leak. Held-out output is an ALERT RATE (unconfirmed).

--days N  : limit training to most recent N days of data (default 7). Use 0 for all data.
"""
import argparse, json
from datetime import timedelta

import pandas as pd
from sqlalchemy.orm import Session

from config import load_config
from database.connection import get_engine
from evaluation.anomaly_metrics import alert_rates
from feature_engineering.rolling_features import build_features, feature_columns
from models.anomaly_detector import AnomalyDetector
from preprocessing.data_validation import validate_observations
from preprocessing.preprocessing import build_grid, load_catalogue, load_gaps, load_observations, select_signals
from training.common import ART, pick_device, record_status


def run(device=None, cfg=None, session=None, days: int = 7):
    cfg = cfg or load_config()
    own = session is None
    session = session or Session(get_engine(), expire_on_commit=False)
    try:
        dev = pick_device(session, device)
        # Limit to recent window to avoid processing months of data on the 5s grid
        start_cutoff = None
        if days > 0:
            from sqlalchemy import func, select
            from database.models import ObservationRow
            t_last = session.scalar(select(func.max(ObservationRow.ts)).where(ObservationRow.device_uuid == dev))
            if t_last is not None:
                import pandas as _pd
                t_last = _pd.Timestamp(t_last, tz="UTC") if _pd.Timestamp(t_last).tzinfo is None else _pd.Timestamp(t_last)
                start_cutoff = t_last - timedelta(days=days)

        cat = load_catalogue(session, dev)
        obs = load_observations(session, dev, start=start_cutoff)
        gaps = load_gaps(session, dev, obs)
        report = validate_observations(obs, cat, gaps)
        if obs.empty:
            record_status(session, "anomaly", "insufficient_data", None, report)
            return {"status": "insufficient_data", "report": report}
        sigs = select_signals(cat, obs, cfg)
        grid = build_grid(obs, cat, gaps, sigs, cfg)
        feats = build_features(grid, sigs, cfg)
        cols = feature_columns(feats)
        print(f"[train] signals={len(sigs)}, grid_rows={len(grid)}, feat_rows={len(feats)}")
        if not feats.empty:
            print(f"[train] state distribution: {feats['__state'].value_counts().to_dict()}")
        if len(feats) < 50 or not sigs:
            res = {"status": "insufficient_data", "signals": sigs, "rows": int(len(feats)), "report": report}
            record_status(session, "anomaly", "insufficient_data", None, res); return res
        cut = feats.index[int(len(feats) * 0.7)]
        embargo = pd.Timedelta(seconds=max(cfg["features"]["windows_seconds"]))
        train, test = feats[feats.index < cut], feats[feats.index >= cut + embargo]
        print(f"[train] train_rows={len(train)}, test_rows={len(test)}")
        det = AnomalyDetector(cfg).fit(train, cols)
        path = ART / "anomaly_detector.joblib"; det.save(path)
        res = {"status": "trained", "signals": sigs, "train_rows": int(len(train)), "heldout_rows": int(len(test)),
               "states_modelled": list(det.models), "states_skipped": det.skipped_states,
               "validation_issues": report["issues"]}
        if len(test):
            res["heldout_alert_rates"] = alert_rates(det.score(test), cfg["preprocessing"]["grid_seconds"])
        record_status(session, "anomaly", "trained", str(path), res)
        return res
    finally:
        if own: session.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--days", type=int, default=7, help="Limit training to most recent N days (0=all)")
    a = ap.parse_args()
    print(json.dumps(run(a.device, days=a.days), indent=2, default=str))
