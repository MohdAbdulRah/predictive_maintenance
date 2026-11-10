"""python -m training.train_failure [--device UUID]

Refuses to train unless verified FailureEvent rows exist in sufficient numbers.
Whole operating runs are held out (train / calibration / test by run, ordered in time)."""
import argparse, json
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import load_config
from database.connection import get_engine
from database.models import FailureEvent, OperatingRun
from evaluation.failure_metrics import evaluate_horizon
from feature_engineering.rolling_features import build_features, feature_columns
from models.failure_predictor import FailurePredictor, InsufficientLabels, assess_trainability, build_failure_labels
from preprocessing.preprocessing import build_grid, load_catalogue, load_gaps, load_observations, select_signals
from training.common import ART, pick_device, record_status


def run(device=None, cfg=None, session=None):
    cfg = cfg or load_config()
    own = session is None
    session = session or Session(get_engine(), expire_on_commit=False)
    try:
        dev = pick_device(session, device)
        runs = session.scalars(select(OperatingRun).order_by(OperatingRun.start_ts)).all()
        fails = session.scalars(select(FailureEvent)).all()
        if not fails or not runs:
            res = {"status": "not_trained", "reason": "no verified failure events and/or operating runs in database. "
                   "Load FailureEvent + OperatingRun rows from maintenance records (see README)."}
            record_status(session, "failure", "not_trained", None, res); return res
        cat = load_catalogue(session, dev)
        frames, run_ids = [], []
        for r in runs:
            obs = load_observations(session, dev, r.start_ts, r.observed_until)
            if obs.empty: continue
            gaps = load_gaps(session, dev, obs)
            sigs = select_signals(cat, obs, cfg)
            feats = build_features(build_grid(obs, cat, gaps, sigs, cfg), sigs, cfg)
            if feats.empty: continue
            f_ts = [f.failure_ts for f in fails if f.run_id == r.run_id]
            lab = build_failure_labels(feats.index, f_ts, pd.Timestamp(r.observed_until), cfg["failure"]["horizons_hours"])
            feats = feats.join(lab); feats["__run"] = r.run_id
            frames.append(feats)
        if not frames:
            return {"status": "insufficient_data", "reason": "no telemetry inside the operating runs"}
        data = pd.concat(frames)
        ycols = [f"y_{h}h" for h in cfg["failure"]["horizons_hours"]]
        trn = assess_trainability(data[ycols + ["hours_to_next_failure"]], data["__run"], cfg)
        if trn["status"] != "trainable":
            record_status(session, "failure", trn["status"], None, trn); return trn
        run_order = list(dict.fromkeys(data["__run"]))
        n = len(run_order); tr_r, ca_r, te_r = run_order[:max(1, int(n*.6))], run_order[int(n*.6):int(n*.8)], run_order[int(n*.8):]
        if not (ca_r and te_r):
            return {"status": "insufficient_data", "reason": f"need >= 5 independent runs to split train/calibration/test; have {n}"}
        cols = feature_columns(data)
        tr, ca, te = (data[data["__run"].isin(x)] for x in (tr_r, ca_r, te_r))
        pred = FailurePredictor(cfg)
        try:
            pred.fit(tr, tr[ycols], ca, ca[ycols], cols)
        except InsufficientLabels as e:
            res = {"status": "insufficient_data", "reason": str(e), "trainability": trn}
            record_status(session, "failure", "insufficient_data", None, res); return res
        P = pred.predict(te)
        res = {"status": "trained", "split_runs": {"train": tr_r, "calibration": ca_r, "test": te_r}, "metrics": {}}
        fts = [f.failure_ts for f in fails if f.run_id in te_r]
        for h in cfg["failure"]["horizons_hours"]:
            res["metrics"][f"{h}h"] = evaluate_horizon(te[f"y_{h}h"], P[f"p_{h}h"], 0.5, fts)
        path = ART / "failure_predictor.joblib"; pred.metrics = res["metrics"]; pred.save(path)
        record_status(session, "failure", "trained", str(path), res)
        return res
    finally:
        if own: session.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device"); a = ap.parse_args()
    print(json.dumps(run(a.device), indent=2, default=str))
