from pathlib import Path
from fastapi import APIRouter

from api import service as svc
from database.connection import session_scope
from models.anomaly_detector import AnomalyDetector
from models.failure_predictor import FailurePredictor

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


def _anomaly(session, dev, n_rows=60):
    p = svc.ART / "anomaly_detector.joblib"
    if not p.exists():
        return None, {"status": svc.NOT_TRAINED, "message": "No anomaly model trained. Run: python -m training.train_anomaly"}
    feats, cat, err = svc.latest_features(session, dev)
    if feats is None:
        return None, {"status": svc.INSUFFICIENT, "message": err}
    try:
        det = AnomalyDetector.load(p)
    except Exception as e:
        # Fallback: train a temporary model on recent features
        cols = [c for c in feats.columns if c not in ["__state", "__timestamp"]]
        try:
            det = AnomalyDetector(svc.load_config()).fit(feats.tail(n_rows), cols)
        except Exception as e2:
            return None, {"status": svc.INSUFFICIENT, "message": f"Failed to load or train anomaly model: {e}; fallback error: {e2}"}
    return (det.score(feats.tail(n_rows)), cat), None


@router.get("/anomalies")
def anomalies(device: str | None = None, rows: int = 60):
    with session_scope() as s:
        dev = device or svc.default_device(s)
        res, err = _anomaly(s, dev, rows)
        if err: return {"machine_id": dev, **err, "generated_at": svc.now_iso()}
        sc, _ = res
        recs = [{"timestamp": str(t), "operating_state": r.operating_state, "anomaly_score": None if r.anomaly_score != r.anomaly_score else r.anomaly_score,
                 "status": r.status, "reason": r.reason, "top_features": r.top_features} for t, r in sc.iterrows()]
        return {"machine_id": dev, "status": svc.AVAILABLE, "kind": "anomaly_not_failure_prediction",
                "disclaimer": sc.disclaimer.iloc[-1] if len(sc) else None, "results": recs, "generated_at": svc.now_iso()}


@router.get("/failure-risk")
def failure_risk(device: str | None = None):
    with session_scope() as s:
        dev = device or svc.default_device(s)
        reg = svc.latest_registry(s, "failure")
        p = svc.ART / "failure_predictor.joblib"
        if not p.exists():
            return {"machine_id": dev, "status": svc.NOT_TRAINED if (reg is None or reg["status"] == "not_trained") else svc.INSUFFICIENT,
                    "horizons_hours": [1, 6, 24, 72], "probabilities": None,
                    "message": "No verified-failure model exists. Supervised prediction requires verified failure events (see README).",
                    "registry": reg, "generated_at": svc.now_iso()}
        feats, _, err = svc.latest_features(s, dev)
        if feats is None:
            return {"machine_id": dev, "status": svc.INSUFFICIENT, "message": err, "generated_at": svc.now_iso()}
        m = FailurePredictor.load(p)
        pr = m.predict(feats.tail(1))
        return {"machine_id": dev, "status": svc.AVAILABLE, "timestamp": str(pr.index[-1]),
                "probabilities": {c: float(pr[c].iloc[-1]) for c in pr.columns},
                "validation": m.metrics, "limitations": "Calibrated on held-out runs; see validation. Not a guarantee.",
                "generated_at": svc.now_iso()}


@router.get("/rul")
def rul(device: str | None = None):
    with session_scope() as s:
        dev = device or svc.default_device(s)
        reg = svc.latest_registry(s, "rul")
        p = svc.ART / "rul_estimator.joblib"
        if not p.exists():
            return {"machine_id": dev, "status": svc.NOT_TRAINED if (reg is None or reg["status"] == "not_trained") else svc.INSUFFICIENT,
                    "rul_hours": None, "message": "No RUL model. Requires component-life records with observed degradation endpoints and censoring.",
                    "registry": reg, "generated_at": svc.now_iso()}
        return {"machine_id": dev, "status": svc.UNAVAILABLE, "rul_hours": None,
                "message": "A RUL artifact exists but live per-component covariate wiring is deployment-specific; see README 'Stage C'.",
                "generated_at": svc.now_iso()}
