from fastapi import APIRouter

from api import service as svc
from api.routes.predictions import _anomaly
from database.connection import session_scope
from models.failure_predictor import FailurePredictor
from recommendations.maintenance_engine import LIMITS, load_rules, recommend

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


@router.get("/recommendations")
def recommendations(device: str | None = None):
    with session_scope() as s:
        dev = device or svc.default_device(s)
        res, err = _anomaly(s, dev)
        if err:
            return {"machine_id": dev, **err, "recommendations": [], "generated_at": svc.now_iso()}
        sc, cat = res
        fp = None
        p = svc.ART / "failure_predictor.joblib"
        if p.exists():
            feats, _, _ = svc.latest_features(s, dev)
            if feats is not None:
                pr = FailurePredictor.load(p).predict(feats.tail(1)); fp = {c: float(pr[c].iloc[-1]) for c in pr.columns}
        recs = recommend(sc, cat, load_rules(), fp)
        return {"machine_id": dev, "status": svc.AVAILABLE, "recommendations": recs,
                "note": "No active anomaly." if not recs else LIMITS, "generated_at": svc.now_iso()}
