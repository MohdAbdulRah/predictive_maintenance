from fastapi import APIRouter

from api import service as svc
from database.connection import session_scope

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status")
def status():
    with session_scope() as s:
        out = {}
        for stage, art in (("anomaly", "anomaly_detector"), ("failure", "failure_predictor"), ("rul", "rul_estimator")):
            reg = svc.latest_registry(s, stage)
            exists = (svc.ART / f"{art}.joblib").exists()
            out[stage] = {"status": (svc.AVAILABLE.replace("prediction_available", "trained") if exists else (reg["status"] if reg else svc.NOT_TRAINED)),
                          "artifact_present": exists, "registry": reg}
        return {"stages": out, "generated_at": svc.now_iso()}
