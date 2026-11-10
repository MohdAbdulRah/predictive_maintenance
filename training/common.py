import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from config import ROOT
from database.models import DataItemRow, ModelRegistry, ObservationRow

ART = ROOT / "artifacts"


def pick_device(session, device: str | None = None) -> str:
    if device:
        return device
    d = session.scalars(select(DataItemRow.device_uuid).distinct()).first()
    if d is None:
        raise SystemExit("No data items in DB. Run scripts.collect first.")
    return d


def record_status(session, stage: str, status: str, artifact: str | None, detail: dict):
    session.add(ModelRegistry(stage=stage, status=status, trained_at=datetime.now(timezone.utc),
                              artifact_path=artifact, detail_json=json.dumps(detail, default=str)))
    session.commit()
