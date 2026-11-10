from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from api.service import default_device, now_iso
from database.connection import session_scope
from database.models import DataItemRow, ObservationRow

router = APIRouter(prefix="/api/machine", tags=["machine"])


@router.get("/live")
def live(device: str | None = None):
    with session_scope() as s:
        dev = device or default_device(s)
        if dev is None:
            return {"status": "no_data", "machine_id": None, "observations": [], "generated_at": now_iso()}
        sub = (select(ObservationRow.data_item_id, func.max(ObservationRow.ts).label("mx"))
               .where(ObservationRow.device_uuid == dev).group_by(ObservationRow.data_item_id).subquery())
        rows = s.execute(select(ObservationRow, DataItemRow).join(
            sub, (ObservationRow.data_item_id == sub.c.data_item_id) & (ObservationRow.ts == sub.c.mx))
            .join(DataItemRow, (DataItemRow.device_uuid == ObservationRow.device_uuid) & (DataItemRow.id == ObservationRow.data_item_id))
            .where(ObservationRow.device_uuid == dev)).all()
        seen, out = set(), []
        for o, d in sorted(rows, key=lambda r: -r[0].sequence):
            if o.data_item_id in seen: continue
            seen.add(o.data_item_id)
            out.append({"data_item_id": o.data_item_id, "name": d.name, "type": d.type, "category": o.category,
                        "value": o.value_num if o.value_num is not None else o.value_text,
                        "unavailable": o.unavailable, "units": d.units, "timestamp": o.ts.isoformat(),
                        "condition_level": o.condition_level, "component_path": d.component_path})
        return {"status": "ok", "machine_id": dev, "observations": out, "generated_at": now_iso()}


@router.get("/history")
def history(data_item_id: str, device: str | None = None, start: datetime | None = None,
            end: datetime | None = None, limit: int = Query(5000, le=50000)):
    with session_scope() as s:
        dev = device or default_device(s)
        d = s.get(DataItemRow, (dev, data_item_id)) if dev else None
        if d is None:
            raise HTTPException(404, f"unknown data item {data_item_id}")
        q = select(ObservationRow).where(ObservationRow.device_uuid == dev, ObservationRow.data_item_id == data_item_id)
        if start: q = q.where(ObservationRow.ts >= start)
        if end: q = q.where(ObservationRow.ts <= end)
        rows = s.scalars(q.order_by(ObservationRow.ts.desc()).limit(limit)).all()[::-1]
        return {"machine_id": dev, "data_item_id": data_item_id, "units": d.units, "type": d.type,
                "points": [{"timestamp": r.ts.isoformat(), "value": r.value_num if r.value_num is not None else r.value_text,
                            "unavailable": r.unavailable} for r in rows], "generated_at": now_iso()}
