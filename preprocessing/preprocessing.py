"""Observation log -> regular, state-aware, gap-aware time grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

from database.models import CollectionGap, DataItemRow, ObservationRow


def load_catalogue(session, device_uuid: str) -> pd.DataFrame:
    rows = session.scalars(select(DataItemRow).where(DataItemRow.device_uuid == device_uuid)).all()
    return pd.DataFrame([{c.name: getattr(r, c.name) for c in DataItemRow.__table__.columns} for r in rows])


def load_observations(session, device_uuid: str, start=None, end=None) -> pd.DataFrame:
    q = select(ObservationRow).where(ObservationRow.device_uuid == device_uuid)
    if start is not None:
        q = q.where(ObservationRow.ts >= start)
    if end is not None:
        q = q.where(ObservationRow.ts <= end)
    rows = session.scalars(q.order_by(ObservationRow.ts, ObservationRow.sequence)).all()
    df = pd.DataFrame([{c.name: getattr(r, c.name) for c in ObservationRow.__table__.columns} for r in rows])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _ts_s(ts) -> pd.Timestamp:
    """Normalise any timestamp to datetime64[s, UTC] to satisfy pandas 3.x strict dtype matching."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.as_unit("s")


def load_gaps(session, device_uuid: str, obs: pd.DataFrame) -> pd.DataFrame:
    rows = session.scalars(select(CollectionGap).where(CollectionGap.device_uuid == device_uuid)).all()
    g = pd.DataFrame([{c.name: getattr(r, c.name) for c in CollectionGap.__table__.columns} for r in rows])
    if g.empty:
        return g
    for c in ("gap_start_ts", "gap_end_ts", "detected_at"):
        g[c] = pd.to_datetime(g[c], utc=True).dt.as_unit("s")
    # sequence-only gaps: bound them with the neighbouring observation timestamps
    for i, r in g.iterrows():
        if pd.isna(r.gap_start_ts) and pd.notna(r.from_sequence) and not obs.empty:
            before = obs[(obs.instance_id == r.instance_id) & (obs.sequence < r.from_sequence)]
            after = obs[(obs.instance_id == r.instance_id) & (obs.sequence > r.to_sequence)]
            g.at[i, "gap_start_ts"] = _ts_s(before.ts.max()) if len(before) else _ts_s(r.detected_at)
            g.at[i, "gap_end_ts"] = _ts_s(after.ts.min()) if len(after) else _ts_s(r.detected_at)
        elif pd.isna(r.gap_start_ts):
            g.at[i, "gap_start_ts"] = g.at[i, "gap_end_ts"] = _ts_s(r.detected_at)
    return g


def select_signals(catalogue: pd.DataFrame, obs: pd.DataFrame, cfg: dict) -> list[str]:
    """Numeric, non-constant, VALUE-representation SAMPLEs that actually carry data."""
    cand = catalogue[(catalogue.category == "SAMPLE") & (catalogue.representation == "VALUE")].id.tolist()
    allow, block = cfg.get("signal_allowlist") or [], set(cfg.get("signal_blocklist") or [])
    out = []
    for did in cand:
        if allow and did not in allow:
            continue
        if did in block:
            continue
        v = obs.loc[obs.data_item_id == did, "value_num"].dropna()
        if len(v) >= 10 and v.nunique() > 1:
            out.append(did)
    return out


def derive_state(obs: pd.DataFrame, catalogue: pd.DataFrame, cfg_state: dict) -> pd.Series | None:
    """Event-driven operating-state series (index = ts) or None when no state item exists."""
    ids = catalogue[catalogue.type == cfg_state["source_type"].upper()].id.tolist()
    if not ids:
        return None
    s = obs[obs.data_item_id == ids[0]].sort_values(["ts", "sequence"])
    val = s.value_text.where(~s.unavailable, "UNAVAILABLE").str.upper()
    act, idle = set(cfg_state["active_values"]), set(cfg_state["idle_values"])
    mapped = val.map(lambda v: "active" if v in act else "idle" if v in idle else "unknown")
    mapped.index = s.ts
    return mapped[~mapped.index.duplicated(keep="last")]


def build_grid(obs: pd.DataFrame, catalogue: pd.DataFrame, gaps: pd.DataFrame, signals: list[str],
               cfg: dict) -> pd.DataFrame:
    """Regular grid with: signal columns (float), __state, __segment, __fault_n, __warn_n.

    Values are carried forward only inside gap-free segments (agents publish changes only).
    Grid points inside a recorded gap are dropped, not interpolated.
    """
    if obs.empty:
        return pd.DataFrame()
    step = f"{int(cfg['preprocessing']['grid_seconds'])}s"
    t0, t1 = obs.ts.min().floor(step), obs.ts.max().ceil(step)
    idx = pd.date_range(t0, t1, freq=step, tz="UTC")
    grid = pd.DataFrame(index=idx)

    for did in signals:
        s = obs[(obs.data_item_id == did)].sort_values(["ts", "sequence"])
        v = s.value_num.where(~s.unavailable)
        v.index = s.ts
        v = v[~v.index.duplicated(keep="last")]
        # UNAVAILABLE is a real event: keep it as NaN *at that instant* and carry the NaN forward
        grid[did] = v.sort_index().reindex(idx, method="ffill")  # label-based ffill keeps NaN (UNAVAILABLE)

    st = derive_state(obs, catalogue, cfg["state"])
    if st is not None:
        grid["__state"] = st.sort_index().reindex(idx, method="ffill").fillna("unknown")
    else:
        grid["__state"] = "unknown"

    cond = obs[obs.category == "CONDITION"]
    for lvl, col in (("FAULT", "__fault_n"), ("WARNING", "__warn_n")):
        c = cond[cond.condition_level == lvl]
        grid[col] = c.groupby(c.ts.dt.floor(step)).size().reindex(idx).fillna(0).astype(int) if len(c) else 0

    # segments: break at every recorded gap
    seg = np.zeros(len(idx), dtype=int)
    in_gap = np.zeros(len(idx), dtype=bool)
    if len(gaps):
        for _, g in gaps.iterrows():
            a, b = g.gap_start_ts, g.gap_end_ts
            m = (idx >= a) & (idx <= b)
            in_gap |= m
            seg += (idx > b).astype(int)
    grid["__segment"] = seg
    grid = grid[~in_gap]
    keep = grid.groupby("__segment")["__segment"].transform("size") >= cfg["preprocessing"]["min_segment_rows"]
    return grid[keep]
