import numpy as np
import pandas as pd


def alert_rates(scored: pd.DataFrame, grid_seconds: int) -> dict:
    """Alerts per operating day. WITHOUT labels these are *alert rates*, not false-alarm rates."""
    active = scored[scored.operating_state == "active"]
    op_days = len(active) * grid_seconds / 86400.0
    s = scored[scored.status != "not_scored"]
    ep = lambda m: int(((m) & ~(m.shift(1, fill_value=False))).sum())  # episodes, not rows
    return {"rows_scored": int(len(s)), "rows_not_scored": int((scored.status == "not_scored").sum()),
            "operating_days_active": round(op_days, 4),
            "anomalous_episodes": ep(s.status == "anomalous"), "watch_episodes": ep(s.status == "watch"),
            "anomalous_episodes_per_active_day": round(ep(s.status == "anomalous") / op_days, 3) if op_days > 0 else None,
            "note": "unconfirmed alerts; false-alarm rate requires human-confirmed labels"}


def precision_with_labels(scored: pd.DataFrame, confirmed: pd.Series) -> dict:
    """confirmed: boolean Series (same index) True where an engineer confirmed a real abnormal condition."""
    pred = scored.status.eq("anomalous")
    tp, fp = int((pred & confirmed).sum()), int((pred & ~confirmed).sum())
    return {"confirmed_anomaly_precision": tp / (tp + fp) if tp + fp else None, "tp": tp, "fp": fp}
