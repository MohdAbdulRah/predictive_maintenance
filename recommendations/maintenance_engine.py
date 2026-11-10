"""Hybrid recommendation engine: anomaly evidence + (optional) trained-model risk + editable rules.
Outputs inspection suggestions only. It never touches machine control."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from config import ROOT

LIMITS = ("Suggestions come from statistical deviation from this machine's own training data and "
          "editable rules. They are not diagnoses; confirm with inspection.")


def load_rules(path: str | None = None) -> dict:
    with open(Path(path) if path else ROOT / "config" / "maintenance_rules.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _feature_to_item(feature: str) -> str:
    return feature.split("__")[0]


def recommend(anomaly_rows: pd.DataFrame, catalogue: pd.DataFrame, rules: dict,
              failure_probs: dict | None = None, recent_conditions: list[dict] | None = None) -> list[dict]:
    """anomaly_rows: recent output of AnomalyDetector.score (index = timestamp), newest last.
    failure_probs: {'p_1h':..,'p_24h':..} ONLY from a trained model, else None."""
    th = rules["thresholds"]
    recs: list[dict] = []
    if anomaly_rows is None or anomaly_rows.empty:
        return recs
    last = anomaly_rows.iloc[-1]
    scored = anomaly_rows[anomaly_rows.status != "not_scored"]
    run = 0
    for s in reversed(scored.status.tolist()):
        if s == "anomalous": run += 1
        else: break
    cat = catalogue.set_index("id")

    if last.status in ("anomalous", "watch"):
        groups: dict[str, list[dict]] = {}
        for f in last.top_features:
            if abs(f["robust_z"]) < th["feature_robust_z_min"]:
                continue
            item = _feature_to_item(f["feature"])
            typ = cat.loc[item, "type"] if item in cat.index else "UNKNOWN"
            groups.setdefault(typ, []).append({**f, "data_item_id": item,
                                               "units": cat.loc[item, "units"] if item in cat.index else None})
        for typ, ev in groups.items():
            rule = rules["signal_rules"].get(typ, rules["default_rule"])
            recs.append({"timestamp": str(anomaly_rows.index[-1]), "detected_issue": rule["issue"],
                         "signal_type": typ, "supporting_evidence": ev,
                         "suggested_inspection": rule["inspect"], "operating_state": last.operating_state,
                         "limitations": LIMITS})

    conds = [c for c in (recent_conditions or []) if c.get("level") in ("WARNING", "FAULT")]
    for c in conds:
        r = rules["condition_rule"]
        recs.append({"timestamp": str(c.get("timestamp")), "detected_issue": r["issue"], "signal_type": "CONDITION",
                     "supporting_evidence": [c], "suggested_inspection": r["inspect"],
                     "operating_state": last.operating_state, "limitations": LIMITS})

    risk, basis = "low", "anomaly status only"
    if last.status == "watch": risk = "low-moderate"
    if last.status == "anomalous": risk = "moderate"
    if run >= th["persistence_rows_high"]: risk = "high"
    if any(c["level"] == "FAULT" for c in conds): risk = "high"
    if failure_probs:
        p = max(failure_probs.values())
        if p >= th["failure_prob_high"]: risk, basis = "high", "trained failure model + anomaly"
        elif p >= th["failure_prob_medium"] and risk in ("low", "low-moderate"): risk, basis = "moderate", "trained failure model + anomaly"
    urgency = {"low": "monitor", "low-moderate": "monitor; review at next planned stop",
               "moderate": "inspect at next planned stop", "high": "inspect promptly; follow manufacturer procedure"}[risk]
    for r in recs:
        r.update(risk_level=risk, risk_basis=basis, recommended_urgency=urgency,
                 confidence=("uncalibrated: relative rarity vs training data" if not failure_probs else "see failure model calibration"),
                 persistence_rows=run)
    return recs
