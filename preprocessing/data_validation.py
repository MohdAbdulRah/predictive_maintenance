"""Data-quality report. Run before any modelling; it never edits or invents data."""
from __future__ import annotations

import pandas as pd


def validate_observations(obs: pd.DataFrame, catalogue: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    rep: dict = {"n_observations": int(len(obs)), "issues": []}
    if obs.empty:
        rep["issues"].append("no observations")
        return rep
    rep["time_span"] = [obs.ts.min().isoformat(), obs.ts.max().isoformat()]
    rep["duplicates_on_key"] = int(obs.duplicated(["instance_id", "sequence"]).sum())
    rep["instances"] = sorted(obs.instance_id.unique().tolist())
    rep["n_gaps"] = int(len(gaps))
    rep["gap_kinds"] = gaps.kind.value_counts().to_dict() if len(gaps) else {}
    if rep["duplicates_on_key"]:
        rep["issues"].append("duplicate (instance, sequence) rows present")
    if len(rep["instances"]) > 1:
        rep["issues"].append("multiple agent instanceIds: sequences are NOT comparable across them")

    per = []
    cat = catalogue.set_index("id")
    for did, g in obs.groupby("data_item_id"):
        c = cat.loc[did] if did in cat.index else None
        row = {"id": did, "category": g.category.iloc[0], "n": int(len(g)),
               "pct_unavailable": round(float(g.unavailable.mean()) * 100, 2)}
        if c is not None:
            row.update(type=c["type"], units=c["units"])
        if g.category.iloc[0] == "SAMPLE":
            v = g.value_num.dropna()
            row["pct_non_numeric"] = round(float(g.value_num.isna().mean() - g.unavailable.mean()) * 100, 2)
            if len(v):
                row.update(min=float(v.min()), p50=float(v.median()), max=float(v.max()),
                           std=float(v.std(ddof=0)), n_unique=int(v.nunique()))
                if v.nunique() <= 1:
                    row["flag"] = "constant"
        per.append(row)
    rep["data_items"] = per
    dead = [r["id"] for r in per if r["pct_unavailable"] >= 99]
    if dead:
        rep["issues"].append(f"always UNAVAILABLE: {dead}")
    const = [r["id"] for r in per if r.get("flag") == "constant"]
    if const:
        rep["issues"].append(f"constant signals (uninformative, excluded from modelling): {const}")
    return rep
