"""Stage C: Remaining Useful Life via survival analysis on COMPONENT-LIFE records.

Requires real records: component install time, end time, and whether the end was a true
degradation endpoint (event_observed=True) or a censoring (still in service, preventive
swap, machine retired).  Without >= min_component_lives such rows, training is refused.

RUL and P(fail within H) come from the same conditional survival curve:
    P(fail in next H h | survived to age a) = 1 - S(a+H) / S(a)
Age is measured in OPERATING hours (spindle/EXECUTION active time), not wall-clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


class InsufficientLives(RuntimeError):
    pass


def operating_hours_between(state: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                            observed_until: pd.Timestamp) -> tuple[float, float]:
    """Active hours inside [start, end] from an event-driven operating-state series (index=ts).
    Returns (active_hours, coverage) where coverage = fraction of the interval the state series
    actually covers. Hours outside monitored time are UNKNOWN, not zero: low coverage => do not use."""
    end = min(end, observed_until)
    if end <= start or state is None or state.empty:
        return 0.0, 0.0
    s = state.sort_index()
    ns = lambda x: pd.Timestamp(x).as_unit("ns").value
    s.index = s.index.as_unit("ns")
    idx = s.index.append(pd.DatetimeIndex([pd.Timestamp(end).as_unit("ns")]))
    lo, hi = ns(start), ns(end)
    a = np.maximum(s.index.asi8, lo)
    b = np.minimum(idx.asi8[1:], hi)
    overlap_h = np.clip(b - a, 0, None) / 3.6e12
    active = float(overlap_h[(s.values == "active")].sum())
    first = max(s.index[0].value, lo)
    coverage = max(0.0, (hi - first) / max(hi - lo, 1))
    return active, float(coverage)


def assess_lives(lives: pd.DataFrame, cfg: dict) -> dict:
    r = cfg["rul"]
    n, ev = len(lives), int(lives["event"].sum()) if len(lives) else 0
    reasons = []
    if n == 0:
        status = "not_trained"; reasons.append("no component-life records")
    else:
        status = "trainable"
        if n < r["min_component_lives"]:
            status = "insufficient_data"; reasons.append(f"{n} lives < {r['min_component_lives']}")
        if ev < r["min_observed_failures"]:
            status = "insufficient_data"; reasons.append(f"{ev} observed endpoints < {r['min_observed_failures']}")
    return {"status": status, "n_lives": n, "n_events": ev, "n_censored": n - ev, "reasons": reasons}


@dataclass
class RULEstimator:
    cfg: dict
    backend: str = "rsf"          # 'rsf' (Random Survival Forest) | 'gbs' (gradient boosted) | 'cox'
    model: object = None
    feature_cols: list[str] = field(default_factory=list)
    trained_at: str | None = None

    def _make(self):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        return {"rsf": lambda: RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1),
                "gbs": lambda: GradientBoostingSurvivalAnalysis(random_state=0),
                "cox": lambda: CoxPHSurvivalAnalysis(alpha=1.0)}[self.backend]()

    def fit(self, X: pd.DataFrame, duration_h: np.ndarray, event: np.ndarray, feature_cols: list[str]):
        """One row per component life: features at a reference age, duration in operating hours,
        event=1 for a true degradation endpoint, 0 if right-censored."""
        from sksurv.util import Surv
        if len(X) < self.cfg["rul"]["min_component_lives"] or int(np.sum(event)) < self.cfg["rul"]["min_observed_failures"]:
            raise InsufficientLives("not enough component lives / observed endpoints")
        self.feature_cols = feature_cols
        self.model = self._make().fit(X[feature_cols].fillna(X[feature_cols].median()),
                                      Surv.from_arrays(np.asarray(event, bool), np.asarray(duration_h, float)))
        self.trained_at = datetime.now(timezone.utc).isoformat()
        return self

    def predict(self, X: pd.DataFrame, age_h: float, horizons_h=(1, 6, 24, 72)) -> list[dict]:
        fns = self.model.predict_survival_function(X[self.feature_cols].astype(float))
        res = []
        for fn in fns:
            s_a = float(fn(min(max(age_h, fn.x[0]), fn.x[-1])))
            if s_a <= 1e-6:
                res.append({"rul_median_h": 0.0, "note": "model says component is past its learned life"}); continue
            grid = fn.x[fn.x >= age_h]
            cond = np.array([float(fn(g)) for g in grid]) / s_a
            def q(p):   # hours until conditional survival drops to p
                hit = np.where(cond <= p)[0]
                return float(grid[hit[0]] - age_h) if len(hit) else None   # None = beyond observed horizon
            pf = {h: float(1 - fn(min(age_h + h, fn.x[-1])) / s_a) for h in horizons_h}
            res.append({"rul_p10_h": q(0.9), "rul_median_h": q(0.5), "rul_p90_h": q(0.1),
                        "p_fail_within_h": pf,
                        "note": "quantiles describe the model's survival distribution; None = beyond training horizon"})
        return res

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RULEstimator":
        return joblib.load(path)
