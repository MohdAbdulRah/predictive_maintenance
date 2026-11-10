"""Stage B: supervised failure prediction from VERIFIED failure events only.

No label is ever derived from sensor thresholds. If verified failures are missing or too
few, `assess_trainability` says so and training is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator


class InsufficientLabels(RuntimeError):
    pass


def build_failure_labels(times: pd.DatetimeIndex, failures: list[pd.Timestamp],
                         observed_until: pd.Timestamp, horizons_h: list[int]) -> pd.DataFrame:
    """y_H(t)=1  if a verified failure occurs in (t, t+H].
       y_H(t)=0  only if (t, t+H] is fully observed (t+H <= observed_until) with no failure.
       y_H(t)=NaN otherwise (censored / unknown).  `times` must belong to ONE operating run."""
    f = np.sort(np.array([pd.Timestamp(x).as_unit("ns").value for x in failures], dtype="int64")) if failures else np.array([], dtype="int64")
    t = times.as_unit("ns").asi8
    end_obs = pd.Timestamp(observed_until).as_unit("ns").value
    out = pd.DataFrame(index=times)
    for h in horizons_h:
        hn = int(h * 3600 * 1e9)
        if len(f):
            nxt_idx = np.searchsorted(f, t, side="right")          # first failure strictly after t
            nxt = np.where(nxt_idx < len(f), f[np.minimum(nxt_idx, len(f) - 1)], np.iinfo("int64").max)
        else:
            nxt = np.full(len(t), np.iinfo("int64").max)
        y = np.full(len(t), np.nan)
        y[nxt <= t + hn] = 1.0
        y[(nxt > t + hn) & (t + hn <= end_obs)] = 0.0
        out[f"y_{h}h"] = y
    out["hours_to_next_failure"] = np.where(nxt < np.iinfo("int64").max, (nxt - t) / 3.6e12, np.nan)
    return out


def assess_trainability(labels: pd.DataFrame, run_ids: pd.Series, cfg: dict) -> dict:
    fc = cfg["failure"]
    rep = {"status": "trainable", "per_horizon": {}, "reasons": []}
    n_fail_events = int((labels["hours_to_next_failure"].notna()).groupby(run_ids).any().sum()) if len(labels) else 0
    for h in fc["horizons_hours"]:
        y = labels[f"y_{h}h"]
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        rep["per_horizon"][h] = {"positives": pos, "negatives": neg, "unknown": int(y.isna().sum())}
        if pos == 0:
            rep["reasons"].append(f"{h}h: no positive examples")
        elif pos < fc["min_verified_failures"]:
            rep["reasons"].append(f"{h}h: only {pos} positive rows (< {fc['min_verified_failures']})")
    if n_fail_events == 0:
        rep["status"] = "not_trained"
        rep["reasons"].insert(0, "no verified failure events in the database")
    elif n_fail_events < fc["min_distinct_runs_with_failure"]:
        rep["status"] = "insufficient_data"
        rep["reasons"].insert(0, f"failures seen in {n_fail_events} run(s); need >= {fc['min_distinct_runs_with_failure']} to hold out whole runs")
    elif rep["reasons"]:
        rep["status"] = "insufficient_data"
    return rep


def enforce_monotone(p: dict[int, float]) -> dict[int, float]:
    """P(fail within 24h) must be >= P(fail within 6h) etc."""
    out, cur = {}, 0.0
    for h in sorted(p):
        cur = max(cur, float(p[h])); out[h] = cur
    return out


@dataclass
class FailurePredictor:
    cfg: dict
    horizons: list[int] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    trained_at: str | None = None
    metrics: dict = field(default_factory=dict)

    def fit(self, X_tr, labels_tr, X_cal, labels_cal, feature_cols):
        """Fit on `tr`, calibrate on a LATER chronological block `cal`. Rows with NaN labels are dropped."""
        self.feature_cols, self.horizons = feature_cols, list(self.cfg["failure"]["horizons_hours"])
        for h in self.horizons:
            ytr, yca = labels_tr[f"y_{h}h"], labels_cal[f"y_{h}h"]
            mtr, mca = ytr.notna(), yca.notna()
            if ytr[mtr].nunique() < 2 or yca[mca].nunique() < 2:
                raise InsufficientLabels(f"{h}h: need both classes in train and calibration blocks")
            base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                  class_weight="balanced", random_state=0)
            base.fit(X_tr.loc[mtr, feature_cols], ytr[mtr].astype(int))
            method = "isotonic" if mca.sum() >= 500 and yca[mca].sum() >= 30 else "sigmoid"
            cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
            cal.fit(X_cal.loc[mca, feature_cols], yca[mca].astype(int))
            self.models[h] = cal
        self.trained_at = datetime.now(timezone.utc).isoformat()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = {h: self.models[h].predict_proba(X[self.feature_cols])[:, 1] for h in self.horizons}
        out = pd.DataFrame(raw, index=X.index)
        arr = np.maximum.accumulate(out[sorted(out.columns)].to_numpy(), axis=1)   # logical consistency
        out = pd.DataFrame(arr, index=X.index, columns=[f"p_{h}h" for h in sorted(raw)])
        return out

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FailurePredictor":
        return joblib.load(path)
