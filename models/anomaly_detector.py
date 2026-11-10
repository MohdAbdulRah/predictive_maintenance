"""Stage A: unsupervised anomaly detection.  An anomaly is NOT a failure.

Per operating state (active / idle) we fit
  1. IsolationForest on robust-scaled window features, and
  2. a robust statistical baseline (median / MAD z-scores) of the same features.
Scores are mapped through the training-score distribution, so the alert threshold is
"rarer than the Xth percentile of the normal-training data" - a statement about rarity,
not about damage. Training data is ASSUMED to be predominantly normal; there are no labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

SCORED_STATES = ("active", "idle", "unknown")
# Discrete count/state features suit tree models but not MAD z-scores (near-constant in training => huge z).
# They stay in the Isolation Forest; the statistical baseline and evidence list use signal features only.
STAT_EXCLUDE = ("frac_active__", "state_changes__", "active_to_idle__", "fault_n__", "warn_n__")
DISCLAIMER = ("Anomaly = behaviour rare relative to this machine's own training data. "
              "It is not a confirmed fault and not a failure probability.")


@dataclass
class _StateModel:
    cols: list[str]
    medians: pd.Series
    scaler: RobustScaler
    iforest: IsolationForest
    mad: pd.Series
    center: pd.Series
    train_scores_sorted: np.ndarray
    threshold_raw: float
    n_train: int


@dataclass
class AnomalyDetector:
    cfg: dict
    models: dict = field(default_factory=dict)
    skipped_states: dict = field(default_factory=dict)
    trained_at: str | None = None
    train_span: list | None = None

    def fit(self, feats: pd.DataFrame, cols: list[str]) -> "AnomalyDetector":
        a = self.cfg["anomaly"]
        for state in SCORED_STATES:
            sub = feats[feats["__state"] == state]
            usable = [c for c in cols if sub[c].notna().mean() >= 0.8 and sub[c].nunique() > 1]
            X = sub[usable].dropna(thresh=int(0.8 * len(usable))) if usable else sub.iloc[0:0]
            if len(X) < a["min_training_rows_per_state"] or len(usable) < 2:
                self.skipped_states[state] = f"insufficient data: {len(X)} usable rows, {len(usable)} usable features"
                continue
            med = X.median()
            Xf = X.fillna(med)
            sc = RobustScaler().fit(Xf)
            Z = sc.transform(Xf)
            iso = IsolationForest(n_estimators=a["n_estimators"], contamination=a["contamination"],
                                  random_state=0, n_jobs=-1).fit(Z)
            raw = -iso.score_samples(Z)                      # higher = more anomalous
            center = Xf.median()
            # robust scale with a floor: avoids infinite z-scores when a feature is almost constant in training
            mad = np.maximum(1.4826 * (Xf - center).abs().median(), np.maximum(0.1 * Xf.std(ddof=0), 1e-9))
            self.models[state] = _StateModel(
                usable, med, sc, iso, mad, center, np.sort(raw),
                float(np.percentile(raw, a["alert_percentile"])), len(X))
        if not self.models:
            raise RuntimeError("No state had enough data: " + str(self.skipped_states))
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.train_span = [feats.index.min().isoformat(), feats.index.max().isoformat()]
        return self

    def score(self, feats: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        a = self.cfg["anomaly"]
        out = []
        if feats.empty:
            return pd.DataFrame(columns=["operating_state", "anomaly_score", "status", "reason", "top_features", "disclaimer"])
        for ts, row in feats.iterrows():
            state = row["__state"]
            rec = {"timestamp": ts, "operating_state": state, "anomaly_score": np.nan,
                   "status": "not_scored", "reason": None, "top_features": [], "disclaimer": DISCLAIMER}
            m = self.models.get(state)
            if m is None:
                rec["reason"] = ("operating state not modelled: " + self.skipped_states.get(state, f"state '{state}'"))
                out.append(rec); continue
            x = row[m.cols].astype(float)
            if x.isna().mean() > 0.2:
                rec["reason"] = "too many missing features in window (warm-up or data gap)"
                out.append(rec); continue
            x = x.fillna(m.medians)
            raw = float(-m.iforest.score_samples(m.scaler.transform(x.to_frame().T))[0])
            pct = float(np.searchsorted(m.train_scores_sorted, raw) / len(m.train_scores_sorted))
            stat_cols = [c for c in m.cols if not c.startswith(STAT_EXCLUDE)]
            z = ((x[stat_cols] - m.center[stat_cols]) / m.mad[stat_cols]).replace([np.inf, -np.inf], np.nan)
            zmax = float(z.abs().max()) if z.notna().any() else 0.0
            top = z.abs().sort_values(ascending=False).head(top_k)
            rec["anomaly_score"] = round(pct, 4)
            rec["top_features"] = [{"feature": k, "robust_z": round(float(z[k]), 2),
                                    "value": round(float(x[k]), 4),
                                    "training_median": round(float(m.center[k]), 4)} for k in top.index]
            n_far = int((z.abs() >= a["robust_z_alert"]).sum())
            iso_flag, stat_flag = raw >= m.threshold_raw, n_far >= a.get("min_features_flagged", 2)
            near = zmax >= 0.6 * a["robust_z_alert"] or pct >= 0.99
            # either detector alone is a rarity flag; agreement raises confidence
            rec["status"] = "anomalous" if (iso_flag or stat_flag) else ("watch" if near else "normal")
            rec["detector_agreement"] = "both" if (iso_flag and stat_flag) else ("one" if (iso_flag or stat_flag) else "none")
            rec["reason"] = f"isolation_forest={'flag' if iso_flag else 'ok'}, statistical={'flag' if stat_flag else 'ok'} (max |z|={zmax:.1f}, {n_far} features beyond threshold)"
            out.append(rec)
        return pd.DataFrame(out).set_index("timestamp")

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "AnomalyDetector":
        return joblib.load(path)
