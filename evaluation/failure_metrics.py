import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)


def evaluate_horizon(y_true: pd.Series, p: pd.Series, threshold: float, failure_times, alarm_cooldown_h=6.0) -> dict:
    m = y_true.notna()
    y, pr = y_true[m].astype(int), p[m]
    if y.nunique() < 2:
        return {"error": "need both classes in the evaluation block"}
    pred = (pr >= threshold).astype(int)
    out = {"n": int(len(y)), "prevalence": float(y.mean()),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "pr_auc": float(average_precision_score(y, pr)), "roc_auc": float(roc_auc_score(y, pr)),
           "brier": float(brier_score_loss(y, pr))}
    bins = pd.cut(pr, [0, .1, .2, .4, .6, .8, 1.0], include_lowest=True)
    out["calibration"] = [{"bin": str(k), "mean_pred": float(g.mean()), "obs_rate": float(y[g.index].mean()), "n": int(len(g))}
                          for k, g in pr.groupby(bins, observed=True)]
    # alarm episodes: consecutive alarms within cooldown merge
    al = pr.index[pred.values == 1]
    eps, last = [], None
    for t in al:
        if last is None or (t - last).total_seconds() / 3600 > alarm_cooldown_h:
            eps.append(t)
        last = t
    ft = pd.DatetimeIndex(failure_times)
    horizon_h = None
    out["alarm_episodes"] = len(eps)
    span_days = max((pr.index.max() - pr.index.min()).total_seconds() / 86400, 1e-9)
    out["alarm_episodes_per_day"] = len(eps) / span_days
    return out


def lead_times_and_misses(alarm_times: pd.DatetimeIndex, failure_times, max_lead_h: float) -> dict:
    leads, missed = [], 0
    for f in pd.DatetimeIndex(failure_times):
        prior = alarm_times[(alarm_times < f) & (alarm_times >= f - pd.Timedelta(hours=max_lead_h))]
        if len(prior): leads.append((f - prior.min()).total_seconds() / 3600)
        else: missed += 1
    return {"missed_failures": missed, "n_failures": len(failure_times),
            "mean_lead_time_h": float(np.mean(leads)) if leads else None}
