import numpy as np


def rul_errors(true_h, pred_h, lo_h=None, hi_h=None) -> dict:
    """Only for components whose end was an OBSERVED degradation endpoint (censored lives have no true RUL)."""
    t, p = np.asarray(true_h, float), np.asarray(pred_h, float)
    e = np.abs(t - p)
    out = {"n": int(len(t)), "mae_h": float(np.mean(e)), "median_ae_h": float(np.median(e))}
    if lo_h is not None and hi_h is not None:
        lo, hi = np.asarray(lo_h, float), np.asarray(hi_h, float)
        out["interval_coverage"] = float(np.mean((t >= lo) & (t <= hi)))
    return out
