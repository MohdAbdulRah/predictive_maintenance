"""State-aware rolling-window features on the regular grid.

* Windows never cross a segment (gap) boundary.
* Each row's signal statistics are computed ONLY over rows in the SAME operating state,
  so idle behaviour is never blended into active behaviour.
* Windows are strictly backward-looking (no future data), so features at time t use <= t only.
* A window with too few valid points yields NaN, never an imputed value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ("active", "idle")
ALL_STATES = ("active", "idle", "unknown")


def _rolling_stats(x: pd.Series, n: int, min_fill: float, t_sec: np.ndarray, pfx: str, min_state_points: int = 12) -> pd.DataFrame:
    # x is already masked to ONE operating state, so a long window legitimately holds few in-state points;
    # require min(fraction of window, min_state_points) valid in-state points.
    need = max(2, min(int(np.ceil(n * min_fill)), min_state_points))
    r = x.rolling(n, min_periods=need)
    out = pd.DataFrame({f"{pfx}_mean": r.mean(), f"{pfx}_std": r.std(ddof=0),
                        f"{pfx}_min": r.min(), f"{pfx}_max": r.max()}, index=x.index)
    # least-squares slope per second via rolling sums (NaN-aware)
    m = x.notna().astype(float)
    xv = x.fillna(0.0)
    t = pd.Series(t_sec, index=x.index)
    tm = t * m
    sn = m.rolling(n, min_periods=1).sum()
    sx, st = xv.rolling(n, min_periods=1).sum(), tm.rolling(n, min_periods=1).sum()
    sxt = (xv * t).rolling(n, min_periods=1).sum()
    stt = (tm * t).rolling(n, min_periods=1).sum()
    den = sn * stt - st ** 2
    slope = (sn * sxt - sx * st) / den.where(den > 1e-9)
    out[f"{pfx}_slope_per_s"] = slope.where(sn >= need)
    return out


def build_features(grid: pd.DataFrame, signals: list[str], cfg: dict) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    step = int(cfg["preprocessing"]["grid_seconds"])
    wins = [max(2, w // step) for w in cfg["features"]["windows_seconds"]]
    labels = [f"{w}s" for w in cfg["features"]["windows_seconds"]]
    min_fill = cfg["features"]["min_window_fill"]
    msp = cfg["features"].get("min_state_points", 12)
    parts = []
    for _, g in grid.groupby("__segment", sort=True):
        t_sec = (g.index - g.index[0]).total_seconds().to_numpy(dtype=float)
        state = g["__state"]
        # Collect all columns into a dict and build the DataFrame once (avoids fragmentation)
        col_dict: dict[str, np.ndarray] = {}
        idx = g.index

        act_mask = (state == "active").to_numpy()[:, None]
        idl_mask = (state == "idle").to_numpy()[:, None]
        unk_mask = (state == "unknown").to_numpy()[:, None]

        for n, lab in zip(wins, labels):
            # state-masked rolling stats, then each row picks the stats of its own state
            for sig in signals:
                per_state = {}
                for s in ALL_STATES:
                    xs = g[sig].where(state == s)
                    per_state[s] = _rolling_stats(xs, n, min_fill, t_sec, f"{sig}__{lab}", msp)
                A, I, U = per_state["active"], per_state["idle"], per_state["unknown"]
                merged = np.where(act_mask, A.values,
                                  np.where(idl_mask, I.values,
                                           np.where(unk_mask, U.values, np.nan)))
                for ci, col in enumerate(A.columns):
                    col_dict[col] = merged[:, ci]

            is_act = (state == "active").astype(float).to_numpy()
            col_dict[f"frac_active__{lab}"] = pd.Series(is_act, index=idx).rolling(n, min_periods=2).mean().to_numpy()
            trans = (state != state.shift(1)).astype(float).to_numpy()
            col_dict[f"state_changes__{lab}"] = pd.Series(trans, index=idx).rolling(n, min_periods=2).sum().to_numpy()
            stops = ((state.shift(1) == "active") & (state == "idle")).astype(float).to_numpy()
            col_dict[f"active_to_idle__{lab}"] = pd.Series(stops, index=idx).rolling(n, min_periods=2).sum().to_numpy()
            col_dict[f"fault_n__{lab}"] = g["__fault_n"].rolling(n, min_periods=1).sum().to_numpy()
            col_dict[f"warn_n__{lab}"] = g["__warn_n"].rolling(n, min_periods=1).sum().to_numpy()

        col_dict["op_hours_in_segment"] = ((state == "active").astype(float).cumsum() * step / 3600.0).to_numpy()
        col_dict["__state"] = state.to_numpy()
        col_dict["__segment"] = g["__segment"].to_numpy()
        parts.append(pd.DataFrame(col_dict, index=idx))
    return pd.concat(parts)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if not c.startswith("__") and c != "op_hours_in_segment"]
