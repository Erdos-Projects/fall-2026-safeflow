"""Paired, quarter-stratified cluster bootstrap for frozen warning policies."""
from __future__ import annotations
import numpy as np
import pandas as pd

COUNT_COLUMNS = ["days", "onsets", "caught", "false_episodes", "wrong_rows", "rare_onsets", "rare_caught"]


def ratios(counts):
    counts = np.asarray(counts, dtype=float)
    def divide(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b != 0)
    return np.stack([divide(counts[...,2], counts[...,1]),
                     divide(counts[...,3], counts[...,0]),
                     divide(counts[...,4] * 5 / 60, counts[...,0]),
                     divide(counts[...,6], counts[...,5])], axis=-1)


def paired_bootstrap(daily, model_a, model_b, *, block="week", repetitions=5000, seed=42):
    """Return A-B intervals with identical within-quarter cluster draws.

    Weekly blocks keep every station and available day in each calendar week
    together. Weeks crossing quarters are split at the fold boundary. Daily
    resampling provides sensitivity to the temporal dependence assumption.
    Fixed policies are not recalibrated; intervals condition on saved model fits.
    """
    if block not in ("week", "day") or repetitions < 2:
        raise ValueError("Need week/day blocks and at least two replicates")
    if daily.duplicated(["model", "fold", "date"]).any():
        raise ValueError("Need one count row per model/fold/day")
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily.date).dt.normalize()
    keys = ["fold", "date"]
    grids = [daily.loc[daily.model.eq(m)].set_index(keys).sort_index() for m in [model_a, model_b]]
    if grids[0].index.has_duplicates or grids[1].index.has_duplicates:
        raise ValueError("Need one count row per model/fold/day")
    if not grids[0].index.equals(grids[1].index):
        raise ValueError("Models must cover identical evaluation dates")
    parts = []
    for model in [model_a, model_b]:
        d = daily.loc[daily.model.eq(model)].copy()
        d["date"] = pd.to_datetime(d.date).dt.normalize()
        d["block"] = d.date if block == "day" else d.date - pd.to_timedelta(d.date.dt.dayofweek, unit="D")
        parts.append(d.groupby(["fold", "block"])[COUNT_COLUMNS].sum().sort_index())
    if not len(parts[0]) or not parts[0].index.equals(parts[1].index):
        raise ValueError("Models must cover identical fold/time blocks")
    for column in ["days", "onsets", "rare_onsets"]:
        if not np.array_equal(parts[0][column], parts[1][column]):
            raise ValueError("Models must have the same evaluation population")
    a, b = (p.to_numpy(dtype=float) for p in parts)
    rng = np.random.default_rng(seed)
    total_a = np.zeros((repetitions,len(COUNT_COLUMNS)))
    total_b = np.zeros_like(total_a)
    folds = parts[0].index.get_level_values("fold")
    for fold in folds.unique():
        positions = np.flatnonzero(folds == fold)
        draws = rng.choice(positions, size=(repetitions,len(positions)), replace=True)
        total_a += a[draws].sum(axis=1)
        total_b += b[draws].sum(axis=1)
    differences = ratios(total_a) - ratios(total_b)
    point = ratios(a.sum(axis=0)) - ratios(b.sum(axis=0))
    metrics = ["catch_15min", "false_episodes_per_day", "wrong_alert_hours_per_day", "rare_catch_15min"]
    rows = []
    for i,metric in enumerate(metrics):
        low,high = np.nanquantile(differences[:,i], [.025,.975])
        rows.append(dict(model_a=model_a,model_b=model_b,metric=metric,block=block,
            difference=point[i],ci_low=low,ci_high=high,blocks=len(a),repetitions=repetitions,seed=seed))
    return pd.DataFrame(rows), differences
