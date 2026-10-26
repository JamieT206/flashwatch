"""Probabilistic scores for onset forecasts, and the year-block bootstrap for their CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def bss(y, p, p_ref) -> float:
    """Brier skill score of p against reference forecast p_ref (e.g. B0 climatology)."""
    return 1.0 - brier(y, p) / brier(y, p_ref)


def reliability(y, p, eps=1e-6) -> tuple[float, float]:
    """(intercept, slope) of logit(P(y=1)) = a + b * logit(p). Perfect calibration is (0, 1)."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    x = np.log(p / (1 - p))[:, None]
    m = LogisticRegression(C=1e6, max_iter=1000).fit(x, np.asarray(y, int))
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def pr_auc(y, p) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y, int), p))


def hit_far(y, p, threshold: float) -> tuple[float, float]:
    """Hit rate (POD) and false alarm ratio when warning at p >= threshold."""
    y, warn = np.asarray(y, bool), np.asarray(p) >= threshold
    hits = (warn & y).sum()
    pod = hits / max(y.sum(), 1)
    far = (warn & ~y).sum() / max(warn.sum(), 1)
    return float(pod), float(far)


def score(y, p, p_ref=None) -> dict:
    out = {"n": int(len(y)), "base_rate": float(np.mean(y)), "brier": brier(y, p), "pr_auc": pr_auc(y, p)}
    out["rel_intercept"], out["rel_slope"] = reliability(y, p)
    if p_ref is not None:
        out["bss"] = bss(y, p, p_ref)
    return out


def year_bootstrap(df: pd.DataFrame, stat, n: int = 1000, seed: int = 0, year_col: str = "year"):
    """Percentile CI of stat(df_resample) resampling whole years with replacement.

    Returns (point estimate, 2.5%, 97.5%). Whole years keep spatial and within-season
    dependence intact, which a row bootstrap would ignore.
    """
    rng = np.random.default_rng(seed)
    groups = {y: g for y, g in df.groupby(year_col)}
    years = np.array(list(groups))
    draws = []
    for _ in range(n):
        pick = rng.choice(years, size=len(years), replace=True)
        draws.append(stat(pd.concat([groups[y] for y in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(stat(df)), float(lo), float(hi)


def year_bootstrap_bss(years, y, p, p_ref, n: int = 1000, seed: int = 0):
    """Year-block bootstrap CI of BSS(p vs p_ref) from per-year squared-error sums.

    Same resampling as year_bootstrap, but O(years) per draw instead of copying every row,
    which matters at ~7M rows. Returns (point estimate, 2.5%, 97.5%).
    """
    df = pd.DataFrame({"yr": np.asarray(years), "e": (np.asarray(p, float) - y) ** 2,
                       "r": (np.asarray(p_ref, float) - y) ** 2})
    g = df.groupby("yr")[["e", "r"]].sum()
    e, r = g.e.to_numpy(), g.r.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    draws = 1 - e[idx].sum(1) / r[idx].sum(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(1 - e.sum() / r.sum()), float(lo), float(hi)
