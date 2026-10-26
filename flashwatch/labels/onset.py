"""Flash-drought onset days and lead-week labels (spec section "Labels").

Onset on day d when all hold, with P the root-zone percentile series:
  1. P(d) <= low and P(d-1) > low
  2. P >= high on at least one day in [d-20, d-5]
  3. P <= low for all of [d+1, d+10]
and day d is not masked (snow, frozen soil). Defaults high=40, low=20; the sensitivity
variants are (30, 20) and (40, 10). Labels may use future data; features never do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEADS = (1, 2, 3, 4)


def onset_days(P: np.ndarray, high: float = 40, low: float = 20, lookback=(20, 5), persist: int = 10,
               mask: np.ndarray | None = None) -> np.ndarray:
    """Boolean array like P (time, ...) marking onset days. NaNs never satisfy a condition."""
    T = P.shape[0]
    with np.errstate(invalid="ignore"):
        is_low = P <= low
        above_low = P > low
        is_high = P >= high
    onset = np.zeros(P.shape, dtype=bool)
    onset[1:] = is_low[1:] & above_low[:-1]

    # 2. any P >= high in [d - lookback[0], d - lookback[1]]
    a, b = lookback
    ch = np.concatenate([np.zeros((1,) + P.shape[1:], int), np.cumsum(is_high, axis=0)])
    prior = np.zeros(P.shape, dtype=bool)
    d = np.arange(a, T)
    prior[a:] = (ch[d - b + 1] - ch[d - a]) > 0

    # 3. P <= low for all of [d+1, d+persist]
    cl = np.concatenate([np.zeros((1,) + P.shape[1:], int), np.cumsum(is_low, axis=0)])
    stays = np.zeros(P.shape, dtype=bool)
    d = np.arange(0, T - persist)
    stays[: T - persist] = (cl[d + persist + 1] - cl[d + 1]) == persist

    onset &= prior & stays
    if mask is not None:
        onset &= ~mask
    return onset


def lead_labels(onset: np.ndarray, time: pd.DatetimeIndex, issue_dates) -> dict[str, np.ndarray]:
    """Labels per issue date: week w is 1 if an onset day falls in (tau + 7(w-1), tau + 7w].

    Returns {"week1".."week4", "any28"}, each (n_issue, ...). NaN-free; the caller applies
    eligibility (P(tau - latency) > low) and drops issue dates whose 28 days run past the data.
    """
    t_index = pd.DatetimeIndex(time)
    c = np.concatenate([np.zeros((1,) + onset.shape[1:], int), np.cumsum(onset, axis=0)])
    out = {}
    pos = t_index.get_indexer(pd.DatetimeIndex(issue_dates))
    if (pos < 0).any():
        raise ValueError("issue date outside the time axis")
    if (pos + 28 >= len(t_index)).any():
        raise ValueError("an issue date's 28-day window runs past the end of the data")
    for w in LEADS:
        lo, hi = pos + 7 * (w - 1), pos + 7 * w      # onset index in (lo, hi]
        out[f"week{w}"] = (c[hi + 1] - c[lo + 1]) > 0
    out["any28"] = np.any([out[f"week{w}"] for w in LEADS], axis=0)
    return out


def mask_days(snow_depth: np.ndarray, soil_temp_k: np.ndarray, snow_m: float = 0.01,
              frozen_k: float = 273.15) -> np.ndarray:
    """Cell-days excluded from onset: snow depth above snow_m, or top soil layer at or below 0 C."""
    with np.errstate(invalid="ignore"):
        return (snow_depth > snow_m) | (soil_temp_k <= frozen_k)
