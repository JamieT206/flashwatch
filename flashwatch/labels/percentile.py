"""Day-of-year percentiles of a daily field against the same calendar window in other years.

P(d) for a value on day d of year y is its empirical percentile among the values within
+/- `half_window` calendar days of d in the reference years, excluding year y itself. The
reference years are a fold's training years, so climatologies never see held-out years.
Ties count half (mid-rank). NaN values stay NaN and are left out of the reference pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def running_mean(x: np.ndarray, n: int = 5) -> np.ndarray:
    """Trailing n-day mean along axis 0 (days d-n+1..d); NaN for the first n-1 days or any NaN."""
    c = np.cumsum(np.nan_to_num(x, nan=0.0, copy=True).astype(np.float64), axis=0)
    bad = np.cumsum(np.isnan(x), axis=0)
    out = np.full(x.shape, np.nan)
    out[n - 1:] = c[n - 1:] - np.concatenate([np.zeros((1,) + x.shape[1:]), c[:-n]])
    nbad = bad[n - 1:] - np.concatenate([np.zeros((1,) + x.shape[1:]), bad[:-n]])
    out[n - 1:] /= n
    out[n - 1:][nbad > 0] = np.nan
    return out


def calendar_day(time: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 365-day calendar: Feb 29 shares Feb 28's day so windows line up."""
    doy = time.day_of_year.to_numpy().copy()
    leap_after = time.is_leap_year & ((time.month > 2) | ((time.month == 2) & (time.day == 29)))
    doy[leap_after] -= 1
    return doy - 1                                   # 0..364


def percentiles(x: np.ndarray, time: pd.DatetimeIndex, ref_years, target_years=None,
                half_window: int = 15) -> np.ndarray:
    """Percentile (0-100) of x[t, ...] for days in target_years, against ref_years.

    x: (time, ...) array. Returns an array like x, NaN outside target_years.
    """
    x = np.asarray(x, dtype=np.float32)
    flat = x.reshape(len(time), -1)
    years = time.year.to_numpy()
    doy = calendar_day(time)
    ref_years = np.asarray(sorted(ref_years))
    target = np.isin(years, np.asarray(sorted(target_years if target_years is not None else ref_years)))
    in_ref = np.isin(years, ref_years)
    out = np.full(flat.shape, np.nan, dtype=np.float32)

    for d in np.unique(doy[target]):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        pool_idx = np.where(in_ref & (dist <= half_window))[0]
        pool = flat[pool_idx]                        # (npool, cells)
        pool_year = years[pool_idx]
        tgt_idx = np.where(target & (doy == d))[0]
        for t in tgt_idx:                            # one target day per year at this doy
            keep = pool_year != years[t]
            p = pool[keep]
            v = flat[t]
            valid = np.isfinite(p)
            n = valid.sum(axis=0)
            below = (p < v).sum(axis=0)
            ties = (p == v).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                pct = 100.0 * (below + 0.5 * ties) / n
            pct[(n == 0) | ~np.isfinite(v)] = np.nan
            out[t] = pct
    return out.reshape(x.shape)
