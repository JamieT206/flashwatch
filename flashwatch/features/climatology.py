"""Calendar-window climatologies that can be refitted per fold at almost no cost.

For a daily series R (e.g. a trailing 30-day mean) and a set of calendar days (the as-of
cutoffs of the issue dates), YearSums stores, for every calendar day k and year j, the sum,
sum of squares and count of R over days k +/- half_window in year j. A standardised anomaly
for an issue in year y, under reference years Ref, then needs only the totals over
Ref minus {y}: no pass over the raw data per fold, and held-out years never enter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.labels.percentile import calendar_day


def trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of days d-n+1..d along axis 0; NaN if any of them is NaN or before the start."""
    from flashwatch.labels.percentile import running_mean
    return running_mean(x, n).astype(np.float32)


class YearSums:
    def __init__(self, R: np.ndarray, time: pd.DatetimeIndex, cal_days, years, half_window: int = 7):
        """R: (time, ...). cal_days: distinct 0-364 calendar days. years: pool years (all candidates)."""
        self.years = np.asarray(years)
        self.cal_days = np.asarray(cal_days)
        doy = calendar_day(time)
        yr = time.year.to_numpy()
        flat = R.reshape(len(time), -1)
        K, J, C = len(self.cal_days), len(self.years), flat.shape[1]
        self.S = np.zeros((K, J, C), dtype=np.float32)
        self.SS = np.zeros((K, J, C), dtype=np.float32)
        self.N = np.zeros((K, J, C), dtype=np.int8)
        for k, d in enumerate(self.cal_days):
            dist = np.abs(doy - d)
            near = np.minimum(dist, 365 - dist) <= half_window
            for j, y in enumerate(self.years):
                vals = flat[near & (yr == y)].astype(np.float64)
                ok = np.isfinite(vals)
                v0 = np.where(ok, vals, 0.0)
                self.S[k, j], self.SS[k, j], self.N[k, j] = v0.sum(0), (v0 ** 2).sum(0), ok.sum(0)
        self.cell_shape = R.shape[1:]

    def zscore(self, values: np.ndarray, k_idx: np.ndarray, own_year: np.ndarray, ref_years) -> np.ndarray:
        """Standardised anomaly of values (n, ...) at calendar-day index k_idx (n,) for issues in
        own_year (n,), against ref_years excluding each issue's own year."""
        ref = np.isin(self.years, list(ref_years))
        W = ref[None, :] & (self.years[None, :] != np.asarray(own_year)[:, None])   # (n, J)
        S = np.einsum("nj,njc->nc", W, self.S[k_idx], dtype=np.float64)
        SS = np.einsum("nj,njc->nc", W, self.SS[k_idx], dtype=np.float64)
        N = np.einsum("nj,njc->nc", W, self.N[k_idx], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = S / N
            sd = np.sqrt(np.maximum(SS / N - mean ** 2, 0))
            z = (values.reshape(len(values), -1) - mean) / sd
        z[(N < 10) | (sd == 0)] = np.nan
        return z.reshape((len(values),) + self.cell_shape).astype(np.float32)

    def mean(self, ref_years) -> np.ndarray:
        """Climatological mean per calendar day over ref_years: (K, ...)."""
        ref = np.isin(self.years, list(ref_years))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.S[:, ref].sum(1, dtype=np.float64) / self.N[:, ref].sum(1, dtype=np.float64)
        return m.reshape((len(self.cal_days),) + self.cell_shape)
