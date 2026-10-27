"""The single gate between data and features: nothing a feature reads may postdate
as_of(source, issue_date), the last day a forecaster could have had on the issue morning.

Features must read observed data only through `history()`. tests/test_asof.py checks this
by perturbing everything after the cutoff and requiring identical features.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# Days between a field's valid date and its availability (spec, Features table).
LATENCY_DAYS = {"era5land": 5, "smap": 3, "s2s": 0, "usdm": 2, "static": 0, "nino": 7, "gistemp": 15, "mjo": 2}


def as_of(source: str, issue_date) -> pd.Timestamp:
    """Last valid date of `source` usable for a forecast issued on issue_date."""
    if source not in LATENCY_DAYS:
        raise KeyError(f"unknown source {source!r}; add its latency to LATENCY_DAYS")
    return pd.Timestamp(issue_date).normalize() - pd.Timedelta(days=LATENCY_DAYS[source])


def history(x: np.ndarray, time: pd.DatetimeIndex, source: str, issue_date, days: int) -> np.ndarray:
    """The last `days` days of x (time, ...) up to and including as_of(source, issue_date).

    Raises if the window starts before the data does, so short records fail loudly instead
    of silently returning fewer days.
    """
    cutoff = as_of(source, issue_date)
    end = time.get_indexer([cutoff])[0]
    if end < 0:
        raise KeyError(f"{source}: cutoff {cutoff.date()} not in the time axis")
    start = end - days + 1
    if start < 0:
        raise ValueError(f"{source}: need {days} days before {cutoff.date()}, data starts {time[0].date()}")
    return x[start:end + 1]


def monthly_as_of(series: pd.Series, source: str, issue_date) -> float:
    """Latest monthly value (index = month start) published by the issue date: month m counts
    as published LATENCY_DAYS[source] days after it ends."""
    cutoff = as_of(source, issue_date)
    month_end = series.index + pd.offsets.MonthEnd(0)
    ok = month_end <= cutoff
    return float(series[ok].iloc[-1]) if ok.any() else float("nan")
