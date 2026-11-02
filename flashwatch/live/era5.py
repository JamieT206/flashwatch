"""ERA5-Land for the live feed: complete years as in flashwatch.ingest.era5land, the current year month by month
up to the latest day Earth Engine has (it runs about 5-10 days behind). Re-running fetches only from the month
of the last day already saved. Same files (data/raw/era5land/<variable>_<year>.npz), same grid.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from flashwatch.ingest.common import get_logger, retry
from flashwatch.ingest.era5land import COLLECTION, OUT, VARIABLES, init, pull, year_task

log = get_logger("live.era5")


def latest_day(ee=None) -> pd.Timestamp:
    ee = ee or init()
    now = pd.Timestamp.now()
    c = ee.ImageCollection(COLLECTION).filterDate((now - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
                                                 (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    return pd.Timestamp(ee.Date(c.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo())


def current_year(var: str, year: int, last: pd.Timestamp) -> bool:
    """Months of `year` up to `last`, appended to what is saved."""
    target = OUT / f"{var}_{year}.npz"
    dates, data = [], None
    if target.exists():
        z = np.load(target)
        dates, data = list(z["dates"]), z["data"]
    start = pd.Timestamp(f"{year}-01-01")
    if dates:
        saved = pd.Timestamp(dates[-1])
        if saved >= last:
            return True
        start = saved.replace(day=1)          # re-pull the last month: it may have been partial
        keep = [i for i, d in enumerate(dates) if pd.Timestamp(d) < start]
        dates, data = [dates[i] for i in keep], data[keep]
    ee = init()
    parts = []
    m = start
    while m <= last:
        end = min(m + pd.offsets.MonthBegin(1), last + pd.Timedelta(days=1))
        part = retry(lambda: pull(ee, var, m.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")), log,
                     f"{var} {m:%Y-%m}", tries=4, wait=20)
        if part is None:
            return False
        parts.append(part)
        m = m + pd.offsets.MonthBegin(1)
    new_dates = sum((p[0] for p in parts), [])
    new = np.concatenate([p[1] for p in parts])
    dates = dates + new_dates
    data = new if data is None or not len(data) else np.concatenate([data, new])
    tmp = target.with_name(target.stem + ".part.npz")
    np.savez_compressed(tmp, data=data, dates=np.array(dates, dtype="U8"))
    os.replace(tmp, target)
    log.info("%s %d: through %s (%d days)", var, year, dates[-1], len(dates))
    return True


def update(years) -> pd.Timestamp:
    """Every variable for `years`; the current year up to the latest day available. Returns that day."""
    OUT.mkdir(parents=True, exist_ok=True)
    last = latest_day()
    tasks = [(v, y) for y in years for v in VARIABLES]

    def one(t):
        v, y = t
        return year_task(v, y) if y < last.year else current_year(v, y, last)
    with ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(one, tasks))
    if not all(ok):
        raise RuntimeError(f"ERA5-Land: {ok.count(False)} variable-years failed; re-run to retry")
    return last
