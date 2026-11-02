"""Drought Monitor maps for the live years: the weekly shapefiles since `start` (same files as
flashwatch.ingest.usdm), gridded with labels.crosscheck.usdm_grid, kept in data/live/usdm_grid.npz."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from flashwatch.config import DATA
from flashwatch.ingest.common import get_logger, http_download, retry
from flashwatch.ingest.usdm import OUT, URL

log = get_logger("live.usdm")
CACHE = DATA / "live" / "usdm_grid.npz"


def update(start: date) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(Tuesdays, category grids) from `start` to the latest map published."""
    from flashwatch.labels.crosscheck import usdm_grid
    OUT.mkdir(parents=True, exist_ok=True)
    d = start + timedelta(days=(1 - start.weekday()) % 7)
    dates, cats = [], []
    if CACHE.exists():
        z = np.load(CACHE)
        dates, cats = list(pd.DatetimeIndex(z["dates"]).date), list(z["cat"])
        if dates:
            d = dates[-1] + timedelta(days=7)
    while d <= date.today() - timedelta(days=2):                       # published two days after it is valid
        f = OUT / f"USDM_{d:%Y%m%d}_M.zip"
        if not f.exists():
            retry(lambda: http_download(URL.format(d), f, log), log, f"USDM {d}", tries=3, wait=10)
        c = usdm_grid(d)
        if c is None:
            break
        dates.append(d)
        cats.append(c)
        d += timedelta(days=7)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.DatetimeIndex(dates)
    np.savez_compressed(CACHE, dates=idx.strftime("%Y-%m-%d").to_numpy(dtype="U10"), cat=np.stack(cats))
    return idx, np.stack(cats)
