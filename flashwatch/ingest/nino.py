"""Monthly Nino 3.4 SST anomaly (NOAA PSL, ERSST v6, 1981-2010 base) to data/raw/nino/."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.config import RAW
from flashwatch.ingest.common import get_logger, http_download, retry

OUT = RAW / "nino"
URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
log = get_logger("nino")


def main():
    target = OUT / "nina34.anom.data"
    target.unlink(missing_ok=True)                    # small and revised monthly: always refresh
    if retry(lambda: http_download(URL, target, log), log, "nino34", tries=3, wait=10):
        log.info("done: latest month %s", load().index[-1].strftime("%Y-%m"))


def load(path=None) -> pd.Series:
    """Monthly series indexed by month start; missing (-99.99) months dropped."""
    lines = (path or OUT / "nina34.anom.data").read_text().splitlines()
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        y = int(parts[0])
        rows += [(pd.Timestamp(y, m + 1, 1), float(v)) for m, v in enumerate(parts[1:])]
    s = pd.Series(dict(rows)).sort_index()
    return s[s > -99]
