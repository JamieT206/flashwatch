"""US Drought Monitor weekly shapefiles (Tuesday valid dates), 2004-2024."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from flashwatch.config import OBS_YEARS, RAW
from flashwatch.ingest.common import get_logger, http_download, retry

OUT = RAW / "usdm"
URL = "https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_{:%Y%m%d}_M.zip"
log = get_logger("usdm")


def tuesdays():
    d = date(OBS_YEARS[0], 1, 1)
    d += timedelta(days=(1 - d.weekday()) % 7)
    while d.year <= OBS_YEARS[-1]:
        yield d
        d += timedelta(days=7)


def main():
    days = list(tuesdays())
    log.info("%d weekly maps", len(days))

    def one(d):
        return retry(lambda: http_download(URL.format(d), OUT / f"USDM_{d:%Y%m%d}_M.zip", log),
                     log, f"USDM {d}", tries=3, wait=10)

    with ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(one, days))
    log.info("finished: %d ok, %d missing or failed", sum(bool(x) for x in ok), sum(not x for x in ok))
