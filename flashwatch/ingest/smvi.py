"""SMVI flash-drought inventory (HydroShare): the event tables (onset dates per GLDAS cell, 1990-2021).

The ~3 GB "Composites" archives are atmospheric anomaly composites, not events; not needed.
"""

from __future__ import annotations

from flashwatch.config import RAW
from flashwatch.ingest.common import get_logger, http_download, retry

OUT = RAW / "smvi"
BASE = "https://www.hydroshare.org/resource/b6d1d853b82247e1aeaf907ac254c99e/data/contents/"
FILES = ["README.txt", "LONLAT.csv", "FD_Events/CSV/SMVI_GLDAS_Events.tar.gz",
         "FD_Events/NetCDF/FD_Events_NC.tar.gz"]
log = get_logger("smvi")


def main():
    for f in FILES:
        target = OUT / f.rsplit("/", 1)[-1]
        if retry(lambda: http_download(BASE + f, target, log), log, f, tries=4, wait=30):
            log.info("done %s (%.0f MB)", target.name, target.stat().st_size / 1e6)
    log.info("finished")
