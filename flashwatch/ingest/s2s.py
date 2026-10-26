"""ECMWF S2S reforecasts (Cy49r1, model-version year 2025) from the ECMWF Data Store.

One file per (issue date, variable group): all 20 hindcast years (2005-2024), control plus
10 perturbed members, days 1-46, over the region plus a margin. Two groups per date:
  daily  - daily means: 2 m temperature, 2 m dewpoint, soil moisture 0-20 / 0-100 cm,
           soil temperature 0-20 cm
  accum  - accumulations from t=0 at 24 h steps: total precipitation, downward solar radiation
Valid dates/years/variables were checked against the ECDS constraints table on Oct 26.
Requests use the dataset's own form keys; if the probe rejects them, MARS-style keys are
used instead (the syntax that worked in check_access.py).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from flashwatch.config import EAST, NORTH, RAW, SOUTH, WEST, YEARS
from flashwatch.ingest.common import get_logger, retry

OUT = RAW / "s2s"
DATES_FILE = Path(__file__).resolve().parents[1] / "config" / "s2s_dates.json"
AREA = [NORTH + 3, WEST - 3, SOUTH - 3, EAST + 3]   # margin of two 1.5-degree cells for regridding
GROUPS = {
    "daily": {"variables": ["2_m_temperature", "2_m_dewpoint_temperature", "soil_moisture_top_20_cm",
                            "soil_moisture_top_100_cm", "soil_temperature_top_20_cm"],
              "leadtime_hour": [f"{h}_{h + 24}" for h in range(0, 1104, 24)],
              "mars_param": "167/168/228086/228087/228095",
              "mars_step": "/".join(f"{h}-{h + 24}" for h in range(0, 1104, 24))},
    "accum": {"variables": ["total_precipitation", "surface_solar_radiation_downwards"],
              "leadtime_hour": [str(h) for h in range(0, 1105, 24)],
              "mars_param": "228228/169",
              "mars_step": "/".join(str(h) for h in range(0, 1105, 24))},
}
# ECDS caps queued requests per user and dataset ("please request data sequentially").
WORKERS = int(os.getenv("FW_S2S_WORKERS", "1"))
QUEUE_FULL = "maximum number queued requests"
log = get_logger("s2s")


def issue_dates() -> list[date]:
    """The 2025 model-version dates in Apr 1 - Oct 15 that carry every variable, both member
    types and hindcast years 2005-2024 (from the ECDS constraints table, config/s2s_dates.json)."""
    dates = json.loads(DATES_FILE.read_text())["model_version_dates"]
    return [date.fromisoformat(d) for d in dates]


def client():
    import cdsapi
    key = os.getenv("ECDS_API_KEY") or os.getenv("CDSAPI_KEY")
    if not key:
        raise SystemExit("CDSAPI_KEY missing: put it in .env (the CDS key works for ECDS)")
    return cdsapi.Client(url=os.getenv("ECDS_API_URL", "https://ecds.ecmwf.int/api"), key=key,
                         quiet=True, progress=False)


def form_request(d: date, group: str, hyears=None, members=("control_forecast", "perturbed_forecast"),
                 leadtimes=None, variables=None) -> dict:
    g = GROUPS[group]
    return {"origin": "ecmwf", "year": str(d.year), "month": f"{d.month:02d}", "day": f"{d.day:02d}",
            "time": "00:00", "hyear": [str(y) for y in (hyears or YEARS)],
            "hmonth": [f"{d.month:02d}"], "hday": [f"{d.day:02d}"], "level_type": "single_level",
            "variable": variables or g["variables"], "forecast_type": list(members),
            "leadtime_hour": leadtimes or g["leadtime_hour"], "area": AREA, "data_format": "grib"}


def mars_requests(d: date, group: str) -> list[tuple[str, dict]]:
    g = GROUPS[group]
    hdates = "/".join(f"{y}-{d:%m-%d}" for y in YEARS)
    base = {"class": "s2", "expver": "prod", "model": "glob", "origin": "ecmf", "stream": "enfh",
            "levtype": "sfc", "date": d.isoformat(), "hdate": hdates, "time": "00:00:00",
            "param": g["mars_param"], "step": g["mars_step"],
            "area": "/".join(str(x) for x in AREA)}
    return [("cf", {**base, "type": "cf"}), ("pf", {**base, "type": "pf", "number": "1/to/10"})]


def probe(c) -> str:
    """Tiny request to pick the request syntax. Returns 'form' or 'mars'."""
    style_file = OUT / "_style.json"
    if style_file.exists():
        return json.loads(style_file.read_text())["style"]
    d = issue_dates()[0]
    target = OUT / "_probe.grib"
    try:
        c.retrieve("s2s-reforecasts", form_request(d, "accum", hyears=[2005], members=("control_forecast",),
                                                   leadtimes=["0", "24"], variables=["total_precipitation"]),
                   str(target))
        style = "form"
    except Exception as e:  # noqa: BLE001
        log.warning("form-style probe failed (%s: %s); trying MARS style", type(e).__name__, str(e)[:300])
        req = mars_requests(d, "accum")[0][1]
        c.retrieve("s2s-reforecasts", {**req, "hdate": f"2005-{d:%m-%d}", "param": "228228", "step": "0/24"},
                   str(target))
        style = "mars"
    log.info("probe OK with %s-style keys: %d bytes", style, target.stat().st_size)
    style_file.write_text(json.dumps({"style": style}))
    return style


def fetch(d: date, group: str, style: str) -> bool:
    jobs = ([(f"{d.isoformat()}_{group}.grib", form_request(d, group))] if style == "form" else
            [(f"{d.isoformat()}_{group}_{t}.grib", r) for t, r in mars_requests(d, group)])
    ok = True
    for name, req in jobs:
        target = OUT / name
        if target.exists():
            continue
        part = target.with_name(name + ".part")

        def go():
            while True:
                try:
                    client().retrieve("s2s-reforecasts", req, str(part))
                    break
                except Exception as e:  # noqa: BLE001
                    if QUEUE_FULL not in str(e).lower():
                        raise
                    log.info("%s: ECDS queue full, waiting 2 min", name)
                    time.sleep(120)
            os.replace(part, target)
            return True

        if retry(go, log, name, tries=4, wait=60):
            log.info("done %s (%.1f MB)", name, target.stat().st_size / 1e6)
        else:
            ok = False
    return ok


def main(limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    style = probe(client())
    # Coarse-to-fine order (every 4th date first), so a partial run still spans the season.
    dates = issue_dates()
    order = dates[0::4] + dates[2::4] + dates[1::4] + dates[3::4]
    tasks = [(d, g) for d in order for g in GROUPS][:limit]
    log.info("%d requests (%d issue dates x %d groups), %d at a time", len(tasks),
             len(issue_dates()), len(GROUPS), WORKERS)
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda t: fetch(*t, style), tasks))
    log.info("finished: %d ok, %d failed (re-run to retry failures)", sum(results), results.count(False))
