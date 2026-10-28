"""ECMWF S2S reforecasts (Cy49r1, model-version year 2025) from the ECMWF Data Store.

One file per (issue date, variable group): all 20 hindcast years (2005-2024), control plus
10 perturbed members, days 1-28, over the region plus a margin. Two groups per date:
  daily  - daily means: 2 m temperature, 2 m dewpoint, soil moisture 0-100 cm
  accum  - accumulations from t=0 at 24 h steps: total precipitation
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
# Only what forecast features use: days 1-28 of 2 m temperature and dewpoint, 0-100 cm soil
# moisture and precipitation. Files from before Oct 28 also hold days 29-46, 0-20 cm soil
# moisture and temperature, and solar radiation; forecast.build ignores those. Trimming cut
# each date to about a third of the ECDS request cost, so more dates fit in a packed request.
GROUPS = {
    "daily": {"variables": ["2_m_temperature", "2_m_dewpoint_temperature", "soil_moisture_top_100_cm"],
              "leadtime_hour": [f"{h}_{h + 24}" for h in range(0, 672, 24)],
              "mars_param": "167/168/228087",
              "mars_step": "/".join(f"{h}-{h + 24}" for h in range(0, 672, 24))},
    "accum": {"variables": ["total_precipitation"],
              "leadtime_hour": [str(h) for h in range(0, 673, 24)],
              "mars_param": "228228",
              "mars_step": "/".join(str(h) for h in range(0, 673, 24))},
}
# ECDS caps queued requests per user and dataset ("please request data sequentially").
WORKERS = int(os.getenv("FW_S2S_WORKERS", "1"))
# Each request waits 1.5-3 h in the ECDS queue but runs in 3-8 min (job timestamps, Oct 27), so
# FW_S2S_PACK > 1 asks for that many same-month model-version dates in one request and splits
# the reply into the usual per-date files (form-style keys only).
PACK = int(os.getenv("FW_S2S_PACK", "1"))
pack_size = {g: PACK for g in GROUPS}      # shrinks per group on "request too large"
QUEUE_FULL = "maximum number queued requests"
TOO_LARGE = "cost limits exceeded"
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


def form_request(d: date | list[date], group: str, hyears=None,
                 members=("control_forecast", "perturbed_forecast"), leadtimes=None, variables=None) -> dict:
    """One model-version date, or several in the same month (a packed request)."""
    g = GROUPS[group]
    ds = d if isinstance(d, list) else [d]
    assert len({(x.year, x.month) for x in ds}) == 1, "packed dates must share a month"
    days = [f"{x.day:02d}" for x in ds]
    return {"origin": "ecmwf", "year": str(ds[0].year), "month": f"{ds[0].month:02d}", "day": days,
            "time": "00:00", "hyear": [str(y) for y in (hyears or YEARS)],
            "hmonth": [f"{ds[0].month:02d}"], "hday": days, "level_type": "single_level",
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


def expected_messages(group: str) -> int:
    g = GROUPS[group]
    return len(YEARS) * 11 * len(g["leadtime_hour"]) * len(g["variables"])


def split_pack(pack: Path, group: str) -> list[date]:
    """Write each model-version date's messages (GRIB key `date`) from a packed reply to
    <date>_<group>.grib. Only dates with the full message count are written."""
    import eccodes

    by_date: dict[int, list[bytes]] = {}
    with open(pack, "rb") as f:
        while (h := eccodes.codes_grib_new_from_file(f)) is not None:
            by_date.setdefault(eccodes.codes_get(h, "date"), []).append(eccodes.codes_get_message(h))
            eccodes.codes_release(h)
    written = []
    for k, msgs in sorted(by_date.items()):
        d = date(k // 10000, k // 100 % 100, k % 100)
        target = OUT / f"{d.isoformat()}_{group}.grib"
        if len(msgs) != expected_messages(group):
            log.warning("%s: %d messages, expected %d; not written", target.name, len(msgs),
                        expected_messages(group))
            continue
        part = target.with_name(target.name + ".part")
        part.write_bytes(b"".join(msgs))
        os.replace(part, target)
        written.append(d)
    return written


def fetch_pack(ds: list[date], group: str) -> bool:
    todo = [d for d in ds if not (OUT / f"{d.isoformat()}_{group}.grib").exists()]
    if not todo:
        return True
    name = f"_pack_{todo[0].isoformat()}_{todo[-1]:%d}_{group}.grib"
    part = OUT / (name + ".part")

    def go():
        while True:
            try:
                client().retrieve("s2s-reforecasts", form_request(todo, group), str(part))
                return True
            except Exception as e:  # noqa: BLE001
                if TOO_LARGE in str(e).lower():
                    return "too large"
                if QUEUE_FULL not in str(e).lower():
                    raise
                log.info("%s: ECDS queue full, waiting 2 min", name)
                time.sleep(120)

    res = retry(go, log, name, tries=1)
    if res == "too large" and len(todo) > 1:
        # Rejected at submission (no queue wait): halve this pack and the group's pack size.
        pack_size[group] = min(pack_size[group], len(todo) // 2)
        log.info("%s over the ECDS cost limit; %s pack size now %d", name, group, pack_size[group])
        h = (len(todo) + 1) // 2
        return all([fetch_pack(todo[:h], group), fetch_pack(todo[h:], group)])
    if res is not True:
        # Each try costs a queue wait, so don't repeat a failing pack: fetch its dates one by one.
        # Later packs still go packed (a one-off failure, e.g. ECDS dismissing a queued job on
        # Oct 28, used to switch the whole run to single dates, about 3 times slower).
        log.warning("%s failed; fetching its dates one by one", name)
        return all([fetch(d, group, "form") for d in todo])
    written = split_pack(part, group)
    part.unlink()
    log.info("done %s: %d of %d dates (%s)", name, len(written), len(todo),
             " ".join(f"{d:%m-%d}" for d in written))
    return len(written) == len(todo)


def main(limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    style = probe(client())
    if PACK > 1 and style == "form":
        return main_packed()
    # Coarse-to-fine order (every 4th date first), so a partial run still spans the season.
    dates = issue_dates()
    order = dates[0::4] + dates[2::4] + dates[1::4] + dates[3::4]
    tasks = [(d, g) for d in order for g in GROUPS][:limit]
    log.info("%d requests (%d issue dates x %d groups), %d at a time", len(tasks),
             len(issue_dates()), len(GROUPS), WORKERS)
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda t: fetch(*t, style), tasks))
    log.info("finished: %d ok, %d failed (re-run to retry failures)", sum(results), results.count(False))


def main_packed():
    """Remaining dates per group, in packs of PACK same-month dates, months in season order."""
    # FW_S2S_MONTHS="8,9,10": only these months, so a second machine (with its own ECDS account)
    # can take part of the season; copy its data/raw/s2s/<date>_*.grib files here afterwards.
    only = {int(m) for m in os.getenv("FW_S2S_MONTHS", "").split(",") if m.strip()}
    tasks = []
    for g in GROUPS:
        left = [d for d in issue_dates() if not (OUT / f"{d.isoformat()}_{g}.grib").exists()
                and (not only or d.month in only)]
        for m in sorted({d.month for d in left}):
            ms = [d for d in left if d.month == m]
            tasks += [(ms[i:i + PACK], g) for i in range(0, len(ms), PACK)]
    # FW_S2S_FIRST="6,7,8": these months first (the irrigation season, DESIGN.md O), then the rest.
    first = [int(m) for m in os.getenv("FW_S2S_FIRST", "").split(",") if m.strip()]
    tasks.sort(key=lambda t: (first.index(t[0][0].month) if t[0][0].month in first else len(first), t[0][0], t[1]))
    log.info("%d packed requests (up to %d dates each), %d at a time", len(tasks), PACK, WORKERS)

    def run(task):
        ds, g = task
        ok, i = True, 0
        while i < len(ds):                  # pack_size may shrink during the run
            n = pack_size[g]
            sub, i = ds[i:i + n], i + n
            ok &= fetch_pack(sub, g) if n > 1 else all([fetch(d, g, "form") for d in sub])
        return ok

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(run, tasks))
    log.info("finished: %d ok, %d incomplete (re-run to retry)", sum(results), results.count(False))
