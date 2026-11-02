"""ECMWF open-data ensemble forecasts as weekly fields that match data/fcst.zarr, for the live feed.

    python -m flashwatch.live.opendata 2026-07-03        # -> data/live/ens/2026-07-03.npz

The open data (IFS ENS, 50 members to 15 days, CC BY 4.0) is read from the Google Cloud mirror, which keeps the
archive; each step file has a .index of byte ranges, so only the needed messages are fetched (about 5 GB per
issue date), each cropped to the region as it arrives. Fields, as in the S2S reforecasts the model was trained on:
- t2m, d2m: daily means of the 00/06/12/18 UTC values (K);
- tp: daily totals (mm), from the accumulation at 168 h and 336 h;
- sm100: water in the top 100 cm (kg m-2), from volumetric layers 1-3 (0-7, 7-28, 28-100 cm) at 12 UTC each day.
Weeks 1-2 only (forecast days 1-7 and 8-14; the reforecasts also have weeks 3-4). Values are taken at the S2S
1.5 degree points (fcst.zarr's lat/lon), which lie on the 0.25 degree grid, as ECMWF's own regridding does.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from flashwatch.config import DATA

BASE = "https://storage.googleapis.com/ecmwf-open-data"
OUT = DATA / "live" / "ens"
LATS = np.arange(51.0, 32.9, -1.5)                 # fcst.zarr's grid
LONS = np.arange(-106.5, -77.9, 1.5)
DAYS = 14
LAYERS = {"1": 0.07, "2": 0.21, "3": 0.72}         # metres of soil in each layer of the top 100 cm
WORKERS = 16


def url(issue: pd.Timestamp, step: int, ext: str) -> str:
    return f"{BASE}/{issue:%Y%m%d}/00z/ifs/0p25/enfo/{issue:%Y%m%d}000000-{step}h-enfo-ef.{ext}"


def fetch(u: str, rng: tuple[int, int] | None = None, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(u, headers={"Range": f"bytes={rng[0]}-{rng[0] + rng[1] - 1}"} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def wanted(issue: pd.Timestamp) -> list[tuple[str, int, int, str, tuple[int, int]]]:
    """(param, member, step, level, byte range) for every message the weekly fields need."""
    steps = sorted({24 * d + h for d in range(DAYS) for h in (0, 6, 12, 18)} | {168, 336})
    out = []
    for step in steps:
        for line in fetch(url(issue, step, "index")).decode().splitlines():
            m = json.loads(line)
            p, lev = m["param"], m.get("levelist", "")
            keep = ((p in ("2t", "2d") and step < 24 * DAYS) or (p == "tp" and step in (168, 336))
                    or (p == "vsw" and lev in LAYERS and step % 24 == 12))
            if keep and m.get("type") == "pf":
                out.append((p, int(m["number"]), step, lev, (int(m["_offset"]), int(m["_length"]))))
    return out


def crop(msg: bytes) -> np.ndarray:
    """One GRIB message's values at the (LATS, LONS) points."""
    import eccodes as ec
    h = ec.codes_new_from_message(msg)
    try:
        ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
        la0, la1 = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees"), ec.codes_get(h, "latitudeOfLastGridPointInDegrees")
        lo0 = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        v = ec.codes_get_values(h).reshape(nj, ni)
    finally:
        ec.codes_release(h)
    dlat, dlon = (la0 - la1) / (nj - 1), 360 / ni
    iy = np.rint((la0 - LATS) / dlat).astype(int)
    ix = np.rint(((LONS - lo0) % 360) / dlon).astype(int) % ni
    return v[np.ix_(iy, ix)].astype(np.float64)


def weekly(issue) -> dict[str, np.ndarray]:
    """{t2m, d2m, tp, sm100}: (member 50, week 2, lat, lon) float32, in fcst.zarr's units."""
    issue = pd.Timestamp(issue)
    todo = wanted(issue)
    print(f"{issue:%Y-%m-%d}: {len(todo)} messages, {sum(r[1] for *_, r in todo) / 1e9:.1f} GB", flush=True)
    sums, counts, tp = defaultdict(float), defaultdict(int), {}

    def one(item):
        p, m, step, lev, rng = item
        return item, crop(fetch(url(issue, step, "grib2"), rng))

    t0, done = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for (p, m, step, lev, _), v in ex.map(one, todo):
            if p == "tp":
                tp[(m, step)] = v
            else:
                name, val = ("sm100", 1000 * LAYERS[lev] * v) if p == "vsw" else ({"2t": "t2m", "2d": "d2m"}[p], v)
                key = (name, m, step // 24 // 7)
                sums[key] = sums[key] + val
                counts[key] += 1 if p != "vsw" else LAYERS[lev]
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(todo)} in {time.time() - t0:.0f} s", flush=True)
    members = sorted({k[1] for k in counts})
    shape = (len(members), 2, len(LATS), len(LONS))
    out = {v: np.full(shape, np.nan, np.float32) for v in ("t2m", "d2m", "tp", "sm100")}
    for (name, m, w), s in sums.items():
        # t2m/d2m: 28 values a week; sm100: 7 days, each the layer-weighted sum (weights add to 1)
        out[name][members.index(m), w] = s / counts[(name, m, w)]
    for m in members:
        a, b = tp.get((m, 168)), tp.get((m, 336))
        if a is not None and b is not None:
            out["tp"][members.index(m), 0] = 1000 * a / 7               # metres accumulated -> mm a day
            out["tp"][members.index(m), 1] = 1000 * (b - a) / 7
    return out


def download(issue) -> "Path":
    issue = pd.Timestamp(issue)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{issue:%Y-%m-%d}.npz"
    if not target.exists():
        f = weekly(issue)
        np.savez_compressed(target, lat=LATS, lon=LONS, **f)
    return target


def load(issue) -> dict[str, np.ndarray]:
    z = np.load(OUT / f"{pd.Timestamp(issue):%Y-%m-%d}.npz")
    return {k: z[k] for k in z.files}


def available(issue) -> bool:
    """The day's 00 UTC run is fully on the mirror (its last step file is there)."""
    try:
        fetch(url(pd.Timestamp(issue), 360, "index"), tries=1)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(download(d))
