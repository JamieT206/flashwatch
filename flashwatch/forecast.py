"""Decode the ECMWF S2S reforecast GRIBs into weekly ensemble fields: data/fcst.zarr.

    python -m flashwatch.forecast inspect data/raw/s2s/2025-06-13_daily.grib   # key dump
    python -m flashwatch.forecast build                                        # all files

For each model-version date (issue month-day), fields are (hyear, member, week, lat, lon)
weekly means on the native 1.5 degree grid: daily means for the 'daily' group, daily totals
(differenced accumulations) for precipitation and solar radiation, averaged over forecast
days 1-7, 8-14, 15-21, 22-28. Week w covers steps 24(7(w-1))..24(7w) hours from 00 UTC on
the issue date. Messages are read key by key with ecCodes rather than through cfgrib's
automatic dimensions, which vary between the S2S hindcast templates.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from flashwatch.config import DATA, RAW

FCST_STORE = DATA / "fcst.zarr"
WEEKS = 4
N_DAYS = 7 * WEEKS
# ecCodes short names seen for S2S fields (GRIB2 time-processed fields get an avg_ prefix).
NAMES = {"2t": "t2m", "avg_2t": "t2m", "t2m": "t2m", "2d": "d2m", "avg_2d": "d2m", "d2m": "d2m",
         "sm20": "sm20", "avg_sm20": "sm20", "sm100": "sm100", "avg_sm100": "sm100",
         "st20": "st20", "avg_st20": "st20", "tp": "tp", "ssrd": "ssrd"}
ACCUMULATED = {"tp", "ssrd"}


def read_messages(path: Path):
    """Yield (name, hindcast date, member, start step h, end step h, lats, lons, values 2-D)."""
    import eccodes as ec
    with open(path, "rb") as f:
        while (h := ec.codes_grib_new_from_file(f)) is not None:
            try:
                g = lambda k: ec.codes_get(h, k)  # noqa: E731
                name = NAMES.get(g("shortName"), g("shortName"))
                ni, nj = g("Ni"), g("Nj")
                lat0, lat1 = g("latitudeOfFirstGridPointInDegrees"), g("latitudeOfLastGridPointInDegrees")
                lon0, lon1 = g("longitudeOfFirstGridPointInDegrees"), g("longitudeOfLastGridPointInDegrees")
                lats = np.linspace(lat0, lat1, nj)
                lons = (np.linspace(lon0, lon1 if lon1 >= lon0 else lon1 + 360, ni) + 180) % 360 - 180
                vals = ec.codes_get_values(h).reshape(nj, ni)
                yield (name, int(g("dataDate")), int(g("number")), int(g("startStep")), int(g("endStep")),
                       lats, lons, vals)
            finally:
                ec.codes_release(h)


def inspect(path: Path):
    """Print the distinct keys found in a file; run this on a real download before building."""
    import eccodes as ec
    seen = defaultdict(set)
    with open(path, "rb") as f:
        n = 0
        while (h := ec.codes_grib_new_from_file(f)) is not None:
            n += 1
            for k in ("shortName", "paramId", "dataDate", "number", "stepRange", "stepType", "dataType",
                      "productDefinitionTemplateNumber", "Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                      "longitudeOfFirstGridPointInDegrees", "iDirectionIncrementInDegrees"):
                try:
                    seen[k].add(str(ec.codes_get(h, k)))
                except Exception:  # noqa: BLE001
                    seen[k].add("<missing>")
            ec.codes_release(h)
    print(f"{path.name}: {n} messages")
    for k, v in seen.items():
        v = sorted(v)
        print(f"  {k}: {v[:8]}{' ...' if len(v) > 8 else ''} ({len(v)} distinct)")


def weekly(path: Path) -> tuple[dict[str, np.ndarray], list[int], np.ndarray, np.ndarray]:
    """Weekly means per variable, (hyear, member, week, lat, lon), from one downloaded file."""
    daily = defaultdict(dict)     # (name) -> {(hdate, member, day_index): field}
    accum = defaultdict(dict)     # (name) -> {(hdate, member, end_step): field}
    lats = lons = None
    for name, hdate, member, s0, s1, la, lo, v in read_messages(path):
        lats, lons = la, lo
        if name in ACCUMULATED:
            accum[name][(hdate, member, s1)] = v
        elif s1 - s0 == 24 and s1 <= 24 * N_DAYS:
            daily[name][(hdate, member, s1 // 24 - 1)] = v
    hdates = sorted({k[0] for d in list(daily.values()) + list(accum.values()) for k in d})
    members = sorted({k[1] for d in list(daily.values()) + list(accum.values()) for k in d})
    out = {}
    shape = (len(hdates), len(members), N_DAYS, len(lats), len(lons))
    for name, d in daily.items():
        a = np.full(shape, np.nan, np.float32)
        for (hd, m, day), v in d.items():
            if day < N_DAYS:
                a[hdates.index(hd), members.index(m), day] = v
        out[name] = a
    for name, d in accum.items():
        a = np.full(shape, np.nan, np.float32)
        for i, hd in enumerate(hdates):
            for j, m in enumerate(members):
                for day in range(N_DAYS):
                    s0, s1 = d.get((hd, m, 24 * day)), d.get((hd, m, 24 * (day + 1)))
                    if s0 is not None and s1 is not None:
                        a[i, j, day] = s1 - s0
        out[name] = a
    wk = {k: v.reshape(shape[:2] + (WEEKS, 7) + shape[3:]).mean(3) for k, v in out.items()}
    years = [hd // 10000 for hd in hdates]
    return wk, years, lats, lons


def build():
    """All downloaded dates into fcst.zarr: dims (issue, hyear, member, week, lat, lon)."""
    import pandas as pd
    import xarray as xr
    from flashwatch.ingest.s2s import GROUPS, issue_dates
    from flashwatch.config import YEARS

    src = RAW / "s2s"
    dates = [d for d in issue_dates() if all(any(src.glob(f"{d.isoformat()}_{g}*.grib")) for g in GROUPS)]
    if not dates:
        sys.exit("no complete issue dates in data/raw/s2s yet")
    data, coords = defaultdict(list), None
    for d in dates:
        fields = {}
        for g in GROUPS:
            for f in sorted(src.glob(f"{d.isoformat()}_{g}*.grib")):
                wk, years, lats, lons = weekly(f)
                if years != list(YEARS):
                    raise ValueError(f"{f.name}: hindcast years {years[0]}..{years[-1]}, expected {YEARS[0]}..{YEARS[-1]}")
                for k, v in wk.items():
                    fields[k] = v if k not in fields else np.where(np.isnan(fields[k]), v, fields[k])
        for k, v in fields.items():
            data[k].append(v)
        coords = (lats, lons, v.shape[1])
        print(f"{d}: {', '.join(sorted(fields))}")
    lats, lons, n_mem = coords
    ds = xr.Dataset({k: (("issue", "hyear", "member", "week", "lat", "lon"), np.stack(v)) for k, v in data.items()},
                    coords={"issue": pd.DatetimeIndex(dates), "hyear": list(YEARS), "member": np.arange(n_mem),
                            "week": np.arange(1, WEEKS + 1), "lat": lats, "lon": lons})
    ds.attrs.update(source="ECMWF S2S reforecasts (Cy49r1, model-version dates 2025) via the ECMWF Data Store",
                    licence="CC BY-NC 4.0; see README for the required S2S attribution")
    ds.to_zarr(FCST_STORE, mode="w")
    print(f"wrote {FCST_STORE}: {dict(ds.sizes)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "inspect":
        for p in sys.argv[2:]:
            inspect(Path(p))
    else:
        build()
