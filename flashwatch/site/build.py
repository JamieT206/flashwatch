"""Data for the static site (site/, DESIGN.md O5): per grid cell and replay year, the season's
forecasts, the normal for each date, soil moisture, Drought Monitor category and the real onsets.

    python -m flashwatch.site.build --oof data/oof/<file>.parquet --model M1

Writes site/data/meta.json (grid, dates, years, counties) and site/data/tiles/<year>/<ty>_<tx>.json,
one file per block of 8 x 8 forecast cells, so the farm page loads only the block around the farm.
Out-of-fold predictions only (each year from models that never saw it); test years are refused.
site/data is git-ignored until the ECMWF licence question (Q2) is settled for forecast-derived data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flashwatch.config import TEST_YEARS
from flashwatch.obs import LAT, LON

SITE = Path(__file__).resolve().parents[2] / "site"
TILE = 8


def _r(a, d=3):
    return [None if not np.isfinite(v) else round(float(v), d) for v in a]


def build(oof: pd.DataFrame, model: str, out: Path = SITE / "data", note: str = "") -> dict:
    from flashwatch.eval.lead import onsets_table
    from flashwatch.features.asof import as_of
    from flashwatch.labels.crosscheck import usdm_cube

    oof = oof[~oof.year.isin(TEST_YEARS)]
    years = sorted(int(y) for y in oof.year.unique())
    md = sorted(pd.DatetimeIndex(oof.issue).strftime("%m-%d").unique())
    cells = oof[["iy", "ix"]].drop_duplicates().sort_values(["iy", "ix"])
    thin = int(np.gcd.reduce(np.r_[cells.iy.to_numpy(), cells.ix.to_numpy()])) or 1
    members = sum(1 for c in oof.columns if c.startswith(f"{model}_week1_m"))
    udates, ucat = usdm_cube(list(range(min(years) - 1, max(years) + 1)))
    frozen = frozen_days(years, md)
    water = daily_water(years)
    ons = onsets_table()
    ons = ons[ons.date.dt.year.isin(years)]
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    for y in years:
        d = oof[oof.year == y]
        dates = pd.DatetimeIndex([f"{y}-{m}" for m in md])
        j = udates.searchsorted([as_of("usdm", t) for t in dates], side="right") - 1
        col = {m: i for i, m in enumerate(md)}
        d = d.assign(t=pd.DatetimeIndex(d.issue).strftime("%m-%d").map(col))
        tiles = {}
        grouped = {k: g for k, g in d.groupby(["iy", "ix"])}
        for iy, ix in cells.itertuples(index=False, name=None):
            g = grouped.get((iy, ix))
            series = {k: np.full(len(md), np.nan) for k in ("p1", "p2", "n1", "n2", "rz")}
            if g is not None:
                t = g.t.to_numpy()
                for k, c in (("p1", f"{model}_week1"), ("p2", f"{model}_week2"), ("n1", "B0_week1"),
                             ("n2", "B0_week2"), ("rz", "rz_pct")):
                    series[k][t] = g[c].to_numpy()
            if g is not None and members:
                # Per-member week-1/2 risk and week-1 weather, for the weather triggers (O2).
                for k, pre in (("mp1", f"{model}_week1_m"), ("mp2", f"{model}_week2_m"),
                               ("mr", "wx_rain1_m"), ("mt", "wx_temp1_m")):
                    arr = [None] * len(md)
                    vals = g[[f"{pre}{j}" for j in range(members)]].to_numpy()
                    for t, row in zip(g.t.to_numpy(), vals):
                        if np.isfinite(row).all():
                            arr[t] = _r(row, 3 if k in ("mp1", "mp2") else (2 if k == "mr" else 0))
                    series[k] = arr
            if g is not None and "fc" in g:               # dates where the ECMWF forecast model gave the numbers
                series["fc"] = [int(t) for t in g.t.to_numpy()[g.fc.to_numpy()]]
            on = ons[(ons.iy == iy) & (ons.ix == ix)].date
            rec = {"iy": int(iy), "ix": int(ix),
                   **{k: (_r(v, 0 if k == "rz" else 3) if isinstance(v, np.ndarray) else v) for k, v in series.items()},
                   "usdm": [int(ucat[k][iy, ix]) if k >= 0 else -1 for k in j],
                   # dates with no forecast because of snow or frozen soil (not drought)
                   "fz": [int(k) for k in np.nonzero(frozen[y][:, iy, ix] & np.isnan(series["rz"]))[0]],
                   # daily rain and reference ET (hundredths of an inch) from WATER_DAYS[0], for the water account;
                   # the page uses only days public by each forecast date (ERA5-Land latency)
                   "wr": water[y][0][:, iy, ix].tolist(), "we": water[y][1][:, iy, ix].tolist(),
                   "onsets": sorted(on[on.dt.year == y].dt.strftime("%m-%d").tolist())}
            tiles.setdefault((iy // thin // TILE, ix // thin // TILE), []).append(rec)
        (out / "tiles" / str(y)).mkdir(parents=True, exist_ok=True)
        for (ty, tx), recs in tiles.items():
            (out / "tiles" / str(y) / f"{ty}_{tx}.json").write_text(json.dumps(recs, separators=(",", ":")))
    meta = {"model": model, "note": note, "years": years, "dates": md, "thin": thin, "tile": TILE, "water_days": list(WATER_DAYS),
            "lat0": float(LAT[0]), "lon0": float(LON[0]), "res": float(LAT[0] - LAT[1]),
            "cells": [[int(a), int(b)] for a, b in cells.itertuples(index=False, name=None)],
            "counties": counties()}
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return meta


def frozen_days(years, md) -> dict:
    """(date, lat, lon) bool per year: snow or frozen soil on the last ERA5-Land day public on each
    forecast date. Those dates have no forecast row, like cells already in drought."""
    from flashwatch.features.asof import as_of
    from flashwatch.labels.onset import mask_days
    from flashwatch.obs import open_obs
    ds = open_obs()
    out = {}
    for y in years:
        t = [as_of("era5land", pd.Timestamp(f"{y}-{m}")) for m in md]
        s = ds.sel(time=t)
        out[y] = mask_days(s["snow_depth"].values, s["soil_temperature_level_1"].values)
    return out


WATER_DAYS = ("04-01", "10-15")


def daily_water(years) -> dict:
    """{year: (rain, eto)}, each (day, lat, lon) int16 in hundredths of an inch, WATER_DAYS inclusive."""
    from flashwatch.decide.waterbook import MM_PER_IN, eto_pm
    from flashwatch.obs import open_obs
    ds = open_obs()
    out = {}
    for y in years:
        s = ds.sel(time=slice(f"{y}-{WATER_DAYS[0]}", f"{y}-{WATER_DAYS[1]}"))
        rain = s["total_precipitation_sum"].values * 1000 / MM_PER_IN
        doy = pd.DatetimeIndex(s.time.values).dayofyear.to_numpy()[:, None, None]
        eto = eto_pm(s["temperature_2m"].values, s["dewpoint_temperature_2m"].values,
                     s["surface_solar_radiation_downwards_sum"].values, s.lat.values[:, None], doy) / MM_PER_IN
        out[y] = tuple(np.nan_to_num(np.round(100 * a), nan=0).astype(np.int16) for a in (rain, eto))
    return out


def counties() -> list:
    """[fips, name, state, lat, lon] for counties in the study region (place names on the farm page)."""
    try:
        from flashwatch.ingest.nass import centroids
        c = centroids()
    except FileNotFoundError:
        return []
    c = c[c.lat.between(LAT[-1] - 0.5, LAT[0] + 0.5) & c.lon.between(LON[0] - 0.5, LON[-1] + 0.5)]
    return [[f, n, s, round(a, 3), round(o, 3)] for f, n, s, a, o in c[["fips", "name", "state", "lat", "lon"]].itertuples(index=False, name=None)]


def with_fill(oof: pd.DataFrame, model: str, fill: pd.DataFrame, fill_model: str) -> pd.DataFrame:
    """`model` (M2: only dates with an ECMWF forecast) where it has a value, `fill_model` (the land-only E1) on
    every other date, as one model "FW"; the fill file's rows, normals (B0) and soil moisture are kept. `fc`
    marks the rows from `model`."""
    key = ["year", "issue", "iy", "ix"]
    cols = [f"{model}_week1", f"{model}_week2"]
    df = fill.merge(oof[key + cols], on=key, how="left")
    fc = df[cols].notna().all(axis=1).to_numpy()
    for w in ("week1", "week2"):
        df[f"FW_{w}"] = np.where(fc, df[f"{model}_{w}"], df[f"{fill_model}_{w}"])
    return df.assign(fc=fc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--model", default="M1")
    ap.add_argument("--fill", help="OOF for dates the --model has no value for (M2 has none without an ECMWF forecast)")
    ap.add_argument("--fill-model", default="E1")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    oof, model = pd.read_parquet(args.oof), args.model
    if args.fill:
        oof, model = with_fill(oof, model, pd.read_parquet(args.fill), args.fill_model), "FW"
        print(f"{model}: {args.model} on {oof.fc.mean():.0%} of rows, {args.fill_model} on the rest")
    meta = build(oof, model, note=args.note)
    n = sum(1 for _ in (SITE / "data" / "tiles").rglob("*.json"))
    print(f"site data: {len(meta['cells'])} cells, years {meta['years']}, {len(meta['dates'])} dates, {n} tile files")


if __name__ == "__main__":
    main()
