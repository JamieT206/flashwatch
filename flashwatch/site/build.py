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
            on = ons[(ons.iy == iy) & (ons.ix == ix)].date
            rec = {"iy": int(iy), "ix": int(ix),
                   **{k: (_r(v, 0 if k == "rz" else 3) if isinstance(v, np.ndarray) else v) for k, v in series.items()},
                   "usdm": [int(ucat[k][iy, ix]) if k >= 0 else -1 for k in j],
                   "onsets": sorted(on[on.dt.year == y].dt.strftime("%m-%d").tolist())}
            tiles.setdefault((iy // thin // TILE, ix // thin // TILE), []).append(rec)
        (out / "tiles" / str(y)).mkdir(parents=True, exist_ok=True)
        for (ty, tx), recs in tiles.items():
            (out / "tiles" / str(y) / f"{ty}_{tx}.json").write_text(json.dumps(recs, separators=(",", ":")))
    meta = {"model": model, "note": note, "years": years, "dates": md, "thin": thin, "tile": TILE,
            "lat0": float(LAT[0]), "lon0": float(LON[0]), "res": float(LAT[0] - LAT[1]),
            "cells": [[int(a), int(b)] for a, b in cells.itertuples(index=False, name=None)],
            "counties": counties()}
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return meta


def counties() -> list:
    """[fips, name, state, lat, lon] for counties in the study region (place names on the farm page)."""
    try:
        from flashwatch.ingest.nass import centroids
        c = centroids()
    except FileNotFoundError:
        return []
    c = c[c.lat.between(LAT[-1] - 0.5, LAT[0] + 0.5) & c.lon.between(LON[0] - 0.5, LON[-1] + 0.5)]
    return [[f, n, s, round(a, 3), round(o, 3)] for f, n, s, a, o in c[["fips", "name", "state", "lat", "lon"]].itertuples(index=False, name=None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--model", default="M1")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    meta = build(pd.read_parquet(args.oof), args.model, note=args.note)
    n = sum(1 for _ in (SITE / "data" / "tiles").rglob("*.json"))
    print(f"site data: {len(meta['cells'])} cells, years {meta['years']}, {len(meta['dates'])} dates, {n} tile files")


if __name__ == "__main__":
    main()
