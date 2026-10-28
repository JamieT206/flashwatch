"""Season replay page: FlashWatch's forecast vs what happened, issue date by issue date.

    python -m flashwatch.viz.replay --oof B0_M1_week1_week2_any28_fast_d0-40_replay2012 --year 2012

Writes data/viz/replay_<year>.html, one self-contained page (data inlined; d3, topojson and the
US state outlines load from CDNs). Per frame (issue date):
- the week-2 onset probability in each eligible cell (from out-of-fold predictions, i.e. a model
  that never saw that year);
- cells already in drought (below the 20th percentile, so not eligible for an onset);
- onsets that actually happened in days 8-14 after the issue date (what the week-2 forecast targets);
- Drought Monitor D1+ cells from the latest map published by the issue date.
Below the map: regional mean forecast vs the observed week-2 onset rate over the season.
Uses ERA5-Land-derived predictions only (no ECMWF forecast data), so the page can be shared.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flashwatch.config import DATA
from flashwatch.obs import LAT, LON

TEMPLATE = Path(__file__).with_name("replay_template.html")


def frames(oof: pd.DataFrame, year: int, depth: str, lead: str = "week2"):
    from flashwatch.features.asof import as_of
    from flashwatch.labels.__main__ import label_file
    from flashwatch.labels.crosscheck import usdm_cube

    z = np.load(DATA / "labels" / label_file(40, 20, depth))
    time = pd.DatetimeIndex(z["time"])
    issues_all = pd.DatetimeIndex(z["issue"])
    sel = np.nonzero(issues_all.year == year)[0]
    issues = issues_all[sel]
    ny, nx = len(LAT), len(LON)
    df = oof[oof.year == year]
    prob = np.full((len(issues), ny, nx), np.nan, np.float32)
    k = issues.get_indexer(pd.DatetimeIndex(df.issue))
    ok = k >= 0
    prob[k[ok], df.iy.to_numpy()[ok], df.ix.to_numpy()[ok]] = df[f"M1_{lead}"].to_numpy()[ok]

    eligible = z["eligible"][sel]
    land = z["eligible"].any(0)
    dry = land[None] & ~eligible                       # land, but already below the 20th percentile

    onset_t = time[z["t"]]
    w = int(lead[-1]) if lead.startswith("week") else None
    onsets = np.zeros((len(issues), ny, nx), bool)
    for i, d in enumerate(issues):
        lo, hi = (d + pd.Timedelta(days=7 * (w - 1)), d + pd.Timedelta(days=7 * w)) if w else (d, d + pd.Timedelta(days=28))
        m = (onset_t > lo) & (onset_t <= hi)
        onsets[i, z["y"][m], z["x"][m]] = True

    udates, ucat = usdm_cube(list(range(2004, 2025)))
    j = udates.searchsorted(pd.DatetimeIndex([as_of("usdm", d) for d in issues]), side="right") - 1
    usdm = ucat[np.maximum(j, 0)] >= 1

    # Season curve: mean forecast vs observed rate over cells with a forecast.
    has = np.isfinite(prob)
    mean_pred = np.nansum(prob, (1, 2)) / np.maximum(has.sum((1, 2)), 1)
    obs_rate = (onsets & has).sum((1, 2)) / np.maximum(has.sum((1, 2)), 1)

    def pack_u8(a):
        return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()

    # Probabilities to 0-254 on a 0-15% scale (255 = no forecast); masks as bytes.
    scale = 0.15
    p8 = np.where(np.isfinite(prob), np.clip(prob / scale, 0, 1) * 254, 255).round()
    return {"year": year, "lead": lead, "depth": depth, "ny": ny, "nx": nx,
            "lat": [float(LAT[0]), float(LAT[-1])], "lon": [float(LON[0]), float(LON[-1])], "res": 0.25,
            "dates": [d.strftime("%Y-%m-%d") for d in issues], "scale": scale,
            "prob": pack_u8(p8), "dry": pack_u8(dry), "onset": pack_u8(onsets), "usdm": pack_u8(usdm),
            "land": pack_u8(land), "mean_pred": [round(float(v), 5) for v in mean_pred],
            "obs_rate": [round(float(v), 5) for v in obs_rate]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="OOF file stem in data/oof/")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--depth", default="0-40")
    ap.add_argument("--lead", default="week2")
    args = ap.parse_args()
    oof = pd.read_parquet(DATA / "oof" / f"{args.oof}.parquet")
    data = frames(oof, args.year, args.depth, args.lead)
    out = DATA / "viz"
    out.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    target = out / f"replay_{args.year}.html"
    target.write_text(page, encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB, {len(data['dates'])} frames)")


if __name__ == "__main__":
    main()
