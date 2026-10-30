"""Training pairs for the Muse Glimmer narrative writer (DESIGN.md G2).

    python -m flashwatch.ingest.usdm_text          # the real weekly narratives (once)
    python -m flashwatch.agent.glimmer_data        # -> data/glimmer/{train,val}.jsonl + stats
    python -m flashwatch.agent.glimmer_data --nomap   # -> data/glimmer/nomap/: without this week's map

Each example is one Drought Monitor week (valid Tuesday). Input: what the data showed over the 7
days to that Tuesday for the High Plains and Midwest, by region and by state (ERA5-Land rain as a
percent of normal, temperature departure in F, root-zone soil moisture as a percent of normal and
its weekly change, new flash-drought onsets at 0-40 cm), and the Drought Monitor's own category
shares and their weekly change. Output: the narrative's High Plains / Midwest sections as written.
"Normal" is the 2005-2019 average for that week of the year (the training years), so validation
years are not used in their own inputs.

Splits: train 2005-2019, validation 2020-2021. Test years (2022-2024) are not written here.
The aim is a small open model that writes the regional summary in the Monitor's voice from
FlashWatch's numbers; it is scored against held-out real narratives plus the number checker.

--nomap: the same weeks, but the input has last week's Drought Monitor shares (public a week earlier) instead of
this week's and their change, so the summary can be written before the Monitor's map comes out. Each row keeps
the real D1+ change by region as `truth`, for scoring only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from flashwatch.config import DATA, RAW
from flashwatch.obs import LAT, LON

OUT = DATA / "glimmer"
TRAIN, VAL = range(2005, 2020), range(2020, 2022)
REGIONS = {"High Plains": ["ND", "SD", "NE", "KS", "CO", "WY", "MT"],
           "Midwest": ["MN", "IA", "MO", "WI", "IL", "IN", "OH", "MI", "KY"]}
SYSTEM = ("You write the US Drought Monitor's weekly narrative for the High Plains and the Midwest. "
          "Use only the numbers and places in the data you are given; describe changes in plain words, "
          "in the Monitor's usual style (what the weather did, then where drought categories changed and why).")
SYSTEM_NOMAP = SYSTEM + (
    " You have last week's Drought Monitor map (the D-category shares marked last_week), not this week's: from the "
    "week's weather and soil moisture, say where drought most likely expanded, eased or held.")


def cell_states() -> np.ndarray:
    """State code of each grid cell (nearest county centroid), '' where none within 50 km."""
    from flashwatch.ingest.nass import centroids
    c = centroids()
    c = c[c.lat.between(35, 50) & c.lon.between(-105, -79)]
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    out = np.full(la.shape, "", dtype=object)
    for i in range(la.shape[0]):
        for j in range(la.shape[1]):
            d = np.hypot(c.lat - la[i, j], (c.lon - lo[i, j]) * np.cos(np.radians(la[i, j]))) * 111
            k = int(np.argmin(d.to_numpy()))
            if d.iloc[k] < 50:
                out[i, j] = c.state.iloc[k]
    return out


def is_ours(heading: str) -> bool:
    h = heading.lower()
    if any(k in h for k in ("central plains", "northern plains", "great lakes", "ohio valley", "corn belt")):
        return True
    return ("midwest" in h or "plains" in h) and not h.startswith(("south", "the south"))


def target(narr: dict) -> str:
    return "\n\n".join(f"{s['heading']}\n{s['text']}" for s in narr["sections"] if is_ours(s["heading"]))


def weekly_series(states: np.ndarray, land: np.ndarray, tuesdays: pd.DatetimeIndex) -> pd.DataFrame:
    """Per Tuesday and area (each region and each state): 7-day rain, mean temperature, soil, onsets."""
    from flashwatch.eval.lead import onsets_table
    from flashwatch.obs import open_obs
    ds = open_obs()
    time = pd.DatetimeIndex(ds.time.values)
    areas = {r: np.isin(states, s) & land for r, s in REGIONS.items()}
    areas.update({s: (states == s) & land for ss in REGIONS.values() for s in ss})
    areas = {k: m for k, m in areas.items() if m.any()}
    tp = ds.total_precipitation_sum.values
    t2 = ds.temperature_2m.values
    sm = ds.rzsm.values
    ons = onsets_table()
    ons_day = pd.DatetimeIndex(ons.date)
    rows = []
    for t in tuesdays:
        k = time.get_indexer([t])[0]
        if k < 7:
            continue
        w = slice(k - 6, k + 1)
        rain, temp = tp[w].sum(0) * 1000, t2[w].mean(0)            # mm, K
        sm_now, sm_prev = sm[k], sm[k - 7]
        new = np.zeros(land.shape, bool)
        e = ons[(ons_day > t - pd.Timedelta(days=7)) & (ons_day <= t)]
        new[e.iy.to_numpy(), e.ix.to_numpy()] = True
        for a, m in areas.items():
            rows.append({"date": t, "area": a, "rain_mm": float(rain[m].mean()), "temp_k": float(temp[m].mean()),
                         "soil": float(sm_now[m].mean()), "soil_prev": float(sm_prev[m].mean()),
                         "onset_share": float(new[m].mean()), "cells": int(m.sum())})
    return pd.DataFrame(rows)


def with_normals(d: pd.DataFrame) -> pd.DataFrame:
    """Anomalies against the 2005-2019 average for the same week of the year (+-1 week)."""
    d = d.copy()
    d["woy"] = d.date.dt.isocalendar().week.astype(int).clip(upper=52)
    base = d[d.date.dt.year.isin(TRAIN)]
    out = []
    for (a, w), g in d.groupby(["area", "woy"]):
        b = base[(base.area == a) & ((base.woy - w).abs().isin([0, 1, 51]))]
        out.append(g.assign(rain_pct_normal=100 * g.rain_mm / max(b.rain_mm.mean(), 0.1),
                            temp_dep_f=(g.temp_k - b.temp_k.mean()) * 1.8,
                            soil_pct_normal=100 * g.soil / b.soil.mean(),
                            soil_change_pct=100 * (g.soil - g.soil_prev) / b.soil.mean()))
    return pd.concat(out).sort_values(["date", "area"])


def monitor_shares(states, land, tuesdays) -> pd.DataFrame:
    from flashwatch.labels.crosscheck import usdm_cube
    dates, cat = usdm_cube(list(range(2005, 2025)))
    idx = pd.DatetimeIndex(dates)
    areas = {r: np.isin(states, s) & land for r, s in REGIONS.items()}
    areas.update({s: (states == s) & land for ss in REGIONS.values() for s in ss})
    rows = []
    for t in tuesdays:
        k = idx.get_indexer([t])[0]
        if k < 1:
            continue
        for a, m in areas.items():
            if not m.any():
                continue
            now, prev = cat[k][m], cat[k - 1][m]
            rows.append({"date": t, "area": a, **{f"D{c}_or_worse_pct": round(100 * float((now >= c).mean()), 1)
                                                for c in range(5)},
                         **{f"D{c}_or_worse_pct_last_week": round(100 * float((prev >= c).mean()), 1) for c in range(5)},
                         "D1_or_worse_change_pct": round(100 * float((now >= 1).mean() - (prev >= 1).mean()), 1)})
    return pd.DataFrame(rows)


def state_json(g: pd.DataFrame, nomap: bool = False) -> dict:
    """The model's input for one week: regions first, then states. `nomap`: last week's map only."""
    end = ("_last_week",) if nomap else ("pct",)
    def area(r):
        return {"rain_percent_of_normal": int(round(r.rain_pct_normal)), "temperature_departure_F": round(r.temp_dep_f, 1),
                "soil_moisture_percent_of_normal": int(round(r.soil_pct_normal)),
                "soil_moisture_change_this_week_pct": round(r.soil_change_pct, 1),
                "new_flash_drought_onsets_percent_of_area": round(100 * r.onset_share, 1),
                **{k: r[k] for k in r.index if k.startswith("D") and k.endswith(end)}}
    regions = {r.area: area(r) for _, r in g.iterrows() if r.area in REGIONS}
    states = {r.area: area(r) for _, r in g.iterrows() if r.area not in REGIONS}
    t = g.date.iloc[0]
    return {"week_ending": f"{t:%B} {t.day}, {t.year}", "regions": regions, "states": states}


def main():
    import argparse
    from flashwatch.obs import land_mask, open_obs
    ap = argparse.ArgumentParser()
    ap.add_argument("--nomap", action="store_true", help="last week's Drought Monitor map instead of this week's")
    nomap = ap.parse_args().nomap
    out = OUT / "nomap" if nomap else OUT
    narr = {}
    for f in sorted((RAW / "usdm_text").glob("*.json")):
        n = json.loads(f.read_text(encoding="utf-8"))
        if int(n["date"][:4]) in TRAIN or int(n["date"][:4]) in VAL:
            narr[pd.Timestamp(n["date"])] = n
    tues = pd.DatetimeIndex(sorted(narr))
    land = land_mask(open_obs())
    states = cell_states()
    d = with_normals(weekly_series(states, land, tues)).merge(monitor_shares(states, land, tues), on=["date", "area"])
    d = d.set_index("area")
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "no_section": 0}
    files = {k: open(out / f"{k}.jsonl", "w", encoding="utf-8") for k in ("train", "val")}
    for t in tues:
        text = target(narr[t])
        if not text:
            counts["no_section"] += 1
            continue
        g = d[d.date == t].reset_index()
        if g.empty:
            continue
        inp = state_json(g, nomap)
        split = "train" if t.year in TRAIN else "val"
        row = {"messages": [{"role": "system", "content": SYSTEM_NOMAP if nomap else SYSTEM},
                            {"role": "user", "content": json.dumps(inp)},
                            {"role": "assistant", "content": text}],
               "date": t.date().isoformat()}
        if nomap:
            row["truth"] = {r.area: float(r.D1_or_worse_change_pct) for _, r in g.iterrows() if r.area in REGIONS}
        files[split].write(json.dumps(row) + "\n")
        counts[split] += 1
    for f in files.values():
        f.close()
    (out / "stats.json").write_text(json.dumps(counts, indent=1))
    print(counts)


if __name__ == "__main__":
    main()
