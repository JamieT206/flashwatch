"""Day-2 cross-checks of the onset labels against independent evidence.

    python -m flashwatch.labels.crosscheck      # after python -m flashwatch.labels
    python -m flashwatch.labels.crosscheck --depth 0-28   # a target-depth variant (DESIGN.md T1)

1. Known events: the largest episode inside each documented event's box and window.
2. SMVI inventory (Osman et al.), 2005-2021: share of our episodes with an SMVI onset in the
   same cells within +/-14 days, against a chance level from SMVI dates of other years.
3. Drought Monitor: share of large episodes whose cells worsen by 2+ categories within
   4 weeks of onset, against the same cells one year earlier/later.
4. Season timing in each known event's box (first onset per cell, Mar-Nov of the event year).
5. Lead over the Drought Monitor: share of onsets followed by a 1+ category worsening within
   8 weeks (vs the same cells and dates 1-2 years away), and the median days to it.
6. Episode coherence: share of onset cell-days in episodes of 10+ cells.
Prints a table for the write-up; the thresholds for "pass" are judgement calls, so the
numbers are shown rather than a verdict.
"""

from __future__ import annotations

import re
import tarfile
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

from flashwatch.config import DATA, RAW
from flashwatch.obs import LAT, LON

KNOWN_EVENTS = [  # name, (lat0, lat1), (lon0, lon1), window
    ("2012 central US", (37, 43), (-98, -86), ("2012-06-01", "2012-07-31")),
    ("2017 Northern Plains", (44, 49), (-104, -96), ("2017-05-01", "2017-07-31")),
    ("2022 central Plains", (37, 43), (-102, -94), ("2022-06-01", "2022-08-31")),
    ("2023 Midwest", (38, 45), (-96, -85), ("2023-05-15", "2023-07-15")),
]
MATCH_DAYS = 14


def load_labels(high=40, low=20, depth="0-100"):
    from flashwatch.labels.__main__ import label_file
    z = np.load(DATA / "labels" / label_file(high, low, depth))
    time = pd.DatetimeIndex(z["time"])
    onset = np.zeros((len(time), len(LAT), len(LON)), bool)
    onset[z["t"], z["y"], z["x"]] = True
    ids = np.zeros(onset.shape, np.int32)
    ids[z["t"], z["y"], z["x"]] = z["episode"]
    return time, onset, ids


def episode_table(time, ids) -> pd.DataFrame:
    t, y, x = np.nonzero(ids)
    df = pd.DataFrame({"episode": ids[t, y, x], "date": time[t], "iy": y, "ix": x,
                       "lat": LAT[y], "lon": LON[x]})
    return df


def known_events(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (la0, la1), (lo0, lo1), (d0, d1) in KNOWN_EVENTS:
        inbox = ep[(ep.lat.between(la0, la1)) & (ep.lon.between(lo0, lo1))
                   & (ep.date >= d0) & (ep.date <= d1)]
        if inbox.empty:
            rows.append({"event": name, "largest_episode_cells": 0})
            continue
        top = inbox.episode.value_counts().index[0]
        e = ep[ep.episode == top]
        box_cells = ((LAT >= la0) & (LAT <= la1)).sum() * ((LON >= lo0) & (LON <= lo1)).sum()
        rows.append({"event": name, "largest_episode_cells": len(e),
                     "episode_onset_median": e.date.median().date(),
                     "episode_onset_range": f"{e.date.min().date()} to {e.date.max().date()}",
                     "box_share_with_onset": round(inbox[["iy", "ix"]].drop_duplicates().shape[0] / box_cells, 2)})
    return pd.DataFrame(rows)


def smvi_onsets(years) -> dict[int, pd.DataFrame]:
    """SMVI onset dates per year on our grid: rows (iy, ix, date)."""
    path = RAW / "smvi" / "SMVI_GLDAS_Events.tar.gz"
    il = np.round((LAT + 59.875) / 0.25).astype(int)
    io = np.round((LON + 179.875) / 0.25).astype(int)
    rows = (il[:, None] * 1440 + io[None, :]).ravel()          # 0-based row in the global table
    iy, ix = np.divmod(np.arange(len(LAT) * len(LON)), len(LON))
    out = {}
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            hit = re.search(r"SMVI_(\d{4})_n\d+\.csv$", m.name)
            if not hit or int(hit.group(1)) not in years:
                continue
            df = pd.read_csv(tar.extractfile(m), index_col=0, low_memory=False)
            sub = df.iloc[rows]
            parts = []
            for k in range(1, 7):
                d = pd.to_datetime(sub[f"fstdate{k}"], errors="coerce").to_numpy()
                ok = ~pd.isna(d)
                parts.append(pd.DataFrame({"iy": iy[ok], "ix": ix[ok], "date": d[ok]}))
            out[int(hit.group(1))] = pd.concat(parts, ignore_index=True)
    return out


def smvi_overlap(ep: pd.DataFrame, smvi: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Share of episodes with an SMVI onset in one of their cells within +/-MATCH_DAYS."""
    def matched(ep_year: pd.DataFrame, sm: pd.DataFrame, shift_years: int) -> pd.Series:
        sm = sm.assign(date=sm.date + pd.DateOffset(years=shift_years))
        j = ep_year.merge(sm, on=["iy", "ix"], suffixes=("", "_s"))
        j = j[(j.date - j.date_s).abs() <= pd.Timedelta(days=MATCH_DAYS)]
        return ep_year.episode.drop_duplicates().isin(j.episode)

    years = sorted(smvi)
    real, chance = [], []
    for y in years:
        e = ep[ep.date.dt.year == y]
        if e.empty:
            continue
        real.append(matched(e, smvi[y], 0))
        for dy in (-2, -1, 1, 2):
            if y - dy in smvi:
                chance.append(matched(e, smvi[y - dy], dy))
    r, c = pd.concat(real), pd.concat(chance)
    return pd.DataFrame([{"check": "SMVI overlap (all episodes)", "observed": round(r.mean(), 3),
                          "chance": round(c.mean(), 3), "n_episodes": len(r)}])


def usdm_grid(day: date) -> np.ndarray | None:
    """Drought Monitor category (-1 none, 0-4 D0-D4) at each cell centre for the map valid on day."""
    import geopandas as gpd
    import shapely
    f = RAW / "usdm" / f"USDM_{day:%Y%m%d}_M.zip"
    if not f.exists():
        return None
    g = gpd.read_file(f"zip://{f}").to_crs(4326)
    gy, gx = np.meshgrid(LAT, LON, indexing="ij")
    cat = np.full(gy.shape, -1, np.int8)
    for dm, geom in zip(g.DM, g.geometry):
        inside = shapely.contains_xy(geom, gx, gy)
        cat[inside] = np.maximum(cat[inside], int(dm))
    return cat


def usdm_cube(years) -> tuple[pd.DatetimeIndex, np.ndarray]:
    cache = DATA / "labels" / "usdm_grid.npz"
    if cache.exists():
        z = np.load(cache)
        return pd.DatetimeIndex(z["dates"]), z["cat"]
    d = date(years[0], 1, 1)
    d += timedelta(days=(1 - d.weekday()) % 7)
    dates, cats = [], []
    while d.year <= years[-1]:
        c = usdm_grid(d)
        if c is not None:
            dates.append(d)
            cats.append(c)
        d += timedelta(days=7)
    dates, cats = pd.DatetimeIndex(dates), np.stack(cats)
    np.savez_compressed(cache, dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"), cat=cats)
    return dates, cats


def usdm_worsening(ep: pd.DataFrame, dates: pd.DatetimeIndex, cat: np.ndarray, min_cells=50) -> pd.DataFrame:
    """Share of large episodes where most cells worsen by 2+ categories within 4 weeks."""
    def worsened(cells: pd.DataFrame, d0: pd.Timestamp) -> float:
        before = dates[dates <= d0]
        after = dates[(dates > d0) & (dates <= d0 + pd.Timedelta(days=28))]
        if before.empty or after.empty:
            return np.nan
        b = cat[dates.get_loc(before[-1])][cells.iy, cells.ix].astype(int)
        a = cat[[dates.get_loc(x) for x in after]][:, cells.iy, cells.ix].max(0).astype(int)
        return float(((a - b) >= 2).mean())

    sizes = ep.groupby("episode").size()
    real, chance = [], []
    for e_id in sizes[sizes >= min_cells].index:
        e = ep[ep.episode == e_id]
        d0 = e.date.median()
        cells = e[["iy", "ix"]].drop_duplicates()
        real.append(worsened(cells, d0))
        chance += [worsened(cells, d0 + pd.DateOffset(years=dy)) for dy in (-1, 1)]
    real, chance = np.array(real, float), np.array(chance, float)
    return pd.DataFrame([{"check": f"Drought Monitor 2+ category worsening (episodes >= {min_cells} cells)",
                          "observed": round(np.nanmean(real >= 0.5), 3) if len(real) else np.nan,
                          "chance": round(np.nanmean(chance >= 0.5), 3) if len(chance) else np.nan,
                          "n_episodes": int(np.isfinite(real).sum())}])


def event_timing(ep: pd.DataFrame) -> pd.DataFrame:
    """Per known event: share of box cells with an onset Mar-Nov of the event year, and the
    10th/50th/90th percentiles of each cell's first onset date."""
    rows = []
    for name, (la0, la1), (lo0, lo1), _ in KNOWN_EVENTS:
        yr = int(name[:4])
        e = ep[ep.lat.between(la0, la1) & ep.lon.between(lo0, lo1) & (ep.date.dt.year == yr)
               & ep.date.dt.month.between(3, 11)]
        box_cells = ((LAT >= la0) & (LAT <= la1)).sum() * ((LON >= lo0) & (LON <= lo1)).sum()
        first = e.groupby(["iy", "ix"]).date.min()
        row = {"event": name, "season_box_share": round(len(first) / box_cells, 2)}
        if len(first):
            q = first.quantile([0.1, 0.5, 0.9])
            row.update(first_onset_p10=q.iloc[0].date(), p50=q.iloc[1].date(), p90=q.iloc[2].date())
        rows.append(row)
    return pd.DataFrame(rows)


def usdm_lead(ep: pd.DataFrame, dates: pd.DatetimeIndex, cat: np.ndarray, weeks=8, n=20000, seed=0):
    """Share of onsets (sampled) whose cell worsens by 1+ USDM category within `weeks`, against
    the same cells and calendar dates 1-2 years away; median days from onset to that worsening."""
    s = ep.sample(min(n, len(ep)), random_state=seed)

    def hits(d, iy, ix):
        i0 = dates.searchsorted(d, "right") - 1
        ok = (i0 >= 0) & (i0 + weeks < len(dates))
        base = cat[np.where(ok, i0, 0), iy, ix]
        lead = np.full(len(d), np.nan)
        for k in range(1, weeks + 1):
            j = np.where(ok, i0 + k, 0)
            worse = ok & np.isnan(lead) & (cat[j, iy, ix] > base)
            lead[worse] = (dates[j[worse]] - d[worse]).days
        return lead, ok

    d, iy, ix = pd.DatetimeIndex(s.date), s.iy.to_numpy(), s.ix.to_numpy()
    lead, ok = hits(d, iy, ix)
    chance = np.mean([np.isfinite(lc)[oc].mean()
                      for lc, oc in (hits(d + pd.DateOffset(years=dy), iy, ix) for dy in (-2, -1, 1, 2))])
    return pd.DataFrame([{"check": f"USDM 1+ category worsening within {weeks} weeks of onset (sample)",
                          "observed": round(np.isfinite(lead)[ok].mean(), 3), "chance": round(chance, 3),
                          "n_episodes": int(ok.sum()),
                          "median_days_to_worsening": float(np.nanmedian(lead))}])


def coherence(ep: pd.DataFrame) -> pd.DataFrame:
    sizes = ep.groupby("episode").size()
    big = ep.episode.map(sizes) >= 10
    return pd.DataFrame([{"check": "onset cell-days in episodes of 10+ cells", "observed": round(big.mean(), 3),
                          "chance": np.nan, "n_episodes": len(sizes)}])


def main():
    import argparse

    from flashwatch.config import YEARS
    from flashwatch.labels.__main__ import DEPTHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="0-100", choices=list(DEPTHS))
    args = ap.parse_args()
    time, onset, ids = load_labels(depth=args.depth)
    ep = episode_table(time, ids)
    print(f"target depth {args.depth} cm: {len(ep)} onset cell-days, {ep.episode.nunique()} episodes")
    print("Known events\n" + known_events(ep).to_string(index=False))
    print("\nSeason timing in the event boxes\n" + event_timing(ep).to_string(index=False))
    dates, cat = usdm_cube(YEARS)
    rows = [smvi_overlap(ep, smvi_onsets(range(2005, 2022))), usdm_worsening(ep, dates, cat),
            usdm_lead(ep, dates, cat), coherence(ep)]
    print("\n" + pd.concat(rows).to_string(index=False))


if __name__ == "__main__":
    main()
