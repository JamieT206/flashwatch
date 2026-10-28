"""How far ahead of the Drought Monitor FlashWatch is (DESIGN.md O0).

    python -m flashwatch.eval.lead [--oof data/oof/<file>.parquet]

The Drought Monitor (USDM) is the map most US drought programmes trigger on. For every onset in
the development years (test years excluded) we find when the Monitor first showed that cell
getting worse than it was before the soil began drying (20 days before onset), and when it first
showed D1 (moderate drought) or worse. Each date counts from the day it is public: Monitor maps
come out 2 days after their valid date, ERA5-Land soil moisture 5 days after.

Our timings: "crossing" = the soil passes the drought line (visible 5 days later); "confirmed" =
it has stayed below for the 10 days the onset rule needs (visible 15 days later). With --oof, also
the forecast warning: the issue date on which the onset's cell was on the top-10% watch list for
week 1 (or week 2). Positive days = FlashWatch earlier.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from flashwatch.config import DATA, TEST_YEARS, YEARS
from flashwatch.features.asof import LATENCY_DAYS

DRYING_DAYS = 20      # the onset rule's look-back: the soil was still wet this long before onset
PERSIST = 10          # days below the line the onset rule needs
WEEKS_AFTER = 12      # how long after onset we wait for the Monitor


def usdm_response(d: pd.DatetimeIndex, iy, ix, dates: pd.DatetimeIndex, cat: np.ndarray,
                  before: int = DRYING_DAYS, weeks: int = WEEKS_AFTER):
    """For onsets on dates d in cells (iy, ix): public dates of the first Monitor map (valid after
    d - before) showing the cell worse than on the last map before d - before, and of the first
    showing D1 or worse; NaT if neither happens within `weeks` of the onset. Also the base
    category."""
    d = pd.DatetimeIndex(d)
    iy, ix = np.asarray(iy), np.asarray(ix)
    i0 = dates.searchsorted(d - pd.Timedelta(days=before), "right") - 1
    ok = i0 >= 0
    base = np.where(ok, cat[np.clip(i0, 0, None), iy, ix], 99).astype(int)
    worse = np.full(len(d), np.datetime64("NaT", "ns"))
    d1 = worse.copy()
    end = d + pd.Timedelta(days=7 * weeks)
    for k in range(1, (before + 7 * weeks) // 7 + 2):
        j = i0 + k
        live = ok & (j < len(dates))
        jj = np.where(live, j, 0)
        live &= dates[jj] <= end
        c = cat[jj, iy, ix].astype(int)
        hit = live & np.isnat(worse) & (c > base)
        worse[hit] = dates[jj[hit]]
        hit = live & np.isnat(d1) & (c >= 1) & (base < 1)
        d1[hit] = dates[jj[hit]]
    pub = pd.Timedelta(days=LATENCY_DAYS["usdm"])
    return pd.DatetimeIndex(worse) + pub, pd.DatetimeIndex(d1) + pub, base


def summarise(ours: pd.DatetimeIndex, monitor: pd.DatetimeIndex, eligible=None) -> dict:
    """Days FlashWatch is ahead of the Monitor (positive = earlier)."""
    e = np.ones(len(ours), bool) if eligible is None else np.asarray(eligible)
    m, o = monitor[e], ours[e]
    shown = ~m.isna()
    ahead = (m[shown] - o[shown]).days.to_numpy()
    return {"onsets": int(e.sum()), "monitor_never_within_12wk": round(1 - shown.mean(), 3),
            "median_days_ahead": float(np.median(ahead)) if len(ahead) else np.nan,
            "p25": float(np.percentile(ahead, 25)) if len(ahead) else np.nan,
            "p75": float(np.percentile(ahead, 75)) if len(ahead) else np.nan,
            "share_flashwatch_first": round(float((ahead > 0).mean()), 3) if len(ahead) else np.nan}


def onsets_table(depth: str = "0-40") -> pd.DataFrame:
    from flashwatch.labels.__main__ import label_file
    z = np.load(DATA / "labels" / label_file(40, 20, depth))
    time = pd.DatetimeIndex(z["time"])
    df = pd.DataFrame({"date": time[z["t"]], "iy": z["y"], "ix": z["x"]})
    return df[~df.date.dt.year.isin(TEST_YEARS)].reset_index(drop=True)


def forecast_warnings(oof: pd.DataFrame, ons: pd.DataFrame, model="M2", top=0.10) -> pd.DataFrame:
    """Onsets that were on the model's top-`top` watch list for week 1 or 2 when issued: one row
    per (lead, onset) with the warning (issue) date and the onset date."""
    rows = []
    for w in (1, 2):
        lead = f"week{w}"
        r = oof[f"{model}_{lead}"].groupby(oof.issue).rank(ascending=False, pct=True)
        hit = oof[(r <= top) & (oof[lead] == 1)][["issue", "iy", "ix"]]
        j = hit.merge(ons, on=["iy", "ix"])
        lo = j.issue + pd.Timedelta(days=7 * (w - 1))
        j = j[(j.date > lo) & (j.date <= lo + pd.Timedelta(days=7))]
        j = j.sort_values("date").drop_duplicates(["issue", "iy", "ix"])
        rows.append(j.assign(lead=lead))
    return pd.concat(rows, ignore_index=True)


def main():
    from flashwatch.labels.crosscheck import usdm_cube
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="0-40")
    ap.add_argument("--oof", help="out-of-fold predictions with M2 (or M1) week1/week2 columns")
    ap.add_argument("--model", default="M2")
    args = ap.parse_args()
    dates, cat = usdm_cube(YEARS)
    ons = onsets_table(args.depth)
    ons = ons[ons.date.dt.year >= dates[0].year + 1]
    worse, d1, base = usdm_response(pd.DatetimeIndex(ons.date), ons.iy, ons.ix, dates, cat)
    grow = ons.date.dt.month.between(5, 9).to_numpy()
    lat = pd.Timedelta(days=LATENCY_DAYS["era5land"])
    timings = {"crossing": pd.DatetimeIndex(ons.date) + lat,
               "confirmed": pd.DatetimeIndex(ons.date) + lat + pd.Timedelta(days=PERSIST)}
    rows = []
    for season, sel in (("all months", np.ones(len(ons), bool)), ("May-Sep", grow)):
        for name, ours in timings.items():
            rows.append({"season": season, "flashwatch": name, "vs monitor": "first worsening",
                         **summarise(ours, worse, sel)})
            rows.append({"season": season, "flashwatch": name, "vs monitor": "first D1+",
                         **summarise(ours, d1, sel & (base < 1))})
    print(f"{len(ons)} onsets ({args.depth} cm, {ons.date.dt.year.min()}-{ons.date.dt.year.max()}, "
          f"test years excluded); base category before drying: "
          + ", ".join(f"{k}: {v:.0%}" for k, v in pd.Series(base).value_counts(normalize=True).sort_index().items()))
    if args.oof:
        oof = pd.read_parquet(args.oof)
        oof = oof[~oof.year.isin(TEST_YEARS)]
        fw = forecast_warnings(oof, ons, args.model)
        w2, d2, b2 = usdm_response(pd.DatetimeIndex(fw.date), fw.iy, fw.ix, dates, cat)
        for lead in ("week1", "week2"):
            s = (fw.lead == lead).to_numpy()
            ours = pd.DatetimeIndex(fw.issue)
            rows.append({"season": "forecast dates", "flashwatch": f"{args.model} {lead} watch list",
                         "vs monitor": "first worsening", **summarise(ours, w2, s)})
            rows.append({"season": "forecast dates", "flashwatch": f"{args.model} {lead} watch list",
                         "vs monitor": "first D1+", **summarise(ours, d2, s & (b2 < 1))})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
