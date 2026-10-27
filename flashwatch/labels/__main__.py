"""Build onset labels from data/obs.zarr and print the Day-2 summary.

    python -m flashwatch.labels                 # reference labels, thresholds 40/20
    python -m flashwatch.labels --variants      # also the 30/20 and 40/10 sensitivity runs

Reference labels use every study year as the climatology (each value against the other
years); they are for cross-checks and the base rate. CV folds rebuild percentiles from
training years only (build_fold), so model scores never see held-out climatology.
Writes data/labels/onsets_<high>_<low>.npz: onset (t, y, x) indices, episode ids, issue-date labels.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from flashwatch.config import CV_YEARS, DATA, YEARS
from flashwatch.features.asof import LATENCY_DAYS
from flashwatch.labels.episodes import episodes
from flashwatch.labels.onset import LEADS, lead_labels, mask_days, onset_days
from flashwatch.labels.percentile import percentiles, running_mean

OUT = DATA / "labels"
SEASON_PAD = ("03-05", "11-30")   # issue dates Apr 1-Oct 15, +28 d labels, -20/+10 d onset rule, -5 d latency


def issue_dates(years=YEARS) -> pd.DatetimeIndex:
    """Hindcast issue dates: each year's copy of the 2025 model-version dates."""
    from flashwatch.ingest.s2s import issue_dates as model_dates
    return pd.DatetimeIndex([date(y, d.month, d.day) for y in years for d in model_dates()])


def load(years=YEARS):
    from flashwatch.obs import open_obs
    ds = open_obs().sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))
    time = pd.DatetimeIndex(ds.time.values)
    rz = running_mean(ds.rzsm.values, 5)
    masked = mask_days(ds.snow_depth.values, ds.soil_temperature_level_1.values)
    land = (ds.land_frac.values >= 0.2)
    return time, rz, masked, land


def season(time):
    md = time.strftime("%m-%d")
    return (md >= SEASON_PAD[0]) & (md <= SEASON_PAD[1])


def build(ref_years, target_years, high=40, low=20, data=None):
    time, rz, masked, land = data or load()
    keep = season(time) & np.isin(time.year, list(target_years))
    P = percentiles(rz, time, ref_years=ref_years, target_years=target_years)
    P[~keep] = np.nan                                  # outside the padded season: no labels
    P[:, ~land] = np.nan
    onset = onset_days(P, high=high, low=low, mask=masked)
    return time, P, onset


def summarise(time, P, onset, low, years):
    ids, n_ep = episodes(onset)
    sizes = np.bincount(ids[ids > 0]) if n_ep else np.array([0])
    issues = issue_dates(years)
    lab = lead_labels(onset, time, issues)
    pos = time.get_indexer(issues - pd.Timedelta(days=LATENCY_DAYS["era5land"]))
    with np.errstate(invalid="ignore"):
        eligible = P[pos] > low                        # (issue, y, x)
    print(f"onset cell-days: {int(onset.sum())}; episodes: {n_ep} "
          f"(median {int(np.median(sizes[1:])) if n_ep else 0} cells, {int((sizes >= 50).sum())} with 50+ cells)")
    print(f"eligible cell-issue rows: {int(eligible.sum())}")
    for k in [f"week{w}" for w in LEADS] + ["any28"]:
        print(f"  base rate {k:6s}: {100 * lab[k][eligible].mean():.2f}%")
    per_year = pd.Series(onset.sum(axis=(1, 2)), index=time).groupby(time.year).sum()
    print("onset cell-days by year:\n" + per_year.to_string())
    return ids, lab, eligible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    for high, low in [(40, 20)] + ([(30, 20), (40, 10)] if args.variants else []):
        print(f"\n=== thresholds {high}/{low} ===")
        time, P, onset = build(YEARS, YEARS, high, low, data)
        ids, lab, eligible = summarise(time, P, onset, low, YEARS)
        t, y, x = np.nonzero(onset)
        np.savez_compressed(OUT / f"onsets_{high}_{low}.npz", t=t, y=y, x=x, episode=ids[t, y, x],
                            time=time.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                            issue=issue_dates().strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                            eligible=eligible, **{k: v for k, v in lab.items()})


def build_fold(held_out: int, high=40, low=20, data=None):
    """Labels for one leave-one-year-out fold: climatology from the other CV years only."""
    ref = [y for y in CV_YEARS if y != held_out]
    return build(ref, CV_YEARS, high, low, data)


if __name__ == "__main__":
    main()
