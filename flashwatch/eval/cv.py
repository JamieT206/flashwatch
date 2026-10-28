"""Leave-one-year-out cross-validation over the development years.

    python -m flashwatch.eval.cv --models B0 B1      # needs data/obs.zarr
    python -m flashwatch.eval.cv --models B0 B1 M1 --fast    # design iteration: minutes, not hours

For each held-out year h, labels and features are rebuilt with climatology from the other
development years only; each model is fitted per lead on those years and predicts h. The
out-of-fold predictions go to data/oof/<models>.parquet and a score table is printed, with
BSS against B0 and year-block bootstrap CIs. Test years are never touched here.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import date

import numpy as np
import pandas as pd

from flashwatch.config import CV_YEARS, DATA, YEARS
from flashwatch.eval.metrics import score, year_bootstrap_bss
from flashwatch.features.build import FeatureBuilder
from flashwatch.labels.onset import LEADS, lead_labels, onset_days
from flashwatch.models.baselines import LIGHT, MODELS

LABELS = [f"week{w}" for w in LEADS] + ["any28"]
LOW = 20
# --fast: 6 held-out years spread over the dev period (incl. 2012), every other cell, light GBMs.
FAST_HOLDOUT = [2006, 2009, 2012, 2015, 2018, 2021]


def model_issue_dates(years) -> pd.DatetimeIndex:
    from flashwatch.ingest.s2s import issue_dates
    return pd.DatetimeIndex([date(y, d.month, d.day) for y in years for d in issue_dates()])


def fold_table(cube, fb: FeatureBuilder, ref_years, target_years, high=40, low=LOW, ff=None) -> pd.DataFrame:
    """Rows = eligible (issue, cell) pairs in target_years: features, labels and ids.
    ff: ForecastFeatures for the same issue dates (adds the f_* columns), or None."""
    P = fb.rz_counts.fold(ref_years)
    P[:, ~cube.land] = np.nan
    onset = onset_days(P, high=high, low=low, mask=cube.masked)
    feats = fb.build(ref_years, P_rz=P)
    if ff is not None:
        feats.update(ff.build(ref_years))
    labels = lead_labels(onset, cube.time, fb.issues)
    keep_issue = np.isin(fb.issues.year, list(target_years))
    with np.errstate(invalid="ignore"):
        eligible = (feats["rz_pct"] > low) & cube.land[None] & keep_issue[:, None, None]
    n, iy, ix = np.nonzero(eligible)
    cols = {"year": fb.issues.year.to_numpy()[n], "issue": fb.issues[n], "iy": iy, "ix": ix,
            "woy": ((fb.issues.day_of_year.to_numpy()[n] - 1) // 7).clip(0, 52)}
    for k in list(feats):
        cols[k] = feats.pop(k)[n, iy, ix]         # pop: free each full grid once copied
    for k, v in labels.items():
        cols[k] = v[n, iy, ix].astype(np.int8)
    return pd.DataFrame(cols, copy=False)         # built once, without copying the columns


def member_frames(ff, fb, ref, train, test):
    """Training rows stacked once per ensemble member with that member's z features (rows with a
    forecast only), one test frame per member, and each member's week-1 weather at the test rows."""
    n = fb.issues.get_indexer(pd.DatetimeIndex(test.issue))
    iy, ix = test.iy.to_numpy(), test.ix.to_numpy()

    def swap(df, g):
        k = fb.issues.get_indexer(pd.DatetimeIndex(df.issue))
        y, x = df.iy.to_numpy(), df.ix.to_numpy()
        return df.assign(**{c: v[k, y, x] for c, v in g.items() if c in df.columns})
    grids = [ff.build_member(ref, m) for m in range(ff.n_members)]
    has = train["f_tp_w1_z"].notna() if "f_tp_w1_z" in train else slice(None)
    tr = pd.concat([swap(train[has], g) for g in grids], ignore_index=True)
    wx = {}
    for m in range(ff.n_members):
        w = ff.member_weather(m)
        wx[f"wx_rain1_m{m}"] = w["rain_in"][n, iy, ix]
        wx[f"wx_temp1_m{m}"] = w["temp_f"][n, iy, ix]
    return tr, [swap(test, g) for g in grids], wx


def run_cv(cube, models=("B0", "B1"), cv_years=CV_YEARS, pool_years=YEARS, fb=None, ff=None, log=print,
           labels=LABELS, holdout=None, thin=1, light=False, gbm_kw=None, drop=(), fc_dates=None, members=False):
    """holdout: the held-out years to run (default all of cv_years; training always uses the
    rest of cv_years). thin: keep every thin-th cell in each direction. light: LIGHT GBMs.
    drop: feature names or prefixes left out of the models (ablations).
    fc_dates: month-days ("04-01", ...) to keep, train and test, e.g. the ECMWF start dates
    downloaded so far.
    members: M2 trained on every ensemble member's own weather (one copy of each row per member)
    and its prediction averaged over the members (DESIGN.md P11); needs ff."""
    fb = fb or FeatureBuilder(cube, model_issue_dates(pool_years), pool_years)
    kw = {m: ({**(LIGHT if light else {}), **(gbm_kw or {})} if m.startswith("M") else {}) for m in models}
    parts = []
    for h in holdout or cv_years:
        t0 = time.time()
        ref = [y for y in cv_years if y != h]
        df = fold_table(cube, fb, ref, cv_years, ff=ff)
        if thin > 1:
            df = df[(df.iy % thin == 0) & (df.ix % thin == 0)]
        if fc_dates:
            df = df[pd.DatetimeIndex(df.issue).strftime("%m-%d").isin(fc_dates)]
        if drop:
            df = df.drop(columns=[c for c in df.columns if c.startswith(tuple(drop)) and c not in LABELS
                                  and c not in ("year", "issue", "iy", "ix", "woy", "rz_pct")])
        is_test = (df.year == h).to_numpy()
        train, test = df[~is_test], df[is_test]
        keep = test[["year", "issue", "iy", "ix", "woy", "rz_pct", *LABELS]].copy()
        mem = member_frames(ff, fb, ref, train, test) if members and ff is not None else None
        for label in labels:
            for m in models:
                if mem and m == "M2":
                    fit = MODELS[m](**kw[m]).fit(mem[0], label)
                    pm = np.stack([fit.predict(t) for t in mem[1]], axis=1)
                    p = pm.mean(1)
                    if label in ("week1", "week2"):         # kept for the triggers (DESIGN.md O2)
                        for j in range(pm.shape[1]):
                            keep[f"M2_{label}_m{j}"] = pm[:, j].astype(np.float32)
                else:
                    p = MODELS[m](**kw[m]).fit(train, label).predict(test)
                keep[f"{m}_{label}"] = p.astype(np.float32)
        if mem:
            for c, v in mem[2].items():
                keep[c] = v.astype(np.float32)
        parts.append(keep)
        log(f"fold {h}: {len(train)} train rows, {len(test)} test rows, "
            f"week-2 base rate {100 * test.week2.mean():.2f}%, {time.time() - t0:.0f}s")
        del df, train, test
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def report(oof: pd.DataFrame, models, n_boot: int = 1000, labels=LABELS) -> pd.DataFrame:
    rows = []
    for label in labels:
        for m in models:
            s = score(oof[label], oof[f"{m}_{label}"], oof.get(f"B0_{label}"))
            if m != "B0" and f"B0_{label}" in oof:
                _, lo, hi = year_bootstrap_bss(oof.year, oof[label].to_numpy(float), oof[f"{m}_{label}"],
                                               oof[f"B0_{label}"], n_boot)
                s["bss_lo"], s["bss_hi"] = lo, hi
            rows.append({"lead": label, "model": m, **s})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["B0", "B1"])
    ap.add_argument("--leads", nargs="+", default=LABELS, choices=LABELS,
                    help="e.g. --leads week1 week2 any28 to run the headline leads first")
    ap.add_argument("--fast", action="store_true",
                    help=f"held-out years {FAST_HOLDOUT}, every other cell, light GBMs ({LIGHT})")
    ap.add_argument("--holdout", nargs="+", type=int, help="held-out years to run (default: all)")
    ap.add_argument("--thin", type=int, help="keep every n-th cell in each direction")
    ap.add_argument("--tag", default="", help="suffix for the OOF file, to keep design variants apart")
    ap.add_argument("--calibration", choices=["refit", "crossfit", "none"], help="GBM calibration (DESIGN.md A7)")
    ap.add_argument("--inner-groups", type=int, help="GBM inner calibration year-groups")
    ap.add_argument("--calibrator", choices=["isotonic", "platt"], help="GBM calibration map (DESIGN.md A7)")
    ap.add_argument("--depth", default="0-40", help="target soil depth (DESIGN.md T1; 0-40 since Oct 28)")
    ap.add_argument("--drop", nargs="+", default=[], help="feature names/prefixes to leave out (ablation)")
    ap.add_argument("--fc-dates", action="store_true", help="only issue dates with ECMWF forecasts")
    ap.add_argument("--members", action="store_true", help="M2 trained and scored member by member (P11)")
    ap.add_argument("--usdm", action="store_true", help="add the Drought Monitor features (F9; on trial)")
    ap.add_argument("--oracle", choices=["native", "coarse"],
                    help="DIAGNOSTIC: give M2 the weather that actually happened (reads the future; eval/oracle.py)")
    ap.add_argument("--oracle-skill", type=float, help="with --oracle: blur the weather to this correlation with the truth")
    args = ap.parse_args()
    holdout = args.holdout or (FAST_HOLDOUT if args.fast else None)
    thin = args.thin or (2 if args.fast else 1)
    from flashwatch.cube import load_cube
    from flashwatch.ingest import indices, nino
    cube = load_cube(depth=args.depth)
    issues = model_issue_dates(YEARS)
    from flashwatch.labels.crosscheck import usdm_cube
    fb = FeatureBuilder(cube, issues, YEARS, nino=nino.load(), usdm=usdm_cube(YEARS) if args.usdm else None,
                        **indices.load_all())
    ff = None
    import xarray as xr
    from flashwatch.forecast import FCST_STORE
    if args.oracle:
        from flashwatch.eval.oracle import PerfectWeather
        ff = PerfectWeather(cube, issues, coarse=args.oracle == "coarse", skill=args.oracle_skill)
    elif FCST_STORE.exists():
        from flashwatch.features.forecast import ForecastFeatures
        ff = ForecastFeatures(xr.open_zarr(FCST_STORE).load(), issues)
    # The builders hold everything the folds need; keep only the masks from the raw cube.
    import dataclasses
    cube = dataclasses.replace(cube, **{f: None for f in ("rzsm", "sfsm", "t2m", "vpd", "precip", "evap", "pet", "ssrd")})
    gc.collect()
    t0 = time.time()
    gbm_kw = {k: v for k, v in (("calibration", args.calibration), ("inner_groups", args.inner_groups),
                                ("calibrator", args.calibrator)) if v}
    oof = run_cv(cube, args.models, fb=fb, ff=ff, labels=args.leads, holdout=holdout, thin=thin, light=args.fast,
                 gbm_kw=gbm_kw, drop=args.drop, members=args.members,
                 fc_dates=set(pd.DatetimeIndex(xr.open_zarr(FCST_STORE).issue.values).strftime("%m-%d"))
                 if args.fc_dates else None)
    out = DATA / "oof"
    out.mkdir(parents=True, exist_ok=True)
    name = ("_".join(args.models + args.leads) + ("_fast" if args.fast else "")
            + ("" if args.depth == "0-100" else f"_d{args.depth}") + (f"_{args.tag}" if args.tag else ""))
    oof.to_parquet(out / f"{name}.parquet")
    print(f"CV took {(time.time() - t0) / 60:.1f} min; OOF in {out / name}.parquet")
    print(report(oof, args.models, labels=args.leads).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
