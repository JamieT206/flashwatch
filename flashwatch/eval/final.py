"""The one-time test-year score (DESIGN.md B; the pre-registered expectations are in docs/writeup.md): freeze the model, then score 2022-2024 once.

    python -m flashwatch.eval.final freeze --models B0 B1 H1 M1 M2 --leads week1 week2 any28 --fc-dates \
        --blend E1=M1,M3+M3b+M3c+M3d          # after the burst; needs a clean, committed tree
    python -m flashwatch.eval.final score     # once: trains on 2005-2021, predicts 2022-2024

`freeze` writes data/final/freeze.json: the models, leads and options, and the git tree hash of the
flashwatch/ code. `score` refuses to run if the code has changed since the freeze, or if the test
years have already been scored (data/final/test_scores.json exists). Training uses the development
years only, with climatology from them; the test years are only predicted.

Writes data/final/test_oof.parquet and test_scores.json, and prints the scores against B0 with
year-bootstrap ranges, M1 against B1 (expectation 1) and skill west of 98 W against east of 90 W
(expectation 4). --fc-dates applies to every model, so all are scored on the same rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time

import numpy as np
import pandas as pd

from flashwatch.config import CV_YEARS, DATA, ROOT, TEST_YEARS
from flashwatch.eval.cv import LABELS, report

DIR = DATA / "final"
WEST_IX, EAST_IX = 24, 56          # cells west of 98 W, east of 90 W (0.25 degree grid from 104 W)


def code_hash() -> str:
    """Git tree hash of the flashwatch/ package at HEAD; refuses uncommitted changes to it."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "flashwatch"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted changes under flashwatch/; commit them first:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD:flashwatch"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def freeze(cfg: dict, d=DIR, code=None) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    if (d / "freeze.json").exists():
        raise SystemExit(f"{d / 'freeze.json'} exists: the model is already frozen")
    rec = {**cfg, "code": code or code_hash(), "frozen_at": dt.datetime.now().isoformat(timespec="seconds")}
    (d / "freeze.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def check(d=DIR, code=None) -> dict:
    """The frozen config, if scoring may go ahead; otherwise stops with the reason."""
    if not (d / "freeze.json").exists():
        raise SystemExit("not frozen: run `python -m flashwatch.eval.final freeze ...` after the burst")
    if (d / "test_scores.json").exists():
        raise SystemExit(f"the test years were already scored ({d / 'test_scores.json'}); they are scored once")
    cfg = json.loads((d / "freeze.json").read_text(encoding="utf-8"))
    now = code or code_hash()
    if now != cfg["code"]:
        raise SystemExit(f"the code changed since the freeze ({cfg['code'][:10]} -> {now[:10]})")
    return cfg


def blends(oof: pd.DataFrame, spec: list[str], leads) -> list[str]:
    """NAME=term,term (each term a model or models joined by +): the equal-weight log-odds mean."""
    from flashwatch.eval.blend import mean_prob
    names = []
    for s in spec:
        name, terms = s.split("=", 1)
        for lead in leads:
            oof[f"{name}_{lead}"] = mean_prob([mean_prob([oof[f"{m}_{lead}"].to_numpy() for m in t.split("+")])
                                               for t in terms.split(",")]).astype(np.float32)
        names.append(name)
    return names


def region_skill(oof: pd.DataFrame, models, leads) -> list[dict]:
    from flashwatch.eval.metrics import year_bootstrap_bss
    rows = []
    for reg, m in (("west of 98 W", oof.ix < WEST_IX), ("east of 90 W", oof.ix >= EAST_IX)):
        s = oof[m]
        for lead in leads:
            y = s[lead].to_numpy(float)
            for mod in models:
                if mod == "B0":
                    continue
                v, lo, hi = year_bootstrap_bss(s.year, y, s[f"{mod}_{lead}"], s[f"B0_{lead}"])
                rows.append({"region": reg, "lead": lead, "model": mod, "bss": v, "lo": lo, "hi": hi})
    return rows


def run_test(cube, models, fb, ff, labels, thin=1, gbm_kw=None, drop=(), fc_dates=None, log=print) -> pd.DataFrame:
    """Train on the development years (climatology from them only), predict the test years."""
    from flashwatch.eval.cv import fold_table
    from flashwatch.models.baselines import MODELS
    t0 = time.time()
    df = fold_table(cube, fb, CV_YEARS, CV_YEARS + TEST_YEARS, ff=ff)
    if thin > 1:
        df = df[(df.iy % thin == 0) & (df.ix % thin == 0)]
    if fc_dates:
        df = df[pd.DatetimeIndex(df.issue).strftime("%m-%d").isin(fc_dates)]
    if drop:
        df = df.drop(columns=[c for c in df.columns if c.startswith(tuple(drop)) and c not in LABELS
                              and c not in ("year", "issue", "iy", "ix", "woy", "rz_pct")])
    is_test = df.year.isin(TEST_YEARS).to_numpy()
    train, test = df[~is_test], df[is_test]
    keep = test[["year", "issue", "iy", "ix", "woy", "rz_pct", *LABELS]].copy()
    for label in labels:
        for m in models:
            kw = (gbm_kw or {}) if m.startswith("M") else {}
            keep[f"{m}_{label}"] = MODELS[m](**kw).fit(train, label).predict(test).astype(np.float32)
    log(f"{len(train)} training rows (2005-2021), {len(test)} test rows (2022-2024), {time.time() - t0:.0f}s")
    return keep


def score(cfg: dict, d=DIR):
    import dataclasses
    import gc

    import xarray as xr

    from flashwatch.cube import load_cube
    from flashwatch.eval.cv import model_issue_dates
    from flashwatch.features.build import FeatureBuilder
    from flashwatch.forecast import FCST_STORE
    from flashwatch.ingest import indices, nino
    from flashwatch.config import YEARS
    cube = load_cube(depth=cfg["depth"])
    issues = model_issue_dates(YEARS)
    fb = FeatureBuilder(cube, issues, YEARS, nino=nino.load(), **indices.load_all())
    ff = None
    if FCST_STORE.exists():
        from flashwatch.features.forecast import ForecastFeatures
        ff = ForecastFeatures(xr.open_zarr(FCST_STORE).load(), issues)
    cube = dataclasses.replace(cube, **{f: None for f in ("rzsm", "sfsm", "t2m", "vpd", "precip", "evap", "pet", "ssrd")})
    gc.collect()
    fc = set(pd.DatetimeIndex(xr.open_zarr(FCST_STORE).issue.values).strftime("%m-%d")) if cfg["fc_dates"] else None
    oof = run_test(cube, cfg["models"], fb, ff, cfg["leads"], thin=cfg["thin"], gbm_kw=cfg["gbm_kw"],
                   drop=cfg["drop"], fc_dates=fc)
    models = cfg["models"] + blends(oof, cfg["blend"], cfg["leads"])
    oof.to_parquet(d / "test_oof.parquet")
    table = report(oof, models, labels=cfg["leads"])
    out = {"scored_at": dt.datetime.now().isoformat(timespec="seconds"), "freeze": cfg,
           "rows": int(len(oof)), "scores": table.to_dict("records"),
           "regions": region_skill(oof, models, cfg["leads"])}
    if {"M1", "B1"} <= set(models):
        from flashwatch.eval.metrics import year_bootstrap_bss
        out["m1_vs_b1"] = {lead: dict(zip(("bss", "lo", "hi"), year_bootstrap_bss(
            oof.year, oof[lead].to_numpy(float), oof[f"M1_{lead}"], oof[f"B1_{lead}"]))) for lead in cfg["leads"]}
    (d / "test_scores.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(out["regions"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "m1_vs_b1" in out:
        print("M1 against B1:", out["m1_vs_b1"])
    print("wrote", d / "test_scores.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--models", nargs="+", required=True)
    f.add_argument("--leads", nargs="+", default=LABELS, choices=LABELS)
    f.add_argument("--fc-dates", action="store_true", help="score on the ECMWF start dates only")
    f.add_argument("--thin", type=int, default=1)
    f.add_argument("--depth", default="0-40")
    f.add_argument("--drop", nargs="+", default=[])
    f.add_argument("--calibration", choices=["refit", "crossfit", "none"])
    f.add_argument("--calibrator", choices=["isotonic", "platt"])
    f.add_argument("--inner-groups", type=int)
    f.add_argument("--blend", nargs="+", default=[], metavar="NAME=TERMS", help="e.g. E1=M1,M3+M3b+M3c+M3d")
    f.add_argument("--note", default="", help="what the burst decided, in a line")
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "freeze":
        gbm_kw = {k: v for k, v in (("calibration", a.calibration), ("inner_groups", a.inner_groups),
                                    ("calibrator", a.calibrator)) if v}
        rec = freeze({"models": a.models, "leads": a.leads, "fc_dates": a.fc_dates, "thin": a.thin, "depth": a.depth,
                      "drop": a.drop, "gbm_kw": gbm_kw, "blend": a.blend, "note": a.note})
        print("frozen:", json.dumps(rec))
    else:
        score(check())


if __name__ == "__main__":
    main()
