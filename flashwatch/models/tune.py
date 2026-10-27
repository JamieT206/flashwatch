"""Hyperparameters tuned once per lead on the development years (QUESTIONS.md Q3, option a).

    python -m flashwatch.models.tune --leads week1 week2 any28          # M1
    python -m flashwatch.models.tune --forecast                          # M2, once fcst.zarr is complete

One table over the CV years (climatology from all of them), thinned to every other cell in
each direction (25% of cells). Random search: each trial is scored by raw log loss over K
year-group folds, with early stopping on the held-out group choosing the rounds; trial 0 is
the default PARAMS. The best parameters per model and lead go to config/gbm_params.json,
which GBM reads. The outer leave-one-year-out CV reuses them, so its scores carry a small
optimistic bias (stated in the model card); the test years are never touched.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from flashwatch.models.gbm import MONOTONE, PARAMS, TUNED_FILE, feature_columns

MAX_ROUNDS = 1500
EARLY_STOP = 50


def sample(rng) -> dict:
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    return {"num_leaves": int(round(lu(15, 255))), "min_data_in_leaf": int(round(lu(100, 5000))),
            "feature_fraction": float(rng.uniform(0.5, 1.0)), "bagging_fraction": float(rng.uniform(0.3, 0.9)),
            "lambda_l2": lu(0.1, 100)}


def evaluate(df: pd.DataFrame, label: str, cols: list[str], params: dict, groups) -> tuple[float, int]:
    """Mean held-out log loss over the year-groups (pooled over rows) and the mean best round."""
    import lightgbm as lgb
    mono = [MONOTONE.get(c, 0) for c in cols]
    X, y = df[cols].to_numpy(np.float32), df[label].to_numpy()
    loss, n, rounds = 0.0, 0, []
    for g in groups:
        va = df.year.isin(g).to_numpy()
        dtr = lgb.Dataset(X[~va], y[~va], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        b = lgb.train({**PARAMS, **params, "monotone_constraints": mono, "metric": "binary_logloss"}, dtr,
                      MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        p = np.clip(b.predict(X[va], num_iteration=b.best_iteration), 1e-7, 1 - 1e-7)
        loss += -(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)).sum()
        n += va.sum()
        rounds.append(b.best_iteration)
    return loss / n, int(np.mean(rounds))


def tune(df: pd.DataFrame, label: str, forecast: bool, trials: int = 30, k: int = 4, seed: int = 0, log=print) -> dict:
    cols = feature_columns(df, forecast)
    rng = np.random.default_rng(seed)
    years = np.array(sorted(df.year.unique()))
    groups = np.array_split(rng.permutation(years), min(k, len(years)))
    best, default = None, None
    for t in range(trials):
        params = {} if t == 0 else sample(rng)
        t0 = time.time()
        loss, rounds = evaluate(df, label, cols, params, groups)
        log(f"{label} trial {t:2d}: logloss {loss:.5f}, {rounds} rounds, {time.time() - t0:.0f}s {params}")
        if t == 0:
            default = loss
        if best is None or loss < best["logloss"]:
            best = {"params": params, "n_rounds": max(rounds, 20), "logloss": loss}
    return {**best, "default_logloss": default, "trials": trials, "rows": len(df)}


def save(model: str, label: str, result: dict):
    tuned = json.loads(TUNED_FILE.read_text()) if TUNED_FILE.exists() else {}
    tuned.setdefault(model, {})[label] = result
    TUNED_FILE.write_text(json.dumps(tuned, indent=1))


def main():
    from flashwatch.eval.cv import LABELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", action="store_true", help="tune M2 (needs data/fcst.zarr)")
    ap.add_argument("--leads", nargs="+", default=LABELS, choices=LABELS)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    import dataclasses
    import gc

    from flashwatch.config import CV_YEARS, YEARS
    from flashwatch.cube import load_cube
    from flashwatch.eval.cv import fold_table, model_issue_dates
    from flashwatch.features.build import FeatureBuilder
    from flashwatch.ingest import indices, nino

    cube = load_cube()
    issues = model_issue_dates(YEARS)
    fb = FeatureBuilder(cube, issues, YEARS, nino=nino.load(), **indices.load_all())
    ff = None
    if args.forecast:
        import xarray as xr
        from flashwatch.features.forecast import ForecastFeatures
        from flashwatch.forecast import FCST_STORE
        ff = ForecastFeatures(xr.open_zarr(FCST_STORE).load(), issues)
    cube = dataclasses.replace(cube, **{f: None for f in ("rzsm", "sfsm", "t2m", "vpd", "precip", "evap", "pet", "ssrd")})
    gc.collect()
    df = fold_table(cube, fb, CV_YEARS, CV_YEARS, ff=ff)
    df = df[(df.iy % 2 == 0) & (df.ix % 2 == 0)].reset_index(drop=True)
    if ff is not None:      # tune only on issue dates that have forecasts
        df = df[df.filter(like="f_").notna().any(axis=1)].reset_index(drop=True)
    del fb, ff
    gc.collect()
    model = "M2" if args.forecast else "M1"
    print(f"{model}: {len(df)} rows (25% of cells), years {df.year.min()}-{df.year.max()}", flush=True)
    for label in args.leads:
        res = tune(df, label, args.forecast, args.trials, log=lambda s: print(s, flush=True))
        save(model, label, res)
        print(f"{model} {label}: best logloss {res['logloss']:.5f} vs default {res['default_logloss']:.5f}, "
              f"{res['n_rounds']} rounds, {res['params']}", flush=True)


if __name__ == "__main__":
    main()
