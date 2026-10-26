import numpy as np
import pandas as pd

from flashwatch.eval.cv import LABELS, fold_table, report, run_cv
from flashwatch.eval.metrics import brier, bss, hit_far, reliability, year_bootstrap
from flashwatch.features.build import FeatureBuilder

YEARS = list(range(2001, 2007))


def issues(years):
    return pd.DatetimeIndex([f"{y}-{m:02d}-{d:02d}" for y in years for m in range(4, 11) for d in (1, 11, 21)])


def test_metrics_basics():
    y = np.array([0, 0, 1, 1])
    assert brier(y, y) == 0 and bss(y, [0.1, 0.1, 0.9, 0.9], [0.5] * 4) > 0.9
    pod, far = hit_far(y, [0.2, 0.6, 0.7, 0.1], 0.5)
    assert pod == 0.5 and far == 0.5
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.5, 20000)
    a, b = reliability(rng.uniform(size=p.size) < p, p)
    assert abs(a) < 0.15 and abs(b - 1) < 0.1


def test_year_bootstrap_ci_brackets_estimate():
    df = pd.DataFrame({"year": np.repeat(range(10), 50), "v": np.random.default_rng(0).normal(size=500)})
    est, lo, hi = year_bootstrap(df, lambda d: d.v.mean(), n=200)
    assert lo <= est <= hi


def test_fold_table_and_cv_end_to_end(cube):
    fb = FeatureBuilder(cube, issues(YEARS), YEARS)
    df = fold_table(cube, fb, YEARS[1:], YEARS)
    assert {"iy", "ix", "woy", "rz_pct", *LABELS}.issubset(df.columns)
    assert (df.rz_pct > 20).all()
    assert df[LABELS].isin([0, 1]).all().all()
    oof = run_cv(cube, ("B0", "B1"), cv_years=YEARS[:4], pool_years=YEARS, fb=fb, log=lambda *_: None)
    assert set(oof.year) == set(YEARS[:4])
    rep = report(oof, ("B0", "B1"), n_boot=20)
    assert len(rep) == 2 * len(LABELS) and rep.brier.between(0, 1).all()


def test_bss_bootstrap_matches_row_bootstrap():
    from flashwatch.eval.metrics import year_bootstrap_bss
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"year": np.repeat(range(12), 300)})
    df["y"] = (rng.uniform(size=len(df)) < 0.1).astype(float)
    df["p"] = np.clip(0.1 + 0.3 * (df.y - 0.1) + rng.normal(0, 0.05, len(df)), 0, 1)
    df["r"] = 0.1
    fast = year_bootstrap_bss(df.year, df.y, df.p, df.r, n=300, seed=3)
    slow = year_bootstrap(df, lambda d: bss(d.y, d.p, d.r), n=300, seed=3)
    assert np.allclose(fast, slow)
