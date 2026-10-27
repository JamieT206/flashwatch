import dataclasses

import numpy as np
import pandas as pd
import pytest

from flashwatch.features.asof import as_of
from flashwatch.features.build import FeatureBuilder, season_days
from flashwatch.features.climatology import trailing_mean
from flashwatch.labels.percentile import PercentileCounts, percentiles

YEARS = list(range(2001, 2007))
ISSUES = pd.DatetimeIndex([f"{y}-{md}" for y in YEARS for md in ("04-01", "06-15", "10-15")])
FIELDS = ["rzsm", "sfsm", "t2m", "vpd", "precip", "evap", "pet", "ssrd"]


@pytest.fixture(scope="module")
def fb(cube):
    return FeatureBuilder(cube, ISSUES, YEARS)


def test_counts_match_direct_percentiles(cube):
    x = trailing_mean(cube.rzsm, 5)
    days = season_days(cube.time)
    pc = PercentileCounts(x, cube.time, YEARS, days)
    ref = [2001, 2002, 2004, 2005]                      # 2003 held out, 2006 unused
    direct = percentiles(x, cube.time, ref_years=ref, target_years=YEARS)
    fold = pc.fold(ref)
    m = days & np.isin(cube.time.year, YEARS)
    np.testing.assert_allclose(fold[m], direct[m], rtol=0, atol=1e-4)


def perturbed(cube, mask_time):
    """Copy of the cube with every field replaced by noise on the days in mask_time."""
    rng = np.random.default_rng(42)
    new = {}
    for f in FIELDS:
        a = getattr(cube, f).copy()
        a[mask_time] = rng.normal(size=a[mask_time].shape) * np.nanstd(a) + np.nanmean(a)
        new[f] = a
    return dataclasses.replace(cube, **new)


def test_features_ignore_own_year_future(cube, fb):
    ref = YEARS[:-1]
    f0 = fb.build(ref)
    for i, issue in enumerate(ISSUES):
        if issue.year == YEARS[-1]:
            continue
        cutoff = as_of("era5land", issue)
        future = (cube.time > cutoff) & (cube.time.year == issue.year)
        fb1 = FeatureBuilder(perturbed(cube, future), ISSUES[i:i + 1], YEARS)
        f1 = fb1.build(ref)
        for k in f0:
            np.testing.assert_array_equal(f0[k][i], f1[k][0], err_msg=f"{k} leaks the future at {issue.date()}")


def test_features_ignore_held_out_year(cube, fb):
    held = 2004
    ref = [y for y in YEARS if y != held]
    train = ISSUES.year != held
    f0 = fb.build(ref)
    fb1 = FeatureBuilder(perturbed(cube, cube.time.year == held), ISSUES, YEARS)
    f1 = fb1.build(ref)
    for k in f0:
        np.testing.assert_array_equal(f0[k][train], f1[k][train], err_msg=f"{k} sees the held-out year")


def test_feature_sanity(fb):
    f = fb.build(YEARS)
    assert len(f) >= 25
    assert np.nanmin(f["rz_pct"]) >= 0 and np.nanmax(f["rz_pct"]) <= 100
    assert np.isfinite(f["t2m30_z"]).mean() > 0.95
    assert f["rz_days_since_p40"].max() <= 61
