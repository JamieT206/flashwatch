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


def test_index_features_ignore_values_published_after_issue(cube):
    """MJO (daily, 2-day latency) and GISTEMP 12-month mean (monthly, 15-day latency): changing
    anything published after the issue date must not change the features."""
    days = pd.date_range("2000-01-01", "2007-12-31", freq="D")
    rng = np.random.default_rng(1)
    mjo = pd.DataFrame(rng.normal(size=(len(days), 4)), index=days, columns=["pc1", "pc2", "amplitude", "angle"])
    months = pd.date_range("1999-01-01", "2007-12-01", freq="MS")
    gis = pd.Series(rng.normal(size=len(months)), index=months)
    f0 = FeatureBuilder(cube, ISSUES, YEARS, gistemp=gis, mjo=mjo).build(YEARS)
    for i, issue in enumerate(ISSUES):
        mjo1, gis1 = mjo.copy(), gis.copy()
        mjo1[mjo1.index > as_of("mjo", issue)] = 99.0
        gis1[gis1.index + pd.offsets.MonthEnd(0) > as_of("gistemp", issue)] = 99.0
        f1 = FeatureBuilder(cube, ISSUES[i:i + 1], YEARS, gistemp=gis1, mjo=mjo1).build(YEARS)
        for k in ("gmst12", "mjo_pc1", "mjo_amplitude"):
            np.testing.assert_array_equal(f0[k][i], f1[k][0], err_msg=f"{k} leaks at {issue.date()}")
        assert abs(f1["mjo_pc1"][0].flat[0]) < 90 and abs(f1["gmst12"][0].flat[0]) < 90


def test_box_mean_directions():
    from flashwatch.features.build import DIRECTIONS, box_mean
    a = np.zeros((1, 20, 20))
    a[0, 10, 4] = 30.0                                     # 6 columns west of (10, 10)
    w = box_mean(a, *DIRECTIONS["W"])[0, 10, 10]
    e = box_mean(a, *DIRECTIONS["E"])[0, 10, 10]
    assert w == 1.0 and e == 0.0                           # 30 cells in the west box, one of them 30
    edge = box_mean(np.ones((1, 5, 5)), *DIRECTIONS["W"])[0, 2, 0]
    assert np.isnan(edge)                                  # nothing to the west of the first column


def test_usdm_features_ignore_maps_published_after_issue(cube):
    dates = pd.date_range("2000-01-04", "2007-12-31", freq="7D")       # Tuesdays
    rng = np.random.default_rng(2)
    cat = rng.integers(-1, 5, size=(len(dates),) + cube.shape).astype(np.int8)
    f0 = FeatureBuilder(cube, ISSUES, YEARS, usdm=(dates, cat)).build(YEARS)
    for i, issue in enumerate(ISSUES):
        c1 = cat.copy()
        c1[dates > as_of("usdm", issue)] = 4
        f1 = FeatureBuilder(cube, ISSUES[i:i + 1], YEARS, usdm=(dates, c1)).build(YEARS)
        for k in ("usdm_cat", "usdm_chg4w", "nbr_usdm_d1"):
            np.testing.assert_array_equal(f0[k][i], f1[k][0], err_msg=f"{k} leaks at {issue.date()}")
