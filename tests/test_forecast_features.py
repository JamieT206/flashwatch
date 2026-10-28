import numpy as np
import pandas as pd
import xarray as xr

from flashwatch.features.forecast import ForecastFeatures, bilinear
from flashwatch.obs import LAT, LON


def fake_fcst(years, seed=0):
    rng = np.random.default_rng(seed)
    lats = np.arange(52.0, 32.9, -1.5)
    lons = np.arange(-107.0, -76.9, 1.5)
    shape = (2, len(years), 11, 4, len(lats), len(lons))
    ds = xr.Dataset({v: (("issue", "hyear", "member", "week", "lat", "lon"), (rng.normal(size=shape) + m).astype("f4"))
                     for v, m in [("tp", 0.003), ("t2m", 290), ("d2m", 280), ("sm100", 0.3)]},
                    coords={"issue": pd.to_datetime(["2025-06-13", "2025-06-15"]), "hyear": years,
                            "member": np.arange(11), "week": [1, 2, 3, 4], "lat": lats, "lon": lons})
    return ds


def test_bilinear_reproduces_a_plane():
    lats, lons = np.arange(52.0, 32.9, -1.5), np.arange(-107.0, -76.9, 1.5)
    plane = (2 * lats[:, None] + 0.5 * lons[None, :])
    out = bilinear(plane[None], lats, lons)[0]
    assert np.allclose(out, 2 * LAT[:, None] + 0.5 * LON[None, :], atol=1e-4)


def test_forecast_features_shapes_and_isolation():
    years = list(range(2005, 2011))
    ds = fake_fcst(years)
    issues = pd.DatetimeIndex([f"{y}-06-13" for y in years] + ["2007-06-14"])     # last has no forecast
    ff = ForecastFeatures(ds, issues)
    ref = [2005, 2006, 2008, 2009, 2010]                                          # 2007 held out
    f0 = ff.build(ref)
    assert f0["f_tp_w1_z"].shape == (len(issues), 52, 96)
    assert np.isnan(f0["f_tp_w1_z"][-1]).all()
    plow = f0["f_tp_w2_plow"][:-1]
    assert np.nanmin(plow) >= 0 and np.nanmax(plow) <= 1 and 0.15 < np.nanmean(plow) < 0.5
    hd = f0["f_hotdry_w12_p"][:-1]                         # independent noise: ~1/3 x 1/3 of members
    assert np.nanmin(hd) >= 0 and np.nanmax(hd) <= 1 and 0.03 < np.nanmean(hd) < 0.25
    assert "f_sm100chg_w3_z" in f0 and "f_sm100chg_w1_z" not in f0 and "f_dryhot_w2_z" in f0
    # Perturbing the held-out year's forecasts must not change training-year features.
    ds2 = ds.copy(deep=True)
    for v in ("tp", "t2m", "d2m", "sm100"):
        ds2[v].loc[dict(hyear=2007)] = ds2[v].sel(hyear=2007) * 3 + 1
    f1 = ForecastFeatures(ds2, issues).build(ref)
    train = np.array([d.year != 2007 for d in issues])
    for k in f0:
        np.testing.assert_array_equal(f0[k][train], f1[k][train], err_msg=k)


def test_member_features_match_names_and_isolation():
    years = list(range(2005, 2011))
    ds = fake_fcst(years)
    issues = pd.DatetimeIndex([f"{y}-06-13" for y in years])
    ff = ForecastFeatures(ds, issues)
    ref = [2005, 2006, 2008, 2009, 2010]
    ens, m0, m1 = ff.build(ref), ff.build_member(ref, 0), ff.build_member(ref, 1)
    assert set(m0) <= set(ens) and "f_tp_w2_z" in m0 and "f_dryhot_w12_z" in m0
    assert not np.allclose(m0["f_tp_w1_z"], m1["f_tp_w1_z"], equal_nan=True)     # members differ
    # Members are noisier than the ensemble mean but on the same z scale (pooled member climatology).
    assert 0.5 < np.nanstd(m0["f_tp_w1_z"]) < 2
    ds2 = ds.copy(deep=True)
    for v in ("tp", "t2m", "d2m", "sm100"):
        ds2[v].loc[dict(hyear=2007)] = ds2[v].sel(hyear=2007) * 3 + 1
    p0 = ForecastFeatures(ds2, issues).build_member(ref, 0)
    train = np.array([d.year != 2007 for d in issues])
    for k in m0:
        np.testing.assert_array_equal(m0[k][train], p0[k][train], err_msg=k)
