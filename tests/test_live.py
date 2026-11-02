import numpy as np
import pandas as pd

from flashwatch.features.forecast import ForecastFeatures
from flashwatch.live.features import live_features, nearest_start, spread_scale
from flashwatch.live.run import plan
from tests.test_forecast_features import fake_fcst


def test_live_features_match_the_reforecast_features():
    """An 'open-data' ensemble that is one reforecast year's own members (weeks 1-2) gives the same features as
    ForecastFeatures.build for that year, so the live M2 sees inputs built exactly as in training."""
    years = list(range(2005, 2011))
    ds = fake_fcst(years)
    ref = years[:-1]
    want = ForecastFeatures(ds, pd.DatetimeIndex(["2010-06-15"])).build(ref)
    sel = ds.sel(issue="2025-06-15", hyear=2010).isel(week=slice(0, 2))
    ens = {v: sel[v].values for v in ("t2m", "d2m", "tp", "sm100")} | {"lat": ds.lat.values, "lon": ds.lon.values}
    got = live_features(ds, ens, "2010-06-15", ref)
    assert got and all("_w3" not in k and "_w4" not in k for k in got)
    for k, v in got.items():
        np.testing.assert_allclose(v, want[k][0], rtol=1e-4, atol=1e-4, err_msg=k)


def test_spread_scale_and_nearest_start():
    assert spread_scale(11, 11) == 1
    assert 0.94 < spread_scale(50, 11) < 0.97          # 11 members understate the spread more than 50 do
    starts = pd.to_datetime(["2025-06-13", "2025-06-15", "2025-07-03"])
    assert nearest_start(starts, pd.Timestamp("2026-06-14")) in (0, 1)
    assert nearest_start(starts, pd.Timestamp("2026-07-02")) == 2


def test_plan_keeps_the_training_latency_and_the_season(monkeypatch):
    monkeypatch.setattr("flashwatch.live.run.site_dates", lambda: ["06-29", "07-01", "07-03", "10-13", "10-15"])
    p = plan(pd.Timestamp("2026-06-28"), None)                 # ERA5-Land through Jun 28: forecast for Jul 3
    assert p == {"year": 2026, "date": "07-03", "offseason": False}
    p = plan(pd.Timestamp("2026-06-29"), None)                 # Jul 4 is not a site date: the latest before it
    assert p["date"] == "07-03"
    p = plan(pd.Timestamp("2026-10-24"), None)                 # late October: show July 3 of this season
    assert p == {"year": 2026, "date": "07-03", "offseason": True}
    p = plan(pd.Timestamp("2027-02-01"), None)                 # winter: still last season
    assert p == {"year": 2026, "date": "07-03", "offseason": True}
