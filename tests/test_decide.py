import numpy as np

from flashwatch.decide import curve, irrigation, load
from flashwatch.decide.engine import Curve, daily, decide, ev_act, two_way


def test_irrigation_notice_helps_until_the_profile_is_banked():
    f = irrigation.Field()                                   # pivot below peak crop use
    assert f.capacity < f.et_peak
    c = irrigation.curve(f, 14)
    s0 = c.attrs["stress_day"]
    assert 0 < s0 < f.spell_days
    assert (np.diff(c.avoided) >= -1e-9).all()               # more notice never hurts
    assert c.avoided[3] > c.avoided[0]                        # and helps at first ...
    assert c.avoided.iloc[-1] == c.avoided.iloc[s0]          # ... but not beyond a full profile
    big = irrigation.curve(irrigation.Field(gpm_per_acre=8.0), 14)   # pivot outruns the crop
    assert np.ptp(big.avoided) < 1e-9 and big.yield_loss_pct.max() == 0


def test_water_balance_never_overfills():
    r = irrigation.simulate(irrigation.Field(gpm_per_acre=8.0), 0)
    assert r["depletion"].min() >= 0 and r["shortfall_in"] == 0


def test_daily_spreads_weeks_and_caps_total():
    p = daily({"week1": 0.14, "week2": 0.07})
    assert np.allclose(p[:7], 0.02) and np.allclose(p[7:], 0.01)
    assert np.isclose(daily({"week1": 0.9, "week2": 0.9}).sum(), 1)


def test_decide_trades_notice_against_news():
    c = Curve(np.array([0.0, 14]), np.array([0.0, 140]), np.zeros(2), cost_noevent=10.0)
    p = daily({"week1": 0.2, "week2": 0.2})
    # With no news coming, acting now beats waiting (waiting only loses notice).
    d = decide(c, p, [(1.0, 1.0)])
    assert d.act_now and d.ev_now > d.ev_wait
    # Low risk but news could double it: not yet, the upside scenario is worth acting on later.
    c2 = Curve(c.notice, c.avoided, c.cost_event, cost_noevent=6.5)
    lo = daily({"week1": 0.02, "week2": 0.04})
    d = decide(c2, lo, two_way(1.0))
    assert not d.act_now and d.ev_wait > 0 and ev_act(c2, lo) < 0
    assert np.isclose(sum(w * m for w, m in two_way(0.6)), 1)


def test_config_builds_every_action():
    for name, a in load(overrides={"gpm_per_acre": 4.0}).items():
        c = curve(a)
        assert len(c.notice) == len(c.avoided) and np.isfinite(c.cost_noevent), name
    assert load(overrides={"gpm_per_acre": 4.0})["irrigation"]["params"]["gpm_per_acre"] == 4.0


def test_backtest_credits_hits_and_charges_false_alarms():
    import pandas as pd
    from flashwatch.decide import backtest
    issues = pd.date_range("2012-06-01", "2012-08-31", freq="2D")
    p = np.where(issues == pd.Timestamp("2012-07-01"), 0.6, 0.0)
    p2 = np.where(issues == pd.Timestamp("2012-08-10"), 0.6, 0.0)       # a second start: false alarm
    oof = pd.DataFrame({"year": 2012, "issue": issues, "iy": 0, "ix": 0,
                        "M1_week1": p + p2, "M1_week2": p + p2})
    ons = pd.DataFrame({"date": pd.to_datetime(["2012-07-06"]), "iy": [0], "ix": [0]})
    act = load()["irrigation"]
    bt = backtest.run(oof, ons, "M1", act, "careful")
    c = curve(act)
    assert bt.hits[0] == 1 and bt.false_alarms[0] == 1 and bt.onsets[0] == 1
    assert np.isclose(bt.value[0], float(c.net(5)) - c.cost_noevent)
    cal = backtest.run(oof, ons, "M1", act, "calendar", calendar="07-03")
    assert cal.hits[0] == 0 and cal.false_alarms[0] == 1   # the calendar start (July 3) already covers the July 6 onset


def test_trigger_names_the_weather_that_tips_it():
    from dataclasses import replace
    from flashwatch.decide.trigger import trigger
    c = replace(curve(load()["irrigation"]), cost_noevent=11.0)     # a costly false start: the dry members tip it
    rain = np.array([0.1, 0.2, 0.3, 0.5, 0.9, 1.2, 1.5, 1.8, 2.0, 2.4, 3.0])
    p1 = np.where(rain < 0.4, 0.45, 0.03)                  # dry members carry the risk
    p2 = np.where(rain < 0.4, 0.30, 0.03)
    temp = np.full(11, 80.0) + np.arange(11) % 2            # temperature says nothing here
    t = trigger(c, p1, p2, {"rain_in": rain, "temp_f": temp}, lam=1.0)
    assert not t["act_now"] and t["members_tipping"] == 3
    assert t["trigger"]["var"] == "rain_in" and 0.3 < t["trigger"]["threshold"] < 0.5
    assert "less than 0.4 in of rain" in t["trigger"]["text"]
    calm = trigger(c, np.full(11, 0.01), np.full(11, 0.01), {"rain_in": rain})
    assert calm["trigger"] is None and calm["members_tipping"] == 0


def test_yield_shortfall_uses_trend_from_other_years():
    import pandas as pd
    from flashwatch.eval.yields import shortfall
    years = list(range(2005, 2017))
    y = [100 + 2 * (t - 2005) for t in years]
    y[7] *= 0.8                                          # one bad year, 20% below trend
    d = pd.DataFrame({"fips": "1", "crop": "corn", "practice": "all", "year": years, "yield_bu": y})
    s = shortfall(d).set_index("year").dev_pct
    assert abs(s[2012] + 20) < 0.5                  # its own trend never sees the bad year
    assert (s.drop(2012).abs() < 5).all()          # the others are only nudged by it


def _days():
    import pandas as pd
    return pd.date_range("2012-04-01", "2012-10-15").strftime("%m-%d").tolist()


def test_false_start_water_is_used_when_the_dry_spell_comes_late_and_lost_when_rain_comes():
    from flashwatch.decide import irrigation
    from flashwatch.decide.falsestart import season_runs
    days = _days()
    f = irrigation.field(**load()["irrigation"]["params"])
    start = days.index("07-01")
    eto = np.full((len(days), 2), 0.25)
    rain = np.zeros((len(days), 2))
    rain[:start, :] = 0.3                                   # full profile on July 1 at both cells
    rain[start + 7, 1] = 4.0                                # cell 1: a downpour the day after the early week
    rain[start + 8:start + 40, 1] = 0.3                     # ... and a wet month after it
    r = season_runs(rain, eto, days, f, np.array([start]), review=7, reaction=2)
    stored = r["stored"][0]
    assert np.allclose(r["dr"][0], 0) and (stored > 0.5).all()
    used, lost = r["extra_pumped"][0] * f.efficiency / stored
    assert used < 0.35                                      # dry weather follows: the crop uses most of it
    assert lost > 0.9                                       # the downpour drains it all


def test_backtest_costs_leave_out_the_year_they_score():
    import pandas as pd
    from flashwatch.decide.falsestart import backtest_costs
    rows = pd.DataFrame({"year": [2012] * 2 + [2018] * 2, "issue": pd.to_datetime(["2012-07-01"] * 2 + ["2018-07-01"] * 2),
                         "iy": 0, "ix": [0, 1, 0, 1], "dr": 0.1, "extra_pumped": [1.0, 1.0, 0.2, 0.2],
                         "onset": False})
    with_rows = pd.concat([rows] * 100, ignore_index=True)  # enough dates per month-and-deficit cell
    out = backtest_costs(10.0, rows=with_rows)
    assert np.allclose(out[out.year == 2012].expected, 2.0)   # 2012 is priced from 2018's weather
    assert np.allclose(out[out.year == 2018].expected, 10.0)
    assert np.allclose(out[out.year == 2012].actual, 10.0)
