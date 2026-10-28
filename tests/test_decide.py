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
