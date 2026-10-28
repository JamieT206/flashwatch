import numpy as np

from flashwatch.decide import irrigation
from flashwatch.decide.waterbook import account, eto_pm, kc, status


def test_kc_follows_the_growth_stage():
    assert kc("corn", "05-01") == 0.3
    assert kc("corn", "07-20") == 1.2
    assert 0.3 < kc("corn", "06-15") < 1.2
    assert kc("soybeans", "08-01") == 1.15


def test_penman_monteith_gives_summer_values_in_range():
    # A hot, sunny Nebraska July day: 27 C, dewpoint 15 C, 27 MJ/m2 of sun -> roughly 6-8 mm.
    e = eto_pm(300.15, 288.15, 27e6, 41.0, 195)
    assert 5 < float(e) < 9
    # Cool and cloudy: much less.
    assert float(eto_pm(288.15, 283.15, 10e6, 41.0, 140)) < 3


def test_account_dries_without_rain_and_refills_with_irrigation():
    days = [f"07-{d:02d}" for d in range(1, 21)]
    dry = account(days, [0] * 20, [0.3] * 20, "corn", taw_in=6.0)
    assert np.all(np.diff(dry["deficit"]) >= 0) and dry["deficit"][-1] > 3
    wet = account(days, [0] * 20, [0.3] * 20, "corn", taw_in=6.0, irrigation={"07-10": 2.0})
    assert wet["deficit"][-1] < dry["deficit"][-1]
    # Rain can't fill beyond full.
    full = account(days[:2], [5, 0], [0.3, 0.3], "corn", taw_in=6.0)
    assert full["deficit"][0] == 0
    s = status(2.0, 3.3, 6.0, 0.3, 0.23)
    assert round(s["days_to_stress"], 1) == round(1.3 / 0.3, 1) and s["share_full"] == 1 - 2 / 6


def test_a_drier_field_gains_more_from_warning():
    full = irrigation.curve(irrigation.field(spell_days=21))
    dry = irrigation.curve(irrigation.field(spell_days=21, dr0=2.5))
    assert irrigation.stress_day(irrigation.field(dr0=2.5)) < irrigation.stress_day(irrigation.field())
    assert dry.avoided.iloc[3] >= full.avoided.iloc[3]
