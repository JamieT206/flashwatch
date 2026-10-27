import numpy as np
import pandas as pd

from flashwatch.labels.episodes import episodes
from flashwatch.labels.onset import lead_labels, onset_days
from flashwatch.labels.percentile import calendar_day, percentiles, running_mean


def series(values):
    return np.array(values, dtype=float)[:, None]


def test_onset_textbook_case():
    # 25 days at 50, then 5 days falling to 15, then 15 for 12 days.
    P = series([50] * 25 + [45, 35, 28, 22, 15] + [15] * 12)
    on = onset_days(P)[:, 0]
    assert np.flatnonzero(on).tolist() == [29]


def test_onset_needs_persistence_and_prior_high():
    recovers = series([50] * 25 + [15] * 5 + [30] * 12)          # P back above 20 within 10 days
    assert not onset_days(recovers).any()
    never_high = series([35] * 25 + [15] * 12)                    # never >= 40 beforehand
    assert not onset_days(never_high).any()
    too_early = series([50] * 10 + [30] * 15 + [15] * 12)         # last >= 40 is 15 days before
    on = onset_days(too_early)[:, 0]
    assert np.flatnonzero(on).tolist() == [25]
    stale = series([50] * 5 + [30] * 21 + [15] * 12)              # last >= 40 is 21 days before
    assert not onset_days(stale).any()


def test_onset_mask_and_nan():
    P = series([50] * 25 + [15] * 12)
    mask = np.zeros_like(P, dtype=bool)
    mask[25] = True
    assert not onset_days(P, mask=mask).any()
    P[30] = np.nan                                                 # NaN breaks persistence
    assert not onset_days(P).any()


def test_lead_labels_windows():
    time = pd.date_range("2020-01-01", periods=60, freq="D")
    onset = np.zeros((60, 1), bool)
    onset[10] = True                                               # Jan 11
    lab = lead_labels(onset, time, [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-04"),
                                    pd.Timestamp("2020-01-11")])
    # tau=Jan 1: onset at +10 days -> week 2. tau=Jan 4: +7 -> week 1. tau=Jan 11: +0 -> none.
    assert lab["week1"][:, 0].tolist() == [False, True, False]
    assert lab["week2"][:, 0].tolist() == [True, False, False]
    assert lab["any28"][:, 0].tolist() == [True, True, False]


def test_running_mean_trailing():
    x = series(range(10))
    rm = running_mean(x, 5)[:, 0]
    assert np.isnan(rm[:4]).all() and rm[4] == 2.0 and rm[9] == 7.0


def test_calendar_day_leap():
    t = pd.DatetimeIndex(["2020-02-28", "2020-02-29", "2020-03-01", "2021-03-01", "2020-12-31"])
    assert calendar_day(t).tolist() == [58, 58, 59, 59, 364]


def test_percentiles_exclude_own_year_and_held_out_years():
    time = pd.date_range("2001-01-01", "2010-12-31", freq="D")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(time), 2))
    ref = range(2001, 2009)
    P = percentiles(x, time, ref_years=ref, target_years=range(2001, 2011))
    assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 100
    # Held-out years must not affect the training-year percentiles.
    x2 = x.copy()
    x2[time.year >= 2009] += 100
    P2 = percentiles(x2, time, ref_years=ref, target_years=range(2001, 2011))
    train = time.year < 2009
    assert np.array_equal(P[train], P2[train])
    # A value's own year is not in its pool: shifting one year leaves that year's ranks alone
    # only if the year is excluded; it changes other years' ranks.
    x3 = x.copy()
    x3[time.year == 2003] += 100
    P3 = percentiles(x3, time, ref_years=ref, target_years=range(2001, 2011))
    y03 = time.year == 2003
    assert (P3[y03] == 100).all()
    assert not np.array_equal(P[~y03 & train], P3[~y03 & train])


def test_episodes_link_space_and_time():
    onset = np.zeros((40, 5, 5), bool)
    onset[10, 1, 1] = True
    onset[15, 2, 2] = True     # diagonal neighbour, 5 days later -> same episode
    onset[30, 3, 3] = True     # neighbour of (2,2) but 15 days later -> new episode
    onset[10, 4, 0] = True     # same day but not adjacent -> new episode
    ids, n = episodes(onset)
    assert n == 3
    assert ids[10, 1, 1] == ids[15, 2, 2] != ids[30, 3, 3]
