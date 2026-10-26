import numpy as np
import pandas as pd

from flashwatch.labels.crosscheck import smvi_overlap, usdm_worsening


def ep_frame(rows):
    return pd.DataFrame(rows, columns=["episode", "date", "iy", "ix"]).assign(
        date=lambda d: pd.to_datetime(d.date), lat=0.0, lon=0.0)


def test_smvi_overlap_counts_matches_within_window():
    ep = ep_frame([(1, "2010-06-10", 5, 5), (1, "2010-06-12", 5, 6), (2, "2010-08-01", 20, 20)])
    smvi = {y: pd.DataFrame({"iy": [5], "ix": [6], "date": pd.to_datetime([f"{y}-06-20"])})
            for y in range(2008, 2013)}
    r = smvi_overlap(ep, smvi).iloc[0]
    assert r.observed == 0.5 and r.n_episodes == 2      # episode 1 matched (8 days), 2 not
    assert r.chance == 0.5                               # same calendar date every year here


def test_usdm_worsening():
    dates = pd.date_range("2010-06-01", periods=10, freq="7D")
    cat = np.full((10, 3, 3), -1, np.int8)
    cat[3:, :2, :] = 1                                   # cells in rows 0-1 go none -> D1
    ep = ep_frame([(1, "2010-06-09", y, x) for y in range(3) for x in range(3)])
    r = usdm_worsening(ep, dates, cat, min_cells=5).iloc[0]
    assert r.observed == 1.0 and r.n_episodes == 1       # 6 of 9 cells worsen by 2
