import numpy as np
import pandas as pd

from flashwatch.eval.lead import summarise, usdm_response


def test_monitor_response_dates_and_lead():
    dates = pd.date_range("2012-04-03", periods=20, freq="7D")       # Tuesdays
    cat = np.full((20, 1, 2), -1, np.int8)
    cat[6:, 0, 0] = 0                                                # cell 0: D0 from 2012-05-15
    cat[8:, 0, 0] = 1                                                # D1 from 2012-05-29
    cat[:, 0, 1] = 2                                                 # cell 1: already D2, never worse
    onset = pd.DatetimeIndex(["2012-05-10", "2012-05-10"])
    worse, d1, base = usdm_response(onset, [0, 0], [0, 1], dates, cat)
    assert list(base) == [-1, 2]
    assert worse[0] == pd.Timestamp("2012-05-17") and d1[0] == pd.Timestamp("2012-05-31")   # public +2 days
    assert pd.isna(worse[1]) and pd.isna(d1[1])
    s = summarise(onset + pd.Timedelta(days=5), worse)              # crossing seen 5 days later
    assert s["median_days_ahead"] == 2 and s["monitor_never_within_12wk"] == 0.5
