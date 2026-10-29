import numpy as np
import pandas as pd


def test_b2_recalibrates_the_raw_ecmwf_share():
    from flashwatch.models.baselines import MODELS
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 12, 4000) / 11                    # share of 11 members
    y = (rng.random(4000) < 0.02 + 0.3 * raw ** 2).astype(int)   # raw overstates the chance
    df = pd.DataFrame({"B2_raw_week1": raw, "week1": y})
    df.loc[:9, "B2_raw_week1"] = np.nan                     # rows without a forecast
    b2raw, b2 = MODELS["B2raw"]().fit(df, "week1"), MODELS["B2"]().fit(df, "week1")
    p_raw, p = b2raw.predict(df), b2.predict(df)
    assert np.allclose(p_raw[10:], raw[10:]) and np.allclose(p[:10], y.mean())
    assert abs(p.mean() - y.mean()) < 0.01 and p_raw[10:].mean() > 2 * y.mean()
    assert np.all(np.diff(p[10:][np.argsort(raw[10:])]) >= -1e-12)    # monotone in the raw share
