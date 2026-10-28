import numpy as np
import pandas as pd
import pytest


def test_m3_learns_a_spatial_pattern_and_returns_one_probability_per_row():
    pytest.importorskip("torch")
    from flashwatch.models.cnn import CNN
    rng = np.random.default_rng(0)
    rows = []
    for year in (2003, 2006, 2009, 2012):
        for d in range(40):
            issue = pd.Timestamp(f"{year}-05-01") + pd.Timedelta(days=2 * d)
            dry = rng.normal(size=(8, 10))
            for iy in range(0, 16, 2):
                for ix in range(0, 20, 2):
                    # onset when the 3x3 neighbourhood is dry: needs spatial context
                    nb = dry[max(0, iy // 2 - 1):iy // 2 + 2, max(0, ix // 2 - 1):ix // 2 + 2].mean()
                    rows.append({"year": year, "issue": issue, "iy": iy, "ix": ix, "woy": 22,
                                 "dryness": dry[iy // 2, ix // 2], "week1": int(nb > 0.4)})
    df = pd.DataFrame(rows)
    train, test = df[df.year != 2012], df[df.year == 2012]
    p = CNN(epochs=30, width=16).fit(train, "week1").predict(test)
    assert p.shape == (len(test),) and np.all((p >= 0) & (p <= 1))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(test.week1, p) > 0.75
