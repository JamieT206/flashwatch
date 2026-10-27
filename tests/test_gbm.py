import numpy as np
import pandas as pd

from flashwatch.models.gbm import GBM, GBMForecast, feature_columns


def toy(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"year": rng.integers(2005, 2013, n), "iy": 0, "ix": 0, "woy": 20,
                       "issue": pd.Timestamp("2010-06-01"),
                       "rz_pct": rng.uniform(20, 100, n), "t2m30_z": rng.normal(size=n),
                       "f_tp_w1_z": rng.normal(size=n), "noise": rng.normal(size=n)})
    logit = -3 - 0.04 * (df.rz_pct - 40) + 0.8 * df.t2m30_z - 0.5 * df.f_tp_w1_z
    df["week1"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["any28"] = df.week1
    return df


def test_feature_columns_exclude_ids_and_labels():
    df = toy(100)
    assert feature_columns(df, forecast=False) == ["rz_pct", "t2m30_z", "noise"]
    assert "f_tp_w1_z" in feature_columns(df, forecast=True)


def test_gbm_learns_calibrated_probabilities():
    df = toy()
    train, test = df[df.year < 2011], df[df.year >= 2011]
    p1 = GBM(n_rounds=60).fit(train, "week1").predict(test)
    p2 = GBMForecast(n_rounds=60).fit(train, "week1").predict(test)
    y = test.week1.to_numpy()
    base = ((y.mean() - y) ** 2).mean()
    assert ((p1 - y) ** 2).mean() < base and ((p2 - y) ** 2).mean() < ((p1 - y) ** 2).mean()
    assert abs(p2.mean() - y.mean()) < 0.01                       # calibrated in the large
    # Monotone constraint: probability never rises with rz_pct.
    grid = test.iloc[:1].loc[[test.index[0]] * 50].copy()
    grid["rz_pct"] = np.linspace(21, 99, 50)
    p = GBM(n_rounds=60).fit(train, "week1").predict(grid)
    assert (np.diff(p) <= 1e-12).all()


def test_tuning_picks_params_and_gbm_uses_them(tmp_path, monkeypatch):
    import flashwatch.models.gbm as gbm
    from flashwatch.models import tune
    monkeypatch.setattr(gbm, "TUNED_FILE", tmp_path / "gbm_params.json")
    monkeypatch.setattr(tune, "TUNED_FILE", tmp_path / "gbm_params.json")
    monkeypatch.setattr(tune, "MAX_ROUNDS", 80)
    df = toy(20000)
    res = tune.tune(df, "week1", forecast=False, trials=3, log=lambda s: None)
    assert res["logloss"] <= res["default_logloss"] and res["n_rounds"] >= 20
    tune.save("M1", "week1", res)
    m = GBM().fit(df, "week1")
    assert m.n_rounds == res["n_rounds"]
    assert all(m.params[k] == v for k, v in res["params"].items())
    assert GBM(n_rounds=30).fit(df, "week1").n_rounds == 30      # explicit settings win


def test_calibration_modes():
    df = toy()
    train, test = df[df.year < 2011], df[df.year >= 2011]
    y = test.week1.to_numpy()
    for mode in ("refit", "crossfit", "none"):
        m = GBM(n_rounds=60, inner_groups=3, calibration=mode).fit(train, "week1")
        p = m.predict(test)
        assert len(m.boosters) == (3 if mode == "crossfit" else 1)
        assert ((p - y) ** 2).mean() < ((y.mean() - y) ** 2).mean() and abs(p.mean() - y.mean()) < 0.01
