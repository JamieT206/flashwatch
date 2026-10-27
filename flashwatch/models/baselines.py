"""Reference forecasts B0 (climatology) and B1 (logistic on current soil-moisture state).

Each model: fit(train_df, label) then predict(df) -> probabilities. Frames carry the columns
built by eval.cv.fold_table: iy, ix (cell indices, never features), woy, and the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Climatology:
    """B0: onset rate per cell and week of year, smoothed over +/-2 weeks and the 3x3
    neighbourhood, then shrunk towards the regional rate for that week (k pseudo-rows)."""

    name = "B0"

    def __init__(self, k: float = 20.0):
        self.k = k

    def fit(self, df: pd.DataFrame, label: str):
        ny, nx = df.iy.max() + 1, df.ix.max() + 1
        pos = np.zeros((53, ny, nx))
        cnt = np.zeros((53, ny, nx))
        np.add.at(pos, (df.woy.to_numpy(), df.iy.to_numpy(), df.ix.to_numpy()), df[label].to_numpy())
        np.add.at(cnt, (df.woy.to_numpy(), df.iy.to_numpy(), df.ix.to_numpy()), 1)

        def smooth(a):
            w = np.array([1, 2, 3, 2, 1], float)
            a = sum(wi * np.roll(a, s, axis=0) for wi, s in zip(w, range(-2, 3)))
            pad = np.pad(a, ((0, 0), (1, 1), (1, 1)))
            return sum(pad[:, 1 + dy:1 + dy + a.shape[1], 1 + dx:1 + dx + a.shape[2]]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1))

        sp, sc = smooth(pos), smooth(cnt)
        regional = sp.sum((1, 2)) / np.maximum(sc.sum((1, 2)), 1)
        self.rate = (sp + self.k * regional[:, None, None]) / (sc + self.k)
        self.default = float(df[label].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        iy, ix = df.iy.to_numpy(), df.ix.to_numpy()
        ok = (iy < self.rate.shape[1]) & (ix < self.rate.shape[2])
        out = np.full(len(df), self.default)
        out[ok] = self.rate[df.woy.to_numpy()[ok], iy[ok], ix[ok]]
        return out


class StateLogistic:
    """B1: logistic regression on the current root-zone percentile and its 10/20-day change.
    Missing inputs are filled with training medians (B1 only; the GBMs keep NaN)."""

    name = "B1"
    inputs = ("rz_pct", "rz_pct_chg10", "rz_pct_chg20")

    def fit(self, df: pd.DataFrame, label: str):
        from sklearn.linear_model import LogisticRegression
        X = df[list(self.inputs)]
        self.fill = X.median()
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(X.fillna(self.fill), df[label].astype(int))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[list(self.inputs)].fillna(self.fill))[:, 1]


def _gbm(name, **kw):
    from flashwatch.models import gbm
    return {"M1": gbm.GBM, "M2": gbm.GBMForecast}[name](**kw)


MODELS = {"B0": Climatology, "B1": StateLogistic, "M1": lambda **kw: _gbm("M1", **kw),
          "M2": lambda **kw: _gbm("M2", **kw)}
# Light GBM training for fast design iteration (eval.cv --fast); heavy training comes later.
LIGHT = {"params": {"learning_rate": 0.1}, "n_rounds": 150}
