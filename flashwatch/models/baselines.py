"""Reference forecasts B0 (climatology), B1 (logistic on current soil-moisture state) and H1
(logistic on recent rain and heat: a farmer's rule of thumb).

Each model: fit(train_df, label) then predict(df) -> probabilities. Frames carry the columns
built by eval.cv.fold_table: iy, ix (cell indices, never features), woy, and the features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


B0_FILE = Path(__file__).resolve().parents[1] / "config" / "b0_params.json"


def tuned_b0(label: str) -> dict | None:
    return json.loads(B0_FILE.read_text()).get(label) if B0_FILE.exists() else None


def b0_rate(pos: np.ndarray, cnt: np.ndarray, k: float, tw: int, sr: int) -> np.ndarray:
    """Smoothed, shrunk onset rate from (week of year, iy, ix) counts of onsets and rows."""
    from scipy.ndimage import convolve, convolve1d

    def smooth(a):
        w = (tw + 1 - np.abs(np.arange(-tw, tw + 1))).astype(float)
        a = convolve1d(a, w, axis=0, mode="wrap")
        return convolve(a, np.ones((1, 2 * sr + 1, 2 * sr + 1)), mode="constant")

    sp, sc = smooth(pos), smooth(cnt)
    regional = sp.sum((1, 2)) / np.maximum(sc.sum((1, 2)), 1)
    return (sp + k * regional[:, None, None]) / (sc + k)


class Climatology:
    """B0: onset rate per cell and week of year, smoothed over +/-tw weeks (triangular weights)
    and the (2 sr + 1)^2 neighbourhood, then shrunk towards the regional rate for that week
    (k pseudo-rows). Built without arguments it uses the per-lead values chosen by
    python -m flashwatch.models.tune_b0 (config/b0_params.json; QUESTIONS.md Q6a), else
    tw=2, sr=1, k=20."""

    name = "B0"

    def __init__(self, k: float | None = None, tw: int | None = None, sr: int | None = None):
        self.use_tuned = k is None and tw is None and sr is None
        self.k, self.tw, self.sr = k or 20.0, 2 if tw is None else tw, 1 if sr is None else sr

    def fit(self, df: pd.DataFrame, label: str):
        if self.use_tuned:
            t = tuned_b0(label)
            if t:
                self.k, self.tw, self.sr = t["k"], t["tw"], t["sr"]
        ny, nx = df.iy.max() + 1, df.ix.max() + 1
        pos = np.zeros((53, ny, nx))
        cnt = np.zeros((53, ny, nx))
        np.add.at(pos, (df.woy.to_numpy(), df.iy.to_numpy(), df.ix.to_numpy()), df[label].to_numpy())
        np.add.at(cnt, (df.woy.to_numpy(), df.iy.to_numpy(), df.ix.to_numpy()), 1)
        self.rate = b0_rate(pos, cnt, self.k, self.tw, self.sr)
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


class WeatherRule(StateLogistic):
    """H1: a farmer's rule of thumb, "it's been hot and it hasn't rained": logistic regression on
    the last two weeks' rain and temperature against normal for the date. The public comparison
    for FlashWatch (James, Oct 29), in place of our own land-only model M1."""

    name = "H1"
    inputs = ("precip14_z", "t2m14_z")


def _gbm(name, **kw):
    from flashwatch.models import gbm
    return {"M1": gbm.GBM, "M2": gbm.GBMForecast}[name](**kw)


MODELS = {"B0": Climatology, "B1": StateLogistic, "H1": WeatherRule, "M1": lambda **kw: _gbm("M1", **kw),
          "M2": lambda **kw: _gbm("M2", **kw)}
# Light GBM training for fast design iteration (eval.cv --fast); heavy training comes later.
LIGHT = {"params": {"learning_rate": 0.1}, "n_rounds": 150}
