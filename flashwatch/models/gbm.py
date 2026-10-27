"""M1 (land state only) and M2 (land state + ECMWF forecast features): LightGBM per lead,
plain log loss (no downsampling, no class weights), monotone constraints where the physics
is unambiguous, and isotonic calibration fitted on inner out-of-fold predictions.

Within each outer CV fold, the training years are split into K year-groups; K inner models
give out-of-fold raw scores on every training row, an isotonic map is fitted on those, and
the final model is refitted on all training rows (K=2 by default: QUESTIONS.md Q4). Test rows
are never seen by either step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS = {"year", "issue", "iy", "ix", "woy"}
LABEL_PREFIXES = ("week", "any28")

# +1: probability may only rise with the feature; -1: only fall. Unlisted: unconstrained.
MONOTONE = {"rz_pct": -1, "rz_pct_chg5": -1, "rz_pct_chg10": -1, "rz_pct_chg20": -1,
            "sf_pct": -1, "sf_pct_chg5": -1, "rzsm5_z": -1,
            "precip14_z": -1, "precip30_z": -1, "precip60_z": -1, "esi14": -1, "esi30": -1,
            "esi14_z": -1, "esi30_z": -1,
            "t2m14_z": 1, "t2m30_z": 1, "vpd14_z": 1, "vpd30_z": 1, "pet14_z": 1, "pet30_z": 1}
for _w in range(1, 5):
    MONOTONE.update({f"f_tp_w{_w}_z": -1, f"f_tp_w{_w}_plow": 1, f"f_t2m_w{_w}_z": 1,
                     f"f_vpd_w{_w}_z": 1, f"f_sm100_w{_w}_z": -1})

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
          "feature_fraction": 0.8, "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_bin": 63, "verbose": -1, "num_threads": 0}
N_ROUNDS = 400


def feature_columns(df: pd.DataFrame, forecast: bool) -> list[str]:
    cols = [c for c in df.columns if c not in ID_COLUMNS and not c.startswith(LABEL_PREFIXES)
            and not c.startswith(("B0_", "B1_", "M1_", "M2_", "B2_"))]
    return [c for c in cols if forecast or not c.startswith("f_")]


class GBM:
    forecast = False
    name = "M1"

    def __init__(self, params=None, n_rounds=N_ROUNDS, inner_groups: int = 2, seed: int = 0):
        self.params = {**PARAMS, **(params or {}), "seed": seed}
        self.n_rounds, self.k, self.seed = n_rounds, inner_groups, seed

    def _fit_raw(self, df, label):
        import lightgbm as lgb
        mono = [MONOTONE.get(c, 0) for c in self.cols]
        ds = lgb.Dataset(df[self.cols].to_numpy(np.float32), df[label].to_numpy(), free_raw_data=True)
        return lgb.train({**self.params, "monotone_constraints": mono}, ds, self.n_rounds)

    def fit(self, df: pd.DataFrame, label: str):
        from sklearn.isotonic import IsotonicRegression
        self.cols = feature_columns(df, self.forecast)
        years = np.array(sorted(df.year.unique()))
        groups = np.array_split(np.random.default_rng(self.seed).permutation(years), min(self.k, len(years)))
        raw = np.empty(len(df))
        for g in groups:
            inner = df.year.isin(g).to_numpy()
            raw[inner] = self._fit_raw(df[~inner], label).predict(df.loc[inner, self.cols].to_numpy(np.float32))
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, df[label].to_numpy())
        self.booster = self._fit_raw(df, label)
        return self

    def predict_raw(self, df):
        return self.booster.predict(df[self.cols].to_numpy(np.float32))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(self.predict_raw(df))


class GBMForecast(GBM):
    forecast = True
    name = "M2"
