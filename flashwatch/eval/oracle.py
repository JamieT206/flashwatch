"""DIAGNOSTIC ONLY: a "perfect weather forecast" for M2 (DESIGN.md P10). Reads the future on purpose.

    python -m flashwatch.eval.cv --models B0 M1 M2 --leads week1 week2 any28 --fast --oracle coarse --tag oracle_coarse

Gives M2 the weather that actually happened in forecast weeks 1-4 (ERA5-Land precipitation,
2 m temperature and VPD), shaped like the real forecast features (f_<var>_w<week>_z: anomaly
z-scores against the same start date in the fold's reference years, own year excluded). No
soil moisture: the future soil state would hand the model the label.
- native: at the 0.25 degree cell, the most a forecast could ever give;
- coarse: smoothed over 7x7 cells (about 1.75 degrees), close to what the 1.5 degree ECMWF
  grid could resolve with perfect skill.
M2 with this minus M1 is the ceiling on what weather forecasts can add. It never feeds a real model.

skill=r blurs the perfect forecast into an imperfect one: z' = r z + sqrt(1 - r^2) e, where e is
large-scale noise (smoothed over about 4 degrees, like real forecast errors, then rescaled to unit
spread), so z' correlates with the truth at about r. Running a few values of r maps how
flash-drought skill depends on forecast quality (DESIGN.md P10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.features.build import nbr_mean

WEEKS = 4
FIELDS = {"tp": "precip", "t2m": "t2m", "vpd": "vpd"}


class PerfectWeather:
    def __init__(self, cube, issues: pd.DatetimeIndex, coarse: bool = False, skill: float | None = None):
        self.skill = skill
        self.issues = pd.DatetimeIndex(issues)
        self.md = np.asarray(self.issues.strftime("%m-%d"))
        self.year = self.issues.year.to_numpy()
        # Forecast week w covers days 7(w-1) .. 7w-1 from the issue date (forecast.py).
        offs = pd.to_timedelta(np.arange(7 * WEEKS), "D")
        pos = np.stack([cube.time.get_indexer(d + offs) for d in self.issues])     # (issue, day)
        self.weekly = {}
        for name, attr in FIELDS.items():
            a = getattr(cube, attr)
            out = np.full((len(self.issues), WEEKS) + a.shape[1:], np.nan, np.float32)
            for k, p in enumerate(pos):
                for w in range(WEEKS):
                    days = p[7 * w:7 * (w + 1)]
                    if (days >= 0).all():
                        out[k, w] = a[days].mean(0)
            self.weekly[name] = nbr_mean(out, 3).astype(np.float32) if coarse else out
        self.noise = {}
        if skill is not None:
            rng = np.random.default_rng(0)                  # same noise in every fold
            for name, a in self.weekly.items():
                e = nbr_mean(rng.standard_normal(a.shape).astype(np.float32), 8)
                self.noise[name] = (e / e.std()).astype(np.float32)

    def build(self, ref_years) -> dict[str, np.ndarray]:
        ref = np.isin(self.year, list(ref_years))
        out = {}
        for name, a in self.weekly.items():
            z = np.full_like(a, np.nan)
            for md in np.unique(self.md):
                idx = np.nonzero(self.md == md)[0]
                for k in idx:
                    pool = idx[ref[idx] & (self.year[idx] != self.year[k])]
                    if len(pool) < 3:
                        continue
                    mu, sd = a[pool].mean(0), a[pool].std(0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        z[k] = (a[k] - mu) / sd
            if self.skill is not None:
                z = self.skill * z + np.sqrt(1 - self.skill ** 2) * self.noise[name]
            for w in range(WEEKS):
                out[f"f_{name}_w{w + 1}_z"] = z[:, w]
        return out
