"""Forecast features from the weekly reforecast fields in data/fcst.zarr.

Per issue date, week 1-4 and variable (precipitation, 2 m temperature, VPD, 0-100 cm soil
moisture): the ensemble-mean anomaly (z against the reforecast's own climatology for that
start date and week), the ensemble spread relative to that climatology, and for
precipitation the fraction of members in the climatological lower tercile. Climatologies
use the fold's reference years minus the issue's own year, matching the observed features.
Computed on the native 1.5 degree grid, then bilinearly interpolated to the 0.25 degree
grid (which adds no information; see the spec).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.cube import vpd_kpa
from flashwatch.obs import LAT, LON

VARS = ("tp", "t2m", "vpd", "sm100")


def bilinear(field: np.ndarray, lats, lons) -> np.ndarray:
    """Interpolate (..., lat, lon) from a regular source grid to the 0.25 degree LAT/LON."""
    from scipy.interpolate import RegularGridInterpolator
    la_o, lo_o = np.argsort(lats), np.argsort(lons)
    src = field[..., la_o, :][..., lo_o]
    lead = src.shape[:-2]
    flat = src.reshape((-1,) + src.shape[-2:])
    gy, gx = np.meshgrid(LAT, LON, indexing="ij")
    pts = np.stack([gy.ravel(), gx.ravel()], -1)
    out = np.empty((len(flat), len(LAT) * len(LON)), np.float32)
    for i, f2 in enumerate(flat):
        out[i] = RegularGridInterpolator((np.asarray(lats)[la_o], np.asarray(lons)[lo_o]), f2,
                                         bounds_error=False, fill_value=np.nan)(pts)
    return out.reshape(lead + (len(LAT), len(LON)))


class ForecastFeatures:
    def __init__(self, fcst, issues: pd.DatetimeIndex):
        """fcst: xarray Dataset from forecast.build (issue, hyear, member, week, lat, lon)."""
        self.issues = pd.DatetimeIndex(issues)
        self.hyears = [int(y) for y in fcst.hyear.values]
        md = pd.DatetimeIndex(fcst.issue.values).strftime("%m-%d")
        self.row = {(m, y): (i, j) for i, m in enumerate(md) for j, y in enumerate(self.hyears)}
        self.lats, self.lons = fcst.lat.values, fcst.lon.values
        f = {v: fcst[v].values for v in ("tp", "t2m", "sm100") if v in fcst}
        if "t2m" in fcst and "d2m" in fcst:
            f["vpd"] = vpd_kpa(fcst["t2m"].values, fcst["d2m"].values)
        self.fields = f                                   # each (issue, hyear, member, week, lat, lon)
        self.idx = np.array([self.row.get((d.strftime("%m-%d"), d.year), (-1, -1)) for d in self.issues])

    def build(self, ref_years) -> dict[str, np.ndarray]:
        ref = np.isin(self.hyears, list(ref_years))
        have = self.idx[:, 0] >= 0
        n = len(self.issues)
        out = {}
        for v, a in self.fields.items():
            ens = a.mean(2)                                         # (I, J, W, la, lo)
            spread = a.std(2)
            res = {f"f_{v}_w{w}_{s}": np.full((n, len(self.lats), len(self.lons)), np.nan, np.float32)
                   for w in range(1, a.shape[3] + 1) for s in ("z", "spread") + (("plow",) if v == "tp" else ())}
            for k in np.where(have)[0]:
                i, j = self.idx[k]
                pool = ref.copy()
                pool[j] = False                                     # never the issue's own year
                clim = ens[i, pool]                                 # (years, W, la, lo)
                mu, sd = clim.mean(0), clim.std(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    z = (ens[i, j] - mu) / sd
                    sr = spread[i, j] / sd
                for w in range(a.shape[3]):
                    res[f"f_{v}_w{w + 1}_z"][k] = z[w]
                    res[f"f_{v}_w{w + 1}_spread"][k] = sr[w]
                if v == "tp":
                    members = a[i, pool]                            # (years, M, W, la, lo)
                    lo_t = np.quantile(members.reshape((-1,) + members.shape[2:]), 1 / 3, axis=0)
                    frac = (a[i, j] < lo_t[None]).mean(0)           # (W, la, lo)
                    for w in range(a.shape[3]):
                        res[f"f_tp_w{w + 1}_plow"][k] = frac[w]
            for name, r in res.items():
                out[name] = bilinear(r, self.lats, self.lons)
        return out
