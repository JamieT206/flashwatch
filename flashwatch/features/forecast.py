"""Forecast features from the weekly reforecast fields in data/fcst.zarr.

Per issue date, week 1-4 and variable (precipitation, 2 m temperature, VPD, 0-100 cm soil
moisture): the ensemble-mean anomaly (z against the reforecast's own climatology for that
start date and week), the ensemble spread relative to that climatology, and for
precipitation the fraction of members in the climatological lower tercile. Climatologies
use the fold's reference years minus the issue's own year, matching the observed features.
Computed on the native 1.5 degree grid, then bilinearly interpolated to the 0.25 degree
grid (which adds no information; see the spec).

Derived features (F8, Oct 28; the early M2 run used the week-2 forecasts poorly):
- weeks 1+2 together ("w12"): the same statistics on the two-week mean, which is less noisy;
- f_hotdry_w<k>_p: share of members that are both dry (precipitation in the lower tercile) and
  hot-and-dry-air (VPD in the upper tercile), i.e. the ensemble's chance of a flash-drought week;
- f_dryhot_w<k>_z: VPD anomaly minus precipitation anomaly (ensemble mean);
- f_sm100chg_w<k>_z: forecast change in 0-100 cm soil moisture from week 1 to week k (k = 2-4, 12).

Member by member (P11, `build_member`): the same z and dry-hot features for one ensemble member, so
M2 can be trained on each member's weather and its predictions averaged over the members. The
ensemble mean blurs the strong hot, dry spells that cause onsets; single members keep them.
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

    NAMES = [f"w{w}" for w in range(1, 5)] + ["w12"]

    def _extended(self) -> dict[str, np.ndarray]:
        """Weeks 1-4 plus the weeks 1+2 mean; sm100chg = soil moisture minus its week-1 value."""
        if not hasattr(self, "_ext"):
            f = {v: np.concatenate([a, a[:, :, :, :2].mean(3, keepdims=True)], 3) for v, a in self.fields.items()}
            if "sm100" in f:
                f["sm100chg"] = f["sm100"] - f["sm100"][:, :, :, :1]
            self._ext = f
        return self._ext

    def build_member(self, ref_years, m: int) -> dict[str, np.ndarray]:
        """Member m's own anomalies, z against all pooled members of the reference years (same names
        as the ensemble-mean z features), and its dry-hot index."""
        ref = np.isin(self.hyears, list(ref_years))
        have = np.where(self.idx[:, 0] >= 0)[0]
        n, grid = len(self.issues), (len(self.lats), len(self.lons))
        f, names, res = self._extended(), self.NAMES, {}
        for k in have:
            i, j = self.idx[k]
            pool = ref.copy()
            pool[j] = False
            z = {}
            for v, a in f.items():
                mem = a[i][pool]                                    # (years, M, W, la, lo)
                with np.errstate(invalid="ignore", divide="ignore"):
                    z[v] = (a[i, j, m] - mem.mean((0, 1))) / mem.std((0, 1))
                for w, nm in enumerate(names):
                    if not (v == "sm100chg" and nm == "w1"):
                        res.setdefault(f"f_{v}_{nm}_z", np.full((n,) + grid, np.nan, np.float32))[k] = z[v][w]
            if "vpd" in z:
                for w, nm in enumerate(names):
                    res.setdefault(f"f_dryhot_{nm}_z", np.full((n,) + grid, np.nan, np.float32))[k] = z["vpd"][w] - z["tp"][w]
        return {name: bilinear(r, self.lats, self.lons) for name, r in res.items()}

    def member_weather(self, m: int, week: int = 1) -> dict[str, np.ndarray]:
        """Member m's own weather in week `week`, in farmer units: rainfall total (inches) and mean
        temperature (F), on the 0.25 degree grid. For the "what are we waiting for" triggers (O2);
        not a model feature."""
        have = np.where(self.idx[:, 0] >= 0)[0]
        n, grid = len(self.issues), (len(self.lats), len(self.lons))
        rain, temp = np.full((n,) + grid, np.nan, np.float32), np.full((n,) + grid, np.nan, np.float32)
        for k in have:
            i, j = self.idx[k]
            rain[k] = self.fields["tp"][i, j, m, week - 1] * 7 / 25.4         # mm/day -> inches/week
            temp[k] = (self.fields["t2m"][i, j, m, week - 1] - 273.15) * 9 / 5 + 32
        return {"rain_in": bilinear(rain, self.lats, self.lons), "temp_f": bilinear(temp, self.lats, self.lons)}

    @property
    def n_members(self) -> int:
        return next(iter(self.fields.values())).shape[2]

    def build(self, ref_years) -> dict[str, np.ndarray]:
        ref = np.isin(self.hyears, list(ref_years))
        have = np.where(self.idx[:, 0] >= 0)[0]
        n, grid = len(self.issues), (len(self.lats), len(self.lons))
        f, names = self._extended(), self.NAMES
        res = {}

        def put(key, k, val):
            res.setdefault(key, np.full((n,) + grid, np.nan, np.float32))[k] = val

        for k in have:
            i, j = self.idx[k]
            pool = ref.copy()
            pool[j] = False                                         # never the issue's own year
            z = {}
            for v, a in f.items():
                ens = a[i].mean(1)                                  # (years, W, la, lo)
                mu, sd = ens[pool].mean(0), ens[pool].std(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    z[v] = (ens[j] - mu) / sd
                    sr = a[i, j].std(0) / sd
                for w, nm in enumerate(names):
                    if v == "sm100chg" and nm == "w1":
                        continue                                    # zero by construction
                    put(f"f_{v}_{nm}_z", k, z[v][w])
                    if v != "sm100chg":
                        put(f"f_{v}_{nm}_spread", k, sr[w])
            # Member-level terciles from the pooled reference members.
            dry, hot = self._tercile(f["tp"][i], pool, j, 1 / 3, below=True), None
            if "vpd" in f:
                hot = self._tercile(f["vpd"][i], pool, j, 2 / 3, below=False)
            for w, nm in enumerate(names):
                put(f"f_tp_{nm}_plow", k, dry[:, w].mean(0))
                if hot is not None:
                    put(f"f_hotdry_{nm}_p", k, (dry[:, w] & hot[:, w]).mean(0))
                    put(f"f_dryhot_{nm}_z", k, z["vpd"][w] - z["tp"][w])
        return {name: bilinear(r, self.lats, self.lons) for name, r in res.items()}

    @staticmethod
    def _tercile(a, pool, j, q, below):
        """Members of year j beyond the q-quantile of all pooled members: (members, W, la, lo) bool."""
        members = a[pool]                                           # (years, M, W, la, lo)
        t = np.quantile(members.reshape((-1,) + members.shape[2:]), q, axis=0)
        return a[j] < t[None] if below else a[j] > t[None]
