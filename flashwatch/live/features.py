"""Forecast features for a live issue date: the open-data ensemble (live/opendata.py) against the reforecast
climatology of the nearest reforecast start date in data/fcst.zarr, as ForecastFeatures.build does for the
reforecasts themselves. Weeks 1, 2 and 1+2 only: the open data runs 15 days, so the live M2 is trained
without the week 3-4 features (live/run.py).

Two differences from training, both small: the live ensemble has 50 members, the reforecasts 11, so its spread
is scaled by the expected ratio of the two sample standard deviations; and the live run is the current IFS
cycle while the climatology comes from the Cy49r1 reforecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.cube import vpd_kpa
from flashwatch.features.forecast import ForecastFeatures, bilinear

WEEKS = {"w1": 0, "w2": 1, "w12": 4}            # positions in ForecastFeatures._extended() (week 1-4, then 1+2)


def spread_scale(n_live: int, n_ref: int) -> float:
    """Ratio of the expected sample (ddof=0) standard deviations of n_ref and n_live normal draws."""
    from math import exp, lgamma, sqrt
    c = lambda n: sqrt(2 / n) * exp(lgamma(n / 2) - lgamma((n - 1) / 2))  # noqa: E731
    return c(n_ref) / c(n_live)


def nearest_start(fcst_issues, issue: pd.Timestamp) -> int:
    """Index of the reforecast start date (month-day) closest to the issue's, within the season."""
    md = pd.DatetimeIndex(fcst_issues)
    doy = np.array([pd.Timestamp(2001, d.month, d.day).day_of_year for d in md])
    return int(np.argmin(np.abs(doy - pd.Timestamp(2001, issue.month, issue.day).day_of_year)))


def live_features(fcst, ens: dict[str, np.ndarray], issue, ref_years) -> dict[str, np.ndarray]:
    """{feature name: (lat, lon) on the 0.25 degree grid} for one issue date. fcst: data/fcst.zarr loaded;
    ens: opendata.load() (member, week 2, lat, lon) on fcst's grid."""
    issue = pd.Timestamp(issue)
    if not (np.allclose(ens["lat"], fcst.lat.values) and np.allclose(ens["lon"], fcst.lon.values)):
        raise ValueError("live ensemble is not on the reforecast grid")
    ff = ForecastFeatures(fcst, pd.DatetimeIndex([issue]))
    i = nearest_start(fcst.issue.values, issue)
    ref = np.isin(ff.hyears, list(ref_years))
    clim = {v: a[i][ref] for v, a in ff._extended().items()}               # (years, 11, W+1, la, lo)
    live = {v: ens[v].astype(np.float32) for v in ("tp", "t2m", "sm100")}
    live["vpd"] = vpd_kpa(ens["t2m"], ens["d2m"])
    live = {v: np.concatenate([a, a.mean(1, keepdims=True)], 1) for v, a in live.items()}   # weeks 1, 2, 1+2
    live["sm100chg"] = live["sm100"] - live["sm100"][:, :1]
    k = spread_scale(ens["t2m"].shape[0], clim["t2m"].shape[1])
    out = {}
    z = {}
    for v, a in live.items():
        ens_ref = clim[v].mean(1)                                          # (years, W+1, la, lo)
        for nm, w in WEEKS.items():
            mu, sd = ens_ref[:, w].mean(0), ens_ref[:, w].std(0)
            lw = a[:, list(WEEKS).index(nm)]
            with np.errstate(invalid="ignore", divide="ignore"):
                z[(v, nm)] = (lw.mean(0) - mu) / sd
                if not (v == "sm100chg" and nm == "w1"):
                    out[f"f_{v}_{nm}_z"] = z[(v, nm)]
                if v != "sm100chg":
                    out[f"f_{v}_{nm}_spread"] = k * lw.std(0) / sd
    for nm, w in WEEKS.items():
        j = list(WEEKS).index(nm)
        members_ref = lambda v: clim[v][:, :, w].reshape((-1,) + clim[v].shape[3:])  # noqa: E731
        dry = live["tp"][:, j] < np.quantile(members_ref("tp"), 1 / 3, axis=0)[None]
        hot = live["vpd"][:, j] > np.quantile(members_ref("vpd"), 2 / 3, axis=0)[None]
        out[f"f_tp_{nm}_plow"] = dry.mean(0)
        out[f"f_hotdry_{nm}_p"] = (dry & hot).mean(0)
        out[f"f_dryhot_{nm}_z"] = z[("vpd", nm)] - z[("tp", nm)]
    names = sorted(out)
    grid = bilinear(np.stack([out[n] for n in names]).astype(np.float32), fcst.lat.values, fcst.lon.values)
    return dict(zip(names, grid))
