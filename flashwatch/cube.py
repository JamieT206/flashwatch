"""In-memory daily cube for the study years, with the derived fields labels and features share.

Arrays are (time, lat, lon) float32 on the 0.25 degree grid. Built from data/obs.zarr by
load_cube(), or directly from arrays by Cube(...) in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flashwatch.config import YEARS


def vpd_kpa(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """Vapour pressure deficit from daily-mean 2 m temperature and dewpoint (Tetens, kPa)."""
    def es(tk):
        tc = tk - 273.15
        return 0.6108 * np.exp(17.27 * tc / (tc + 237.3))
    return (es(t2m_k) - es(d2m_k)).astype(np.float32)


@dataclass
class Cube:
    time: pd.DatetimeIndex
    rzsm: np.ndarray                 # target-depth soil moisture (0-100 cm unless load_cube(depth=...))
    sfsm: np.ndarray                 # surface layer (0-7 cm)
    t2m: np.ndarray                  # K
    vpd: np.ndarray                  # kPa
    precip: np.ndarray               # m/day
    evap: np.ndarray                 # m/day, ERA5 sign: negative = evaporation
    pet: np.ndarray                  # m/day, same sign convention
    ssrd: np.ndarray                 # J/m2/day
    masked: np.ndarray               # bool: snow or frozen soil that day
    land: np.ndarray                 # bool (lat, lon): cell used at all
    extra: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.rzsm.shape[1:]


def load_cube(years=YEARS, min_land=0.2, depth: str = "0-100") -> Cube:
    """depth: the target soil depth (DESIGN.md T1); rzsm, its percentiles, the onset labels and
    the rz_* features all follow it."""
    from flashwatch.labels.__main__ import REFERENCE_DEPTH, soil_moisture
    from flashwatch.labels.onset import mask_days
    from flashwatch.obs import open_obs

    ds = open_obs().sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))
    v = lambda name: ds[name].values.astype(np.float32)  # noqa: E731
    t2m = v("temperature_2m")
    rz = v("rzsm") if depth == REFERENCE_DEPTH else soil_moisture(ds, depth).astype(np.float32)
    return Cube(time=pd.DatetimeIndex(ds.time.values), rzsm=rz,
                sfsm=v("volumetric_soil_water_layer_1"), t2m=t2m,
                vpd=vpd_kpa(t2m, v("dewpoint_temperature_2m")), precip=v("total_precipitation_sum"),
                evap=v("total_evaporation_sum"), pet=v("potential_evaporation_sum"),
                ssrd=v("surface_solar_radiation_downwards_sum"),
                masked=mask_days(v("snow_depth"), v("soil_temperature_level_1")),
                land=ds.land_frac.values >= min_land)
