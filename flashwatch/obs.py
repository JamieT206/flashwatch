"""Assemble the ERA5-Land downloads into data/obs.zarr: (time, lat, lon) on the 0.25 degree grid.

    python -m flashwatch.obs           # build or rebuild from data/raw/era5land/*.npz

Adds the derived root-zone soil moisture `rzsm` (0-100 cm, depth-weighted layers 1-3:
0-7, 7-28, 28-100 cm) and a static `land_frac`. Missing variable-years are left as NaN
and listed, so the store can be built while downloads are still running.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from flashwatch.config import DATA, EAST, NORTH, OBS_YEARS, RAW, RES, SOUTH, WEST

OBS_STORE = DATA / "obs.zarr"
LAT = np.arange(NORTH - RES / 2, SOUTH, -RES)       # 48.875 ... 36.125 (52, north first)
LON = np.arange(WEST + RES / 2, EAST, RES)          # -103.875 ... -80.125 (96)
RZ_LAYERS = {"volumetric_soil_water_layer_1": 7, "volumetric_soil_water_layer_2": 21,
             "volumetric_soil_water_layer_3": 72}


def root_zone(ds):
    """Depth-weighted 0-100 cm volumetric soil moisture from ERA5-Land layers 1-3."""
    total = sum(RZ_LAYERS.values())
    return sum(ds[v] * (w / total) for v, w in RZ_LAYERS.items()).rename("rzsm")


def build(years=OBS_YEARS):
    import xarray as xr
    from flashwatch.ingest.era5land import VARIABLES

    src = RAW / "era5land"
    time = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    shape = (len(time), len(LAT), len(LON))
    data, missing = {}, []
    for var in VARIABLES:
        arr = np.full(shape, np.nan, dtype=np.float32)
        for y in years:
            f = src / f"{var}_{y}.npz"
            if not f.exists():
                missing.append(f.name)
                continue
            z = np.load(f)
            idx = time.get_indexer(pd.to_datetime(z["dates"], format="%Y%m%d"))
            if (idx < 0).any():
                raise ValueError(f"{f.name}: dates outside the store's time axis")
            arr[idx] = z["data"]
        data[var] = (("time", "lat", "lon"), arr)
    ds = xr.Dataset(data, coords={"time": time, "lat": LAT, "lon": LON})
    ds["rzsm"] = root_zone(ds)
    lf = src / "land_frac.npz"
    ds["land_frac"] = (("lat", "lon"), np.load(lf)["data"] if lf.exists()
                       else np.full(shape[1:], np.nan, np.float32))
    ds.attrs.update(source="ERA5-Land daily aggregates via Google Earth Engine (ECMWF/ERA5_LAND/DAILY_AGGR), "
                           "area-mean regridded to 0.25 deg",
                    credit="Contains modified Copernicus Climate Change Service information",
                    missing=", ".join(missing))
    enc = {v: {"chunks": (366, len(LAT), len(LON))} for v in ds.data_vars if "time" in ds[v].dims}
    ds.to_zarr(OBS_STORE, mode="w", encoding=enc)
    print(f"wrote {OBS_STORE}: {dict(ds.sizes)}; {len(missing)} variable-years missing")
    return ds


def open_obs():
    import xarray as xr
    return xr.open_zarr(OBS_STORE)


if __name__ == "__main__":
    build()
    sys.exit(0)
