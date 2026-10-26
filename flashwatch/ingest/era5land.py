"""ERA5-Land daily aggregates from Earth Engine, area-averaged onto the fixed 0.25 degree grid.

One .npz per (variable, year) in data/raw/era5land/: `data` float32 (days, 52, 96), NaN where no
land, and `dates` (YYYYMMDD strings). Also `land_frac.npz`: fraction of each cell covered by
ERA5-Land land pixels. Uses reduceResolution(mean) at native resolution before reprojecting,
so cells are true area means, not nearest-neighbour samples. A year that fails as one call is
retried month by month.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from flashwatch.config import GRID_025, OBS_YEARS, RAW, RES, NORTH, WEST
from flashwatch.ingest.common import get_logger, retry

OUT = RAW / "era5land"
COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
VARIABLES = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
             "volumetric_soil_water_layer_3", "temperature_2m", "dewpoint_temperature_2m",
             "total_precipitation_sum", "total_evaporation_sum", "potential_evaporation_sum",
             "surface_solar_radiation_downwards_sum", "soil_temperature_level_1", "snow_depth"]
NODATA = -9999.0
WORKERS = int(os.getenv("FW_EE_WORKERS", "4"))
log = get_logger("era5land")


def init():
    import ee
    project = os.getenv("EE_PROJECT")
    try:
        ee.Initialize(project=project)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Earth Engine init failed ({e}). On a new machine run `earthengine authenticate` "
                         "once, and set EE_PROJECT in .env") from e
    return ee


def to_grid(ee, img):
    """Area-mean an image (any band count) onto GRID_025 and fetch it as (bands, 52, 96) float32."""
    native = ee.Image(img).select(0).projection()
    img = (img.setDefaultProjection(native)
           .reduceResolution(ee.Reducer.mean(), maxPixels=256)
           .reproject(crs="EPSG:4326", crsTransform=[RES, 0, WEST, 0, -RES, NORTH])
           .unmask(NODATA).toFloat())
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": GRID_025})
    names = list(arr.dtype.names)
    out = np.stack([arr[n] for n in names]).astype(np.float32)
    out[out <= NODATA + 1] = np.nan
    return names, out


def pull(ee, var: str, start: str, end: str):
    coll = ee.ImageCollection(COLLECTION).filterDate(start, end).select(var)
    names, data = to_grid(ee, coll.toBands())
    return [n.split("_")[0] for n in names], data


def year_task(var: str, year: int) -> bool:
    target = OUT / f"{var}_{year}.npz"
    if target.exists():
        return True
    ee = init()

    def whole_year():
        return pull(ee, var, f"{year}-01-01", f"{year + 1}-01-01")

    res = retry(whole_year, log, f"{var} {year}", tries=2, wait=20)
    if res is None:
        log.info("%s %d: falling back to monthly calls", var, year)
        parts = []
        for m in range(1, 13):
            end = f"{year + 1}-01-01" if m == 12 else f"{year}-{m + 1:02d}-01"
            part = retry(lambda: pull(ee, var, f"{year}-{m:02d}-01", end), log, f"{var} {year}-{m:02d}")
            if part is None:
                return False
            parts.append(part)
        res = (sum((p[0] for p in parts), []), np.concatenate([p[1] for p in parts]))
    dates, data = res
    tmp = target.with_name(target.stem + ".part.npz")
    np.savez_compressed(tmp, data=data, dates=np.array(dates))
    os.replace(tmp, target)
    log.info("done %s %d: %d days, %.0f%% valid", var, year, data.shape[0], 100 * np.isfinite(data).mean())
    return True


def land_fraction():
    target = OUT / "land_frac.npz"
    if target.exists():
        return
    ee = init()
    img = ee.ImageCollection(COLLECTION).filterDate("2020-07-01", "2020-07-02").first()
    land = img.select("volumetric_soil_water_layer_1").mask().rename("land_frac")
    _, data = to_grid(ee, land)
    np.savez_compressed(target, data=data[0])
    log.info("land fraction: %d of %d cells have land", int((data[0] > 0).sum()), data[0].size)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    init()
    land_fraction()
    tasks = [(v, y) for y in OBS_YEARS for v in VARIABLES]
    log.info("%d variable-years, %d at a time", len(tasks), WORKERS)
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda t: year_task(*t), tasks))
    log.info("finished: %d ok, %d failed (re-run to retry failures)", sum(results), results.count(False))
