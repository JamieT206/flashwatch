"""FlashWatch data-access check (pre-kickoff diagnostic, not project code).

Runs one small request against each data source in the design spec and writes
access_report.json next to this file. Each check is independent: a missing
credential or library skips that check with instructions instead of failing.

Usage (from this folder):
    pip install -r requirements-check.txt
    python check_access.py            # quick checks only
    python check_access.py --slow     # also time a CDS and an ECMWF S2S download

Credentials (set only the ones you have):
    EARTHDATA_USERNAME / EARTHDATA_PASSWORD   NASA Earthdata (or a ~/.netrc entry)
    EE_PROJECT                                Google Cloud project registered for Earth Engine
    NASS_API_KEY                              USDA Quick Stats key
    ~/.cdsapirc                               Copernicus CDS key file
    ~/.ecmwfapirc                             ECMWF Web API key file
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Region from the spec: 36-49N, 104-80W. CDS/MARS order is N/W/S/E.
NORTH, WEST, SOUTH, EAST = 49, -104, 36, -80
TEST_DAY = "2023-06-15"          # inside the 2023 Midwest flash drought
TIMEOUT = 60

HERE = Path(__file__).resolve().parent
RESULTS: list[dict] = []


def load_env(path: Path = HERE / ".env") -> int:
    """Load KEY='value' lines from .env into os.environ (existing variables win)."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value and key.strip() not in os.environ:
            os.environ[key.strip()] = value
            n += 1
    return n


def record(name, status, detail="", seconds=None, **extra):
    row = {"source": name, "status": status, "detail": detail}
    if seconds is not None:
        row["seconds"] = round(seconds, 1)
    row.update(extra)
    RESULTS.append(row)
    mark = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}.get(status, status)
    print(f"[{mark}] {name}: {detail}")


def check(name):
    """Decorator: run a check, time it, and turn exceptions into FAIL rows."""
    def wrap(fn):
        def run(args):
            t0 = time.time()
            try:
                fn(args, t0)
            except ModuleNotFoundError as e:
                record(name, "SKIP", f"missing library {e.name}; run pip install -r requirements-check.txt")
            except Exception as e:  # noqa: BLE001 - report everything
                msg = f"{type(e).__name__}: {e}"
                record(name, "FAIL", msg[:400], time.time() - t0,
                       trace=traceback.format_exc(limit=3)[-1500:])
        run.check_name = name
        return run
    return wrap


def http_get(url, **kw):
    import requests
    r = requests.get(url, timeout=TIMEOUT, **kw)
    return r


# ---------------------------------------------------------------- NASA ----

@check("NASA CMR search (SMAP L4 v008 granules)")
def cmr_smap(args, t0):
    url = ("https://cmr.earthdata.nasa.gov/search/granules.json"
           f"?short_name=SPL4SMGP&version=008&temporal={TEST_DAY}T00:00:00Z,{TEST_DAY}T23:59:59Z"
           "&page_size=20")
    r = http_get(url)
    r.raise_for_status()
    entries = r.json()["feed"]["entry"]
    sizes = [float(e.get("granule_size", 0) or 0) for e in entries]
    record("NASA CMR search (SMAP L4 v008 granules)", "PASS" if entries else "FAIL",
           f"{len(entries)} granules on {TEST_DAY}; median size {sorted(sizes)[len(sizes)//2]:.0f} MB each"
           if entries else "no granules found", time.time() - t0,
           granules=len(entries), granule_mb=sizes[:3])


@check("NASA Earthdata login")
def earthdata_login(args, t0):
    import earthaccess
    have_env = os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")
    netrc = Path.home() / (("_netrc") if os.name == "nt" else ".netrc")
    if not have_env and not netrc.exists():
        record("NASA Earthdata login", "SKIP",
               "set EARTHDATA_USERNAME and EARTHDATA_PASSWORD (or create ~/.netrc)")
        return
    auth = earthaccess.login(strategy="environment" if have_env else "netrc")
    ok = bool(auth and auth.authenticated)
    record("NASA Earthdata login", "PASS" if ok else "FAIL",
           "authenticated" if ok else "login rejected; check username/password", time.time() - t0)


@check("NASA AppEEARS login")
def appeears_login(args, t0):
    import requests
    user, pw = os.getenv("EARTHDATA_USERNAME"), os.getenv("EARTHDATA_PASSWORD")
    if not (user and pw):
        record("NASA AppEEARS login", "SKIP", "needs EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
        return
    r = requests.post("https://appeears.earthdatacloud.nasa.gov/api/login",
                      auth=(user, pw), timeout=TIMEOUT)
    ok = r.status_code == 200 and "token" in r.text
    record("NASA AppEEARS login", "PASS" if ok else "FAIL",
           "token issued" if ok else f"HTTP {r.status_code}: {r.text[:200]}", time.time() - t0)


@check("NASA SMAP L4 one-granule download")
def smap_download(args, t0):
    if not args.slow:
        record("NASA SMAP L4 one-granule download", "SKIP", "run with --slow to time one full granule")
        return
    import earthaccess
    earthaccess.login()
    results = earthaccess.search_data(short_name="SPL4SMGP", version="008",
                                      temporal=(f"{TEST_DAY}T10:00", f"{TEST_DAY}T13:00"),
                                      count=1)
    with tempfile.TemporaryDirectory() as tmp:
        files = earthaccess.download(results[:1], tmp)
        mb = sum(Path(f).stat().st_size for f in files) / 1e6
    record("NASA SMAP L4 one-granule download", "PASS",
           f"{mb:.0f} MB in {time.time() - t0:.0f} s (global granule; we will subset instead)",
           time.time() - t0, mb=round(mb))


# -------------------------------------------------------- Earth Engine ----

@check("Earth Engine (ERA5-Land daily + SMAP L4 v008)")
def earth_engine(args, t0):
    import ee
    project = os.getenv("EE_PROJECT")
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as e:  # noqa: BLE001
        record("Earth Engine (ERA5-Land daily + SMAP L4 v008)", "SKIP" if not project else "FAIL",
               ("set EE_PROJECT in .env to your Earth Engine Cloud project ID. " if not project else "")
               + f"Initialize failed: {str(e)[:200]}", time.time() - t0)
        return
    region = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])
    day0, day1 = TEST_DAY, "2023-06-16"

    era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(day0, day1).first()
    bands = era.bandNames().getInfo()
    wanted = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
              "volumetric_soil_water_layer_3", "temperature_2m", "dewpoint_temperature_2m",
              "total_precipitation_sum", "total_evaporation_sum",
              "potential_evaporation_sum", "surface_solar_radiation_downwards_sum",
              "soil_temperature_level_1", "snow_depth"]
    missing = [b for b in wanted if b not in bands]
    era_mean = era.select("volumetric_soil_water_layer_2").reduceRegion(
        ee.Reducer.mean(), region, 25000).getInfo()

    smap = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(day0, day1)
    n_smap = smap.size().getInfo()
    smap_bands = smap.first().bandNames().getInfo() if n_smap else []
    smap_mean = (smap.select("sm_rootzone").mean().reduceRegion(ee.Reducer.mean(), region, 9000)
                 .getInfo() if n_smap else {})

    record("Earth Engine (ERA5-Land daily + SMAP L4 v008)",
           "PASS" if not missing and n_smap else "FAIL",
           f"ERA5-Land {len(bands)} bands, missing {missing or 'none'}; "
           f"layer-2 mean {era_mean}; SMAP images that day: {n_smap}, root-zone mean {smap_mean}",
           time.time() - t0, era5land_bands=bands, smap_bands=smap_bands)


# ---------------------------------------------------------- Copernicus ----

@check("Copernicus CDS (ERA5-Land, one day)")
def cds(args, t0):
    import cdsapi
    if not ((Path.home() / ".cdsapirc").exists() or os.getenv("CDSAPI_KEY")):
        record("Copernicus CDS (ERA5-Land, one day)", "SKIP",
               "put your CDS API token in CDSAPI_KEY in .env (cds.climate.copernicus.eu/profile) "
               "and accept the ERA5-Land licence")
        return
    if not args.slow:
        c = cdsapi.Client(quiet=True)
        record("Copernicus CDS (ERA5-Land, one day)", "PASS",
               "key file loads; run with --slow to time a real request", time.time() - t0)
        return
    c = cdsapi.Client(quiet=True)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "era5land.nc"
        c.retrieve("reanalysis-era5-land", {
            "variable": ["volumetric_soil_water_layer_2"],
            "year": "2023", "month": "06", "day": "15",
            "time": ["00:00", "12:00"],
            "area": [NORTH, WEST, SOUTH, EAST],
            "data_format": "netcdf", "download_format": "unarchived",
        }, str(target))
        kb = target.stat().st_size / 1e3
    record("Copernicus CDS (ERA5-Land, one day)", "PASS",
           f"{kb:.0f} KB in {time.time() - t0:.0f} s including queue", time.time() - t0)


# --------------------------------------------------------------- ECMWF ----

@check("ECMWF S2S reforecast via ECMWF Data Store (one start date)")
def ecmwf_s2s(args, t0):
    # The old Web API (apps.ecmwf.int) closed for S2S on 27 May 2026. S2S now lives in
    # the ECMWF Data Store (ECDS), reached with the CDS client and the same CDS key.
    name = "ECMWF S2S reforecast via ECMWF Data Store (one start date)"
    key = os.getenv("ECDS_API_KEY") or os.getenv("CDSAPI_KEY")
    if not key:
        record(name, "SKIP", "needs CDSAPI_KEY in .env (the CDS key also works for the ECMWF Data Store)")
        return
    import cdsapi
    client = cdsapi.Client(url=os.getenv("ECDS_API_URL", "https://ecds.ecmwf.int/api"),
                           key=key, quiet=True)
    if not args.slow:
        record(name, "PASS", "client configured; run with --slow to time a real reforecast request",
               time.time() - t0)
        return
    steps = "/".join(str(h) for h in range(0, 1105, 24))   # daily to day 46
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "s2s.grib"
        # ECMWF reforecast (stream enfh) issued with the 2024-06-13 real-time run, for 2004,
        # in ECDS's MARS-like syntax. If a keyword is wrong, the error message names it.
        client.retrieve("s2s-reforecasts", {
            "class": "s2", "expver": "prod", "model": "glob",
            "origin": "ecmf", "stream": "enfh", "type": "cf", "levtype": "sfc",
            "date": "2024-06-13", "hdate": "2004-06-13", "time": "00:00:00",
            "param": "228228",              # total precipitation
            "step": steps,
            "area": f"{NORTH}/{WEST}/{SOUTH}/{EAST}",
        }, str(target))
        kb = target.stat().st_size / 1e3
    record(name, "PASS",
           f"control member, 1 variable, 46 days: {kb:.0f} KB in {time.time() - t0:.0f} s including queue",
           time.time() - t0)


# ------------------------------------------------------- Drought & crops --

@check("US Drought Monitor (weekly shapefile)")
def usdm(args, t0):
    url = "https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_20230613_M.zip"
    r = http_get(url)
    ok = r.status_code == 200 and r.content[:2] == b"PK"
    record("US Drought Monitor (weekly shapefile)", "PASS" if ok else "FAIL",
           f"{len(r.content)/1e3:.0f} KB zip for 2023-06-13" if ok else f"HTTP {r.status_code} at {url}",
           time.time() - t0)


@check("US Drought Monitor (statistics API)")
def usdm_api(args, t0):
    url = ("https://usdmdataservices.unl.edu/api/StateStatistics/"
           "GetDroughtSeverityStatisticsByAreaPercent?aoi=IL&startdate=6/1/2023"
           "&enddate=6/30/2023&statisticsType=1")
    r = http_get(url, headers={"Accept": "application/json"})
    ok = r.status_code == 200 and len(r.text) > 50
    record("US Drought Monitor (statistics API)", "PASS" if ok else "FAIL",
           f"{len(r.text)} bytes for Illinois, June 2023" if ok else f"HTTP {r.status_code}: {r.text[:150]}",
           time.time() - t0)


@check("SMVI flash drought inventory (HydroShare)")
def smvi(args, t0):
    rid = "b6d1d853b82247e1aeaf907ac254c99e"
    r = http_get(f"https://www.hydroshare.org/hsapi/resource/{rid}/files/")
    if r.status_code != 200:
        record("SMVI flash drought inventory (HydroShare)", "FAIL",
               f"HTTP {r.status_code}; may need a HydroShare login", time.time() - t0)
        return
    data = r.json()
    files = data.get("results", data if isinstance(data, list) else [])
    total = sum(float(f.get("size", 0) or 0) for f in files) / 1e6
    record("SMVI flash drought inventory (HydroShare)", "PASS",
           f"{len(files)} files listed, {total:.0f} MB total (first page)", time.time() - t0,
           sample=[f.get("file_name") or f.get("url") for f in files[:5]])


@check("USDA NASS Quick Stats (crop condition)")
def nass(args, t0):
    key = os.getenv("NASS_API_KEY")
    if not key:
        record("USDA NASS Quick Stats (crop condition)", "SKIP",
               "set NASS_API_KEY (free at https://quickstats.nass.usda.gov/api)")
        return
    url = ("https://quickstats.nass.usda.gov/api/api_GET/"
           f"?key={key}&commodity_desc=CORN&statisticcat_desc=CONDITION"
           "&year=2023&state_alpha=IL&agg_level_desc=STATE&format=JSON")
    r = http_get(url)
    rows = r.json().get("data", []) if r.status_code == 200 else []
    record("USDA NASS Quick Stats (crop condition)", "PASS" if rows else "FAIL",
           f"{len(rows)} rows for Illinois corn condition 2023" if rows else f"HTTP {r.status_code}: {r.text[:150]}",
           time.time() - t0)


# ------------------------------------------------------------ fallbacks --

@check("NOAA GEFSv12 reforecast (AWS, anonymous)")
def gefs(args, t0):
    r = http_get("https://noaa-gefs-retrospective.s3.amazonaws.com/?list-type=2&max-keys=5&delimiter=/")
    ok = r.status_code == 200 and "<ListBucketResult" in r.text
    record("NOAA GEFSv12 reforecast (AWS, anonymous)", "PASS" if ok else "FAIL",
           "bucket listable" if ok else f"HTTP {r.status_code}", time.time() - t0)


@check("Hugging Face (Muse Glimmer weights)")
def hf(args, t0):
    r = http_get("https://huggingface.co/api/models?search=glimmer&author=facebook&limit=5")
    if r.status_code == 200 and not r.json():
        r = http_get("https://huggingface.co/api/models?search=muse-glimmer&limit=5")
    names = [m.get("id") for m in r.json()] if r.status_code == 200 else []
    record("Hugging Face (Muse Glimmer weights)", "PASS" if names else "FAIL",
           f"found {names}" if names else "no matching repo found; check the model name on Meta's page",
           time.time() - t0)


# ------------------------------------------------------------ Meta API --

@check("Meta Model API (key, model list, one call)")
def meta_api(args, t0):
    import requests
    name = "Meta Model API (key, model list, one call)"
    key = os.getenv("META_API_KEY")
    if not key:
        record(name, "SKIP", "add META_API_KEY to .env (from your Meta developer dashboard)")
        return
    base = os.getenv("META_API_BASE", "https://api.meta.ai/v1").rstrip("/")
    auth = {"Authorization": f"Bearer {key}"}
    r = requests.get(f"{base}/models", headers=auth, timeout=TIMEOUT)
    models = [m.get("id") for m in r.json().get("data", [])] if r.ok else []
    model = (os.getenv("META_MODEL")
             or next((m for m in models if "wonder" in m.lower() and "full" not in m.lower()), None)
             or next((m for m in models if "spark" in m.lower()), "muse-spark-1.1"))
    r2 = requests.post(f"{base}/chat/completions", headers=auth, timeout=TIMEOUT, json={
        "model": model, "max_tokens": 50,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}]})
    if not r2.ok:
        record(name, "FAIL", f"model list HTTP {r.status_code}; call to {model} HTTP {r2.status_code}: "
               f"{r2.text[:250]}", time.time() - t0, models=models)
        return
    reply = r2.json()["choices"][0]["message"].get("content", "")
    record(name, "PASS", f"{len(models)} models visible ({', '.join(models[:8])}); "
           f"{model} replied {reply.strip()[:40]!r}", time.time() - t0, models=models)


# ---------------------------------------------- throughput (--throughput) --

GRID_025 = {"dimensions": {"width": int((EAST - WEST) / 0.25), "height": int((NORTH - SOUTH) / 0.25)},
            "affineTransform": {"scaleX": 0.25, "shearX": 0, "translateX": WEST,
                                "shearY": 0, "scaleY": -0.25, "translateY": NORTH},
            "crsCode": "EPSG:4326"}


@check("Earth Engine throughput (ERA5-Land 1 year, SMAP 1 month, 0.25 deg grid)")
def ee_throughput(args, t0):
    name = "Earth Engine throughput (ERA5-Land 1 year, SMAP 1 month, 0.25 deg grid)"
    if not args.throughput:
        record(name, "SKIP", "run with --throughput")
        return
    import ee
    import numpy as np
    project = os.getenv("EE_PROJECT")
    ee.Initialize(project=project) if project else ee.Initialize()

    def pull(img):
        t = time.time()
        a = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": GRID_025})
        vals = np.stack([a[n] for n in a.dtype.names])
        return time.time() - t, vals

    era = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate("2023-01-01", "2024-01-01")
           .select("volumetric_soil_water_layer_2").toBands())
    t_era, v_era = pull(era)
    smap = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate("2023-06-01", "2023-07-01")
            .select("sm_rootzone").toBands())
    t_smap, v_smap = pull(smap)
    full_min = t_era * 10 * 22 / 60          # 10 variables x 22 years, one call each
    record(name, "PASS",
           f"ERA5-Land 1 variable x {v_era.shape[0]} days on a {v_era.shape[1]}x{v_era.shape[2]} grid: "
           f"{t_era:.0f} s, {np.isfinite(v_era).mean():.0%} valid. SMAP root zone, {v_smap.shape[0]} "
           f"3-hourly images: {t_smap:.0f} s. Full ERA5-Land cube (10 vars x 22 yrs) ~ {full_min:.0f} min "
           "if run one call at a time", time.time() - t0,
           era_seconds=round(t_era, 1), smap_seconds=round(t_smap, 1))


S2S_DAILY_STEPS = "/".join(str(h) for h in range(0, 1105, 24))
S2S_BASE = {"class": "s2", "expver": "prod", "model": "glob", "origin": "ecmf", "stream": "enfh",
            "levtype": "sfc", "date": "2024-06-13", "time": "00:00:00",
            "area": f"{NORTH}/{WEST}/{SOUTH}/{EAST}"}


def ecds_client():
    import cdsapi
    return cdsapi.Client(url=os.getenv("ECDS_API_URL", "https://ecds.ecmwf.int/api"),
                         key=os.getenv("ECDS_API_KEY") or os.getenv("CDSAPI_KEY"), quiet=True)


@check("ECMWF S2S bundled request (10 members x 20 years) + GRIB decode")
def s2s_bundle(args, t0):
    name = "ECMWF S2S bundled request (10 members x 20 years) + GRIB decode"
    if not args.throughput:
        record(name, "SKIP", "run with --throughput")
        return
    hdates = "/".join(f"{y}-06-13" for y in range(2004, 2024))
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "bundle.grib"
        ecds_client().retrieve("s2s-reforecasts", {**S2S_BASE, "type": "pf", "number": "1/to/10",
                                                    "hdate": hdates, "param": "228228",
                                                    "step": S2S_DAILY_STEPS}, str(target))
        t_dl, mb = time.time() - t0, target.stat().st_size / 1e6
        try:
            import xarray as xr
            ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"indexpath": ""})
            decoded = f"decoded with cfgrib: {dict(ds.sizes)}"
        except Exception as e:  # noqa: BLE001
            decoded = f"GRIB decode FAILED ({type(e).__name__}: {str(e)[:150]}); install cfgrib and eccodes"
    record(name, "PASS" if "FAILED" not in decoded else "FAIL",
           f"precipitation, 10 members x 20 hindcast years x 46 days: {mb:.1f} MB in {t_dl:.0f} s "
           f"including queue; {decoded}", time.time() - t0, mb=round(mb, 2), seconds_download=round(t_dl))


@check("ECMWF S2S daily-averaged variable (2 m temperature)")
def s2s_daily_avg(args, t0):
    name = "ECMWF S2S daily-averaged variable (2 m temperature)"
    if not args.throughput:
        record(name, "SKIP", "run with --throughput")
        return
    ranges = "/".join(f"{h}-{h + 24}" for h in range(0, 1104, 24))
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "t2m.grib"
        ecds_client().retrieve("s2s-reforecasts", {**S2S_BASE, "type": "cf", "hdate": "2004-06-13",
                                                    "param": "167", "step": ranges}, str(target))
        kb = target.stat().st_size / 1e3
    record(name, "PASS", f"control member, 46 daily means: {kb:.0f} KB in {time.time() - t0:.0f} s; "
           "daily-averaged request syntax confirmed", time.time() - t0)


CHECKS = [cmr_smap, earthdata_login, appeears_login, smap_download, earth_engine,
          cds, ecmwf_s2s, usdm, usdm_api, smvi, nass, gefs, hf, meta_api,
          ee_throughput, s2s_bundle, s2s_daily_avg]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slow", action="store_true", help="also run timed downloads (CDS, S2S, one SMAP granule)")
    ap.add_argument("--throughput", action="store_true",
                    help="time bulk pulls: Earth Engine year/month, bundled S2S request, GRIB decoding")
    ap.add_argument("--only", nargs="*", help="run only checks whose name contains one of these words")
    args = ap.parse_args()
    loaded = load_env()

    print(f"FlashWatch access check, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, "
          f"Python {platform.python_version()} on {platform.system()}; "
          f"{loaded} settings loaded from .env\n")
    for fn in CHECKS:
        if args.only and not any(w.lower() in fn.check_name.lower() for w in args.only):
            continue
        fn(args)

    out = HERE / "access_report.json"
    out.write_text(json.dumps({
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "os": platform.platform(),
        "slow": args.slow, "results": RESULTS,
    }, indent=2))
    counts = {s: sum(r["status"] == s for r in RESULTS) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped. "
          f"Report written to {out}")


if __name__ == "__main__":
    sys.exit(main())
