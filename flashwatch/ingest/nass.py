"""Crop acres for the roll-up (DESIGN.md O9): USDA NASS Census of Agriculture, county and national
harvested acres of corn for grain and soybeans, irrigated and all, plus county centroids (Census
Gazetteer) to place counties on our grid.

    python -m flashwatch.ingest nass          # needs NASS_API_KEY in flashwatch.env

Writes data/raw/nass/acres_<year>.csv (fips, state, county, crop, practice, acres) and the Gazetteer
file. NASS data are public domain; withheld county values ("(D)") are left out, so county sums fall
short of state totals (reported alongside).
"""

from __future__ import annotations

import io
import os
import zipfile

import pandas as pd

from flashwatch.config import RAW, load_env
from flashwatch.ingest.common import get_logger, http_download, retry

OUT = RAW / "nass"
API = "https://quickstats.nass.usda.gov/api/api_GET/"
GAZ = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_counties_national.zip"
SERIES = {("corn", "irrigated"): "CORN, GRAIN, IRRIGATED - ACRES HARVESTED",
          ("corn", "all"): "CORN, GRAIN - ACRES HARVESTED",
          ("soybeans", "irrigated"): "SOYBEANS, IRRIGATED - ACRES HARVESTED",
          ("soybeans", "all"): "SOYBEANS - ACRES HARVESTED"}
log = get_logger("nass")


def query(short_desc: str, level: str, year: int) -> pd.DataFrame:
    import requests
    params = {"key": os.environ["NASS_API_KEY"], "source_desc": "CENSUS", "year": year,
              "agg_level_desc": level, "short_desc": short_desc, "domain_desc": "TOTAL", "format": "JSON"}
    r = requests.get(API, params=params, timeout=120)
    if r.status_code != 200:
        # Never log the URL: it carries the key.
        raise RuntimeError(f"NASS {short_desc} {level}: HTTP {r.status_code}")
    return pd.DataFrame(r.json()["data"])


def fetch(year: int = 2022) -> pd.DataFrame:
    rows = []
    for (crop, practice), desc in SERIES.items():
        for level in ("COUNTY", "STATE", "NATIONAL"):
            d = retry(lambda: query(desc, level, year), log, f"{desc} {level}", tries=3, wait=20)
            if d is None or not len(d):
                continue
            v = pd.to_numeric(d["Value"].str.replace(",", ""), errors="coerce")
            out = pd.DataFrame({"level": level.lower(), "state": d.get("state_alpha"),
                                "fips": d.get("state_fips_code", "") + d.get("county_code", ""),
                                "county": d.get("county_name"), "crop": crop, "practice": practice,
                                "acres": v})
            rows.append(out[out.acres.notna()])
            log.info("%s %s %s: %d rows", crop, practice, level, len(out))
    return pd.concat(rows, ignore_index=True)


def centroids() -> pd.DataFrame:
    """County FIPS -> internal point lat/lon (Census Gazetteer)."""
    z = OUT / "2022_Gaz_counties_national.zip"
    with zipfile.ZipFile(z) as f:
        name = [n for n in f.namelist() if n.endswith(".txt")][0]
        g = pd.read_csv(io.BytesIO(f.read(name)), sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    return g.rename(columns={"GEOID": "fips", "INTPTLAT": "lat", "INTPTLONG": "lon", "NAME": "name",
                             "USPS": "state"})[["fips", "state", "name", "lat", "lon"]]


def load(year: int = 2022) -> pd.DataFrame:
    return pd.read_csv(OUT / f"acres_{year}.csv", dtype={"fips": str})


def main():
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    http_download(GAZ, OUT / "2022_Gaz_counties_national.zip", log)
    d = fetch(2022)
    d.to_csv(OUT / "acres_2022.csv", index=False)
    log.info("done: %d rows", len(d))


if __name__ == "__main__":
    main()
