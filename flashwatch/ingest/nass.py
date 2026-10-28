"""Crop acres for the roll-up (DESIGN.md O9): USDA NASS Census of Agriculture, county and national
harvested acres of corn for grain and soybeans, irrigated and all, plus county centroids (Census
Gazetteer) to place counties on our grid.

    python -m flashwatch.ingest nass          # needs NASS_API_KEY in flashwatch.env
    python -m flashwatch.ingest.nass yields   # county yields 2005-2021 (survey), for eval/yields.py
    python -m flashwatch.ingest.nass local    # state prices and farm counts, for the farm page defaults

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


YIELDS = {("corn", "all"): "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
          ("corn", "irrigated"): "CORN, GRAIN, IRRIGATED - YIELD, MEASURED IN BU / ACRE",
          ("corn", "non-irrigated"): "CORN, GRAIN, NON-IRRIGATED - YIELD, MEASURED IN BU / ACRE",
          ("soybeans", "all"): "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
          ("soybeans", "irrigated"): "SOYBEANS, IRRIGATED - YIELD, MEASURED IN BU / ACRE",
          ("soybeans", "non-irrigated"): "SOYBEANS, NON-IRRIGATED - YIELD, MEASURED IN BU / ACRE"}


def query(short_desc: str, level: str, year: int, source: str = "CENSUS", period: str = "YEAR") -> pd.DataFrame:
    import requests
    params = {"key": os.environ["NASS_API_KEY"], "source_desc": source, "year": year,
              "agg_level_desc": level, "short_desc": short_desc, "domain_desc": "TOTAL", "format": "JSON"}
    if source == "SURVEY":
        params["reference_period_desc"] = period
    r = requests.get(API, params=params, timeout=120)
    if r.status_code == 400 and source == "SURVEY":     # NASS answers 400 when nothing matches
        return pd.DataFrame()
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


def fetch_yields(years=range(2005, 2022)) -> pd.DataFrame:
    """County survey yields (bu/acre); combined "other counties" rows (county code 998) dropped."""
    rows = []
    for (crop, practice), desc in YIELDS.items():
        for y in years:
            d = retry(lambda: query(desc, "COUNTY", y, "SURVEY"), log, f"{desc} {y}", tries=3, wait=20)
            if d is None or not len(d):
                continue
            d = d[d.county_code != "998"]
            v = pd.to_numeric(d["Value"].str.replace(",", ""), errors="coerce")
            rows.append(pd.DataFrame({"year": y, "fips": d.state_fips_code + d.county_code, "state": d.state_alpha,
                                      "crop": crop, "practice": practice, "yield_bu": v}).dropna())
        log.info("%s %s yields: %d rows", crop, practice, sum(len(r) for r in rows if r.crop.iloc[0] == crop
                                                               and r.practice.iloc[0] == practice))
    return pd.concat(rows, ignore_index=True)


PRICES = {"corn": "CORN, GRAIN - PRICE RECEIVED, MEASURED IN $ / BU",
          "soybeans": "SOYBEANS - PRICE RECEIVED, MEASURED IN $ / BU",
          "hay": "HAY - PRICE RECEIVED, MEASURED IN $ / TON"}
OPS = {("corn", "irrigated"): "CORN, GRAIN, IRRIGATED - OPERATIONS WITH AREA HARVESTED",
       ("corn", "all"): "CORN, GRAIN - OPERATIONS WITH AREA HARVESTED",
       ("soybeans", "irrigated"): "SOYBEANS, IRRIGATED - OPERATIONS WITH AREA HARVESTED",
       ("soybeans", "all"): "SOYBEANS - OPERATIONS WITH AREA HARVESTED"}


def fetch_local(price_years=(2010, 2011, 2012, 2013, 2022, 2023, 2024)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """State marketing-year prices received (survey) and 2022 Census farm counts (operations)."""
    num = lambda s: pd.to_numeric(s.str.replace(",", ""), errors="coerce")  # noqa: E731
    prices = []
    for crop, desc in PRICES.items():
        for y in price_years:
            d = retry(lambda: query(desc, "STATE", y, "SURVEY", "MARKETING YEAR"), log, f"{desc} {y}", tries=3, wait=20)
            if d is not None and len(d):
                prices.append(pd.DataFrame({"year": y, "state": d.state_alpha, "item": crop, "price": num(d.Value)}).dropna())
    ops = []
    for (crop, practice), desc in OPS.items():
        d = retry(lambda: query(desc, "STATE", 2022), log, desc, tries=3, wait=20)
        if d is not None and len(d):
            ops.append(pd.DataFrame({"state": d.state_alpha, "crop": crop, "practice": practice, "farms": num(d.Value)}).dropna())
    return pd.concat(prices, ignore_index=True), pd.concat(ops, ignore_index=True)


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
    import sys
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["local"]:
        prices, ops = fetch_local()
        prices.to_csv(OUT / "prices.csv", index=False)
        ops.to_csv(OUT / "farms_2022.csv", index=False)
        log.info("done: %d price rows, %d farm-count rows", len(prices), len(ops))
        return
    if sys.argv[1:] == ["yields"]:
        d = fetch_yields()
        d.to_csv(OUT / "yields_2005_2021.csv", index=False)
        log.info("done: %d yield rows", len(d))
        return
    http_download(GAZ, OUT / "2022_Gaz_counties_national.zip", log)
    d = fetch(2022)
    d.to_csv(OUT / "acres_2022.csv", index=False)
    log.info("done: %d rows", len(d))


if __name__ == "__main__":
    main()
