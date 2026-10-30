"""The headline number (DESIGN.md O9): what if every farm used FlashWatch?

    python -m flashwatch.decide.rollup --oof data/oof/<file>.parquet --model M1 [--tag name]

Three scopes, kept separate (James, Oct 28):
1. Irrigated corn and soybeans in the study region, from the day-by-day backtest (decide/backtest.py)
   per grid cell, weighted by each county's irrigated acres (NASS Census 2022, county centroids).
2. The same per-acre value scaled up to all US irrigated corn and soybeans. This is an
   extrapolation: flash-drought exposure outside the region is not measured here.
3. Rain-fed corn and soybeans in the region: trimming side-dress nitrogen (June) and skipping the
   fungicide pass (July-August) ahead of a flash drought (config/actions.json; rough values).

Each against both usual practices (careful, calendar). Per year: mean, median and worst/best of the
development years in the OOF file. Food: bushels saved (crop value saved / price), calories (USDA
FoodData Central: yellow corn 365 kcal/100 g, soybeans 446 kcal/100 g), people fed for a year at
2,000 kcal a day, and the weight against US retail food loss (USDA ERS, 2010: 43 billion lb).
Writes site/data/rollup_<tag>.json for the region page.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from flashwatch.decide import backtest, load
from flashwatch.obs import LAT, LON

KCAL_PER_BU = {"corn": 56 * 0.4536 * 3650, "soybeans": 60 * 0.4536 * 4460}    # lb/bu x kg/lb x kcal/kg
LB_PER_BU = {"corn": 56, "soybeans": 60}
PERSON_YEAR_KCAL = 2000 * 365
RETAIL_LOSS_LB = 43e9


def county_cells(cells: pd.DataFrame, max_km: float = 60) -> pd.DataFrame:
    """County fips -> nearest backtest cell (iy, ix) within max_km of the county's centroid."""
    from flashwatch.ingest.nass import centroids
    c = centroids()
    lat, lon = LAT[cells.iy.to_numpy()], LON[cells.ix.to_numpy()]
    rows = []
    for f, a, o in c[["fips", "lat", "lon"]].itertuples(index=False, name=None):
        d = np.hypot(lat - a, (lon - o) * np.cos(np.radians(a))) * 111
        k = int(d.argmin())
        if d[k] <= max_km:
            rows.append((f, int(cells.iy.iloc[k]), int(cells.ix.iloc[k])))
    return pd.DataFrame(rows, columns=["fips", "iy", "ix"])


def per_cell(bt: pd.DataFrame) -> pd.DataFrame:
    return bt.groupby(["iy", "ix"])[["value", "saved"]].mean().reset_index()


def scope(acres: pd.DataFrame, cell_val: pd.DataFrame, cc: pd.DataFrame, price: float) -> dict:
    """$ per year for these county acres, plus bushels saved."""
    a = acres.merge(cc, on="fips").merge(cell_val, on=["iy", "ix"])
    return {"acres": float(a.acres.sum()), "dollars": float((a.acres * a.value).sum()),
            "saved_dollars": float((a.acres * a.saved).sum()),
            "bushels": float((a.acres * a.saved).sum() / price)}


def yearly(bt: pd.DataFrame, acres: pd.DataFrame, cc: pd.DataFrame, price: float) -> pd.DataFrame:
    rows = []
    for y, g in bt.groupby("year"):
        s = scope(acres, g[["iy", "ix", "value", "saved"]], cc, price)
        rows.append({"year": int(y), **s})
    return pd.DataFrame(rows)


def run(oof: pd.DataFrame, model: str, overrides: dict | None = None, water: bool = True) -> dict:
    """water: the farmer knows the field's deficit (a water account) when deciding; else only the month."""
    from flashwatch.decide.falsestart import backtest_costs
    from flashwatch.ingest.nass import load as nass
    acres = nass(2022)
    cfg = load(overrides=overrides)
    ons = backtest.onset_days()
    cells = oof[["iy", "ix"]].drop_duplicates()
    cc = county_cells(cells)
    county = acres[acres.level == "county"]
    nat = acres[acres.level == "national"].set_index(["crop", "practice"]).acres
    out = {"model": model, "overrides": overrides or {}, "water_account": water, "years": sorted(int(y) for y in oof.year.unique()), "scopes": {}}
    for base in ("careful", "calendar"):
        res = {}
        for crop, key in (("corn", "irrigation"), ("soybeans", "irrigation_soybeans")):
            act = cfg[key]
            price = act["params"]["price"]
            bt = backtest.run(oof, ons, model, act, base,
                              false_starts=backtest_costs(act["params"]["pump_cost_per_in"], water))
            a = county[(county.crop == crop) & (county.practice == "irrigated")][["fips", "acres"]]
            yr = yearly(bt, a, cc, price)
            yr["kcal"] = yr.bushels * KCAL_PER_BU[crop]
            yr["lb"] = yr.bushels * LB_PER_BU[crop]
            region_acres = scope(a, per_cell(bt), cc, price)["acres"]
            per_acre = yr.dollars.mean() / region_acres if region_acres else np.nan
            res[crop] = {"region": yr.mean().to_dict() | {"acres": region_acres},
                         "region_by_year": yr.to_dict("records"), "per_acre": per_acre,
                         "per_acre_bushels": yr.bushels.mean() / region_acres if region_acres else np.nan,
                         "us_acres": float(nat.get((crop, "irrigated"), np.nan))}
            res[crop]["us"] = {"dollars": per_acre * res[crop]["us_acres"],
                               "bushels": res[crop]["per_acre_bushels"] * res[crop]["us_acres"]}
        # Scope 3: rain-fed corn and soybeans (all minus irrigated): side-dress nitrogen (June), the
        # fungicide pass (corn mid-July, soybeans late July), each a one-off decision in its window.
        def rainfed(crop):
            allc = county[(county.crop == crop) & (county.practice == "all")][["fips", "acres"]]
            irr = county[(county.crop == crop) & (county.practice == "irrigated")][["fips", "acres"]]
            rf = allc.merge(irr, on="fips", how="left", suffixes=("", "_irr")).fillna(0)
            return rf.assign(acres=(rf.acres - rf.acres_irr).clip(lower=0))[["fips", "acres"]]
        for name, crop in (("nitrogen", "corn"), ("fungicide_corn", "corn"), ("fungicide_soy", "soybeans")):
            act = cfg[name]
            bt = backtest.run(oof, ons, model, act, base, season=tuple(act["window"]))
            yr = yearly(bt, rainfed(crop), cc, act["params"]["price"])
            res[f"rainfed_{name}"] = {"region": yr.mean().to_dict(), "region_by_year": yr.to_dict("records")}
        res["rainfed_nitrogen"] = res.pop("rainfed_nitrogen")
        out["scopes"][base] = res
    out["sources"] = {"acres": "USDA NASS Census of Agriculture 2022 (county, harvested acres)",
                      "kcal": "USDA FoodData Central: corn grain yellow 365 kcal/100 g (fdcId 170288); soybeans mature raw 446 kcal/100 g (fdcId 174270)",
                      "retail_loss": "USDA ERS: retail-level food loss 43 billion lb in 2010 (10% of the food supply)"}
    return out


def headline(r: dict, base: str) -> dict:
    s = r["scopes"][base]
    reg = s["corn"]["region"]["dollars"] + s["soybeans"]["region"]["dollars"]
    us = s["corn"]["us"]["dollars"] + s["soybeans"]["us"]["dollars"]
    kcal_reg = s["corn"]["region"]["kcal"] + s["soybeans"]["region"]["kcal"]
    bu_us = {c: s[c]["us"]["bushels"] for c in ("corn", "soybeans")}
    kcal_us = sum(bu_us[c] * KCAL_PER_BU[c] for c in bu_us)
    lb_us = sum(bu_us[c] * LB_PER_BU[c] for c in bu_us)
    return {"baseline": base, "region_irrigated_dollars": reg, "us_irrigated_dollars": us,
            "rainfed_nitrogen_dollars": s["rainfed_nitrogen"]["region"]["dollars"],
            "region_people_fed": kcal_reg / PERSON_YEAR_KCAL, "us_people_fed": kcal_us / PERSON_YEAR_KCAL,
            "us_share_of_retail_loss_by_weight": lb_us / RETAIL_LOSS_LB,
            "per_acre_corn": s["corn"]["per_acre"], "per_acre_soy": s["soybeans"]["per_acre"]}


def main():
    from flashwatch.site.build import SITE
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--model", default="M1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-water", action="store_true", help="false-start cost by month only (no water account)")
    ap.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                    help="override action settings, e.g. spell_days=21 (calibrated to county yields, eval/yields.py)")
    args = ap.parse_args()
    over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    r = run(pd.read_parquet(args.oof), args.model, over, water=not args.no_water)
    r["headline"] = [headline(r, b) for b in ("careful", "calendar")]
    r["tag"] = args.tag
    print(pd.DataFrame(r["headline"]).T.to_string())
    for b in ("careful", "calendar"):
        print(f"\n{b}: region irrigated corn by year")
        print(pd.DataFrame(r["scopes"][b]["corn"]["region_by_year"])[["year", "dollars", "bushels"]].round(0).to_string(index=False))
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / f"rollup_{args.tag or args.model}.json").write_text(json.dumps(r, default=float))


if __name__ == "__main__":
    main()
