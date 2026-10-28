"""Local starting values for the farm page (James, Oct 29: "pre-fill with typical numbers for the
area, and label them"): crop and hay prices by state, county yields, typical farm sizes, and the
corn and soybean acres FlashWatch covers.

    python -m flashwatch.ingest.nass local     # first: state prices and farm counts
    python -m flashwatch.site.defaults

Writes site/data/defaults.json. Sources (all USDA NASS, public domain): marketing-year prices
received (2024, the latest complete year), the 2012 hay price against 2011 (the last big flash
drought), county survey yields (median of 2017-2021), and 2022 Census acres and farm counts.
Fertilizer and pumping costs are not in NASS: see FIXED below.
"""

from __future__ import annotations

import json

import pandas as pd

from flashwatch.ingest.nass import OUT, centroids
from flashwatch.site.build import SITE

STATES = ["ND", "SD", "NE", "KS", "CO", "WY", "MT", "MN", "IA", "MO", "WI", "IL", "IN", "OH", "MI", "KY"]
BOX = {"lat": (36, 49), "lon": (-104, -80)}
PRICE_YEAR = 2024
YIELD_YEARS = (2017, 2021)
# Not in NASS. Anhydrous ammonia (82% N) at $1,085 a ton: USDA AMS Iowa Production Cost Report,
# June 26, 2026. Pumping: UNL's diesel table (about 2.5-4.7 gallons an acre-inch) at ~$3.50 a gallon,
# plus repairs; a round estimate.
FIXED = {"n_price": {"value": 0.66, "source": "USDA AMS Iowa production cost report, June 2026 (anhydrous ammonia, $1,085 a ton)"},
         "pump_cost_per_in": {"value": 10, "source": "Estimate from University of Nebraska diesel-use figures"},
         "n_cut_lb": {"value": 40, "source": "Typical side-dress rate"},
         "n_wrong_pct": {"value": 5, "source": "Estimate: yield lost if the rain comes and the crop is short of nitrogen"}}


def county_yields() -> dict:
    y = pd.read_csv(OUT / "yields_2005_2021.csv", dtype={"fips": str})
    y = y[y.year.between(*YIELD_YEARS) & y.state.isin(STATES)]
    med = y.groupby(["fips", "crop", "practice"]).yield_bu.median().unstack(["crop", "practice"])
    st = y.assign(s=y.fips.str[:2]).groupby(["s", "crop", "practice"]).yield_bu.median().unstack(["crop", "practice"])
    out = {}
    for f, row in med.iterrows():
        s = st.loc[f[:2]]
        def pick(crop, prac, fallback):
            for v in (row.get((crop, prac)), s.get((crop, prac)), row.get((crop, fallback))):
                if v is not None and pd.notna(v):
                    return round(float(v))
            return None
        out[f] = {"corn": {"irrigated": pick("corn", "irrigated", "all"), "rainfed": pick("corn", "non-irrigated", "all")},
                  "soybeans": {"irrigated": pick("soybeans", "irrigated", "all"), "rainfed": pick("soybeans", "non-irrigated", "all")}}
    return out


def state_values() -> dict:
    p = pd.read_csv(OUT / "prices.csv")
    piv = p.pivot_table(index=["state", "item"], columns="year", values="price")
    out = {}
    for s in STATES:
        get = lambda item, yr: piv.loc[(s, item), yr] if (s, item) in piv.index and pd.notna(piv.loc[(s, item), yr]) else None  # noqa: E731
        hay, h11, h12 = get("hay", PRICE_YEAR), get("hay", 2011), get("hay", 2012)
        out[s] = {"corn_price": get("corn", PRICE_YEAR), "soy_price": get("soybeans", PRICE_YEAR), "hay_price": hay,
                  "hay_rise": round(100 * (h12 / h11 - 1)) if h11 and h12 else None}
    # Soybean prices are not published for CO, MT, WY: use the regional median.
    soy = pd.Series([v["soy_price"] for v in out.values() if v["soy_price"]]).median()
    for v in out.values():
        v["soy_price"] = v["soy_price"] or round(float(soy), 2)
    return out


def coverage() -> dict:
    """Corn and soybean acres harvested (2022 Census) in counties whose centre is in the study box,
    scaled up per state for withheld county values."""
    a = pd.read_csv(OUT / "acres_2022.csv", dtype={"fips": str})
    c = centroids()
    cty = a[a.level == "county"].merge(c[["fips", "lat", "lon"]], on="fips")
    st = a[a.level == "state"].set_index(["state", "crop", "practice"]).acres
    scale = (st / cty.groupby(["state", "crop", "practice"]).acres.sum()).fillna(1).clip(1, 3)
    box = cty[cty.lat.between(*BOX["lat"]) & cty.lon.between(*BOX["lon"])]
    s = box.groupby(["state", "crop", "practice"]).acres.sum()
    s = (s * scale.reindex(s.index).fillna(1)).groupby(["crop", "practice"]).sum()
    out = {f"{crop}_{prac}": round(float(s[(crop, prac)])) for crop, prac in s.index}
    for crop in ("corn", "soybeans"):
        out[f"{crop}_rainfed"] = out[f"{crop}_all"] - out[f"{crop}_irrigated"]
    out["total"] = out["corn_all"] + out["soybeans_all"]
    return out


def farm_sizes() -> dict:
    """Average acres per farm in the study states (2022 Census), corn and soybeans, irrigated and all."""
    a = pd.read_csv(OUT / "acres_2022.csv")
    f = pd.read_csv(OUT / "farms_2022.csv")
    st = a[(a.level == "state") & a.state.isin(STATES)].merge(f, on=["state", "crop", "practice"])
    g = st.groupby(["crop", "practice"])[["acres", "farms"]].sum()
    return {f"{c}_{p}": round(float(r.acres / r.farms)) for (c, p), r in g.iterrows()}


def main():
    out = {"price_year": PRICE_YEAR, "yield_years": list(YIELD_YEARS), "states": state_values(),
           "counties": county_yields(), "coverage": coverage(), "farm_sizes": farm_sizes(), "fixed": FIXED}
    (SITE / "data" / "defaults.json").write_text(json.dumps(out, separators=(",", ":")))
    print("coverage (million acres):", {k: round(v / 1e6, 1) for k, v in out["coverage"].items()})
    print("farm sizes:", out["farm_sizes"])


if __name__ == "__main__":
    main()
