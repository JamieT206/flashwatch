"""Tools the analyst agent calls (DESIGN.md G1). Each reads one weekly pack (agent/pack.py) and
returns small JSON-ready dicts of rounded numbers, which the number checker later traces.

Probabilities and shares are returned in percent (keys ending _pct); "times_normal" is the forecast
divided by the normal chance for that place and date.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LEADS = ["week1", "week2", "week3", "week4", "any28"]
USDM = {-1: "none", 0: "D0 (abnormally dry)", 1: "D1 (moderate drought)", 2: "D2 (severe drought)",
        3: "D3 (extreme drought)", 4: "D4 (exceptional drought)"}
# Plain-language names for the features most likely to show up as drivers; others pass through.
FEATURE_TEXT = {
    "rz_pct": "root-zone soil moisture percentile (low = dry)",
    "rz_pct_max15": "highest soil-moisture percentile in the last 15 days (wet soil has further to fall)",
    "rz_pct_chg5": "soil moisture percentile change over 5 days",
    "rz_pct_chg10": "soil moisture percentile change over 10 days",
    "rz_pct_chg20": "soil moisture percentile change over 20 days",
    "rz_days_to_p20": "days until soil moisture reaches the drought line at the current rate of drying",
    "sf_pct": "surface soil moisture percentile",
    "precip14_z": "rainfall over the last 14 days vs normal",
    "precip30_z": "rainfall over the last 30 days vs normal",
    "dry_days_2mm": "days without meaningful rain",
    "hot_days14": "hot days in the last 14 days",
    "vpd14_z": "air dryness (VPD) over the last 14 days vs normal",
    "t2m30_z": "temperature over the last 30 days vs normal",
    "esi14": "evaporative stress over the last 14 days",
    "nbr_share_dry": "share of nearby land already in drought",
    "nbr_share_newly_dry": "share of nearby land that just entered drought",
    "nbr_rz_pct": "soil moisture percentile of the surrounding area",
    "nbr_rz_pct_chg10": "soil moisture change over 10 days in the surrounding area",
    "deep_pct": "deeper soil moisture percentile (0-100 cm)",
    "deep_pct_chg10": "deeper soil moisture change over 10 days",
    "rz_pct_acc5": "whether soil drying is speeding up",
    "rz_days_since_p40": "days since soil moisture was above the 40th percentile",
    "sf_minus_rz_pct": "surface soil drier or wetter than the root zone",
    "vpd14_x_dryness": "dry air combined with already-dry soil",
}


VAR_TEXT = {"tp": "rainfall", "t2m": "temperature", "vpd": "air dryness (VPD)", "sm100": "soil moisture (0-100 cm)",
            "sm100chg": "soil drying (0-100 cm, change from week 1)", "dryhot": "hot-and-dry index"}
WEEK_TEXT = {"w1": "next week", "w2": "in week 2", "w3": "in week 3", "w4": "in week 4", "w12": "over the next 2 weeks"}


LAND_TEXT = {"precip": "rainfall", "t2m": "temperature", "vpd": "air dryness (VPD)", "ssrd": "sunshine (solar radiation)",
             "rzsm": "root-zone soil moisture", "evap": "evaporation", "pet": "evaporative demand"}


def describe(name: str) -> str:
    """Plain-language name of a feature (forecast features are spelled out from their parts)."""
    if name in FEATURE_TEXT:
        return FEATURE_TEXT[name]
    m = re.fullmatch(r"([a-z]+?)(\d+)_z", name)
    if m and m.group(1) in LAND_TEXT:
        return f"{LAND_TEXT[m.group(1)]} over the last {m.group(2)} days vs normal (standard deviations)"
    parts = name.split("_")
    if parts[0] == "f" and len(parts) == 4:
        v, w, stat = parts[1:]
        when = WEEK_TEXT.get(w, w)
        if v == "hotdry":
            return f"share of forecast scenarios with a hot, dry spell {when}"
        if stat == "plow":
            return f"share of forecast scenarios with low rainfall {when}"
        if stat == "spread":
            return f"forecast uncertainty in {VAR_TEXT.get(v, v)} {when}"
        return f"forecast {VAR_TEXT.get(v, v)} {when} vs normal (standard deviations)"
    return name


def _r(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def _pct(x, d=1):
    return _r(100 * x, d)


def _risk(cells: pd.DataFrame) -> dict:
    ok = cells[~cells.in_drought]
    out = {}
    for lead in LEADS:
        p, n = ok[f"p_{lead}"].mean(), ok[f"normal_{lead}"].mean()
        out[lead] = {"chance_pct": _pct(p), "normal_pct": _pct(n), "times_normal": _r(p / n if n else np.nan, 1)}
    return out


def _drivers(cells: pd.DataFrame, k: int = 3) -> list[dict]:
    """Features that most often push the 28-day risk up, weighted by how hard they push."""
    ok = cells[~cells.in_drought]
    rows = []
    for r in (1, 2, 3):
        rows.append(ok[[f"driver{r}", f"driver{r}_push", f"driver{r}_value"]].set_axis(["f", "push", "value"], axis=1))
    d = pd.concat(rows)
    d = d[d.push > 0]
    if d.empty:
        return []
    g = d.groupby("f").agg(push=("push", "sum"), value=("value", "median"), cells=("push", "size"))
    g = g.sort_values("push", ascending=False).head(k)
    return [{"feature": f, "meaning": describe(f), "typical_value": _r(v.value, 2),
             "share_of_area_where_top_driver_pct": _pct(v.cells / len(ok), 0)} for f, v in g.iterrows()]


def _outlook(cells: pd.DataFrame) -> dict:
    cols = [c for c in cells.columns if c.startswith("f_")]
    return {describe(c): _r(cells[c].mean(), 2) for c in cols}


class Tools:
    def __init__(self, pack: pd.DataFrame):
        self.p = pack

    def near(self, lat: float, lon: float, radius_km: float = 60) -> pd.DataFrame:
        dy = (self.p.lat - lat) * 111.0
        dx = (self.p.lon - lon) * 111.0 * np.cos(np.radians(lat))
        return self.p[np.hypot(dx, dy) <= radius_km]

    def area_summary(self, lat: float, lon: float, radius_km: float = 60) -> dict:
        """Risk, drivers, soil state, Drought Monitor status and weather outlook around a point."""
        c = self.near(lat, lon, radius_km)
        if c.empty:
            return {"error": "no land cells within that radius of the point (outside the study region?)"}
        cat = int(c.usdm_cat.mode().iloc[0])
        return {"lat": lat, "lon": lon, "radius_km": radius_km, "land_cells": len(c),
                "share_already_in_drought_pct": _pct(c.in_drought.mean(), 0),
                "root_zone_soil_moisture_percentile_0_40cm_median": _r(c.rz_pct.median(), 0),
                "drought_monitor_most_common": USDM.get(cat, str(cat)),
                "onset_risk": _risk(c), "top_drivers": _drivers(c), "weather_outlook": _outlook(c)}

    def hotspots(self, n: int = 5, lead: str = "any28") -> list[dict]:
        """The ~150 km blocks where onset risk is highest relative to normal."""
        ok = self.p[~self.p.in_drought].assign(by=lambda d: d.iy // 6, bx=lambda d: d.ix // 6)
        g = ok.groupby(["by", "bx"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), cells=("lat", "size"),
                                         p=(f"p_{lead}", "mean"), normal=(f"normal_{lead}", "mean"))
        g = g[g.cells >= 10].assign(ratio=lambda d: d.p / d.normal).sort_values("ratio", ascending=False)
        return [{"lat": _r(v.lat, 2), "lon": _r(v.lon, 2), "chance_pct": _pct(v.p), "normal_pct": _pct(v.normal),
                 "times_normal": _r(v.ratio, 1)} for _, v in g.head(n).iterrows()]

    def overview(self) -> dict:
        """Whole study region (36-49N, 104-80W): risk vs normal, share in drought."""
        return {"land_cells": len(self.p), "share_already_in_drought_pct": _pct(self.p.in_drought.mean(), 0),
                "share_drought_monitor_D1_or_worse_pct": _pct((self.p.usdm_cat >= 1).mean(), 0),
                "onset_risk": _risk(self.p)}

    def specs(self) -> dict:
        """name -> (JSON schema, function), for Chat.run."""
        point = {"type": "object", "properties": {
            "lat": {"type": "number", "description": "latitude, degrees north"},
            "lon": {"type": "number", "description": "longitude, degrees east (negative in the US)"},
            "radius_km": {"type": "number", "description": "radius around the point, default 60"}},
            "required": ["lat", "lon"]}
        return {
            "area_summary": ({"description": self.area_summary.__doc__, "parameters": point}, self.area_summary),
            "hotspots": ({"description": self.hotspots.__doc__, "parameters": {"type": "object", "properties": {
                "n": {"type": "integer"}, "lead": {"type": "string", "enum": LEADS}}}}, self.hotspots),
            "overview": ({"description": self.overview.__doc__, "parameters": {"type": "object", "properties": {}}},
                         self.overview),
        }
