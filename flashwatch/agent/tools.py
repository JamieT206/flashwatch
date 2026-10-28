"""Tools the analyst agent calls (DESIGN.md G1). Each reads one weekly pack (agent/pack.py) and
returns small JSON-ready dicts of rounded numbers, which the number checker later traces.

Probabilities are returned as fractions (0.12 = 12%); "times_normal" is the forecast divided by the
normal chance for that place and date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEADS = ["week1", "week2", "week3", "week4", "any28"]
USDM = {-1: "none", 0: "D0 (abnormally dry)", 1: "D1 (moderate drought)", 2: "D2 (severe drought)",
        3: "D3 (extreme drought)", 4: "D4 (exceptional drought)"}
# Plain-language names for the features most likely to show up as drivers; others pass through.
FEATURE_TEXT = {
    "rz_pct": "root-zone soil moisture percentile (low = dry)",
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
    "f_tp_w1_z": "forecast rainfall next week vs normal",
    "f_tp_w2_z": "forecast rainfall in week 2 vs normal",
    "f_tp_w12_z": "forecast rainfall over the next 2 weeks vs normal",
    "f_t2m_w1_z": "forecast temperature next week vs normal",
    "f_t2m_w12_z": "forecast temperature over the next 2 weeks vs normal",
    "f_vpd_w12_z": "forecast air dryness over the next 2 weeks vs normal",
    "f_hotdry_w1_p": "share of forecast scenarios with a hot, dry week next week",
    "f_hotdry_w12_p": "share of forecast scenarios with a hot, dry fortnight",
    "f_dryhot_w12_z": "forecast hot-and-dry index over the next 2 weeks",
    "f_sm100chg_w3_z": "forecast soil drying by week 3",
}


def _r(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def _risk(cells: pd.DataFrame) -> dict:
    ok = cells[~cells.in_drought]
    out = {}
    for lead in LEADS:
        p, n = ok[f"p_{lead}"].mean(), ok[f"normal_{lead}"].mean()
        out[lead] = {"chance": _r(p), "normal": _r(n), "times_normal": _r(p / n if n else np.nan, 1)}
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
    return [{"feature": f, "meaning": FEATURE_TEXT.get(f, f), "typical_value": _r(v.value, 2),
             "share_of_cells": _r(v.cells / len(ok), 2)} for f, v in g.iterrows()]


def _outlook(cells: pd.DataFrame) -> dict:
    cols = [c for c in cells.columns if c.startswith("f_")]
    return {FEATURE_TEXT.get(c, c): _r(cells[c].mean(), 2) for c in cols}


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
                "share_already_in_drought": _r(c.in_drought.mean(), 2),
                "soil_moisture_percentile_median": _r(c.rz_pct.median(), 0),
                "drought_monitor_most_common": USDM.get(cat, str(cat)),
                "onset_risk": _risk(c), "top_drivers": _drivers(c), "weather_outlook": _outlook(c)}

    def hotspots(self, n: int = 5, lead: str = "any28") -> list[dict]:
        """The ~150 km blocks where onset risk is highest relative to normal."""
        ok = self.p[~self.p.in_drought].assign(by=lambda d: d.iy // 6, bx=lambda d: d.ix // 6)
        g = ok.groupby(["by", "bx"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), cells=("lat", "size"),
                                         p=(f"p_{lead}", "mean"), normal=(f"normal_{lead}", "mean"))
        g = g[g.cells >= 10].assign(ratio=lambda d: d.p / d.normal).sort_values("ratio", ascending=False)
        return [{"lat": _r(v.lat, 2), "lon": _r(v.lon, 2), "chance": _r(v.p), "normal": _r(v.normal),
                 "times_normal": _r(v.ratio, 1)} for _, v in g.head(n).iterrows()]

    def overview(self) -> dict:
        """Whole study region (36-49N, 104-80W): risk vs normal, share in drought."""
        return {"land_cells": len(self.p), "share_already_in_drought": _r(self.p.in_drought.mean(), 2),
                "share_drought_monitor_D1_or_worse": _r((self.p.usdm_cat >= 1).mean(), 2),
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
