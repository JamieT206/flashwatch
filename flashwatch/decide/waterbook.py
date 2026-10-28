"""The field's water account (the irrigator's "checkbook"; James, Oct 29, QUESTIONS Q15/Q16).

Each day the root zone loses what the crop uses (reference ET x a crop coefficient for the growth
stage, FAO-56) and gains rain and irrigation. The deficit says how far the soil is from full, when
the crop starts to feel short of water, and how much irrigation it would take to fill it. Water the
farmer has already put on (typed in, or read from the pivot through a connector) counts.

Reference ET: FAO-56 Penman-Monteith from ERA5-Land daily means (temperature, dewpoint, solar
radiation), with wind set to the FAO default of 2 m/s and net radiation estimated from solar radiation
(FAO-56 eqs 21-39). Daily means instead of max/min understate the vapour-pressure deficit a little.
Mirrored in site/assets/engine.js (FW.account); the site data carry daily rain and ETo per cell.
"""

from __future__ import annotations

import numpy as np

MM_PER_IN = 25.4
GAMMA = 0.066            # psychrometric constant, kPa/C, near 300 m elevation
DTR = 12.0               # assumed daily temperature range, C (see eto_pm)

# FAO-56 crop coefficients by date for a typical Corn Belt planting: (month-day, Kc) points,
# straight lines between. Corn planted ~May 1 (Kc ini 0.3, mid 1.2 from ~V16, end 0.6 at black layer);
# soybeans ~May 15 (0.4, 1.15, 0.5).
KC = {"corn": [("04-01", 0.3), ("05-31", 0.3), ("07-05", 1.2), ("08-25", 1.2), ("09-25", 0.6), ("10-31", 0.6)],
      "soybeans": [("04-01", 0.4), ("06-15", 0.4), ("07-20", 1.15), ("08-30", 1.15), ("09-30", 0.5), ("10-31", 0.5)]}


def _doy(md: str) -> int:
    m, d = int(md[:2]), int(md[3:])
    return [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][m - 1] + d


def kc(crop: str, md: str) -> float:
    pts = KC[crop]
    return float(np.interp(_doy(md), [_doy(a) for a, _ in pts], [v for _, v in pts]))


def eto_pm(t2m_k, d2m_k, ssrd_j, lat_deg, doy, u2: float = 2.0, z_m: float = 300.0) -> np.ndarray:
    """FAO-56 Penman-Monteith reference ET, mm/day. Arrays broadcast: (day, lat, lon) with lat (lat, 1)
    and doy (day, 1, 1)."""
    t = np.asarray(t2m_k, float) - 273.15
    svp = lambda x: 0.6108 * np.exp(17.27 * x / (x + 237.3))  # noqa: E731
    # FAO-56 averages the saturation pressure at the day's max and min; with daily means only, assume a
    # typical summer range of 12 C (mean +/- 6), or the deficit (and ETo) comes out ~30% low.
    es = (svp(t + DTR / 2) + svp(t - DTR / 2)) / 2
    td = np.asarray(d2m_k, float) - 273.15
    ea = 0.6108 * np.exp(17.27 * td / (td + 237.3))
    delta = 4098 * es / (t + 237.3) ** 2
    rs = np.asarray(ssrd_j, float) / 1e6
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * z_m) * ra
    rnl = 4.903e-9 * (t + 273.16) ** 4 * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0, None))) * (1.35 * np.clip(rs / np.maximum(rso, 1e-6), 0.3, 1) - 0.35)
    rn = 0.77 * rs - rnl
    num = 0.408 * delta * rn + GAMMA * 900 / (t + 273) * u2 * np.clip(es - ea, 0, None)
    return np.clip(num / (delta + GAMMA * (1 + 0.34 * u2)), 0, None)


def eto_makkink(t2m_k, ssrd_j) -> np.ndarray:
    """Makkink reference ET, mm/day (radiation only; kept for comparison, low in the windy Plains)."""
    t = np.asarray(t2m_k, float) - 273.15
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    delta = 4098 * es / (t + 237.3) ** 2
    rs = np.asarray(ssrd_j, float) / 1e6
    return np.clip(0.61 * delta / (delta + GAMMA) * rs / 2.45 - 0.12, 0, None)


def account(days: list[str], rain_in, eto_in, crop: str, taw_in: float, p: float = 0.55,
            irrigation: dict | None = None, efficiency: float = 0.85) -> dict:
    """Day-by-day root-zone deficit (inches below full) starting full on days[0].

    days: month-day strings; rain_in, eto_in: inches a day; irrigation: {month-day: inches applied}.
    Returns deficit after each day, crop water use, and the stress line (readily available water)."""
    irrigation = irrigation or {}
    raw = p * taw_in
    dr, out, use = 0.0, [], []
    for md, r, e in zip(days, rain_in, eto_in):
        etc = kc(crop, md) * e
        ks = 1.0 if dr <= raw else max(0.0, (taw_in - dr) / (taw_in - raw))
        dr = min(taw_in, max(0.0, dr + ks * etc - r - irrigation.get(md, 0.0) * efficiency))
        out.append(dr)
        use.append(ks * etc)
    return {"deficit": np.array(out), "use": np.array(use), "raw": raw, "taw": taw_in}


def status(deficit: float, raw: float, taw: float, use_per_day: float, capacity_in_day: float) -> dict:
    """Where the field stands: share full, days of no rain until stress, and hours of pivot time to fill."""
    to_stress = max(0.0, raw - deficit) / use_per_day if use_per_day > 0 else float("inf")
    return {"share_full": 1 - deficit / taw, "days_to_stress": to_stress, "inches_to_fill": deficit,
            "pivot_days_to_fill": deficit / capacity_in_day if capacity_in_day > 0 else float("inf")}
