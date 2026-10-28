"""The watcher keeps score of itself (DESIGN.md G1: "keeps a track record of its past alerts").

    python -m flashwatch.agent.scorecard             # all replay years, every 8th land cell (~25 min)
    python -m flashwatch.agent.scorecard --every 40  # fewer farms, quicker

Every text the watcher sends on a forecast (start, rising risk, act now on a rain-fed decision, risk
eased) is checked against what happened at that farm: did a flash drought begin within the next 14
days? Texts about the field itself (the pivot isn't keeping up, the soil is in drought) aren't
forecasts and aren't scored. The replay years are cross-validation years, so each was predicted by a
model that never saw it.

The track record runs the watcher (template texts, no API) on a spread of farms across the region,
irrigated and rain-fed corn, for every replay year, and sets the hit rate of each kind of text
against the normal chance: how often a flash drought begins within 14 days of any summer forecast
date at those farms while the soil isn't already in drought. It also counts how many onsets had a
warning in the 14 days before. Writes site/data/scorecard.json (read by site/agents.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from flashwatch.agent.farm import FarmData
from flashwatch.site.build import SITE

WINDOW = 14
WARN = ("start", "watch")                     # irrigated texts that say risk is up
CLEAR = ("ease",)


def _day(year: int, md: str) -> dt.date:
    return dt.date(year, int(md[:2]), int(md[3:]))


def began_within(year: int, md: str, onsets: list[str], days: int = WINDOW) -> bool:
    d = _day(year, md)
    return any(0 < (_day(year, o) - d).days <= days for o in onsets)


def is_warning(a: dict) -> bool:
    """A text that says a flash drought may be coming. A rain-fed "act now" sent because the soil is
    already in drought ("note" in its facts) describes the soil now, not a forecast."""
    return a["kind"] in WARN or (a["kind"].startswith("act_") and "note" not in a["facts"])


def score_alerts(alerts: list[dict], sub: dict, year: int, data: FarmData | None = None) -> list[dict]:
    """Adds "outcome" to each forecast text: what happened in the next 14 days, and whether the text was right."""
    onsets = (data or FarmData(year)).record(sub["lat"], sub["lon"])["onsets"]
    for a in alerts:
        if not (is_warning(a) or a["kind"] in CLEAR):
            continue
        came = began_within(year, a["date"][5:], onsets)
        a["outcome"] = {"flash_drought_within_14_days": came, "right": came if is_warning(a) else not came}
    return alerts


def farms(meta: dict, every: int) -> list[tuple[float, float]]:
    cells = sorted(meta["cells"])
    return [(round(meta["lat0"] - iy * meta["res"], 2), round(meta["lon0"] + ix * meta["res"], 2))
            for iy, ix in cells[::every]]


def track_record(years: list[int], every: int = 8) -> dict:
    from flashwatch.agent.watch import follow
    kinds: dict[str, list[bool]] = {}
    base, onsets_n, warned_n, seasons = [], 0, 0, 0
    for year in years:
        data = FarmData(year)
        ds = [d for d in data.dates if "06-01" <= d <= "08-31"]
        for lat, lon in farms(data.meta, every):
            r = data.record(lat, lon)
            on = [o for o in r["onsets"] if "06-01" <= o <= "09-14"]
            for watering, gpm in (("irrigated", 5.0), ("rain-fed", 0)):
                sub = {"name": "farm", "lat": lat, "lon": lon, "crop": "corn", "watering": watering, "soil": "Silt loam", "gpm": gpm}
                alerts, _, _ = follow(sub, year, ds)
                seasons += 1
                for a in score_alerts(alerts, sub, year, data):
                    if "outcome" in a:
                        k = "act" if a["kind"].startswith("act_") else a["kind"]
                        kinds.setdefault(f"{watering}:{k}", []).append(a["outcome"]["flash_drought_within_14_days"])
                warn_days = [_day(year, a["date"][5:]) for a in alerts if is_warning(a)]
                for o in on:
                    onsets_n += 1
                    warned_n += any(0 < (_day(year, o) - w).days <= WINDOW for w in warn_days)
            for md in ds:
                rz = r["rz"][data.dates.index(md)]
                if rz is not None and rz >= 20:
                    base.append(began_within(year, md, r["onsets"]))
    normal = sum(base) / len(base)
    rows = []
    for k, v in sorted(kinds.items()):
        watering, kind = k.split(":")
        rate = sum(v) / len(v)
        rows.append({"watering": watering, "kind": kind, "sent": len(v), "followed_by_flash_drought_pct": round(100 * rate, 1),
                     "times_normal": round(rate / normal, 1) if normal else None})
    return {"years": years, "farms": len(farms(FarmData(years[0]).meta, every)), "farm_seasons": seasons,
            "window_days": WINDOW, "normal_pct": round(100 * normal, 1), "kinds": rows,
            "onsets": onsets_n // 2, "onsets_warned_pct": round(100 * warned_n / max(onsets_n, 1), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int)
    ap.add_argument("--every", type=int, default=8, help="take every n-th land cell as a farm")
    a = ap.parse_args()
    years = a.years or sorted(int(p.name) for p in (SITE / "data" / "tiles").iterdir() if p.name.isdigit())
    out = track_record(years, a.every)
    (SITE / "data" / "scorecard.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
