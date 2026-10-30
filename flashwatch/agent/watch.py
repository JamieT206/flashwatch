"""The watcher: follows each subscribed farm through the season and speaks up only when something changes.

    python -m flashwatch.agent.watch --year 2012                      # template texts, no API
    python -m flashwatch.agent.watch --year 2012 --llm 3             # Muse Spark writes the first 3 alerts

Every forecast date (every other day here; daily when live) it reads the farm's state through the same tools
as Ask FlashWatch (agent/chat.py): the irrigation advice, the rain-fed decisions, the water account. It
compares with the last date and raises an alert when:
- the irrigation advice turns to "start" (urgent) or to "not yet" (heads-up), or eases back afterwards;
- a rain-fed decision turns to "act now" inside its window;
- the soil drops below the drought line;
- the root zone is within 2 days of stress (irrigated; at most once a week).
Quiet otherwise: most days need no message. With --llm, Muse Spark writes the alert from the event's facts
and the number checker vets it; otherwise a plain template. Writes site/data/alerts/<year>.json.
"""

from __future__ import annotations

import argparse
import json

from flashwatch.agent.chat import Farm
from flashwatch.agent.validate import unsupported
from flashwatch.site.build import SITE

SUBSCRIBERS = [
    {"name": "North Place", "lat": 45.38, "lon": -99.38, "crop": "corn", "watering": "irrigated", "soil": "Sandy loam", "gpm": 4.5},
    {"name": "Story County", "lat": 42.03, "lon": -93.47, "crop": "corn", "watering": "rain-fed", "soil": "Silt loam", "gpm": 0},
    {"name": "Hall County", "lat": 40.92, "lon": -98.34, "crop": "soybeans", "watering": "irrigated", "soil": "Silt loam", "gpm": 5.0},
]
RAINFED = ("fungicide", "nitrogen", "hay")
WRITER = """You write one short text message (at most 160 characters) for a farmer from FlashWatch, about the event
in the JSON. Start with what to do. Plain words, no dates (it is sent that day), whole numbers, "today" or
"now" instead of 0 days. Every number must come from the JSON. times_normal_next_7_days is the flash-drought
chance as a multiple of the normal chance for this date (not rain). A negative dollar value means starting
early costs more than it saves: say so in words or leave it out. Reply with the message only."""


def usual_irrigation(sub: dict, year: int, upto: str) -> dict:
    """The farm's usual schedule up to `upto`, standing in for the pivot's as-applied records: a full day of
    pumping whenever the root zone passes 80% of the stress line, from mid-May (decide/waterbook.py)."""
    from flashwatch.agent.farm import FarmData
    from flashwatch.agent.operator import SOILS, root_zone
    data = FarmData(year)
    piv = {"lat": sub["lat"], "lon": sub["lon"], "crop": sub["crop"], "soil": sub["soil"], "gpm_per_acre": sub["gpm"], "acres": 130}
    gross = sub["gpm"] * 24 * 60 / 27154
    taw = SOILS.get(sub["soil"], 2.0) * 3.0
    raw = (0.5 if sub["crop"].startswith("soy") else 0.55) * taw
    irr: dict = {}
    for md in [d for d in data.dates if "05-15" <= d <= upto]:
        rz = root_zone(data, piv, md, [{"date": f"{year}-{k}", "depth_in": v} for k, v in irr.items()])
        if rz["deficit_in"] > 0.8 * raw:
            irr[md] = round(2 * gross, 2)            # forecast dates are 2 days apart: two days' pumping
    return irr


def state(farm: Farm, sub: dict) -> dict:
    s = {"drought": False}
    adv = farm.irrigation_advice()
    s["drought"] = adv.get("soil_moisture_percentile") is None and "below the drought line" in adv.get("verdict", "")
    if sub["watering"] == "irrigated":
        s["verdict"] = adv.get("verdict", "")
        s["advice"] = adv
        wa = farm.water_account()
        s["days_to_stress"], s["share_full_pct"] = wa["days_until_stress_if_no_rain"], wa["root_zone_share_full_pct"]
    else:
        s["decisions"] = {d: farm.drought_decision(d) for d in RAINFED}
    return s


def events(prev: dict | None, s: dict, sub: dict, last_stress: str | None, date: str) -> list[dict]:
    ev = []
    if s["drought"] and not (prev or {}).get("drought") and not (prev or {}).get("drought_recent"):
        ev.append({"kind": "drought", "level": "info", "facts": {"what": "the soil around your farm is now in drought"}})
    if sub["watering"] == "irrigated" and "verdict" in s:
        # What the farmer was last told (start / watch / calm). Risk easing must hold for two forecasts in a
        # row before we say so, so a one-off dip doesn't send "eased" and then "rising" again two days later.
        v, told = s["verdict"], (prev or {}).get("told", "calm")
        level = "start" if v.startswith("start") else "watch" if v.startswith("not yet") else "calm"
        s["level"], s["told"] = level, told
        a = s["advice"]
        facts = {k: a[k] for k in ("chance_flash_drought_next_7_days_pct", "times_normal_next_7_days", "value_of_starting_today_dollars_per_acre")
                 if k in a}
        if level == "start" and told != "start":
            ev.append({"kind": "start", "level": "urgent", "facts": {"advice": v, **facts}})
            s["told"] = "start"
        elif level == "watch" and told == "calm":
            ev.append({"kind": "watch", "level": "heads-up", "facts": {"advice": v, **facts}})
            s["told"] = "watch"
        elif level == "calm" and told != "calm" and (prev or {}).get("level") == "calm":
            if not s["drought"]:
                ev.append({"kind": "ease", "level": "info", "facts": {"advice": v, **facts}})
            s["told"] = "calm"
        dts = s.get("days_to_stress")
        if dts is not None and dts <= 2 and not s["drought"] and (last_stress is None or _days(last_stress, date) >= 7):
            ev.append({"kind": "water_now", "level": "urgent",
                       "facts": {"root_zone_full_pct": s["share_full_pct"], "days_until_stress_if_no_rain": dts}})
    for d, x in (s.get("decisions") or {}).items():
        px = ((prev or {}).get("decisions") or {}).get(d, {})
        acting = x.get("in_window_today") and (x.get("verdict") == "act now" or "already in drought" in x.get("note", ""))
        was = px.get("in_window_today") and (px.get("verdict") == "act now" or "already in drought" in px.get("note", ""))
        if acting and not was:
            ev.append({"kind": f"act_{d}", "level": "urgent", "facts": {k: x[k] for k in x if k not in ("per",)}})
    return ev


def _days(a: str, b: str) -> int:
    import datetime as dt
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def template(e: dict, sub: dict) -> str:
    f = e["facts"]
    if e["kind"] == "start":
        v = f.get("value_of_starting_today_dollars_per_acre")
        return (f"{sub['name']}: start extra irrigation today. Flash-drought chance {f.get('chance_flash_drought_next_7_days_pct')}% "
                f"this week, {f.get('times_normal_next_7_days')}x normal." + (f" Worth about ${v:.0f} an acre on average." if v else ""))
    if e["kind"] == "watch":
        v = f.get("value_of_starting_today_dollars_per_acre")
        return (f"{sub['name']}: flash-drought risk rising ({f.get('chance_flash_drought_next_7_days_pct')}% this week). Not yet: we'll text if it's time."
                + (f" Starting today: {'-' if v < 0 else ''}${abs(v):.1f} an acre." if v is not None else ""))
    if e["kind"] == "ease":
        return f"{sub['name']}: risk has eased. Irrigate as usual."
    if e["kind"] == "water_now":
        d = f["days_until_stress_if_no_rain"]
        when = "today" if d < 1 else "tomorrow" if d < 2 else f"in {round(d)} days"
        return (f"{sub['name']}: your pivot isn't keeping up. Root zone {f['root_zone_full_pct']}% full, crop stress {when} "
                f"without rain. Run it full time.")
    if e["kind"] == "drought":
        return f"{sub['name']}: the soil around you is now in drought (rain alone isn't keeping up)."
    if e["kind"].startswith("act_"):
        if "note" in f:
            return f"{sub['name']}: {f['decision'].lower()}: your soil is already in drought, so the dry-weather case applies."
        return (f"{sub['name']}: {f['decision'].lower()}: flash-drought chance {f.get('chance_of_flash_drought_next_14_days_pct')}% "
                f"over 14 days, above the {f.get('acting_pays_when_chance_tops_pct')}% where it pays.")
    return f"{sub['name']}: {e['kind']}"


def follow(sub: dict, year: int, dates=None, chat=None, llm: int = 0) -> tuple[list[dict], int, int]:
    """One farm through one season: (alerts, forecast dates checked, Muse Spark drafts left)."""
    farm0 = Farm(sub["lat"], sub["lon"], year, "06-01", sub["crop"], sub["soil"], sub["gpm"] or 5.0, irrigation={})
    ds = dates or [d for d in farm0.data.dates if "06-01" <= d <= "08-31"]
    prev, last_stress, last_drought, alerts = None, None, None, []
    usual = usual_irrigation(sub, year, ds[-1]) if sub["watering"] == "irrigated" else {}
    for md in ds:
        irr = {k: v for k, v in usual.items() if k < md}
        farm = Farm(sub["lat"], sub["lon"], year, md, sub["crop"], sub["soil"], sub["gpm"] or 5.0, irrigation=irr)
        s = state(farm, sub)
        if prev is not None:
            prev["drought_recent"] = last_drought is not None and _days(last_drought, f"{year}-{md}") < 28
        evs = events(prev, s, sub, last_stress, f"{year}-{md}")
        if any(e["kind"] == "water_now" for e in evs):
            evs = [e for e in evs if e["kind"] != "ease"]
        for e in evs:
            if e["kind"] == "drought":
                last_drought = f"{year}-{md}"
            if e["kind"] == "water_now":
                last_stress = f"{year}-{md}"
            text, by = template(e, sub), "template"
            if chat and llm > 0:
                draft = (chat.complete([{"role": "system", "content": WRITER},
                                        {"role": "user", "content": json.dumps({"farm": sub["name"], "date": f"{year}-{md}", **e})}]).get("content") or "").strip()
                if draft and not unsupported(draft, [e["facts"], {"date": [year, int(md[:2]), int(md[3:])]}]):
                    text, by = draft[:200], chat.model
                llm -= 1
            alerts.append({"date": f"{year}-{md}", "kind": e["kind"], "level": e["level"], "text": text, "by": by, "facts": e["facts"]})
        prev = s
    return alerts, len(ds), llm


def run(year: int, llm: int = 0, dates=None) -> dict:
    from flashwatch.agent.scorecard import score_alerts
    chat = None
    if llm:
        from flashwatch.agent.llm import Chat
        chat = Chat(timeout=300)
    out = {"year": year, "farms": []}
    for sub in SUBSCRIBERS:
        alerts, n, llm = follow(sub, year, dates, chat, llm)
        out["farms"].append({**sub, "dates_checked": n, "alerts": score_alerts(alerts, sub, year)})
    if chat:
        out["usage"] = chat.usage
    (SITE / "data" / "alerts").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / "alerts" / f"{year}.json").write_text(json.dumps(out, default=float))
    return out


def main():
    from flashwatch.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2012)
    ap.add_argument("--llm", type=int, default=0, help="how many alerts Muse Spark writes (the rest use templates)")
    a = ap.parse_args()
    r = run(a.year, a.llm)
    for f in r["farms"]:
        print(f"\n{f['name']} ({f['watering']} {f['crop']}): {len(f['alerts'])} alerts in {f['dates_checked']} forecast dates")
        for x in f["alerts"]:
            print(f"  {x['date']} [{x['level']}] {x['text']}  ({x['by']})")
    if "usage" in r:
        print("tokens:", r["usage"])


if __name__ == "__main__":
    main()
