"""Muse Spark writes the farm's daily note and text alert (DESIGN.md O6).

    python -m flashwatch.agent.farm --lat 40.87 --lon -97.99 --year 2012 --dates 06-20 07-01

The agent calls one tool, `farm_status`, which reads the site's replay data (site/data, from
flashwatch.site.build) and runs the decision engine with the default farm settings: risk this week
and next vs normal, soil moisture, Drought Monitor, the verdict (start the pivot / not yet / no
need), the expected values of today vs the next forecast, what would change it. It writes a text
message (at most 160 characters) and a short note. Every number is checked against the tool result
(agent/validate.py), with up to 2 redrafts. Output: site/data/notes/<year>_<iy>_<ix>_<MM-DD>.json,
which the farm page shows.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from flashwatch.agent.llm import Chat
from flashwatch.agent.validate import drop_sentences, unsupported
from flashwatch.decide import curve, load
from flashwatch.decide.engine import daily, decide, risk_threshold, two_way
from flashwatch.decide.trigger import trigger
from flashwatch.site.build import SITE

USDM = {-1: "none", 0: "D0 abnormally dry", 1: "D1 moderate drought", 2: "D2 severe drought",
        3: "D3 extreme drought", 4: "D4 exceptional drought"}

SYSTEM = """You are FlashWatch, a flash-drought adviser for an irrigated corn farm in the US Corn Belt.
Each morning you tell the farmer whether to start the centre pivot early. Call the farm_status tool
first, then reply with JSON only: {"sms": "...", "note": "..."}.
- "sms": a text message of at most 160 characters: the verdict and the one number that matters.
- "note": at most 90 words for the farmer: the verdict, why (risk vs normal, soil, the forecast),
  what would change it, and the money at stake per acre.
Rules: every number must come from the tool result (percentages are already in percent; round them
to whole numbers). Do not invent numbers or use outside knowledge of what happened later.
Plain words, no jargon. The farmer decides; you advise."""


class FarmData:
    """The site's replay data for one year (site/data/meta.json and the tiles)."""

    def __init__(self, year: int, site=SITE):
        self.meta = json.loads((site / "data" / "meta.json").read_text())
        self.year, self.site = year, site
        self.dates = self.meta["dates"]

    def cell(self, lat, lon):
        m = self.meta
        return min(m["cells"], key=lambda c: (m["lat0"] - c[0] * m["res"] - lat) ** 2
                   + ((m["lon0"] + c[1] * m["res"] - lon) * np.cos(np.radians(lat))) ** 2)

    def record(self, lat, lon):
        m, (iy, ix) = self.meta, self.cell(lat, lon)
        f = (self.site / "data" / "tiles" / str(self.year)
             / f"{iy // m['thin'] // m['tile']}_{ix // m['thin'] // m['tile']}.json")
        return next(r for r in json.loads(f.read_text()) if r["iy"] == iy and r["ix"] == ix)


def status(data: FarmData, lat: float, lon: float, md: str, settings: dict | None = None,
           action: str = "irrigation") -> dict:
    """Everything the note may say, rounded, for one farm and date. action: irrigation or
    irrigation_soybeans (config/actions.json)."""
    r, i = data.record(lat, lon), data.dates.index(md)
    c = curve(load(overrides=settings)[action])
    p1, p2, n1 = r["p1"][i], r["p2"][i], r["n1"][i]
    out = {"date": f"{data.year}-{md}", "soil_moisture_percentile": r["rz"][i],
           "drought_monitor": USDM.get(r["usdm"][i], "none")}
    if p1 is None:
        out["verdict"] = "irrigate as usual: the soil is already below the drought line"
        return out
    pd_ = daily({"week1": p1, "week2": p2})
    if r.get("mp1") and r["mp1"][i]:
        t = trigger(c, r["mp1"][i], r["mp2"][i], {"rain_in": r["mr"][i], "temp_f": r["mt"][i]})
        act_now, ev_now, ev_wait, trig = t["act_now"], t["ev_now"], t["ev_wait"], t["trigger"]
    else:
        d = decide(c, pd_, two_way(0.5), 2)
        act_now, ev_now, ev_wait, trig = d.act_now, d.ev_now, d.ev_wait, None
    m = risk_threshold(c, pd_)
    verdict = ("start the pivot today" if act_now else "not yet: wait for the next forecast" if ev_wait > 0
               else "no need to start early")
    out.update({
        "chance_flash_drought_next_7_days_pct": round(100 * p1, 1),
        "normal_chance_next_7_days_pct": round(100 * n1, 1),
        "times_normal_next_7_days": round(p1 / n1, 1) if n1 else None,
        "chance_days_8_to_14_pct": round(100 * p2, 1),
        "verdict": verdict,
        "value_of_starting_today_dollars_per_acre": round(ev_now, 1),
        "value_of_deciding_at_next_forecast_dollars_per_acre": round(ev_wait, 1),
        "next_forecast_in_days": 2,
        "loss_avoided_with_7_days_warning_dollars_per_acre": round(float(np.interp(7, c.notice, c.avoided))),
        "how_to_read": "value_* are average dollars per acre against starting the pivot the usual way (after the "
                       "crop shows stress); negative means starting today costs more in pumping than it saves on "
                       "average. The loss avoided is what a week's warning saves if a flash drought does come.",
    })
    if trig:
        out["what_would_change_it"] = f"next forecast shows {trig['text']}"
    elif not act_now and np.isfinite(m) and m * p1 < 1:
        out["what_would_change_it"] = f"next forecast shows a 7-day chance of at least {round(100 * m * p1)}%"
    return out


def write(data: FarmData, lat: float, lon: float, md: str, place: str, chat: Chat | None = None,
          max_redrafts: int = 2) -> dict:
    chat = chat or Chat()
    spec = {"farm_status": ({"description": "Today's flash-drought risk, soil, Drought Monitor and the "
                                            "irrigation decision for the farm.",
                              "parameters": {"type": "object", "properties": {}}},
                             lambda: status(data, lat, lon, md))}
    ask = f"Write this morning's note for the farm near {place}."
    text, messages, log = chat.run([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}], spec)
    results = [r for _, _, r in log] or [status(data, lat, lon, md)]
    drafts = []
    for _ in range(max_redrafts):
        bad = unsupported(text, results)
        drafts.append({"draft": text, "untraced": bad})
        if not bad:
            break
        fix = (f"These figures are not in the tool result: {', '.join(bad)}. Rewrite using only its numbers. "
               "Reply with the JSON only.")
        text, messages, more = chat.run(messages + [{"role": "user", "content": fix}], spec)
        log += more
        results = [r for _, _, r in log] or results
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        out = json.loads(m.group(0)) if m else {"sms": "", "note": text}
    except json.JSONDecodeError:
        out = {"sms": "", "note": text}
    bad = unsupported(" ".join(str(v) for v in out.values()), results)
    if bad:
        out = {k: drop_sentences(str(v), bad) for k, v in out.items()}
    return {**out, "status": results[-1], "model": chat.model, "cut_untraced": bad, "drafts": drafts,
            "usage": dict(chat.usage), "place": place}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--dates", nargs="+", required=True, help="MM-DD forecast dates")
    ap.add_argument("--place", default="")
    args = ap.parse_args()
    data = FarmData(args.year)
    iy, ix = data.cell(args.lat, args.lon)
    outdir = SITE / "data" / "notes"
    outdir.mkdir(parents=True, exist_ok=True)
    chat = Chat()
    for md in args.dates:
        note = write(data, args.lat, args.lon, md, args.place or f"{args.lat}, {args.lon}", chat)
        (outdir / f"{args.year}_{iy}_{ix}_{md}.json").write_text(json.dumps(note, indent=1, default=float))
        print(f"--- {md}: {note['status'].get('verdict')}\nSMS: {note.get('sms')}\n{note.get('note')}\n"
              f"[checker: {[d['untraced'] for d in note['drafts']]}, cut {note['cut_untraced']}]")
    print(f"tokens: {chat.usage}")


if __name__ == "__main__":
    main()
