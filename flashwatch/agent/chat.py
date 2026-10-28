"""Ask FlashWatch: a farmer's questions answered by Muse Spark through FlashWatch's tools.

    python -m flashwatch.agent.chat --lat 40.87 --lon -97.99 --year 2012 --date 07-03       # type questions
    python -m flashwatch.agent.chat ... --ask "Should I spray fungicide on my beans?"       # one question

The agent answers from tool results only: the irrigation advice, the rain-fed decisions (fungicide,
nitrogen, silage, hay), the field's water account, the multi-pivot farm, and what the terms mean. It can
also log irrigation the farmer reports ("I put an inch on the north pivot yesterday"), which the water
account then counts. Every number in a reply is checked against the tool results (agent/validate.py);
a reply with an untraced figure goes back for a redraft, and anything still untraced is cut.
The same agent serves the farm page through flashwatch.agent.serve.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from flashwatch.agent.farm import FarmData, status
from flashwatch.agent.validate import drop_sentences, unsupported
from flashwatch.config import DATA
from flashwatch.decide import curve, load
from flashwatch.decide.engine import daily, decide, two_way

SYSTEM = """You are FlashWatch, an assistant for a farmer in the US Corn Belt or northern Plains.
FlashWatch forecasts the chance of a flash drought (soil drying into drought within two or three
weeks) at the farm, one and two weeks ahead, and works out what extra irrigation and other drought
decisions are worth in dollars.
Answer the farmer's question using the tools. Call the tool that fits before answering:
- irrigation_advice: start extra irrigation today, wait, or irrigate as usual, with dollars;
- drought_decision: fungicide, side-dress nitrogen, silage, hay (mostly for rain-fed farms);
- water_account: how full the root zone is, days until stress, irrigation logged so far;
- log_irrigation: when the farmer says they put water on (date and inches), record it, then
  check water_account;
- explain: what a FlashWatch term means.
Rules: every number you give must come from a tool result; round dollars and percentages to whole
numbers. Say plainly when something is uncertain. At most 120 words, plain words, no jargon, no
lists unless the farmer asks for steps. If a question is outside FlashWatch (markets, chemicals,
legal), say so briefly and suggest who to ask."""

HELP = {
    "flash drought": "Soil moisture in the top 40 cm falling from normal (40th percentile or above) to drought (20th or below) and staying there 10 days or more, usually within two or three weeks.",
    "percentile": "How wet the soil is compared with the same time of year in 2005-2024 at this place: 50 is typical, below 20 is drought.",
    "normal": "How often a flash drought begins in that week at this place and time of year, from 2005-2024 records.",
    "start today": "The average gain per acre from starting extra irrigation today, against starting a couple of days after the crop shows stress.",
    "decide at the next forecast": "The average gain per acre from waiting two days for the next forecast, then deciding.",
    "water account": "A day-by-day count of the water in the root zone: rain and irrigation in, crop water use out (FAO-56).",
    "stress line": "The point where the crop can't take up water fast enough: about 55% of the root zone's water used for corn.",
}
ACTIONS = {"fungicide": "fungicide_corn", "fungicide_soy": "fungicide_soy", "nitrogen": "nitrogen", "silage": "silage", "hay": "hay"}
LOG = DATA / "agent" / "irrigation_log.json"


class Farm:
    """One farmer's session: place, date and the water they've logged."""

    def __init__(self, lat: float, lon: float, year: int, md: str, crop: str = "corn", soil: str = "Silt loam",
                 gpm_per_acre: float = 5.0, irrigation: dict | None = None):
        self.data, self.lat, self.lon, self.year, self.md = FarmData(year), lat, lon, year, md
        self.crop, self.soil, self.gpm = crop, soil, gpm_per_acre
        self.key = f"{lat:.2f},{lon:.2f},{year}"
        self.log = json.loads(LOG.read_text()) if LOG.exists() else {}
        if irrigation is not None:                      # the farm page sends its own log
            self.log[self.key] = {k: float(v) for k, v in irrigation.items()}
            self.persist = False
        else:
            self.persist = True

    def pivot(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "crop": self.crop, "soil": self.soil, "gpm_per_acre": self.gpm, "acres": 130}

    def irrigation(self) -> dict:
        return self.log.get(self.key, {})

    # ----- tools -----
    def irrigation_advice(self) -> dict:
        from flashwatch.agent.operator import root_zone
        piv = self.pivot()
        rz = root_zone(self.data, piv, self.md, [{"date": f"{self.year}-{d}", "depth_in": v} for d, v in self.irrigation().items()])
        from flashwatch.agent.operator import SOILS
        s = status(self.data, self.lat, self.lon, self.md, {"dr0": round(rz["deficit_in"], 2) if self.irrigation() else 0.0,
                                                            "gpm_per_acre": self.gpm, "awc_in_per_ft": SOILS.get(self.soil, 2.0)},
                   "irrigation_soybeans" if self.crop.startswith("soy") else "irrigation")
        return {k: v for k, v in s.items() if k != "how_to_read"}

    def drought_decision(self, decision: str) -> dict:
        name = ACTIONS.get(decision.lower().split()[0], decision)
        if self.crop.startswith("soy") and name == "fungicide_corn":
            name = "fungicide_soy"
        cfg = load()
        if name not in cfg:
            return {"error": f"unknown decision {decision!r}; one of {sorted(ACTIONS)}"}
        a = cfg[name]
        r, i = self.data.record(self.lat, self.lon), self.data.dates.index(self.md)
        win = a.get("window", ["04-01", "10-31"])
        out = {"decision": a["label"], "decided_between": f"{win[0]} and {win[1]}", "in_window_today": win[0] <= self.md <= win[1],
               "per": a.get("per", "acre")}
        if r["p1"][i] is None:
            out["note"] = "the soil is already in drought, so the drought case applies"
            return out
        c = curve(a)
        d = decide(c, daily({"week1": r["p1"][i], "week2": r["p2"][i]}), two_way(0.5), 2)
        gain = float(c.avoided.max())
        out.update({"chance_of_flash_drought_next_14_days_pct": round(100 * min(1.0, r["p1"][i] + r["p2"][i])),
                    "acting_pays_when_chance_tops_pct": round(100 * c.cost_noevent / (gain + c.cost_noevent)) if gain + c.cost_noevent > 0 else 0,
                    "dollars_if_flash_drought_comes": round(gain), "dollars_if_it_does_not": round(-c.cost_noevent),
                    "average_value_of_acting_today_dollars": round(d.ev_now), "verdict": "act now" if d.act_now else ("not yet" if d.ev_wait > 0 else "keep your plan")})
        return out

    def water_account(self) -> dict:
        from flashwatch.agent.operator import root_zone
        piv = self.pivot()
        rz = root_zone(self.data, piv, self.md, [{"date": f"{self.year}-{d}", "depth_in": v} for d, v in self.irrigation().items()])
        rz.pop("deficit_in")
        full = rz.pop("share_full")
        return {**rz, "root_zone_share_full_pct": round(100 * full),
                "irrigation_logged_in": self.irrigation(), "as_of": f"{self.year}-{self.md}"}

    def log_irrigation(self, date: str, inches: float) -> dict:
        md = date[-5:] if len(date) >= 5 else date
        try:
            dt.date.fromisoformat(f"{self.year}-{md}")
        except ValueError:
            return {"error": "date must be MM-DD or YYYY-MM-DD"}
        if not 0 < float(inches) <= 3:
            return {"error": "inches must be between 0 and 3 for one day"}
        if md > self.md:
            return {"error": f"{md} is after today ({self.md})"}
        log = self.log.setdefault(self.key, {})
        log[md] = round(log.get(md, 0.0) + float(inches), 2)
        if self.persist:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            LOG.write_text(json.dumps(self.log, indent=1))
        return {"logged": {"date": md, "inches": float(inches)}, "total_logged_in": round(sum(log.values()), 2)}

    def explain(self, term: str) -> dict:
        t = term.lower()
        hit = {k: v for k, v in HELP.items() if k in t or t in k}
        return hit or {"terms": sorted(HELP)}

    def tools(self) -> dict:
        s = lambda props=None, req=None: {"parameters": {"type": "object", "properties": props or {}, "required": req or []}}  # noqa: E731
        return {
            "irrigation_advice": ({"description": "Today's extra-irrigation advice for the farm, with dollars per acre.", **s()}, self.irrigation_advice),
            "drought_decision": ({"description": "A drought decision: fungicide, nitrogen, silage or hay.",
                                  **s({"decision": {"type": "string", "enum": sorted(ACTIONS)}}, ["decision"])}, self.drought_decision),
            "water_account": ({"description": "The field's root-zone water account today.", **s()}, self.water_account),
            "log_irrigation": ({"description": "Record irrigation the farmer put on.",
                                **s({"date": {"type": "string", "description": "MM-DD"}, "inches": {"type": "number"}}, ["date", "inches"])}, self.log_irrigation),
            "explain": ({"description": "What a FlashWatch term means.", **s({"term": {"type": "string"}}, ["term"])}, self.explain),
        }


def ask(farm: Farm, question: str, chat, history: list | None = None, max_redrafts: int = 1) -> dict:
    """One question in, one checked answer out (plus the tool calls, for the record)."""
    msgs = [{"role": "system", "content": SYSTEM + f"\nToday is {farm.year}-{farm.md}. The farm is at {farm.lat}, {farm.lon}; crop: {farm.crop}."}]
    msgs += history or []
    msgs.append({"role": "user", "content": question})
    log_all = []
    for attempt in range(max_redrafts + 1):
        text, msgs, log = chat.run(msgs, farm.tools())
        log_all += log
        bad = unsupported(text, [r for _, _, r in log_all])
        if not bad or attempt == max_redrafts:
            break
        msgs.append({"role": "user", "content": f"These figures aren't in the tool results: {', '.join(bad)}. Rewrite using only tool numbers."})
    bad = unsupported(text, [r for _, _, r in log_all])
    if bad:
        text = drop_sentences(text, bad)
    return {"answer": text, "tools": [{"tool": n, "args": a} for n, a, _ in log_all], "cut_untraced": bad}


def main():
    from flashwatch.agent.llm import Chat
    from flashwatch.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=40.87)
    ap.add_argument("--lon", type=float, default=-97.99)
    ap.add_argument("--year", type=int, default=2012)
    ap.add_argument("--date", default="07-03")
    ap.add_argument("--crop", default="corn")
    ap.add_argument("--ask", nargs="*", help="questions to ask in order (default: type them)")
    a = ap.parse_args()
    farm, chat = Farm(a.lat, a.lon, a.year, a.date, a.crop), Chat(timeout=300)
    hist: list = []
    qs = a.ask if a.ask else iter(lambda: input("you> "), "")
    for q in qs:
        r = ask(farm, q, chat, hist)
        print(f"you> {q}\nFlashWatch> {r['answer']}\n  (tools: {', '.join(t['tool'] for t in r['tools'])})")
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
    print("tokens:", chat.usage)


if __name__ == "__main__":
    main()
