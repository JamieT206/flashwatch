"""The irrigation operator: Muse Spark plans which pivots to start, the farmer approves, the
connector carries it out (DESIGN.md O10).

    python -m flashwatch.agent.operator --year 2012 --dates 07-01 07-03 07-05 --approve "demo farmer"
    python -m flashwatch.agent.operator --year 2012 --dates 07-03 --no-llm      # rule-based plan, no API

Each morning, for a farm with several pivots (config/farm_demo.json):
1. `farm_overview` (tool): each pivot's advice from the decision engine with its own crop, soil
   and pivot capacity, whether it is running and how much water it put down (from the connector),
   which pivots share a well and how much water allocation is left.
2. `propose_commands` (tool): the agent's list of starts, most valuable first. `connect.guard`
   checks each against the farm's rules (advice, season, depth, allocation, well capacity) and
   returns what it refused and why; the agent can revise.
3. The agent writes a text message asking for approval and a short note. Every number is checked
   against the tool results (agent/validate.py), as for the farm note.
4. Only on the farmer's YES (`--approve`) are the commands sent through the connector. With the
   simulated connector the pivots then run, and their as-applied water shows up in the next
   morning's overview.
Output: site/data/ops/<year>_<MM-DD>.json, which the farm page can show.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from flashwatch.agent.farm import FarmData, status
from flashwatch.agent.validate import drop_sentences, unsupported
from flashwatch.connect import make
from flashwatch.connect.base import Command
from flashwatch.connect.guard import STARTABLE, check, flow_gpm
from flashwatch.site.build import SITE

FARM = Path(__file__).resolve().parents[1] / "config" / "farm_demo.json"
SOILS = {"Sand": 1.0, "Sandy loam": 1.4, "Loam": 1.8, "Silt loam": 2.0, "Clay loam": 1.9}   # in/ft, as engine.js
DEPTH_IN = 0.75

SYSTEM = """You are FlashWatch's irrigation operator for a farm with several centre pivots.
Each morning:
1. Call farm_overview: each pivot's flash-drought advice, whether it is running, its well and water
   allocation, and any recent voice report from the farmer about the crop.
2. Decide which pivots to start today. Start only pivots whose advice says to start, or whose crop
   the farmer reports stressed (say so, quoting the farmer), or whose root zone is at the stress line
   (advice.root_zone_at_stress_line: the field's water account says water now, warning or not). Where pivots
   share a well that cannot run them all, start the one worth more money first. Call
   propose_commands with your list, most valuable first. The tool checks every command against the
   farm's rules and says what it refused and why; propose again if a change would help (for
   example a smaller depth per pass when the allocation is short). Stop a running pivot whose
   water allocation is used up, and one whose root zone is already full (root_zone.share_full 0.95
   or more) unless its advice says to start: pumping into a full profile is wasted. For each start,
   use root_zone.inches_to_fill as the depth (the tool keeps it between 0.3 and 1.5 inches).
3. Reply with JSON only: {"sms": "...", "note": "..."}
   - sms: at most 140 characters: which pivots to start and the main reason. The system adds the
     approval request.
   - note: at most 120 words for the farmer: what to start and why, what you held back and why,
     and the money at stake.
Rules: every number must come from the tool results; round percentages and dollars to whole
numbers. Nothing runs until the farmer replies YES. Plain words, no jargon."""


def load_farm(path=FARM) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def connector_for(farm: dict, today: str):
    cfg = dict(farm.get("connector", {"kind": "simulated"}))
    kind = cfg.pop("kind")
    if kind == "simulated":
        return make("simulated", pivots={k: {"gpm_per_acre": p["gpm_per_acre"]} for k, p in farm["pivots"].items()},
                    today=today, **cfg)
    if kind == "manual":
        return make("manual", pivots=farm["pivots"])
    return make(kind, **cfg)


def root_zone(data: FarmData, piv: dict, md: str, records: list[dict]) -> dict:
    """The pivot's water account (decide/waterbook.py) up to the morning: ERA5-Land rain and crop water use
    from the site data, public 5 days late (the last 5 days: no rain, the past week's use), and the water
    the connector says the pivot put down."""
    from flashwatch.decide.waterbook import account, kc
    rec, d0 = data.record(piv["lat"], piv["lon"]), data.meta["water_days"][0]
    start, today = dt.date.fromisoformat(f"{data.year}-{d0}"), dt.date.fromisoformat(f"{data.year}-{md}")
    n = (today - start).days + 1
    known = max(1, min(n - 5, len(rec["wr"])))
    week = sum(rec["we"][max(0, known - 7):known]) / min(7, known) / 100
    days = [(start + dt.timedelta(days=k)).strftime("%m-%d") for k in range(n)]
    rain = [rec["wr"][k] / 100 if k < known else 0.0 for k in range(n)]
    eto = [rec["we"][k] / 100 if k < known else week for k in range(n)]
    irr: dict = {}
    for r in records:
        irr[r["date"][5:10]] = irr.get(r["date"][5:10], 0.0) + r["depth_in"]
    taw = SOILS.get(piv["soil"], 2.0) * 3.0
    crop = "soybeans" if piv["crop"].startswith("soy") else "corn"
    acc = account(days, rain, eto, crop, taw, 0.5 if crop == "soybeans" else 0.55, irr)
    dr, use = float(acc["deficit"][-1]), kc(crop, md) * week
    cap = piv["gpm_per_acre"] * 24 * 60 / 27154 * 0.85
    return {"share_full": round(1 - dr / taw, 2), "inches_to_fill": round(dr, 2),
            "days_until_stress_if_no_rain": round(max(0.0, acc["raw"] - dr) / use, 1) if use > 0 else None,
            "full_time_pumping_days_to_fill": round(dr / cap, 1), "deficit_in": dr}


def pivot_advice(data: FarmData, piv: dict, md: str, deficit: float = 0.0) -> dict:
    """The decision engine's advice for one pivot, with its own crop, soil and capacity, starting from
    its root zone's deficit."""
    action = "irrigation_soybeans" if piv["crop"].startswith("soy") else "irrigation"
    s = status(data, piv["lat"], piv["lon"], md, {"gpm_per_acre": piv["gpm_per_acre"], "dr0": round(deficit, 2),
                                                   "awc_in_per_ft": SOILS.get(piv["soil"], 2.0)}, action)
    out = {k: v for k, v in s.items() if k not in ("how_to_read", "date")}
    if "value_of_starting_today_dollars_per_acre" in s:
        out["value_of_starting_today_dollars_whole_pivot"] = round(s["value_of_starting_today_dollars_per_acre"] * piv["acres"])
    return out


def with_allocation_used(farm: dict, conn, year: int) -> dict:
    """The farm with each allocation reduced by the water the connector says was put down this season."""
    farm = json.loads(json.dumps(farm))
    for k, piv in farm["pivots"].items():
        if piv.get("allocation_in_left") is not None:
            used = sum(r["depth_in"] for r in conn.applied(k, f"{year}-01-01"))
            piv["allocation_in_left"] = round(max(0.0, piv["allocation_in_left"] - used), 2)
    return farm


def field_reports(year: int, date: str, days: int = 7) -> dict:
    """pivot -> its latest voice field report in the last `days` days (agent/voice.py)."""
    from flashwatch.agent.voice import load_reports
    d0 = (dt.date.fromisoformat(date) - dt.timedelta(days=days)).isoformat()
    out = {}
    for rep in load_reports(year):
        if d0 <= rep["date"] <= date:
            for r in rep.get("reports", []):
                if r.get("pivot"):
                    out[r["pivot"]] = {"date": rep["date"], "severity": r["severity"], "symptoms": r.get("symptoms", []),
                                       "farmer_said": r.get("quote", "")}
    return out


def overview(data, farm, md, conn) -> dict:
    pivots = {}
    date = f"{data.year}-{md}"
    reports = field_reports(data.year, date)
    recent = (dt.date.fromisoformat(date) - dt.timedelta(days=3)).isoformat()
    for k, piv in farm["pivots"].items():
        st = conn.status(k)
        rz = root_zone(data, piv, md, conn.applied(k, f"{data.year}-04-01"))
        pivots[k] = {"crop": piv["crop"], "acres": piv["acres"], "soil": piv["soil"],
                     "pivot_capacity_gpm_per_acre": piv["gpm_per_acre"], "pumps_gpm": round(flow_gpm(piv)),
                     "running": st.running, "water_put_down_last_7_days_in": st.applied_in_7d,
                     "water_allocation_left_in": piv.get("allocation_in_left"),
                     "root_zone": {kk: v for kk, v in rz.items() if kk != "deficit_in"},
                     "advice": pivot_advice(data, piv, md, rz["deficit_in"])}
        # The water account's own schedule: a root zone within 2 days of stress needs water now, warning or not.
        if rz["days_until_stress_if_no_rain"] is not None and rz["days_until_stress_if_no_rain"] <= 2:
            pivots[k]["advice"]["root_zone_at_stress_line"] = True
        if st.note:
            pivots[k]["connection"] = st.note
        if k in reports:
            pivots[k]["field_report"] = reports[k]
            pivots[k]["advice"]["farmer_reports_stress"] = reports[k]["severity"] >= 2 and reports[k]["date"] >= recent
    return {"date": f"{data.year}-{md}", "farm": farm["name"], "pivots": pivots,
            "wells": {w: {"gives_gpm": s["gpm"], "pivots": s["pivots"]} for w, s in farm.get("wells", {}).items()},
            "connection": conn.name, "connection_can_start_pivots": conn.can_control,
            "suggested_order": ranked(pivots)}


def ranked(pivots: dict) -> list[str]:
    """Pivots the advice says to start, most money first (soil already dry counts as most urgent)."""
    ok = [k for k, p in pivots.items() if not p["running"] and (p["advice"].get("verdict", "").startswith(STARTABLE)
                                                                or p["advice"].get("farmer_reports_stress")
                                                                or p["advice"].get("root_zone_at_stress_line"))]
    # Early starts by the money they're worth, then pivots the water account says are at the stress line,
    # driest first.
    return sorted(ok, key=lambda k: (not pivots[k]["advice"].get("verdict", "").startswith("start"),
                                     -pivots[k]["advice"].get("value_of_starting_today_dollars_whole_pivot", 0),
                                     pivots[k].get("root_zone", {}).get("share_full", 1)))


def rule_plan(ov: dict, farm: dict, date: str) -> list[Command]:
    """The plan without a language model: stop pivots out of allocation, then starts in the suggested
    order with the depth cut to the allocation."""
    cmds = [Command(k, "stop", 0.0, "water allocation used up", date) for k, p in ov["pivots"].items()
            if p["running"] and p["water_allocation_left_in"] is not None and p["water_allocation_left_in"] < 0.3]
    cmds += [Command(k, "stop", 0.0, "root zone full", date) for k, p in ov["pivots"].items()
             if p["running"] and k not in {c.pivot for c in cmds} and p.get("root_zone", {}).get("share_full", 0) >= 0.95
             and not p["advice"].get("verdict", "").startswith(STARTABLE)]
    for k in ov["suggested_order"]:
        left = farm["pivots"][k].get("allocation_in_left")
        fill = ov["pivots"][k].get("root_zone", {}).get("inches_to_fill")
        depth = round(min(1.5, max(0.3, fill)), 2) if fill else DEPTH_IN
        depth = depth if left is None else round(min(depth, left), 2)
        cmds.append(Command(k, "start", depth, "advice says start today", date))
    return cmds


class Morning:
    """One morning's planning: tools for the agent, and the plan it settles on."""

    def __init__(self, data, farm, md, conn):
        self.data, self.md, self.conn = data, md, conn
        self.farm = with_allocation_used(farm, conn, data.year)
        self.date = f"{data.year}-{md}"
        self.ov = overview(data, self.farm, md, conn)
        self.accepted, self.rejected = [], []

    def propose(self, commands: list) -> dict:
        cmds = [Command(c["pivot"], c.get("action", "start"), float(c.get("depth_in", DEPTH_IN)),
                        c.get("reason", ""), self.date) for c in commands]
        status = {k: self.conn.status(k) for k in self.farm["pivots"]}
        advice = {k: p["advice"] for k, p in self.ov["pivots"].items()}
        self.accepted, self.rejected = check(cmds, self.farm, advice, status)
        return {"accepted_waiting_for_farmer_approval": [{"pivot": c.pivot, "action": c.action, "depth_in": c.depth_in}
                                                         for c in self.accepted],
                "refused": [{"pivot": c.pivot, "action": c.action, "depth_in": c.depth_in, "why": why}
                            for c, why in self.rejected]}

    def tools(self):
        cmd_schema = {"type": "object", "properties": {
            "pivot": {"type": "string"}, "action": {"type": "string", "enum": ["start", "stop"]},
            "depth_in": {"type": "number", "description": "inches per pass, usually 0.5-1.0"},
            "reason": {"type": "string"}}, "required": ["pivot", "action"]}
        return {
            "farm_overview": ({"description": "Every pivot's flash-drought advice, running state, water put down, "
                                              "wells and water allocation, and the suggested order of starts.",
                               "parameters": {"type": "object", "properties": {}}}, lambda: self.ov),
            "propose_commands": ({"description": "Propose today's pivot commands, most valuable first. Replaces any "
                                                 "earlier proposal. Returns what the farm's rules accept and refuse.",
                                  "parameters": {"type": "object", "properties": {
                                      "commands": {"type": "array", "items": cmd_schema}}, "required": ["commands"]}},
                                 lambda commands: self.propose(commands)),
        }


def plan_with_agent(m: Morning, chat, max_redrafts: int = 2) -> dict:
    tools = m.tools()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Plan this morning ({m.date}) for {m.farm['name']}."}]
    text, msgs, log = chat.run(msgs, tools)
    drafts = []
    for _ in range(max_redrafts):
        bad = unsupported(text, [r for _, _, r in log])
        drafts.append({"draft": text, "untraced": bad})
        if not bad:
            break
        text, msgs, more = chat.run(msgs + [{"role": "user", "content":
            f"These figures are not in the tool results: {', '.join(bad)}. Rewrite with only their numbers. JSON only."}], tools)
        log += more
    mt = re.search(r"\{.*\}", text or "", re.S)
    try:
        out = json.loads(mt.group(0)) if mt else {"sms": "", "note": text}
    except json.JSONDecodeError:
        out = {"sms": "", "note": text}
    bad = unsupported(" ".join(str(v) for v in out.values()), [r for _, _, r in log])
    if bad:
        out = {k: drop_sentences(str(v), bad) for k, v in out.items()}
    if not any(n == "propose_commands" for n, _, _ in log):   # the agent must go through the guard
        m.propose([])
    return {**out, "tool_calls": [n for n, _, _ in log], "drafts": drafts, "cut_untraced": bad, "model": chat.model}


def _what(cmds) -> str:
    starts = [c.pivot for c in cmds if c.action == "start"]
    stops = [c.pivot for c in cmds if c.action == "stop"]
    return " and ".join(x for x in (f"start {', '.join(starts)}" if starts else "",
                                    f"stop {', '.join(stops)}" if stops else "") if x)


def rule_note(m: Morning) -> dict:
    what = _what(m.accepted)
    held = "; ".join(f"{c.pivot}: {why}" for c, why in m.rejected)
    return {"sms": f"FlashWatch: {what} today." if what else "FlashWatch: no pivot changes today.",
            "note": (f"{what[0].upper() + what[1:]}." if what else "No changes.") + (f" Held back: {held}." if held else "")}


def approval_line(m: Morning) -> str:
    return f" Reply YES to {_what(m.accepted)}." if m.accepted else ""


def approve(m: Morning, who: str) -> list[dict]:
    """The farmer said YES: send every accepted command through the connector."""
    sent = []
    for c in m.accepted:
        c.state, c.approved_by = "approved", who
        sent.append(m.conn.send(c).to_dict())
    return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", default=str(FARM))
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--dates", nargs="+", required=True, help="MM-DD mornings, in order")
    ap.add_argument("--approve", default="", help="who approves (simulates the farmer's YES); empty = don't send")
    ap.add_argument("--no-llm", action="store_true", help="rule-based plan and message, no API calls")
    ap.add_argument("--llm-dates", nargs="*", help="use Muse only on these MM-DD dates (rules on the rest)")
    args = ap.parse_args()
    farm, data = load_farm(args.farm), FarmData(args.year)
    conn = connector_for(farm, f"{args.year}-{args.dates[0]}")
    chat = None
    if not args.no_llm:
        from flashwatch.agent.llm import Chat
        chat = Chat()
    outdir = SITE / "data" / "ops"
    outdir.mkdir(parents=True, exist_ok=True)
    for j, md in enumerate(args.dates):
        if j and hasattr(conn, "advance"):
            gap = (dt.date.fromisoformat(f"{args.year}-{md}") - dt.date.fromisoformat(f"{args.year}-{args.dates[j - 1]}")).days
            conn.advance(gap)
        m = Morning(data, farm, md, conn)
        if chat and (args.llm_dates is None or md in args.llm_dates):
            msg = plan_with_agent(m, chat)
        else:
            m.propose([c.__dict__ for c in rule_plan(m.ov, farm, m.date)])
            msg = rule_note(m)
        msg["sms"] = (msg.get("sms") or "").strip() + approval_line(m)
        sent = approve(m, args.approve) if args.approve and m.accepted else []
        rec = {"date": m.date, "farm": farm["name"], "place": farm.get("place", ""), "overview": m.ov,
               "accepted": [c.to_dict() for c in m.accepted],
               "refused": [{**c.to_dict(), "why": why} for c, why in m.rejected],
               "sms": msg["sms"], "note": msg.get("note", ""), "approved_by": args.approve if sent else "",
               "sent": sent, "model": msg.get("model", "rules"),
               **{k: msg[k] for k in ("tool_calls", "drafts", "cut_untraced") if k in msg}}
        (outdir / f"{args.year}_{md}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(f"--- {m.date}\nSMS: {rec['sms']}\n{rec['note']}")
        for c in rec["refused"]:
            print(f"  refused {c['pivot']}: {c['why']}")
        for c in sent:
            print(f"  sent {c['pivot']} {c['action']} {c['depth_in']} in: {c['state']} {c['log']}")
    if chat:
        print(f"tokens: {chat.usage}")


if __name__ == "__main__":
    main()
