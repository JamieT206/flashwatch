"""Data for the Agents page (site/agents.html): real runs of each Meta-model agent, gathered in one file.

    python -m flashwatch.site.agents

Reads what the agents wrote (data/agent/, site/data/ops/, site/data/notes/) and writes
site/data/agents.json. Nothing here calls a model.
"""

from __future__ import annotations

import json

from flashwatch.config import DATA
from flashwatch.site.build import SITE


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    out = {}
    piv = sorted((SITE / "data" / "pivots").glob("*.json"))
    if piv:
        best = max((_load(p) | {"stem": p.stem} for p in piv), key=lambda r: len(r["pivots"]))
        out["pivots"] = best
    ops = _load(SITE / "data" / "ops" / "2012_07-03.json")
    if ops:
        out["operator"] = {k: ops[k] for k in ("date", "farm", "place", "sms", "note", "accepted", "refused", "sent",
                                               "approved_by", "model", "tool_calls", "cut_untraced") if k in ops}
        out["operator"]["drafts"] = len(ops.get("drafts", []))
        out["operator"]["pivots"] = {k: {"crop": p["crop"], "acres": p["acres"], "running": p["running"],
                                         "root_zone": p.get("root_zone"), "verdict": p["advice"].get("verdict"),
                                         "value": p["advice"].get("value_of_starting_today_dollars_whole_pivot"),
                                         "allocation_left_in": p.get("water_allocation_left_in"),
                                         "farmer": (p.get("field_report") or {}).get("farmer_said")}
                                     for k, p in ops["overview"]["pivots"].items()}
        out["operator"]["wells"] = ops["overview"].get("wells")
    rep = _load(SITE / "data" / "ops" / "reports_2012.json")
    if rep:
        out["voice"] = rep[-1]
    brief = sorted((DATA / "agent").glob("brief_*.json"))
    if brief:
        b = _load(brief[-1])
        out["brief"] = {k: b.get(k) for k in ("issue", "place", "model", "briefing", "cut_untraced", "tool_calls")}
        out["brief"]["drafts"] = len(b.get("drafts") or [])
        out["brief"]["tool_calls"] = [{"tool": c["tool"], "args": c.get("args", {})} for c in b.get("tool_calls") or []]
    # A week ahead of the Drought Monitor: FlashWatch's outlook next to the Monitor's narrative a week later (agent/ahead.py).
    ah = _load(DATA / "agent" / "ahead.json")
    if ah:
        out["ahead"] = {"scores": ah["scores"], "examples": [{k: e[k] for k in ("issue", "monitor_date", "monitor_before",
                        "model", "outlook", "monitor", "named_states_then", "numbers")} for e in ah["examples"] if e["outlook"]]}
    # Ask FlashWatch on the demo farm's 3 July 2012 (Muse Spark), for the WhatsApp thread on the page.
    ask = _load(DATA / "agent" / "ask_compare_muse-spark-1.1.json")
    if ask:
        out["ask"] = [{"q": a["q"].strip(), "answer": a["answer"]} for a in ask["answers"]]
    al = _load(SITE / "data" / "alerts" / "2012.json")
    if al:
        from flashwatch.agent.scorecard import score_alerts
        out["alerts"] = {"year": al["year"], "farms": [{k: f[k] for k in ("name", "crop", "watering", "dates_checked")} |
                                                        {"alerts": [{k: x[k] for k in ("date", "kind", "level", "text", "by", "outcome", "facts") if k in x}
                                                                    for x in score_alerts(f["alerts"], f, al["year"])]}
                                                        for f in al["farms"]]}
    sc = _load(SITE / "data" / "scorecard.json")
    if sc:
        out["scorecard"] = sc
    (SITE / "data" / "agents.json").write_text(json.dumps(out))
    print("agents.json:", ", ".join(out))


if __name__ == "__main__":
    main()
