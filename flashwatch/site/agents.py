"""Data for the Agents page (site/agents.html): real runs of each Meta-model agent, gathered in one file.

    python -m flashwatch.site.agents

Reads what the agents wrote (data/agent/, site/data/ops/, site/data/notes/, data/glimmer/) and writes
site/data/agents.json. Nothing here calls a model.
"""

from __future__ import annotations

import json

from flashwatch.config import DATA
from flashwatch.site.build import SITE


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def glimmer_table(evals: dict) -> dict:
    """Each narrative writer scored on the held-out weeks every one of them finished (some API calls time
    out), plus one growing-season week from the best-styled run next to the real text."""
    weeks = {k: {w["date"]: w for w in e["weeks"]} for k, e in evals.items()}
    common = sorted(set.intersection(*[set(w) for w in weeks.values()]))
    rows = []
    for k, e in evals.items():
        ws = [weeks[k][d] for d in common]
        rows.append({"model": e["model"], "local": "glimmer" in e["model"], "tuned": "lora" in e["model"],
                     "shots": e.get("shots", 0),
                     **{m: round(sum(w[m] for w in ws) / len(ws), 3) for m in ("untraced", "states", "direction", "rouge_l")},
                     "words": round(sum(len(w["draft"].split()) for w in ws) / len(ws))})
    best = max(rows, key=lambda r: r["rouge_l"])
    key = next(k for k, e in evals.items() if e["model"] == best["model"] and e.get("shots", 0) == best["shots"])
    # The example: a growing-season week (May-September) with no untraced figure, the most wording shared.
    clean = [w for w in weeks[key].values() if not w["untraced_figures"] and "05" <= w["date"][5:7] <= "09"]         or [w for w in weeks[key].values() if not w["untraced_figures"]] or list(weeks[key].values())
    wk = max(clean, key=lambda w: w["rouge_l"])
    return {"weeks": len(common), "rows": rows, "example": {"model": best["model"], "shots": best["shots"],
            **{k: wk[k] for k in ("date", "draft", "real")}}}


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
    g = {f.stem[5:]: _load(f) for f in sorted((DATA / "glimmer").glob("eval_*.json"))}
    if g:
        out["glimmer"] = glimmer_table(g)
        out["glimmer"]["stats"] = _load(DATA / "glimmer" / "stats.json")
        # Without this week's Drought Monitor map (last week's instead): written before the Monitor comes out.
        gn = {f.stem[5:]: _load(f) for f in sorted((DATA / "glimmer" / "nomap").glob("eval_*.json"))}
        if gn:
            out["glimmer"]["nomap"] = glimmer_table(gn)
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
