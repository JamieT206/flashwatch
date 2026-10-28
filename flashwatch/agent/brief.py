"""Write a checked weekly briefing for one place (DESIGN.md G1, HANDOVER 6.5).

    python -m flashwatch.agent.brief --issue 2012-05-03 --place "Story County, Iowa" --lat 42.04 --lon -93.46

The model gathers what it needs through the tools (agent/tools.py over the weekly pack), drafts the
briefing, and the number checker traces every figure to a tool result. Untraced figures go back to
the model with a request to fix them (at most 2 redrafts); any still left are cut with their
sentences. Output: data/agent/brief_<issue>_<place>.json with the text, the tool calls it relied on
and the checker's report.

Replay guard: the model is told the date only as "this week" and must not use outside knowledge of
what happened later (it may know famous droughts from its training data).
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from flashwatch.agent import pack as packs
from flashwatch.agent.llm import Chat
from flashwatch.agent.tools import Tools
from flashwatch.agent.validate import drop_sentences, unsupported

SYSTEM = """You are FlashWatch, a flash-drought early-warning analyst writing for a county agricultural
extension office in the US Midwest. A flash drought is a rapid drop in root-zone soil moisture (from
the 40th to below the 20th percentile within 20 days, lasting at least 10 days).

Use the tools to look up the place you are asked about, the region overview and, if useful, the
hotspots. Then write a briefing of at most 180 words with these headings:
Risk this week / Why / What to watch / Suggested actions.

Rules:
- Every number you write must come from a tool result (percentages are already in percent). Round
  percentages to whole numbers. Do not compute new numbers. Describe standard-deviation values in
  words ("well below normal", "far above normal") or to one decimal, and give at most a few of them.
- Say how the risk compares with normal for this time of year; that matters more than the raw chance.
- Write for farmers: short plain sentences, no model jargon, no coordinates.
- You do not know what happens after this week. Do not use outside knowledge of past events or years.
- Be honest about uncertainty: these are probabilities, and most weeks with elevated risk do not end
  in a flash drought."""


def write(issue, place: str, lat: float, lon: float, chat: Chat | None = None, max_redrafts: int = 2) -> dict:
    tools = Tools(pd.read_parquet(packs.path(issue)))
    chat = chat or Chat()
    ask = f"Write this week's briefing for {place} (latitude {lat}, longitude {lon})."
    text, messages, log = chat.run([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}],
                                   tools.specs())
    results = [r for _, _, r in log]
    history = []
    for _ in range(max_redrafts):
        bad = unsupported(text, results)
        history.append({"draft": text, "untraced": bad})
        if not bad:
            break
        fix = (f"These figures do not appear in any tool result: {', '.join(bad)}. Rewrite the briefing using "
               "only numbers from the tool results (call tools again if you need to). Reply with the briefing only.")
        text, messages, more = chat.run(messages + [{"role": "user", "content": fix}], tools.specs())
        log += more
        results = [r for _, _, r in log]
    bad = unsupported(text, results)
    if bad:
        text = drop_sentences(text, bad)
    return {"issue": str(pd.Timestamp(issue).date()), "place": place, "lat": lat, "lon": lon,
            "model": chat.model, "briefing": text, "cut_untraced": bad, "drafts": history,
            "tool_calls": [{"tool": n, "args": a, "result": r} for n, a, r in log], "usage": chat.usage}


def nearest_city(lat: float, lon: float) -> str:
    from flashwatch.agent.mapimg import CITIES
    return min(CITIES, key=lambda c: (CITIES[c][0] - lat) ** 2 + ((CITIES[c][1] - lon) * np.cos(np.radians(lat))) ** 2)


def map_check(issue, chat: Chat | None = None, k: int = 3) -> dict:
    """Multimodal cross-check: the model reads the rendered risk map (no numbers) and names the k
    areas of highest risk vs normal by their nearest marked city; we compare with the hotspots tool."""
    import base64
    from flashwatch.agent.mapimg import CITIES, render
    p = pd.read_parquet(packs.path(issue))
    chat = chat or Chat()
    png = base64.b64encode(render(p)).decode()
    ask = (f"This map shows flash-drought onset risk relative to normal. Name the {k} separate areas with "
           f"the highest risk (brightest yellow/orange), each by the nearest marked city. Reply as JSON: "
           f'{{"areas": ["<city>", ...]}} using only these city names: {", ".join(CITIES)}.')
    msg = chat.complete([{"role": "user", "content": [{"type": "text", "text": ask},
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}])
    m = re.search(r"\{.*\}", msg.get("content") or "", re.S)
    seen = json.loads(m.group(0)).get("areas", []) if m else []
    truth = []
    for h in Tools(p).hotspots(n=3 * k):
        c = nearest_city(h["lat"], h["lon"])
        if c not in truth:
            truth.append(c)
    truth = truth[:k]
    return {"model_saw": seen, "tool_hotspots": truth, "agree": sorted(set(seen) & set(truth))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--model", help="override META_MODEL")
    args = ap.parse_args()
    out = write(pd.Timestamp(args.issue), args.place, args.lat, args.lon, Chat(model=args.model))
    slug = re.sub(r"[^a-z0-9]+", "-", args.place.lower()).strip("-")
    target = packs.OUT / f"brief_{out['issue']}_{slug}.json"
    target.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(out["briefing"])
    print(f"\n[{len(out['tool_calls'])} tool calls, {len(out['drafts'])} draft(s), cut: {out['cut_untraced']}; {target}]")


if __name__ == "__main__":
    main()
