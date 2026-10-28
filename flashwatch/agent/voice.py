"""Voice field reports (DESIGN.md G3): the farmer talks, FlashWatch listens.

    python -m flashwatch.agent.voice report.wav --year 2012 --date 07-03

1. Muse Voice Transcribe (`muse-voice-transcribe-1.0`, POST {base}/asr/transcribe, multipart with
   a JSON "request" part and the "audio" part; 16-bit mono WAV, 16 or 24 kHz) turns the voice note
   into text.
2. Muse Spark turns the text into a structured report, using a tool that lists the farm's pivots so
   "the sandy pivot" becomes N1: which pivot, what the crop shows, how severe (0 fine, 1 mild,
   2 stress showing, 3 severe), rain since when. Every number must come from the transcript or the
   tool (agent/validate.py).
3. The report is saved to site/data/ops/reports_<year>.json. The operator shows it in the farm
   overview, and the guard accepts a start for a pivot whose crop the farmer reports stressed
   (severity 2+, last 3 days) even before the soil data shows it: the farmer's own eyes are the
   "usual practice" FlashWatch is compared with.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import wave

from flashwatch.agent.validate import numbers_in, unsupported
from flashwatch.config import load_env
from flashwatch.site.build import SITE

ASR_MODEL = "muse-voice-transcribe-1.0"
SYSTEM = """You turn a farmer's voice note into a field report for FlashWatch. Call farm_pivots to see
the farm's pivots, then reply with JSON only:
{"reports": [{"pivot": "<id or null>", "crop": "...", "stage": "...", "symptoms": ["..."],
  "severity": 0-3, "rain_in_since": <inches or null>, "since": "<as said, or null>", "quote": "<the words it rests on>"}],
 "summary": "<one sentence>"}
One entry per field mentioned. Match fields to pivots by what the farmer says (name, crop, soil,
direction); use null if unsure. Severity: 0 fine, 1 mild signs, 2 stress showing (curling, greying),
3 severe (firing, dying). Use only numbers the farmer said. Do not add advice."""


def transcribe(path, key: str | None = None, base: str | None = None, post=None) -> str:
    import requests
    load_env()
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() not in (16000, 24000):
            raise ValueError("need 16-bit mono WAV at 16 or 24 kHz")
        rate = w.getframerate()
    base = (base or os.getenv("META_API_BASE", "https://api.meta.ai/v1")).rstrip("/")
    req = {"model": ASR_MODEL, "audioEncoding": "WAV", "sampleRateHertz": rate}
    post = post or requests.post
    with open(path, "rb") as f:
        r = post(base + "/asr/transcribe", headers={"Authorization": f"Bearer {key or os.environ['META_API_KEY']}"},
                 files={"request": (None, json.dumps(req), "application/json"),
                        "audio": ("note.wav", f, "audio/wav")}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"transcribe: HTTP {r.status_code}")
    return r.json()["transcript"]


def pivots_tool(farm: dict) -> dict:
    return {"farm": farm["name"], "pivots": [{"id": k, "crop": p["crop"], "soil": p["soil"], "acres": p["acres"],
                                              "make": p.get("make", "")} for k, p in farm["pivots"].items()]}


def structure(text: str, farm: dict, chat, max_redrafts: int = 1) -> dict:
    tools = {"farm_pivots": ({"description": "The farm's pivots: id, crop, soil, acres.",
                              "parameters": {"type": "object", "properties": {}}}, lambda: pivots_tool(farm))}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Voice note: {text}"}]
    out, msgs, log = chat.run(msgs, tools)
    sources = [text, pivots_tool(farm)]
    for _ in range(max_redrafts):
        bad = [b for b in unsupported(out, sources) if b not in {"0", "1", "2", "3"}]
        if not bad:
            break
        out, msgs, _ = chat.run(msgs + [{"role": "user", "content":
            f"These figures are not in the note or the pivot list: {', '.join(bad)}. Fix, JSON only."}], tools)
    m = re.search(r"\{.*\}", out or "", re.S)
    rep = json.loads(m.group(0)) if m else {"reports": [], "summary": out}
    valid = set(farm["pivots"])
    for r in rep.get("reports", []):
        if r.get("pivot") not in valid:
            r["pivot"] = None
        r["severity"] = int(r.get("severity") or 0)
    rep["numbers_in_note"] = sorted({str(x) for x in numbers_in(text)})
    return rep


def save(rep: dict, year: int, md: str, transcript: str) -> list:
    f = SITE / "data" / "ops" / f"reports_{year}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(f.read_text()) if f.exists() else []
    allr.append({"date": f"{year}-{md}", "transcript": transcript, **rep})
    f.write_text(json.dumps(allr, indent=1))
    return allr


def load_reports(year: int) -> list:
    f = SITE / "data" / "ops" / f"reports_{year}.json"
    return json.loads(f.read_text()) if f.exists() else []


def main():
    from flashwatch.agent.llm import Chat
    from flashwatch.agent.operator import FARM, load_farm
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--date", required=True, help="MM-DD the note was left")
    ap.add_argument("--farm", default=str(FARM))
    args = ap.parse_args()
    text = transcribe(args.wav)
    print("transcript:", text)
    chat = Chat()
    rep = structure(text, load_farm(args.farm), chat)
    save(rep, args.year, args.date, text)
    print(json.dumps(rep, indent=1))
    print("tokens:", chat.usage)


if __name__ == "__main__":
    main()
