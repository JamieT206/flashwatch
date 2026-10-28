"""Weekly US Drought Monitor narratives ("National Drought Summary"), for the Glimmer narrative
writer (DESIGN.md G2).

    python -m flashwatch.ingest.usdm_text              # Tuesdays 2005-2024, resumable

The summary page (droughtmonitor.unl.edu/Summary.aspx) is an ASP.NET form: we read its hidden
fields once, then post each date. One request a second. Writes data/raw/usdm_text/<YYYYMMDD>.json:
{"date", "title", "sections": [{"heading", "text"}]}. The Drought Monitor is produced by the National
Drought Mitigation Center (UNL), USDA and NOAA; credit them wherever the text is used.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import time

from flashwatch.config import RAW
from flashwatch.ingest.common import get_logger

URL = "https://droughtmonitor.unl.edu/Summary.aspx"
OUT = RAW / "usdm_text"
SELECT = "ctl00$ContentPlaceHolder1$seldate"
REGIONS = ("Summary", "Northeast", "Mid-Atlantic", "Southeast", "South", "Midwest", "High Plains", "West",
           "Plains", "Pacific", "Alaska", "Hawaii", "Puerto Rico", "Looking Ahead", "Caribbean", "Virgin")
log = get_logger("usdm_text")


def hidden_fields(page: str) -> dict:
    return dict(re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', page))


def parse(page: str) -> dict | None:
    """Title and sections from a summary page; headings are short lines naming a region."""
    i, j = page.find("</select>"), page.find("<footer")
    if i < 0:
        return None
    text = html.unescape(re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", "\n", page[i:j]))).replace("�", "'")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    t = next((k for k, ln in enumerate(lines) if ln.startswith("National Drought Summary for")), None)
    if t is None:
        return None
    sections, cur = [], None
    for ln in lines[t + 1:]:
        if len(ln) < 70 and any(r.lower() in ln.lower() for r in REGIONS) and not ln.endswith("."):
            cur = {"heading": ln, "text": ""}
            sections.append(cur)
        elif cur is not None:
            cur["text"] = (cur["text"] + "\n" + ln).strip()
    return {"title": lines[t], "sections": [s for s in sections if s["text"]]}


def tuesdays(y0=2005, y1=2024):
    d = dt.date(y0, 1, 1)
    d += dt.timedelta(days=(1 - d.weekday()) % 7)
    while d.year <= y1:
        yield d
        d += dt.timedelta(days=7)


def main():
    import requests
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "FlashWatch research (hackathon; github.com) - one request per second"
    fields = hidden_fields(s.get(URL, timeout=60).text)
    ok = miss = 0
    for d in tuesdays():
        f = OUT / f"{d:%Y%m%d}.json"
        if f.exists():
            continue
        try:
            page = s.post(URL, data={**fields, SELECT: f"{d:%Y%m%d}", "__EVENTTARGET": SELECT}, timeout=60).text
            r = parse(page)
        except Exception as e:
            log.error("%s: %s", d, type(e).__name__)
            time.sleep(10)
            continue
        if r and r["sections"]:
            f.write_text(json.dumps({"date": d.isoformat(), **r}, indent=1), encoding="utf-8")
            ok += 1
        else:
            miss += 1
            log.info("%s: no summary", d)
        time.sleep(1)
    log.info("finished: %d new, %d missing", ok, miss)


if __name__ == "__main__":
    main()
