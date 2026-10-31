"""Scripted screen recordings of the site for the demo video (docs/video.md).

    python video/record.py all              # every paced shot; or name some: farm ask agents ...
    python video/record.py july3            # the July 3 test clip (not paced to the narration)
    python video/record.py --list

Needs the site served on 127.0.0.1:8812 (`python -m http.server 8812 --directory site`) and, for the Ask shot,
`python -m flashwatch.agent.serve`. Playwright's Chromium records a 1745x982 window, scaled up to 1920x1080 (the
page at 110%, as in the shot list); a drawn cursor glides between targets on eased, slightly curved paths and
dips on each click, as in a live demo. ffmpeg (winget Gyan.FFmpeg) turns the recording into H.264.

Shots are paced to the narration: video/out/timeline.json (from `video/edit.py audio`) gives each beat's start,
and a shot waits for a beat's time before acting. Each shot writes <name>.json with `t0`, the point in its video
where its first beat starts, plus any marks the edit needs (the Ask shot's question and answer).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE = "http://127.0.0.1:8812/"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
W, H, ZOOM = 1920, 1080, 1.1
FARM = "index.html#lat=45.38&lon=-99.38&mode=replay&year=2012"
REPO = "https://github.com/JamieT206/flashwatch"
random.seed(7)

# A drawn arrow cursor: headless Chromium records no real one.
CURSOR = """
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.id = '__cursor';
  c.innerHTML = `<svg width="22" height="32" viewBox="0 0 22 32" xmlns="http://www.w3.org/2000/svg">
    <path d="M1 1 L1 25 L7 19.5 L11.2 29.5 L15 27.9 L10.9 18.2 L19 18.2 Z" fill="#111" stroke="#fff" stroke-width="1.6"
      stroke-linejoin="round"/></svg>`;
  Object.assign(c.style, {position: 'fixed', left: '0', top: '0', zIndex: 2147483647, pointerEvents: 'none',
    transform: 'translate(-100px,-100px)', transition: 'scale 90ms ease-out', transformOrigin: '1px 1px',
    filter: 'drop-shadow(0 1px 1.5px rgba(0,0,0,.35))'});
  document.body.appendChild(c);
  addEventListener('mousemove', e => { c.style.transform = `translate(${e.clientX}px,${e.clientY}px)`; }, true);
  addEventListener('mousedown', () => { c.style.scale = '0.85'; }, true);
  addEventListener('mouseup', () => { c.style.scale = '1'; }, true);
});
"""

SCROLL = """([y, ms]) => new Promise(done => {
  const y0 = scrollY, t0 = performance.now();
  const step = now => { const t = Math.min(1, (now - t0) / ms), e = t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2;
    scrollTo(0, y0 + (y - y0) * e); t < 1 ? requestAnimationFrame(step) : done(); };
  requestAnimationFrame(step); })"""


class Hand:
    """Moves the mouse like a person: eased, slightly curved, a little off-centre on each target."""

    def __init__(self, page, start=(W * 0.62 / ZOOM, H * 0.55 / ZOOM)):
        self.page, self.x, self.y = page, *start
        page.mouse.move(self.x, self.y)

    def move(self, x, y, seconds=None):
        d = math.hypot(x - self.x, y - self.y)
        seconds = seconds or min(1.0, 0.35 + d / 1800)
        n = max(8, int(seconds * 60))
        bend = random.uniform(-0.12, 0.12) * d            # a bend to one side, proportional to the distance
        nx, ny = (-(y - self.y) / d, (x - self.x) / d) if d else (0, 0)
        for i in range(1, n + 1):
            e = (t := i / n) * t * (3 - 2 * t)              # ease in and out
            arc = math.sin(math.pi * e) * bend
            self.page.mouse.move(self.x + (x - self.x) * e + nx * arc, self.y + (y - self.y) * e + ny * arc)
            self.page.wait_for_timeout(1000 / 60)
        self.x, self.y = x, y

    def to(self, target, seconds=None, jitter=0.15, fx=0.5, fy=0.5):
        loc = self.page.locator(target).first if isinstance(target, str) else target
        b = loc.bounding_box()
        x = b["x"] + b["width"] * (fx + random.uniform(-jitter, jitter) * min(1, 200 / max(b["width"], 1)))
        y = b["y"] + b["height"] * (fy + random.uniform(-jitter, jitter) * min(1, 60 / max(b["height"], 1)))
        self.move(x, y, seconds)

    def click(self, target, seconds=None, pause=180):
        self.to(target, seconds, jitter=0.1)
        self.page.wait_for_timeout(pause)
        self.page.mouse.down()
        self.page.wait_for_timeout(90)
        self.page.mouse.up()

    def drag(self, target, dx, seconds=0.8):
        self.to(target, jitter=0)
        self.page.mouse.down()
        self.move(self.x + dx, self.y, seconds)
        self.page.mouse.up()


class Clock:
    """The narration's clock: `at(beat, s)` waits until `s` seconds after that beat starts (negative: before)."""

    def __init__(self, page, first: str, rec0: float):
        self.page, self.rec0 = page, rec0
        self.tl = json.loads((OUT / "timeline.json").read_text())
        self.base, self.t0, self.marks = self.tl[first]["start"], None, {}

    def go(self):
        self.t0 = time.monotonic()
        self.marks["t0"] = round(self.t0 - self.rec0, 3)

    def at(self, beat: str, s: float = 0.0):
        wait = self.tl[beat]["start"] - self.base + s - (time.monotonic() - self.t0)
        if wait > 0:
            self.page.wait_for_timeout(wait * 1000)

    def until_end(self, beat: str, extra: float = 0.5):
        wait = self.tl[beat]["end"] - self.base + extra - (time.monotonic() - self.t0)
        if wait > 0:
            self.page.wait_for_timeout(wait * 1000)

    def mark(self, name: str):
        self.marks[name] = round(time.monotonic() - self.rec0, 3)


def scroll_to(page, target, ms=1200, top=90):
    y = page.evaluate("([s, top]) => document.querySelector(s).getBoundingClientRect().top + scrollY - top", [target, top])
    page.evaluate(SCROLL, [max(0, y), ms])


def jump_to(page, target, top):
    page.evaluate("([s, top]) => scrollTo(0, document.querySelector(s).getBoundingClientRect().top + scrollY - top)",
                  [target, top])


def strip_cell(page, md: str):
    """The date bar's cell for MM-DD in the replay year."""
    s = page.evaluate("""md => { const st = stops();
        return st.findIndex(x => x.kind === "day" && S.meta.dates[x.i] === md); }""", md)
    return page.locator(f'#strip [data-s="{s}"]')


def farm_page(page, md: str):
    page.goto(SITE + FARM + f"&date={md}")
    page.wait_for_selector("#vBig:not(:empty)")
    page.wait_for_timeout(1500)


# --- shots ---------------------------------------------------------------------------------------

def july3(page, rec0):
    """Test clip: the demo farm on July 1 ("as usual"), then a click on July 3 ("start extra irrigation today")."""
    farm_page(page, "07-01")
    hand = Hand(page)
    page.wait_for_timeout(1200)
    hand.to("#vBig", 0.9)
    page.wait_for_timeout(1500)
    hand.click(strip_cell(page, "07-03"))
    page.wait_for_timeout(1800)
    hand.to("#vBig", 0.8)
    page.wait_for_timeout(1200)
    hand.to("#evNow", 0.6)
    page.wait_for_timeout(2500)
    return {"t0": 0}


def region2012(page, rec0):
    """The 2012 risk map stepping through late June and July (beats drydown, july2012)."""
    page.goto(SITE + "region.html")
    page.wait_for_selector("#year option", state="attached", timeout=20000)
    page.select_option("#year", "2012")
    page.wait_for_timeout(1500)
    page.evaluate("""() => { const r = document.getElementById('date'); r.value = 37;
                             r.dispatchEvent(new Event('input', {bubbles: true})); r.dispatchEvent(new Event('change', {bubbles: true})); }""")
    page.wait_for_timeout(1500)
    jump_to(page, "#rmap", 150)
    page.wait_for_timeout(1200)
    hand = Hand(page)
    clk = Clock(page, "drydown", rec0)
    clk.go()
    clk.at("drydown", 1.5)
    hand.to("#rmap", 1.2)
    clk.at("drydown", 4.0)
    hand.to("#next", 1.0)
    clk.at("july2012", -0.2)
    end = clk.tl["july2012"]["end"] - clk.base - 0.8
    while time.monotonic() - clk.t0 < end:
        hand.click("#next", 0.12, pause=60)
        page.wait_for_timeout(420)
    clk.until_end("july2012")
    return clk.marks


def farm(page, rec0):
    """The demo farm: the question, the risk, July 1, July 3, July 5, then the water account."""
    farm_page(page, "07-01")
    hand = Hand(page)
    clk = Clock(page, "mitigate", rec0)
    clk.go()
    clk.at("mitigate", 7.0)
    hand.to("#vQ", 1.0, fx=0.3)
    clk.at("predicts", 0.2)
    hand.to("#t1")
    clk.at("predicts", 3.9)
    hand.to("#t2")
    clk.at("july1", 0.0)
    hand.to("#vBig", fx=0.4)
    clk.at("july3", -0.4)
    hand.click(strip_cell(page, "07-03"), 0.8)
    clk.at("july3", 2.9)
    hand.to("#t1x", 0.8)
    clk.at("july3", 5.0)
    hand.to("#vBig", 0.8, fx=0.45)
    clk.at("twodays", -1.0)
    hand.click(strip_cell(page, "07-05"), 0.8)
    clk.at("water", -0.6)
    scroll_to(page, "#waterCard", 1400)
    clk.at("water", 3.0)
    hand.click("#irrAdd", 0.9)
    clk.at("water", 6.6)
    hand.click("#pivotImport", 0.8)
    clk.at("water", 9.4)
    hand.to("#waterChart", 0.9, fx=0.75, fy=0.45)
    clk.at("water", 11.6)
    hand.to("#waterNow", 0.8)
    clk.until_end("water")
    return clk.marks


def ask(page, rec0):
    """Ask FlashWatch on July 3: the question typed, then (after a cut in the edit) Muse Spark's answer."""
    farm_page(page, "07-03")
    page.wait_for_selector("#askCard:not([hidden])", timeout=15000)
    jump_to(page, "#askCard", 520)
    page.wait_for_timeout(1000)
    hand = Hand(page)
    clk = Clock(page, "ask", rec0)
    clk.go()
    scroll_to(page, "#askCard", 1200, top=160)
    page.wait_for_timeout(1300)
    hand.click("#askQ", 0.8, pause=150)
    page.keyboard.type("I put an inch on yesterday. Do I still need to start?", delay=42)
    page.wait_for_timeout(350)
    hand.click("#askForm button", 0.5)
    clk.mark("asked")
    page.wait_for_selector("#askLog .msg.bot:not(.pending)", timeout=120000)
    clk.mark("answer")
    page.wait_for_timeout(600)
    hand.to("#askLog .msg.bot:not(.pending)", 1.0, fx=0.6, fy=0.3)
    page.wait_for_timeout(9000)
    return clk.marks


def agents(page, rec0):
    """Agents page: SAM's pivots, the operator's checked plan (the phone clip goes on top), the voice note."""
    page.goto(SITE + "agents.html")
    page.wait_for_selector("#pivImg img")
    page.wait_for_timeout(2000)
    hand = Hand(page)
    clk = Clock(page, "setup", rec0)
    clk.go()
    scroll_to(page, "#aPivots", 1300, top=40)
    clk.at("setup", 1.6)
    hand.to("#pivImg img", 1.0, fx=0.45, fy=0.45)
    clk.at("setup", 4.6)
    hand.to("#pivList", 1.0, fy=0.35)
    clk.at("operator", -0.6)
    scroll_to(page, "#aOperator", 1300, top=40)
    clk.at("operator", 1.6)
    hand.to("#opTrace table", 0.9, fx=0.35)
    clk.at("operator", 5.2)
    hand.to("#opTrace .tstep:nth-of-type(2)", 0.9, fx=0.3)
    clk.at("voice", -0.6)
    scroll_to(page, "#aVoice", 1300, top=40)
    clk.at("voice", 2.4)
    hand.to("#voice blockquote", 1.0, fx=0.4)
    clk.at("voice", 5.6)
    hand.to("#voice table", 1.0, fx=0.4)
    clk.until_end("voice")
    return clk.marks


def rainfed(page, rec0):
    """Rain-fed: the switch, then the decision cards; a slider moves and the card updates."""
    farm_page(page, "07-19")
    hand = Hand(page)
    clk = Clock(page, "rainfed", rec0)
    clk.go()
    hand.click("#wRain", 0.9)
    clk.at("rainfed", 4.0)
    hand.to("#vBig", 1.0, fx=0.4)
    clk.at("rainfed", 6.8)
    scroll_to(page, "#decisions", 1500, top=40)
    clk.at("rainfed", 10.4)
    hand.to("#dc-fungicide_corn h4", 0.9)
    clk.at("rainfed", 12.0)
    hand.to("#dc-fungicide_corn .dout", 0.8)
    clk.at("rainfed", 14.4)
    scroll_to(page, "#dc-nitrogen", 1200, top=60)
    hand.to("#dc-nitrogen h4", 0.9)
    clk.at("rainfed", 16.6)
    hand.to("#dc-nitrogen .dout", 0.9)
    clk.at("rainfed", 19.4)
    scroll_to(page, "#dc-fungicide_corn", 1300, top=40)
    clk.at("rainfed", 21.2)
    hand.drag("#dc-fungicide_corn .dparams input[type=range]", 70, 1.0)
    page.wait_for_timeout(700)
    hand.drag("#dc-fungicide_corn .dparams input[type=range]", -70, 0.9)
    clk.at("rainfed", 25.6)
    hand.to("#dc-fungicide_corn .dpill", 0.9)
    clk.until_end("rainfed")
    return clk.marks


def bigpicture(page, rec0):
    """The big picture: forecast quality (six times normal), the lead over the Monitor, the dollar totals."""
    page.goto(SITE + "region.html")
    page.wait_for_selector("#regionTotals:not(:empty)")
    page.wait_for_timeout(2500)
    hand = Hand(page)
    clk = Clock(page, "howgood", rec0)
    clk.go()
    clk.at("howgood", 1.2)
    scroll_to(page, "#flagKpis", 1400, top=140)
    clk.at("howgood", 3.4)
    hand.to("#flagKpis", 1.0, fx=0.2)
    clk.at("howgood", 7.4)
    hand.to("#flagChart", 1.0, fx=0.5)
    clk.at("howgood", 13.4)
    scroll_to(page, "#monitor", 1300, top=200)
    hand.to("#monitor", 0.9, fx=0.25)
    clk.at("howgood", 17.0)
    scroll_to(page, "#regionTotals", 1400, top=160)
    hand.to("#regionTotals", 1.0, fx=0.3)
    clk.until_end("howgood")
    return clk.marks


def hood(page, rec0):
    """Under the hood: the README's architecture diagram on GitHub."""
    page.goto(REPO)
    pre = page.locator("article.markdown-body pre", has_text="Muse").first
    pre.wait_for(timeout=30000)
    page.wait_for_timeout(1500)
    hand = Hand(page)
    clk = Clock(page, "hood", rec0)
    clk.go()
    y = pre.evaluate("p => p.getBoundingClientRect().top + scrollY - 120")
    page.evaluate(SCROLL, [y, 2200])
    clk.at("hood", 1.6)
    hand.to(pre, 1.0, fx=0.25, fy=0.15)
    clk.at("hood", 7.6)
    hand.to(pre, 1.0, fx=0.35, fy=0.45)
    clk.at("hood", 13.6)
    hand.to(pre, 1.0, fx=0.3, fy=0.8)
    clk.until_end("hood")
    return clk.marks


SHOTS = {f.__name__: f for f in (july3, region2012, farm, ask, agents, rainfed, bigpicture, hood)}
PACED = [n for n in SHOTS if n != "july3"]


def record(name: str):
    OUT.mkdir(exist_ok=True)
    raw = OUT / "raw"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        size = {"width": round(W / ZOOM), "height": round(H / ZOOM)}
        ctx = browser.new_context(viewport=size, color_scheme="light", record_video_dir=str(raw), record_video_size=size)
        ctx.add_init_script(CURSOR)
        page = ctx.new_page()
        rec0 = time.monotonic()
        marks = SHOTS[name](page, rec0)
        video = page.video
        ctx.close()
        browser.close()
        src = Path(video.path())
    ffmpeg = shutil.which("ffmpeg") or str(next(Path.home().glob(
        "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe")))
    dst = OUT / f"{name}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale={W}:{H}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-r", "30", str(dst)], check=True)
    src.unlink()
    (OUT / f"{name}.json").write_text(json.dumps(marks))
    print(dst, marks)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print(" ".join(SHOTS))
    else:
        for name in (PACED if sys.argv[1:] == ["all"] else sys.argv[1:]):
            record(name)
