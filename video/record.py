"""Scripted screen recordings of the site for the demo video (docs/video.md).

    python video/record.py july3            # one shot; writes video/out/<shot>.mp4
    python video/record.py --list

Needs the site served on 127.0.0.1:8812 (`python -m http.server 8812 --directory site`) and, for the Ask shot,
`python -m flashwatch.agent.serve`. Playwright's Chromium records a 1745x982 window, scaled up to 1920x1080 (the
page at 110%, as in the shot list); a drawn cursor glides between targets on eased, slightly curved paths and dips on each
click, as in a live demo. ffmpeg (winget Gyan.FFmpeg) turns the recording into H.264.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE = "http://127.0.0.1:8812/"
OUT = Path(__file__).resolve().parent / "out"
W, H, ZOOM = 1920, 1080, 1.1
FARM = "index.html#lat=45.38&lon=-99.38&mode=replay&year=2012"

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


class Hand:
    """Moves the mouse like a person: eased, slightly curved, a little off-centre on each target."""

    def __init__(self, page, start=(W * 0.62, H * 0.55)):
        self.page, self.x, self.y = page, *start
        page.mouse.move(self.x, self.y)

    def move(self, x, y, seconds=None):
        d = math.hypot(x - self.x, y - self.y)
        seconds = seconds or min(1.1, 0.35 + d / 1800)
        n = max(8, int(seconds * 60))
        # a bend to one side, proportional to the distance
        bend = random.uniform(-0.12, 0.12) * d
        nx, ny = (-(y - self.y) / d, (x - self.x) / d) if d else (0, 0)
        for i in range(1, n + 1):
            t = i / n
            e = t * t * (3 - 2 * t)                      # ease in and out
            arc = math.sin(math.pi * e) * bend
            self.page.mouse.move(self.x + (x - self.x) * e + nx * arc, self.y + (y - self.y) * e + ny * arc)
            self.page.wait_for_timeout(1000 / 60)
        self.x, self.y = x, y

    def to(self, locator, seconds=None, jitter=0.18):
        b = locator.bounding_box()
        x = b["x"] + b["width"] * (0.5 + random.uniform(-jitter, jitter))
        y = b["y"] + b["height"] * (0.5 + random.uniform(-jitter, jitter))
        self.move(x, y, seconds)

    def click(self, locator, seconds=None, pause=180):
        self.to(locator, seconds)
        self.page.wait_for_timeout(pause)
        self.page.mouse.down()
        self.page.wait_for_timeout(90)
        self.page.mouse.up()


def strip_cell(page, md: str):
    """The date bar's cell for MM-DD in the replay year."""
    s = page.evaluate("""md => { const st = stops();
        return st.findIndex(x => x.kind === "day" && S.meta.dates[x.i] === md); }""", md)
    return page.locator(f'#strip [data-s="{s}"]')


def july3(page):
    """Shot 1-2: the demo farm on July 1 ("as usual"), then a click on July 3 ("start extra irrigation today")."""
    page.goto(SITE + FARM + "&date=07-01")
    page.wait_for_selector("#vBig:not(:empty)")
    page.wait_for_timeout(1500)
    hand = Hand(page)
    page.wait_for_timeout(1200)
    hand.to(page.locator("#vBig"), 0.9)
    page.wait_for_timeout(1500)
    hand.click(strip_cell(page, "07-03"))
    page.wait_for_timeout(1800)
    hand.to(page.locator("#vBig"), 0.8)
    page.wait_for_timeout(1200)
    hand.to(page.locator("#evNow"), 0.6)
    page.wait_for_timeout(2500)


SHOTS = {"july3": july3}


def record(name: str):
    OUT.mkdir(exist_ok=True)
    raw = OUT / "raw"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        size = {"width": round(W / ZOOM), "height": round(H / ZOOM)}
        ctx = browser.new_context(viewport=size, color_scheme="light", record_video_dir=str(raw), record_video_size=size)
        ctx.add_init_script(CURSOR)
        page = ctx.new_page()
        SHOTS[name](page)
        video = page.video
        ctx.close()
        browser.close()
        src = Path(video.path())
    ffmpeg = shutil.which("ffmpeg") or str(next(Path.home().glob(
        "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe")))
    dst = OUT / f"{name}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale={W}:{H}:flags=lanczos",
                    "-c:v", "libx264", "-crf", "16",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-r", "30", str(dst)], check=True)
    src.unlink()
    print(dst)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print(" ".join(SHOTS))
    else:
        for name in sys.argv[1:]:
            record(name)
