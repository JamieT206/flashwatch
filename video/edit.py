"""Cuts the narration into the script's beats and assembles the demo video (docs/video.md), in two cuts:
`hack` (the hackathon entry) and `general` (FlashWatch the product: no hackathon, no Meta model names).

    $env:CUT = "general"              # PowerShell; default "hack". Each cut has its own folder, video/out/<cut>/
    python video/edit.py audio        # narration.wav + timeline.json (the beats on the final clock)
    python video/record.py all        # the browser shots, paced to timeline.json
    python video/edit.py video        # flashwatch_<cut>.mp4
    python video/edit.py thumb        # video/out/thumbnail.jpg (YouTube, both cuts)

The narration is two takes: video/in/hack.m4a (every line, hackathon wording) and general.m4a (the lines that
differ, plus a re-recorded "farmers" line used in both cuts). Each beat is one or more (take, start, end) ranges in
seconds, from word-timed transcripts (faster-whisper, video/in/<take>.json). Each take is scaled to the hack take's
speech loudness. Cut points move to the quietest 20 ms nearby so no word is clipped, except ends marked exact
(a 4th element "lo", "hi" or "both": cuts inside a sentence, placed by hand and checked by transcribing the join).
Beats are joined with pauses; a pause given as ("min", s) holds the beat to at least s seconds, for shots that
need longer than the line. Shots are recorded against the same clock, so each shot is laid under its beats from
the moment its first beat starts.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CUT = os.environ.get("CUT", "hack")
IN, OUT = HERE / "in", HERE / "out" / CUT
FPS, W, H, SR = 30, 1920, 1080, 48000

# (beat id, source ranges, pause after in seconds or ("min", beat length))
HACK = [
    ("lead", None, 1.5),
    ("hello", [("hack", 1.8, 10.5)], 0.8),
    ("farmers", [("general", 72.3, 80.1)], 1.0),          # "we should be building more tools" (Nov 2 re-record)
    ("drydown", [("hack", 22.5, 25.95)], ("min", 6.0)),   # time to read the definition caption
    ("monitor", [("hack", 26.45, 32.4)], ("min", 9.0)),   # the Drought Monitor's weekly maps, and the 22-day lead
    ("mitigate", [("hack", 42.5, 46.0)], 0.8),             # second take
    ("predicts", [("hack", 47.2, 48.65, "hi"), ("hack", 49.13, 57.9, "lo")], 0.8),   # cut: "satellite"
    ("july1", [("hack", 59.8, 62.7)], 0.6),
    ("july3", [("hack", 63.55, 70.5)], 0.6),
    ("twodays", [("hack", 71.05, 73.8)], 1.0),
    ("water", [("hack", 77.2, 80.25, "hi"), ("hack", 82.85, 85.1, "lo")], ("min", 9.5)),   # cut: "reading ... pivot"
    ("ask", [("hack", 87.1, 91.5)], ("min", 10.5)),
    ("setup", [("hack", 93.0, 99.6)], 0.6),
    ("operator", [("hack", 111.4, 118.95)], 0.6),         # second take
    ("taps", [("hack", 121.1, 124.6)], 3.5),               # long enough for "Done" to come back on the phone
    ("voice", [("hack", 127.55, 132.4)], 1.0),
    ("rainfed", [("hack", 145.4, 155.9)], ("min", 14.5)),  # second take
    ("howgood", [("hack", 168.4, 183.4)], 1.0),            # second take
    ("hood", [("hack", 185.8, 195.7)], 1.0),               # first take ("carry it to the farmer")
    ("outro", [("hack", 214.0, 219.5)], 2.0),
]
GENERAL_LINES = {
    "hello": [("general", 2.2, 11.0)],
    "ask": [("general", 13.7, 16.5)],
    "setup": [("general", 19.3, 25.4)],
    "voice": [("general", 27.55, 33.85)],                  # first take ("on the correct pivot")
    "hood": [("general", 49.2, 61.4)],                     # second take
    "outro": [("general", 63.1, 69.6)],
}
BEATS = {"hack": HACK,
         "general": [(b, GENERAL_LINES.get(b, r), p) for b, r, p in HACK]}[CUT]

# Which picture runs under which beats: (source, first beat). A source runs until the next one starts.
#   shot:<name>  video/out/<name>.mp4 from record.py (its .json says where the first beat starts in it)
#   img:<path>   a still, slowly zoomed
#   seq:usdm     the Drought Monitor's weekly maps, July 3-24 2012 (video/in/usdm/, from droughtmonitor.unl.edu)
SHOTS = [
    ("img:title", "lead"),
    ("img:aerial", "farmers"),
    ("shot:region2012", "drydown"),
    ("seq:usdm", "monitor"),
    ("shot:farm", "mitigate"),
    ("shot:ask", "ask"),
    ("shot:agents", "setup"),
    ("shot:rainfed", "rainfed"),
    ("shot:bigpicture", "howgood"),
    ("shot:hood", "hood"),
    ("img:end", "outro"),
]
# The WhatsApp screen recording (in/phone_orig.mov, 1206x2622) goes over the Agents shot. The plan arrives as the
# operator beat starts; the clip holds on it until the narration says "taps", then the tap (drawn: screen recordings
# show no finger), YES, and "Done" coming back. Times are seconds in the recording.
PHONE = {"src": "phone_orig.mov", "from": 12.58, "hold": 16.75, "tap": 16.95, "yes": (500, 2158),
         "start": ("operator", 2.0), "tap_at": ("taps", 1.9), "end": ("taps", 0.4), "height": 960}


def ffmpeg() -> str:
    return shutil.which("ffmpeg") or str(next(Path.home().glob(
        "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe")))


def run(*args):
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", *map(str, args)], check=True)


def load_take(name: str) -> np.ndarray:
    wav = OUT / "take.f32"
    run("-i", IN / f"{name}.m4a", "-ac", "1", "-ar", SR, "-f", "f32le", wav)
    a = np.fromfile(wav, dtype=np.float32)
    wav.unlink()
    return a


def speech_level(a: np.ndarray) -> float:
    """RMS of the louder half of 50 ms frames: the voice, not the room."""
    w = int(0.05 * SR)
    rms = np.sqrt(np.mean(a[:len(a) // w * w].reshape(-1, w) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.sort(rms)[len(rms) // 2:] ** 2)))


def quiet(a: np.ndarray, t: float, span: float = 0.2) -> float:
    """The quietest 20 ms window within `span` of t (seconds)."""
    w = int(0.02 * SR)
    lo, hi = max(0, int((t - span) * SR)), min(len(a) - w, int((t + span) * SR))
    starts = np.arange(lo, hi, w // 2)
    rms = [float(np.sqrt(np.mean(a[s:s + w] ** 2))) for s in starts]
    return (starts[int(np.argmin(rms))] + w / 2) / SR


def audio():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted({r[0] for _, ranges, _ in BEATS for r in ranges or []})
    takes = {n: load_take(n) for n in names}
    ref = speech_level(load_take("hack"))
    takes = {n: a * ref / speech_level(a) for n, a in takes.items()}
    fade = int(0.01 * SR)
    parts, timeline, t = [], {}, 0.0
    for beat, ranges, pause in BEATS:
        start = t
        for j, (take, lo, hi, *exact) in enumerate(ranges or []):
            a, exact = takes[take], exact[0] if exact else ""
            lo = lo if exact in ("lo", "both") else quiet(a, lo)
            hi = hi if exact in ("hi", "both") else quiet(a, hi)
            seg = a[int(lo * SR):int(hi * SR)].copy()
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
            parts.append(seg)
            t += len(seg) / SR
            if j < len(ranges) - 1:   # a breath where words were cut; next to nothing inside a sentence
                gap = 0.03 if exact in ("hi", "both") else 0.12
                parts.append(np.zeros(int(gap * SR), np.float32))
                t += gap
        speech_end = t
        if isinstance(pause, tuple):
            pause = max(0.6, pause[1] - (t - start))
        parts.append(np.zeros(int(pause * SR), np.float32))
        t += pause
        timeline[beat] = {"start": round(start, 3), "speech_end": round(speech_end, 3), "end": round(t, 3),
                          "source": list(ranges[0][:3]) if ranges else None}
    track = np.concatenate(parts)
    track /= max(1e-6, float(np.abs(track).max())) / 0.89          # peak at about -1 dBFS
    track.astype(np.float32).tofile(OUT / "narration.f32")
    run("-f", "f32le", "-ar", SR, "-ac", 1, "-i", OUT / "narration.f32", OUT / "narration.wav")
    (OUT / "narration.f32").unlink()
    (OUT / "timeline.json").write_text(json.dumps(timeline, indent=1))
    print(f"narration {t:.1f} s, {len(timeline)} beats -> {OUT / 'timeline.json'}")


CARD = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>html,body{margin:0;width:1920px;height:1080px;background:#fff;font-family:'Public Sans',sans-serif;color:#1b1b1b}
.wrap{position:absolute;left:220px;top:50%;transform:translateY(-50%)}
.brand{display:flex;align-items:center;gap:28px;font-size:112px;font-weight:800;letter-spacing:-1px}
.dot{width:72px;height:72px;background:#d83933}
.sub{font-size:46px;margin-top:28px;color:#3d4551;max-width:1400px;line-height:1.25}
.links{font-size:38px;margin-top:56px;color:#1a4480;line-height:1.6}
.links b{display:inline-block;width:250px;font-weight:600;color:#3d4551}
.rule{position:absolute;left:0;right:0;bottom:0;height:18px;background:#d83933}</style></head>
<body><div class="wrap"><div class="brand"><span class="dot"></span>FlashWatch</div>__BODY__</div><div class="rule"></div></body></html>"""
CARDS = {
    "title": '<div class="sub">A week\'s warning of flash drought, and what to do about it.</div>',
    "end": '<div class="sub">Built by Jamie Twiss</div><div class="links"><b>Live demo</b>jamiet206.github.io/flashwatch<br><b>Code</b>github.com/JamieT206/flashwatch</div>',
}
# The Drought Monitor's national maps, one a week, like a page being updated; the lead (README; evidence.json
# monitor.crossing_vs_d1: median 22 days over the 2006-2021 onsets) comes in with the third map.
USDM = [("20120703", "July 3, 2012", 2.0), ("20120710", "July 10", 1.4), ("20120717", "July 17", 1.4),
        ("20120724", "July 24", None)]
USDM_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>html,body{margin:0;width:1920px;height:1080px;background:#fff;font-family:'Public Sans',sans-serif;color:#1b1b1b}
img{position:absolute;left:40px;top:50px;height:960px}
.col{position:absolute;left:1340px;top:120px;width:500px}
.k{font-size:30px;font-weight:800;letter-spacing:1px;color:#3d4551}
.s{font-size:30px;line-height:1.35;color:#3d4551;margin-top:14px}
.d{font-size:76px;font-weight:800;margin-top:44px}
.c{font-size:34px;line-height:1.35;margin-top:56px;border-left:10px solid #d83933;padding-left:22px}
.cr{position:absolute;left:1340px;bottom:56px;font-size:20px;color:#71767a}
.rule{position:absolute;left:0;right:0;bottom:0;height:18px;background:#d83933}</style></head>
<body><img src="__IMG__"><div class="col"><div class="k">U.S. DROUGHT MONITOR</div>
<div class="s">The government's drought map: one map a week, of drought already on the ground.</div>
<div class="d">__DATE__</div>__CAP__</div>
<div class="cr">Maps: U.S. Drought Monitor (NDMC, USDA, NOAA)</div><div class="rule"></div></body></html>"""
USDM_CAP = ('<div class="c">In 2006-2021 the soil reached drought a median <b>22 days</b> before this map showed it. '
            'FlashWatch warns one to two weeks before that.</div>')
# Captions over a shot: shot name -> (png name, seconds in, HTML)
CAPTIONS = {"region2012": ("cap_flash", 0.5, "<b>Flash drought:</b> the soil goes from normal to drought in a few "
                                             "weeks, not months, usually under heat and little rain.")}
CAP_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>html,body{margin:0;width:1920px;height:1080px;background:transparent;font-family:'Public Sans',sans-serif}
div{position:absolute;left:70px;bottom:80px;max-width:1080px;background:rgba(255,255,255,.95);color:#1b1b1b;
font-size:40px;line-height:1.35;padding:26px 34px;border-left:12px solid #d83933;box-shadow:0 4px 18px rgba(0,0,0,.18)}
</style></head><body><div>__TEXT__</div></body></html>"""
STILLS = {"aerial": Path(__file__).resolve().parents[1] / "data" / "agent" / "pivots" / "42.450_-98.600.jpg"}


def cards():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": W, "height": H})
        for name, body in CARDS.items():
            page.set_content(CARD.replace("__BODY__", body))
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / f"{name}.png"))
        for k, (day, label, _) in enumerate(USDM):
            img = "data:image/png;base64," + base64.b64encode((IN / "usdm" / f"{day}_usdm.png").read_bytes()).decode()
            page.set_content(USDM_PAGE.replace("__IMG__", img).replace("__DATE__", label)
                             .replace("__CAP__", USDM_CAP if k >= 2 else ""))
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / f"usdm{k}.png"))
        for png, _, text in CAPTIONS.values():
            page.set_content(CAP_PAGE.replace("__TEXT__", text))
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / f"{png}.png"), omit_background=True)
        b.close()


# The YouTube thumbnail (1280x720, both cuts): the aerial pivots, the tagline, and the July 3 advice card in the farm
# page's own colours (site/assets/style.css: --act, --act-bg).
THUMB = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>html,body{margin:0;width:1280px;height:720px;overflow:hidden;font-family:'Public Sans',sans-serif;color:#1b1b1b}
.bg{position:absolute;inset:0;background:url(__IMG__) right center/cover}
.fade{position:absolute;inset:0;background:linear-gradient(90deg,#fff 0%,#fff 38%,rgba(255,255,255,.85) 52%,rgba(255,255,255,0) 72%)}
.txt{position:absolute;left:64px;top:62px;width:640px}
.brand{display:flex;align-items:center;gap:18px;font-size:58px;font-weight:800;letter-spacing:-1px}
.dot{width:40px;height:40px;background:#d83933}
h1{font-size:76px;line-height:1.02;font-weight:800;margin:34px 0 0;letter-spacing:-1.5px}
.card{position:absolute;left:64px;bottom:78px;background:#fbeceb;border-left:14px solid #b50909;
  padding:22px 28px 24px;box-shadow:0 10px 34px rgba(0,0,0,.35)}
.q{font-size:26px;color:#3d4551;font-weight:600}
.big{font-size:46px;line-height:1.05;font-weight:800;color:#b50909;margin:8px 0 10px}
.sub{font-size:26px;font-weight:600}
.rule{position:absolute;left:0;right:0;bottom:0;height:16px;background:#d83933}</style></head>
<body><div class="bg"></div><div class="fade"></div>
<div class="txt"><div class="brand"><span class="dot"></span>FlashWatch</div><h1>A week's warning of flash drought</h1></div>
<div class="card"><div class="q">Extra irrigation today?</div><div class="big">Start extra irrigation today</div>
<div class="sub">Flash-drought risk 5.3 times normal</div></div><div class="rule"></div></body></html>"""


def thumbnail():
    from PIL import Image
    from playwright.sync_api import sync_playwright
    img = "data:image/jpeg;base64," + base64.b64encode(STILLS["aerial"].read_bytes()).decode()
    png, jpg = OUT.parent / "thumbnail.png", OUT.parent / "thumbnail.jpg"
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1280, "height": 720})
        page.set_content(THUMB.replace("__IMG__", img))
        page.wait_for_timeout(1500)
        page.screenshot(path=str(png))
        b.close()
    Image.open(png).convert("RGB").save(jpg, quality=92)       # YouTube: 1280x720, under 2 MB
    print(jpg)


def usdm(dur: float, dst: Path):
    """The weekly maps in turn, the last held to the end of the segment."""
    times = [d for *_, d in USDM[:-1]]
    times.append(dur - sum(times))
    lst = OUT / "usdm.txt"
    lst.write_text("".join(f"file 'usdm{k}.png'\nduration {d:.3f}\n" for k, d in enumerate(times))
                   + f"file 'usdm{len(times) - 1}.png'\n")
    encode(["-f", "concat", "-safe", 0, "-i", lst, "-vf", f"scale={W}:{H},fps={FPS}"], dst, dur)


def encode(args: list, dst: Path, dur: float):
    run(*args, "-t", f"{dur:.3f}", "-r", FPS, "-c:v", "libx264", "-crf", "17", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an", dst)


def still(name: str, dur: float, dst: Path, zoom: bool):
    src = STILLS.get(name, OUT / f"{name}.png")
    n = int(dur * FPS) + 1
    if zoom:   # slow push in, drawn at 4K and scaled down so the motion stays smooth
        vf = (f"scale=3840:3840:force_original_aspect_ratio=increase,crop=3840:2160,"
              f"zoompan=z='1+0.10*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS}")
    else:      # a card: still, faded in from white
        vf = f"scale={W}:{H},fade=in:st=0:d=0.6:color=white"
    encode(["-loop", "1", "-i", src, "-vf", vf], dst, dur)


def shot(name: str, dur: float, dst: Path):
    marks = json.loads((OUT / f"{name}.json").read_text())
    src = OUT / f"{name}.mp4"
    pad = f"tpad=stop_mode=clone:stop_duration={dur:.2f}"
    if name == "ask":   # the question, then a cut to the answer (Muse Spark takes 10-20 s)
        a = marks["asked"] + 1.8 - marks["t0"]
        b = marks["answer"] - 0.4
        vf = (f"[0:v]trim={marks['t0']}:{marks['asked'] + 1.8},setpts=PTS-STARTPTS[a];"
              f"[0:v]trim=start={b},setpts=PTS-STARTPTS,{pad}[b];[a][b]concat=n=2:v=1[v]")
        encode(["-i", src, "-filter_complex", vf, "-map", "[v]"], dst, dur)
        return
    vf = pad
    if "zoom" in marks:   # push in on part of the page once it has scrolled there (the README diagram)
        vf += "," + push_in(marks["zoom"])
    if name in CAPTIONS:  # a caption box faded in over the shot
        png, at, _ = CAPTIONS[name]
        fc = f"[0:v]{vf}[b];[1:v]format=rgba,fade=in:st={at}:d=0.4:alpha=1[c];[b][c]overlay=0:0:shortest=1[v]"
        encode(["-ss", f"{marks['t0']:.3f}", "-i", src, "-loop", 1, "-framerate", FPS, "-i", OUT / f"{png}.png",
                "-filter_complex", fc, "-map", "[v]"], dst, dur)
        return
    encode(["-ss", f"{marks['t0']:.3f}", "-i", src, "-vf", vf], dst, dur)


def push_in(box: dict, ramp: float = 1.5, margin: float = 1.12) -> str:
    """An eased zoom from the full frame to `box` (frame pixels), starting `box['at']` seconds in."""
    z = min(W / (box["width"] * margin), H / (box["height"] * margin), 2.2)
    cx, cy = 2 * (box["x"] + box["width"] / 2), 2 * (box["y"] + box["height"] / 2)   # drawn at 4K, as the stills
    p = f"clip((it-{box['at']})/{ramp},0,1)"
    e = f"({p}*{p}*(3-2*{p}))"
    return (f"scale={2 * W}:{2 * H}:flags=lanczos,zoompan=z='1+{z - 1:.3f}*{e}'"
            f":x='clip({W}+({cx - W:.1f})*{e}-iw/zoom/2,0,iw-iw/zoom)'"
            f":y='clip({H}+({cy - H:.1f})*{e}-ih/zoom/2,0,ih-ih/zoom)':d=1:s={W}x{H}:fps={FPS}")


def ripple(r: int) -> Path:
    """A grey touch mark, as phones draw for taps."""
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((1, 1, 2 * r - 2, 2 * r - 2), fill=(90, 90, 90, 110), outline=(255, 255, 255, 200), width=2)
    path = OUT / "ripple.png"
    im.save(path)
    return path


def phone_over(base: Path, tl: dict, seg_t0: float, dst: Path):
    """The WhatsApp screen recording over the Agents shot (see PHONE)."""
    P = PHONE
    at = lambda k: tl[P[k][0]]["start"] + P[k][1]
    start, tap = at("start") - seg_t0, at("tap_at") - seg_t0
    end = tl[P["end"][0]]["end"] + P["end"][1] - seg_t0
    resume = tap - (P["tap"] - P["hold"])                     # the clip carries on from `hold` at this time
    freeze = resume - start - (P["hold"] - P["from"])
    src_end = P["hold"] + end - resume
    k = P["height"] / 2622
    r = 24
    tx, ty = P["yes"][0] * k - r, P["yes"][1] * k - r
    t = tap - start                                           # the tap, in the clip's own time
    vf = (f"[1:v]fps={FPS},split[s1][s2];"
          f"[s1]trim={P['from']}:{P['hold']},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={freeze:.3f}[a];"
          f"[s2]trim={P['hold']}:{src_end:.3f},setpts=PTS-STARTPTS[b];"
          f"[a][b]concat=n=2:v=1:a=0,scale=-2:{P['height']}:flags=lanczos[ph];"
          f"[2:v]format=rgba,fade=in:st={t - 0.12:.3f}:d=0.06:alpha=1,fade=out:st={t + 0.02:.3f}:d=0.1:alpha=1[rp];"
          f"[ph][rp]overlay=x={tx:.0f}:y={ty:.0f}:shortest=1,pad=iw+16:ih+16:8:8:color=0x1b1b1b,"
          f"setpts=PTS-STARTPTS+{start:.3f}/TB[p];"
          f"[0:v][p]overlay=x=W-w-150:y=(H-h)/2:eof_action=pass[v]")
    run("-i", base, "-i", IN / P["src"], "-loop", 1, "-framerate", FPS, "-t", f"{end - start + 1:.2f}", "-i", ripple(r),
        "-filter_complex", vf, "-map", "[v]", "-r", FPS, "-c:v", "libx264",
        "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", dst)


def video():
    tl = json.loads((OUT / "timeline.json").read_text())
    total = list(tl.values())[-1]["end"]
    cards()
    segs = []
    for k, (src, first) in enumerate(SHOTS):
        t0 = tl[first]["start"]
        t1 = tl[SHOTS[k + 1][1]]["start"] if k + 1 < len(SHOTS) else total
        dst = OUT / f"seg{k:02d}.mp4"
        kind, name = src.split(":")
        if kind == "img":
            still(name, t1 - t0, dst, zoom=name not in CARDS)
        elif kind == "seq":
            usdm(t1 - t0, dst)
        else:
            shot(name, t1 - t0, dst)
        if name == "agents":
            over = OUT / f"seg{k:02d}p.mp4"
            phone_over(dst, tl, t0, over)
            dst = over
        segs.append(dst)
        print(f"{src:20s} {t0:6.1f}-{t1:6.1f}")
    lst = OUT / "segs.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in segs))
    final = OUT / f"flashwatch_{CUT}.mp4"
    run("-f", "concat", "-safe", 0, "-i", lst, "-i", OUT / "narration.wav", "-map", "0:v", "-map", "1:a",
        "-vf", f"fade=out:st={total - 1.2:.2f}:d=1.2:color=white", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final)
    print(final, f"{total:.1f} s")


if __name__ == "__main__":
    {"audio": audio, "video": lambda: video(), "thumb": thumbnail}[sys.argv[1]]()
