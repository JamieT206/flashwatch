"""Cuts the narration into the script's beats and assembles the demo video (docs/video.md).

    python video/edit.py audio        # video/out/narration.wav + timeline.json (the beats on the final clock)
    python video/record.py all        # the browser shots, paced to timeline.json
    python video/edit.py video        # video/out/flashwatch_demo.mp4

The narration is one long take (video/in/narration.m4a) plus re-recorded lines (fix1, fix2); each beat is one
or more (take, start, end) ranges in seconds, from word-timed transcripts (faster-whisper). Each take is
scaled to the main take's speech loudness. Cut points move to the quietest 20 ms nearby so no
word is clipped. Beats are joined with short pauses; shots are recorded against the same clock, so each
shot is laid under its beats from the moment its first beat starts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IN, OUT = HERE / "in", HERE / "out"
FPS, W, H, SR = 30, 1920, 1080, 48000

# (beat id, source ranges in the narration take, pause after in seconds)
BEATS = [
    ("lead", None, 1.5),
    ("hello", [(3.8, 12.6)], 0.8),
    ("farmers", [("fix2", 0.6, 11.6), ("fix1", 37.9, 40.8)], 1.0),   # re-recorded; "redress that balance"
    ("drydown", [(34.5, 40.3)], 0.5),
    ("july2012", [(40.9, 49.8)], 0.8),
    ("mitigate", [(51.5, 64.6)], 1.0),
    ("predicts", [(68.0, 71.75), (73.3, 75.5)], 0.6),     # cut: "based on its exact location"
    ("july1", [(76.4, 78.9)], 0.6),
    ("july3", [("fix1", 3.0, 11.4)], 0.6),                # re-recorded: "says to start extra irrigation today"
    ("twodays", [(105.6, 108.0)], 1.0),                   # second take
    ("water", [(110.0, 123.3)], 1.0),
    ("ask", [(132.1, 134.7), (135.3, 145.1)], 1.0),        # third take of the first sentence
    ("setup", [("fix1", 14.6, 21.6)], 0.6),               # re-recorded: "an aerial photo"
    ("operator", [(155.4, 163.8)], 0.6),
    ("example", [(164.9, 165.5), (166.2, 174.2)], 0.4),    # cut: "in this example,"
    ("taps", [(175.0, 178.1)], 2.5),
    ("voice", [("fix1", 24.8, 34.0)], 1.0),               # re-recorded: "the operator agent takes it into account"
    ("rainfed", [(197.9, 226.7)], 1.0),
    ("howgood", [(229.7, 251.9)], 1.0),
    ("hood", [(255.8, 274.0)], 1.0),
    ("outro", [(294.9, 297.0), (300.35, 304.9)], 2.0),   # "My name is Jamie Twiss" from the take before (no "ah")
]

# Which picture runs under which beats: (source, first beat). A source runs until the next one starts.
#   shot:<name>  video/out/<name>.mp4 from record.py (its .json says where the first beat starts in it)
#   img:<path>   a still, slowly zoomed
SHOTS = [
    ("img:title", "lead"),
    ("img:aerial", "farmers"),
    ("shot:region2012", "drydown"),
    ("shot:farm", "mitigate"),
    ("shot:ask", "ask"),
    ("shot:agents", "setup"),
    ("shot:rainfed", "rainfed"),
    ("shot:bigpicture", "howgood"),
    ("shot:hood", "hood"),
    ("img:end", "outro"),
]
PHONE = ("operator", 2.0, 12.5)   # the WhatsApp clip goes over the Agents shot: from this beat + seconds; source start


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
    OUT.mkdir(exist_ok=True)
    takes = {n: load_take(n) for n in ("narration", "fix1", "fix2")}
    ref = speech_level(takes["narration"])
    takes = {n: a * ref / speech_level(a) for n, a in takes.items()}
    fade = int(0.01 * SR)
    parts, timeline, t = [], {}, 0.0
    for beat, ranges, pause in BEATS:
        start = t
        ranges = [r if len(r) == 3 else ("narration", *r) for r in ranges or []]
        for j, (take, lo, hi) in enumerate(ranges):
            a = takes[take]
            lo, hi = quiet(a, lo), quiet(a, hi)
            seg = a[int(lo * SR):int(hi * SR)].copy()
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
            parts.append(seg)
            t += len(seg) / SR
            if j < len(ranges) - 1:
                parts.append(np.zeros(int(0.12 * SR), np.float32))   # a breath where words were cut
                t += 0.12
        speech_end = t
        parts.append(np.zeros(int(pause * SR), np.float32))
        t += pause
        timeline[beat] = {"start": round(start, 3), "speech_end": round(speech_end, 3), "end": round(t, 3),
                          "source": list(ranges[0]) if ranges else None}
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
.rule{position:absolute;left:0;right:0;bottom:0;height:18px;background:#d83933}</style></head>
<body><div class="wrap"><div class="brand"><span class="dot"></span>FlashWatch</div>__BODY__</div><div class="rule"></div></body></html>"""
CARDS = {
    "title": '<div class="sub">A week\'s warning of flash drought, and what to do about it.</div>',
    "end": '<div class="sub">Built by Jamie Twiss</div><div class="links">jamiet206.github.io/flashwatch<br>github.com/JamieT206/flashwatch</div>',
}
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
        b.close()


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
    encode(["-ss", f"{marks['t0']:.3f}", "-i", src, "-vf", pad], dst, dur)


def phone_over(base: Path, start: float, dst: Path):
    """The WhatsApp screen recording over the Agents shot, from `start` seconds in."""
    src_from = PHONE[2]
    vf = (f"[1:v]trim=start={src_from},setpts=PTS-STARTPTS+{start}/TB,scale=-2:940,"
          f"pad=iw+16:ih+16:8:8:color=0x1b1b1b[p];"
          f"[0:v][p]overlay=x=W-w-150:y=(H-h)/2:eof_action=pass[v]")
    run("-i", base, "-i", IN / "phone.mp4", "-filter_complex", vf, "-map", "[v]", "-r", FPS, "-c:v", "libx264",
        "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", dst)


def video():
    tl = json.loads((OUT / "timeline.json").read_text())
    total = list(tl.values())[-1]["end"]
    if not (OUT / "title.png").exists():
        cards()
    segs = []
    for k, (src, first) in enumerate(SHOTS):
        t0 = tl[first]["start"]
        t1 = tl[SHOTS[k + 1][1]]["start"] if k + 1 < len(SHOTS) else total
        dst = OUT / f"seg{k:02d}.mp4"
        kind, name = src.split(":")
        if kind == "img":
            still(name, t1 - t0, dst, zoom=name not in CARDS)
        else:
            shot(name, t1 - t0, dst)
        if name == "agents":
            over = OUT / f"seg{k:02d}p.mp4"
            phone_over(dst, tl[PHONE[0]]["start"] + PHONE[1] - t0, over)
            dst = over
        segs.append(dst)
        print(f"{src:20s} {t0:6.1f}-{t1:6.1f}")
    lst = OUT / "segs.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in segs))
    final = OUT / "flashwatch_demo.mp4"
    run("-f", "concat", "-safe", 0, "-i", lst, "-i", OUT / "narration.wav", "-map", "0:v", "-map", "1:a",
        "-vf", f"fade=out:st={total - 1.2:.2f}:d=1.2:color=white", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final)
    print(final, f"{total:.1f} s")


if __name__ == "__main__":
    {"audio": audio, "video": lambda: video()}[sys.argv[1]]()
