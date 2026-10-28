"""Find a farm's centre pivots from aerial imagery with Meta's SAM 3.1 (farm setup without typing).

    python -m flashwatch.agent.pivots --lat 40.83 --lon -97.85            # needs META_API_KEY

Fetches a USDA NAIP aerial image (public domain, USGS National Map image service) centred on the farm,
asks SAM 3.1 to segment "center pivot irrigation circle", and turns each circular detection into a pivot:
its centre (lat, lon), diameter and irrigated acres (a full circle, pi r^2). Boxes are enough for that, so
masks aren't decoded. Writes data/agent/pivots/<lat>_<lon>.json and an annotated image beside it.
Request and response format: dev.meta.ai/docs/sam (Responses API, streamed box and mask records).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import urllib.parse
import urllib.request

from flashwatch.config import DATA, load_env

NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
R_EARTH = 6378137.0
PHRASE = "center pivot irrigation circle"
M2_PER_ACRE = 4046.86
BOX = re.compile(r"<\|box;([^|]*)\|>")


def mercator(lat: float, lon: float) -> tuple[float, float]:
    return math.radians(lon) * R_EARTH, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R_EARTH


def naip(lat: float, lon: float, half_m: float = 2400, px: int = 1024) -> bytes:
    """A px x px JPEG covering 2 x half_m metres of ground on each side, centred on the farm."""
    x, y = mercator(lat, lon)
    h = half_m / math.cos(math.radians(lat))           # Web Mercator stretches distances by 1/cos(lat)
    q = urllib.parse.urlencode({"bbox": f"{x - h},{y - h},{x + h},{y + h}", "bboxSR": 3857, "imageSR": 3857,
                                "size": f"{px},{px}", "format": "jpg", "f": "image"})
    req = urllib.request.Request(f"{NAIP}?{q}", headers={"User-Agent": "FlashWatch/0.1 (research prototype)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def segment(image: bytes, phrase: str = PHRASE, key: str | None = None, base: str | None = None, post=None) -> str:
    """SAM 3.1 through the Meta Model API (non-streaming): the output text of box and mask records."""
    base = (base or os.getenv("META_API_BASE", "https://api.meta.ai/v1")).rstrip("/")
    body = {"model": "sam-3.1", "metadata": {"mask_encoding": "one_bit"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}]}]}
    if post:                                            # tests
        return post(base + "/responses", body)
    req = urllib.request.Request(base + "/responses", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key or os.environ['META_API_KEY']}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    if out.get("output_text") is not None:
        return out["output_text"]
    return "".join(c.get("text", "") for item in out.get("output", []) for c in item.get("content", []) if isinstance(c, dict))


def boxes(text: str) -> list[dict]:
    out = []
    for m in BOX.finditer(text):
        kv = dict(part.split("=", 1) for part in m.group(1).split(";") if "=" in part)
        out.append({k: float(v) for k, v in kv.items()})
    return out


def to_pivots(bx: list[dict], lat: float, lon: float, half_m: float = 2400) -> list[dict]:
    """Circular detections (width ~ height, 150-1000 m across) as pivots with centre, diameter, acres."""
    pivots = []
    for b in bx:
        mpp = 2 * half_m / b["w"]                         # metres of ground per pixel
        wm, hm = (b["x2"] - b["x1"]) * mpp, (b["y2"] - b["y1"]) * mpp
        if not (150 <= wm <= 1000 and 0.8 <= wm / max(hm, 1) <= 1.25):
            continue
        cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
        dn, de = (b["h"] / 2 - cy) * mpp, (cx - b["w"] / 2) * mpp
        d = (wm + hm) / 2
        pivots.append({"lat": round(lat + dn / 111320, 5), "lon": round(lon + de / (111320 * math.cos(math.radians(lat))), 5),
                       "diameter_m": round(d), "acres": round(math.pi * (d / 2) ** 2 / M2_PER_ACRE),
                       "box_px": [int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])], "confidence": b.get("c")})
    return sorted(pivots, key=lambda p: (-p["lat"], p["lon"]))


def annotate(image: bytes, pivots: list[dict], path) -> None:
    import io
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(image)).convert("RGB")
    dr = ImageDraw.Draw(im)
    for k, p in enumerate(pivots, 1):
        x1, y1, x2, y2 = p["box_px"]
        dr.ellipse([x1, y1, x2, y2], outline=(216, 57, 51), width=4)
        dr.text(((x1 + x2) / 2 - 18, (y1 + y2) / 2 - 8), f"P{k} {p['acres']} ac", fill=(255, 255, 255))
    im.save(path)


def farm_config(res: dict, place: str) -> dict:
    """A farm file (same shape as config/farm_demo.json) from the pivots found; crop, soil and capacity are
    starting guesses for the farmer to correct."""
    return {"_note": f"Pivots found by SAM 3.1 in USDA NAIP imagery around {res['lat']}, {res['lon']}. Crop, soil and capacity are guesses: correct them.",
            "name": f"Farm near {place}", "place": place, "approver": "the farmer", "connector": {"kind": "simulated"}, "wells": {},
            "pivots": {f"P{k}": {"lat": p["lat"], "lon": p["lon"], "acres": p["acres"], "crop": "corn", "soil": "Loam",
                                 "gpm_per_acre": 5.0, "make": "unknown"} for k, p in enumerate(res["pivots"], 1)}}


def find(lat: float, lon: float, half_m: float = 2400) -> dict:
    img = naip(lat, lon, half_m)
    text = segment(img)
    bx = boxes(text)
    piv = to_pivots(bx, lat, lon, half_m)
    out = DATA / "agent" / "pivots"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{lat:.3f}_{lon:.3f}"
    (out / f"{stem}.jpg").write_bytes(img)
    annotate(img, piv, out / f"{stem}_pivots.png")
    res = {"lat": lat, "lon": lon, "half_m": half_m, "phrase": PHRASE, "detections": len(bx), "pivots": piv,
           "image": "USDA NAIP via USGS National Map (public domain)"}
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1))
    # For the site: the result and the annotated image (NAIP is public domain).
    from flashwatch.site.build import SITE
    (SITE / "data" / "pivots").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / "pivots" / f"{stem}.json").write_text(json.dumps(res))
    annotate(img, piv, SITE / "data" / "pivots" / f"{stem}.png")
    return res


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--half", type=float, default=2400, help="metres of ground each side of the farm")
    a = ap.parse_args()
    r = find(a.lat, a.lon, a.half)
    print(f"{r['detections']} detections, {len(r['pivots'])} pivots:")
    for k, p in enumerate(r["pivots"], 1):
        print(f"  P{k}: {p['acres']} acres, {p['diameter_m']} m across, at {p['lat']}, {p['lon']}")


if __name__ == "__main__":
    main()
