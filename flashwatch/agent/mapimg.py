"""Risk map image for the agent's multimodal cross-check (DESIGN.md G1).

Draws one weekly pack as a PNG: each land cell coloured by forecast risk relative to normal
(grey = normal or below, through orange to yellow at 4x normal and above), cells already in drought
in purple, latitude/longitude lines and a few reference cities. The agent describes what it sees
and its description is compared with the numeric tools (hotspots), so it can't just echo numbers.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

CITIES = {"Minneapolis": (44.98, -93.27), "Des Moines": (41.59, -93.62), "Omaha": (41.26, -95.93),
          "Kansas City": (39.10, -94.58), "St. Louis": (38.63, -90.20), "Chicago": (41.88, -87.63),
          "Indianapolis": (39.77, -86.16), "Nashville": (36.16, -86.78), "Detroit": (42.33, -83.05),
          "Sioux Falls": (43.55, -96.73), "Bismarck": (46.81, -100.78), "Columbus": (39.96, -83.00)}
RAMP = [(0.0, (60, 66, 76)), (1.0, (60, 66, 76)), (1.5, (120, 70, 30)), (2.5, (200, 110, 20)), (4.0, (255, 225, 140))]
NORTH, WEST, RES = 49.0, -104.0, 0.25


def _colour(ratio: float):
    if not np.isfinite(ratio):
        return (40, 44, 52)
    for (x0, c0), (x1, c1) in zip(RAMP, RAMP[1:]):
        if ratio <= x1:
            t = 0 if x1 == x0 else (max(ratio, x0) - x0) / (x1 - x0)
            return tuple(int(a + t * (b - a)) for a, b in zip(c0, c1))
    return RAMP[-1][1]


def render(pack: pd.DataFrame, lead: str = "any28", px: int = 8) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    ny, nx = 52, 96
    pad_l, pad_t, pad_b = 40, 30, 50
    img = Image.new("RGB", (pad_l + nx * px + 10, pad_t + ny * px + pad_b), (16, 18, 22))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    ratio = pack[f"p_{lead}"] / pack[f"normal_{lead}"]
    for (iy, ix), r, dry in zip(pack[["iy", "ix"]].to_numpy(), ratio, pack.in_drought):
        c = (90, 70, 140) if dry else _colour(r)
        x0, y0 = pad_l + ix * px, pad_t + iy * px
        d.rectangle([x0, y0, x0 + px - 1, y0 + px - 1], fill=c)
    for lat in range(37, 49, 2):
        y = pad_t + (NORTH - lat) / RES * px
        d.line([pad_l, y, pad_l + nx * px, y], fill=(80, 80, 90))
        d.text((4, y - 6), f"{lat}N", fill=(200, 200, 200), font=font)
    for lon in range(-102, -80, 4):
        x = pad_l + (lon - WEST) / RES * px
        d.line([x, pad_t, x, pad_t + ny * px], fill=(80, 80, 90))
        d.text((x - 12, pad_t + ny * px + 4), f"{-lon}W", fill=(200, 200, 200), font=font)
    for name, (lat, lon) in CITIES.items():
        x, y = pad_l + (lon - WEST) / RES * px, pad_t + (NORTH - lat) / RES * px
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))
        d.text((x + 5, y - 7), name, fill=(255, 255, 255), font=font)
    d.text((pad_l, 8), f"Flash-drought onset risk vs normal ({lead}); purple = already in drought; "
           "yellow = 4x normal or more", fill=(230, 230, 230), font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
