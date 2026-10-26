"""Fixed study settings (locked in HANDOVER.md section 5) and local paths."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("FW_DATA", ROOT / "data"))
RAW = DATA / "raw"
LOGS = DATA / "logs"

# Region 36-49N, 104-80W on a 0.25 degree grid: 52 x 96 cells.
NORTH, WEST, SOUTH, EAST = 49.0, -104.0, 36.0, -80.0
RES = 0.25
GRID_025 = {"dimensions": {"width": 96, "height": 52},
            "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": WEST,
                                "shearY": 0, "scaleY": -RES, "translateY": NORTH},
            "crsCode": "EPSG:4326"}

# D1 (decided Oct 26): study years 2005-2024 on one ECMWF cycle, the Cy49r1 reforecasts
# attached to the 2025 model-version dates (every other day; hindcast years 2005-2024).
# CV years 2005-2021, test years 2022-2024. 2004 is pulled for spin-up only.
YEARS = list(range(2005, 2025))
OBS_YEARS = list(range(2004, 2025))
CV_YEARS = list(range(2005, 2022))
TEST_YEARS = [2022, 2023, 2024]
S2S_MODEL_YEAR = 2025
SEASON = ("04-01", "10-15")


def load_env() -> int:
    """Load KEY=value lines from .env (or flashwatch.env) in the repo root; existing vars win."""
    n = 0
    for name in (".env", "flashwatch.env"):
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip().removeprefix("export "), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if value and key not in os.environ:
                os.environ[key] = value
                n += 1
    return n
