"""From forecast to action (DESIGN.md O): value of each action by days of notice, and when to act.

    python -m flashwatch.decide --week1 0.28 --week2 0.15            # all actions, default settings
    python -m flashwatch.decide --action irrigation --set gpm_per_acre=4 --week1 0.10 --week2 0.20

Settings come from config/actions.json; --set overrides any parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flashwatch.decide import irrigation
from flashwatch.decide.engine import Curve

ACTIONS = Path(__file__).resolve().parents[1] / "config" / "actions.json"


def load(path=ACTIONS, overrides: dict | None = None) -> dict:
    """{action: {"label", "per", "params": {name: value}, ...}} with overrides applied to every action."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, a in cfg.items():
        if name.startswith("_"):
            continue
        params = {k: v["value"] for k, v in a["params"].items()}
        params.update({k: v for k, v in (overrides or {}).items() if k in params})
        out[name] = {**a, "params": params}
    return out


def curve(action: dict, max_notice: int = 21) -> Curve:
    p, kind = action["params"], action["kind"]
    if kind == "irrigation":
        f = irrigation.field(**p)
        c = irrigation.curve(f, max_notice)
        noevent = irrigation.false_alarm_cost(f, p["review_days"]) * p["false_alarm_waste"]
        return Curve(c.notice_days.to_numpy(float), c.avoided.to_numpy(), c.extra_cost.to_numpy(), noevent)
    k = np.array(action["notice_days"], float)
    if "tons" in p:                                       # hay / feed
        rise = p["tons"] * p["price_per_ton"] * p["drought_rise"]
        share = np.array(action["avoided_share_of_rise"])
        return Curve(k, rise * share, np.zeros_like(k), p["tons"] * p["price_per_ton"] * p["carry_cost_share"])
    if "n_cut_lb" in p:                                   # nitrogen
        saved = p["n_cut_lb"] * p["n_price_per_lb"]
        wrong = p["yield_cost_if_wrong_pct"] / 100 * p["yield_bu"] * p["price"]
        return Curve(k, saved * np.array(action["avoided_share"]), np.zeros_like(k), wrong - saved)
    raise ValueError(f"unknown action kind {kind!r}")
