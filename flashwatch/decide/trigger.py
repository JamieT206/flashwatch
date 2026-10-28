"""What are we waiting for? (DESIGN.md O2, James Oct 28)

When the engine says "not yet", name the forecast change that would tip it, in weather a farmer
can check: each ECMWF ensemble member is a possible next forecast; members in which starting at the
next decision would pay are separated from the rest by the coming week's rainfall or temperature
at the farm. The split that separates them best becomes the trigger, e.g. "if the next forecast
shows less than 0.4 in of rain this week". When no single threshold separates them well enough,
the trigger falls back to the forecast risk itself.
"""

from __future__ import annotations

import numpy as np

from flashwatch.decide.engine import Curve, decide_members

VARS = {"rain_in": ("less than", "in of rain over the coming week", -1),
        "temp_f": ("an average temperature above", "F over the coming week", +1)}


def best_split(x: np.ndarray, act: np.ndarray, sign: int):
    """Threshold t so that 'act' ~ (sign * (x - t) > 0); returns (t, share of members agreeing)."""
    xs = np.sort(np.unique(x))
    best = (np.nan, 0.0)
    for a, b in zip(xs[:-1], xs[1:]):
        t = (a + b) / 2
        agree = float(((sign * (x - t) > 0) == act).mean())
        if agree > best[1]:
            best = (t, agree)
    return best


def trigger(c: Curve, p1m, p2m, weather: dict, lam: float = 0.5, step: int = 2, min_agree: float = 0.9) -> dict:
    """The decision plus the weather trigger. weather: {"rain_in": per-member values, "temp_f": ...}."""
    d = decide_members(c, p1m, p2m, lam, step)
    act_next = np.array([ev > 0 for _, _, ev in d.scenarios])
    out = {"act_now": d.act_now, "ev_now": d.ev_now, "ev_wait": d.ev_wait, "members_tipping": int(act_next.sum()),
           "members": len(act_next), "trigger": None}
    if d.act_now or act_next.all() or not act_next.any():
        return out
    cands = []
    for var, (pre, unit, sign) in VARS.items():
        if var in weather:
            t, agree = best_split(np.asarray(weather[var], float), act_next, sign)
            if np.isfinite(t):
                cands.append((agree, var, t, pre, unit))
    if cands:
        agree, var, t, pre, unit = max(cands)
        if agree >= min_agree:
            out["trigger"] = {"var": var, "threshold": round(float(t), 2 if var == "rain_in" else 0),
                              "agree": agree, "text": f"{pre} {t:.1f} {unit}" if var == "rain_in" else f"{pre} {t:.0f} {unit}"}
    return out
