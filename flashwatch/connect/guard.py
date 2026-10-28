"""Rules every pivot command must pass before the farmer is even asked (DESIGN.md O10).

The agent proposes; this code decides what is allowed. A command is refused, with a reason the
agent sees and must work around, if:
- the pivot is unknown, offline, or already in the state asked for;
- a start isn't backed by the forecast (today's advice must be "start the pivot today", or the soil
  is already in drought) or by the farmer's own field report of crop stress (agent/voice.py);
- it's outside the irrigation season, or the depth per pass is outside the pivot's range;
- the pivot has less than one pass of its water allocation left (e.g. a Nebraska NRD allocation);
- the well would be over-drawn: pivots sharing a well can't together pump more than it gives.
  Commands are taken in the order given, so the agent puts the most valuable pivot first.
Passing the guard only makes a command "proposed"; nothing is sent without the farmer's approval.
"""

from __future__ import annotations

from flashwatch.connect.base import Command, PivotStatus

SEASON = ("06-01", "08-31")
DEPTH_RANGE = (0.3, 1.5)               # inches per pass
STARTABLE = ("start the pivot today", "irrigate as usual")


def flow_gpm(piv: dict) -> float:
    return piv["gpm_per_acre"] * piv["acres"]


def check(cmds: list[Command], farm: dict, advice: dict, status: dict[str, PivotStatus]):
    """-> (accepted, rejected); rejected is [(command, reason)]."""
    pivots, wells = farm["pivots"], farm.get("wells", {})
    running = {k for k, s in status.items() if s.running}
    accepted, rejected = [], []
    for c in cmds:
        why = _why_not(c, pivots, wells, advice, status, running)
        if why:
            c.state = "rejected"
            rejected.append((c, why))
        else:
            accepted.append(c)
            running = running | {c.pivot} if c.action == "start" else running - {c.pivot}
    return accepted, rejected


def _why_not(c, pivots, wells, advice, status, running):
    if c.pivot not in pivots:
        return f"no pivot called {c.pivot}"
    if c.action not in ("start", "stop"):
        return f"unknown action {c.action}"
    st = status.get(c.pivot)
    if st is not None and not st.reachable and st.note.startswith("offline"):
        return f"pivot {c.pivot} is offline"
    if c.action == "stop":
        return None if c.pivot in running else f"pivot {c.pivot} is not running"
    if c.pivot in running:
        return f"pivot {c.pivot} is already running"
    if not (SEASON[0] <= c.date[5:] <= SEASON[1]):
        return "outside the irrigation season (June to August)"
    verdict = advice.get(c.pivot, {}).get("verdict", "")
    if not verdict.startswith(STARTABLE) and not advice.get(c.pivot, {}).get("farmer_reports_stress"):
        return f"today's advice for {c.pivot} is '{verdict}', not a start"
    lo, hi = pivots[c.pivot].get("depth_range_in", DEPTH_RANGE)
    if not lo <= c.depth_in <= hi:
        return f"depth {c.depth_in} in per pass is outside {lo}-{hi} in"
    left = pivots[c.pivot].get("allocation_in_left")
    if left is not None and left < c.depth_in:
        return f"pivot {c.pivot} has only {left} in of its water allocation left"
    for w, spec in wells.items():
        if c.pivot in spec["pivots"]:
            need = sum(flow_gpm(pivots[p]) for p in spec["pivots"] if p in running or p == c.pivot)
            if need > spec["gpm"]:
                return (f"well {w} gives {spec['gpm']} gpm; running {c.pivot} with the pivots already on it "
                        f"needs {need:.0f} gpm")
    return None
