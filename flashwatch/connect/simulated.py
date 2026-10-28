"""A pivot that exists only in software: for the demo, tests and replays."""

from __future__ import annotations

import datetime as dt

from flashwatch.connect.base import Command, Connector, PivotStatus
from flashwatch.decide.irrigation import GPM_PER_ACRE_TO_IN_PER_DAY as GPM_TO_IN_DAY


class SimulatedPivot(Connector):
    """Pivots that start and stop on command and record the water they put down each day.

    pivots: {id: {"gpm_per_acre": .., "efficiency": .., "running": bool}}. `advance(days)` moves the
    clock and writes as-applied records for running pivots.
    """
    name = "simulated"
    can_control = True

    def __init__(self, pivots: dict, today: str = "2012-07-01", offline: tuple = ()):
        self.p = {k: {"running": False, **v} for k, v in pivots.items()}
        self.today = dt.date.fromisoformat(today)
        self.offline = set(offline)
        self.records: list[dict] = []

    def pivots(self):
        return list(self.p)

    def status(self, pivot):
        week_ago = (self.today - dt.timedelta(days=7)).isoformat()
        put = sum(r["depth_in"] for r in self.applied(pivot, week_ago))
        off = pivot in self.offline
        return PivotStatus(pivot, self.p[pivot]["running"], not off, round(put, 2), "offline" if off else "")

    def applied(self, pivot, since):
        return [r for r in self.records if r["pivot"] == pivot and r["date"] >= since]

    def send(self, cmd: Command) -> Command:
        self._require_approved(cmd)
        if cmd.pivot in self.offline:
            cmd.state = "failed"
            cmd.log.append("pivot not reachable")
            return cmd
        self.p[cmd.pivot]["running"] = cmd.action == "start"
        cmd.state = "sent"
        cmd.log.append(f"simulated pivot {cmd.pivot}: {cmd.action}")
        return cmd

    def advance(self, days: int = 1):
        for _ in range(days):
            for k, v in self.p.items():
                if v["running"]:
                    depth = v.get("gpm_per_acre", 5.0) * GPM_TO_IN_DAY * v.get("efficiency", 0.85)
                    self.records.append({"pivot": k, "date": self.today.isoformat(), "depth_in": round(depth, 3)})
            self.today += dt.timedelta(days=1)
