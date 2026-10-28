"""No connection: the farmer starts the pivot in the maker's own app, and the agent says how."""

from __future__ import annotations

from flashwatch.connect.base import Command, Connector, PivotStatus

APPS = {"lindsay": "FieldNET", "valley": "Valley 365", "reinke": "ReinCloud", "tl": "the T-L panel", "": "your pivot app"}


class Manual(Connector):
    name = "manual"
    can_control = False

    def __init__(self, pivots: dict):
        """pivots: {id: {"make": "lindsay" | "valley" | ...}}"""
        self.p = pivots

    def pivots(self):
        return list(self.p)

    def status(self, pivot):
        return PivotStatus(pivot, running=False, reachable=False, note="no connection: running state unknown")

    def send(self, cmd: Command) -> Command:
        self._require_approved(cmd)
        app = APPS.get(self.p.get(cmd.pivot, {}).get("make", ""), "your pivot app")
        verb = "Start" if cmd.action == "start" else "Stop"
        cmd.state = "sent"
        cmd.log.append(f"to do by hand: {verb} pivot {cmd.pivot} in {app}"
                       + (f" at {cmd.depth_in:.2f} in per pass" if cmd.action == "start" else ""))
        return cmd
