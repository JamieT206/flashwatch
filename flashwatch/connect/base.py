"""Shared types for equipment connectors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class PivotStatus:
    pivot: str
    running: bool
    reachable: bool = True
    applied_in_7d: float = 0.0          # inches put down in the last 7 days
    note: str = ""


@dataclass
class Command:
    """One instruction to one pivot. Lifecycle: proposed -> approved -> sent (or rejected/failed)."""
    pivot: str
    action: str                          # "start" or "stop"
    depth_in: float = 0.0                # inches per pass (start only)
    reason: str = ""
    date: str = ""                       # YYYY-MM-DD the command is for
    state: str = "proposed"
    approved_by: str = ""
    log: list = field(default_factory=list)

    @property
    def id(self) -> str:
        key = json.dumps([self.pivot, self.action, self.depth_in, self.date])
        return hashlib.sha1(key.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {"id": self.id, **asdict(self)}


class Connector:
    """What every connector offers. `send` must refuse a command that is not approved."""
    name = "base"
    can_control = False                  # True if send() really moves the pivot

    def pivots(self) -> list[str]:
        raise NotImplementedError

    def status(self, pivot: str) -> PivotStatus:
        raise NotImplementedError

    def applied(self, pivot: str, since: str) -> list[dict]:
        """[{date, depth_in}] water put down since a date (as-applied records)."""
        return []

    def send(self, cmd: Command) -> Command:
        raise NotImplementedError

    @staticmethod
    def _require_approved(cmd: Command):
        if cmd.state != "approved" or not cmd.approved_by:
            raise PermissionError(f"command {cmd.id} for {cmd.pivot} is {cmd.state}; it needs the farmer's approval")
