"""Connections to the farm's equipment and software (DESIGN.md O10).

FlashWatch decides *whether* to start a pivot; these connectors let the agent act on it, always
with the farmer's approval:

- `simulated.SimulatedPivot`: a pivot that runs the FAO-56 water balance. Used for the demo and
  tests; no account needed.
- `webhook.Webhook`: posts a signed JSON command to a URL the farmer chooses (their own automation,
  a dealer's system, a relay controller). Works with anything that accepts HTTP.
- `leaf.LeafIrrigation`: reads pivots and the water they actually applied from Lindsay FieldNET and
  Valley through Leaf's farm-data API (read-only; needs a Leaf account; untested).
- `manual.Manual`: no connection; the text message tells the farmer what to press in the pivot
  maker's own app (FieldNET, Valley 365). This is the honest default today: neither maker
  publishes a public API to start a pivot.

All connectors share `base.Connector`: `pivots()`, `status(pivot)`, `applied(pivot, since)` and
`send(command)`. Commands are created by `flashwatch.agent.operator`, checked by `guard.check`, and
sent only after the farmer approves.
"""

from __future__ import annotations

from flashwatch.connect.base import Command, Connector, PivotStatus


def make(kind: str, **cfg) -> Connector:
    """A connector by name: simulated, webhook, leaf or manual."""
    if kind == "simulated":
        from flashwatch.connect.simulated import SimulatedPivot
        return SimulatedPivot(**cfg)
    if kind == "webhook":
        from flashwatch.connect.webhook import Webhook
        return Webhook(**cfg)
    if kind == "leaf":
        from flashwatch.connect.leaf import LeafIrrigation
        return LeafIrrigation(**cfg)
    if kind == "manual":
        from flashwatch.connect.manual import Manual
        return Manual(**cfg)
    raise ValueError(f"unknown connector {kind!r}")


__all__ = ["Command", "Connector", "PivotStatus", "make"]
