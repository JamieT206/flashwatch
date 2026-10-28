"""Read pivots and as-applied irrigation from Lindsay FieldNET and Valley through Leaf (read-only).

Leaf (withleaf.io) is a farm-data API that already connects to both pivot makers. Endpoints, from
Leaf's irrigation docs (Oct 2026):
    GET {base}/users/{leafUserId}/irrigation-equipment
    GET {base}/users/{leafUserId}/irrigation/applied-irrigation
with base https://api.withleaf.io/services/irrigation/api. Settings: LEAF_TOKEN and LEAF_USER_ID
in flashwatch.env. **Untested**: we have no Leaf account; field names below follow the docs and
may need adjusting. Leaf cannot start a pivot, so `send` refuses: approved starts go out through
`manual` (the maker's app) or `webhook`.

Why it matters even read-only: the water the pivot actually put down goes back into the field's
water balance, so the next morning's advice knows the pivot is already running.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from flashwatch.config import load_env
from flashwatch.connect.base import Command, Connector, PivotStatus

BASE = "https://api.withleaf.io/services/irrigation/api"
MM_PER_IN = 25.4


class LeafIrrigation(Connector):
    name = "leaf"
    can_control = False

    def __init__(self, user_id: str | None = None, token: str | None = None, base: str = BASE,
                 get=None, timeout: int = 30):
        load_env()
        self.user = user_id or os.getenv("LEAF_USER_ID", "")
        self.token = token or os.getenv("LEAF_TOKEN", "")
        self.base, self.timeout = base.rstrip("/"), timeout
        self.get = get or self._get                       # injectable for tests

    def _get(self, path: str, params: dict | None = None):
        url = f"{self.base}/users/{self.user}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def equipment(self) -> list[dict]:
        out = self.get("/irrigation-equipment")
        return out if isinstance(out, list) else out.get("content", [])

    def pivots(self):
        return [str(e.get("id")) for e in self.equipment() if str(e.get("type", "pivot")).lower().startswith("pivot")]

    def applied(self, pivot, since):
        out = self.get("/irrigation/applied-irrigation", {"startTime": since, "equipmentId": pivot})
        rows = out if isinstance(out, list) else out.get("content", [])
        recs = []
        for r in rows:
            depth = r.get("depth") or r.get("summary", {}).get("depth") or {}
            v, unit = (depth.get("value"), depth.get("unit", "in")) if isinstance(depth, dict) else (depth, "in")
            if v is None:
                continue
            inches = float(v) / MM_PER_IN if str(unit).lower().startswith("mm") else float(v)
            recs.append({"pivot": pivot, "date": str(r.get("startTime", r.get("date", "")))[:10], "depth_in": inches})
        return recs

    def status(self, pivot):
        import datetime as dt
        week_ago = (dt.date.today() - dt.timedelta(days=7)).isoformat()
        recs = self.applied(pivot, week_ago)
        last = max((r["date"] for r in recs), default="")
        running = last >= (dt.date.today() - dt.timedelta(days=1)).isoformat()
        return PivotStatus(pivot, running, True, round(sum(r["depth_in"] for r in recs), 2), "from as-applied records")

    def send(self, cmd: Command) -> Command:
        raise PermissionError("Leaf is read-only; send starts through the maker's app (manual) or a webhook")
