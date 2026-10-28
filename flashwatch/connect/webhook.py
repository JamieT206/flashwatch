"""Send approved commands to a URL the farmer chooses, signed so the receiver can trust them.

The body is JSON: {"id", "pivot", "action", "depth_in", "date", "reason", "approved_by"}. The header
`X-FlashWatch-Signature` is the HMAC-SHA256 of the body with the farmer's shared secret (hex).
The receiver (the farmer's automation, a dealer's system, a relay controller) checks the signature
and does the rest. The secret comes from the environment (FW_WEBHOOK_SECRET), never from the page.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.request

from flashwatch.config import load_env
from flashwatch.connect.base import Command, Connector, PivotStatus


def sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class Webhook(Connector):
    name = "webhook"
    can_control = True

    def __init__(self, url: str, pivots: list[str], secret: str | None = None, timeout: int = 15):
        load_env()
        self.url, self._pivots, self.timeout = url, list(pivots), timeout
        self.secret = secret if secret is not None else os.getenv("FW_WEBHOOK_SECRET", "")
        if not self.secret:
            raise ValueError("webhook needs a shared secret (FW_WEBHOOK_SECRET)")

    def pivots(self):
        return self._pivots

    def status(self, pivot):
        return PivotStatus(pivot, running=False, reachable=True, note="webhook: status not reported")

    def send(self, cmd: Command) -> Command:
        self._require_approved(cmd)
        body = json.dumps({"id": cmd.id, "pivot": cmd.pivot, "action": cmd.action, "depth_in": cmd.depth_in,
                           "date": cmd.date, "reason": cmd.reason, "approved_by": cmd.approved_by}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "X-FlashWatch-Signature": sign(body, self.secret)})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                cmd.state = "sent" if 200 <= r.status < 300 else "failed"
                cmd.log.append(f"webhook HTTP {r.status}")
        except Exception as e:                            # the agent reports it; nothing retries silently
            cmd.state = "failed"
            cmd.log.append(f"webhook error: {type(e).__name__}")
        return cmd
