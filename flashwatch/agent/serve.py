"""Local server for Ask FlashWatch on the farm page (the page shows the chat box only when this is running).

    python -m flashwatch.agent.serve            # http://127.0.0.1:8813, needs META_API_KEY

GET /health -> {"ok": true, "model": ...}
POST /ask {"lat", "lon", "year", "date", "crop", "question", "history": [...]} -> {"answer", "tools", "cut_untraced"}
POST /pivots {"lat", "lon"} -> the pivots SAM 3.1 finds around the farm (agent/pivots.py)
GET /whatsapp (Meta's subscription check) and POST /whatsapp (incoming messages, signature checked):
the WhatsApp loop in agent/whatsapp.py. Meta reaches it through a tunnel (see there).
Binds to 127.0.0.1 only: the API key stays on this machine, and the static site never sees it.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flashwatch.agent.chat import Farm, ask
from flashwatch.agent.llm import Chat
from flashwatch.config import load_env

PORT = 8813


class Handler(BaseHTTPRequestHandler):
    chat: Chat | None = None

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.startswith("/whatsapp"):
            from urllib.parse import parse_qsl, urlsplit
            from flashwatch.agent.whatsapp import verify
            challenge = verify(dict(parse_qsl(urlsplit(self.path).query)))
            data = (challenge or "forbidden").encode()
            self.send_response(200 if challenge else 403)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/health":
            self._send(200, {"ok": True, "model": Handler.chat.model})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/whatsapp":
            self._whatsapp()
            return
        if self.path not in ("/ask", "/pivots"):
            self._send(404, {"error": "not found"})
            return
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/pivots":
                from flashwatch.agent.pivots import find
                r = find(round(float(q["lat"]), 3), round(float(q["lon"]), 3))
                self._send(200, {**r, "image": f"data/pivots/{float(q['lat']):.3f}_{float(q['lon']):.3f}.png"})
                return
            farm = Farm(float(q["lat"]), float(q["lon"]), int(q["year"]), q["date"], q.get("crop", "corn"),
                        q.get("soil", "Silt loam"), float(q.get("gpm_per_acre", 5.0)), q.get("irrigation"))
            hist = [m for m in q.get("history", [])[-8:] if m.get("role") in ("user", "assistant")]
            self._send(200, ask(farm, str(q["question"])[:500], Handler.chat, hist))
        except Exception as e:                          # the page shows the message
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _whatsapp(self):
        from flashwatch.agent.whatsapp import WhatsApp, handle, incoming, signed
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not signed(raw, self.headers.get("X-Hub-Signature-256", "")):
            print("whatsapp: bad signature, refused", flush=True)
            self._send(403, {"error": "bad signature"})
            return
        self._send(200, {"ok": True})                    # answer Meta at once; replies go out on their own
        wa, msgs = WhatsApp(), incoming(json.loads(raw or b"{}"))
        print(f"whatsapp: {len(msgs)} message(s)", *(m["type"] for m in msgs), flush=True)
        for msg in msgs:
            try:
                handle(msg, wa, Handler.chat)
            except Exception as e:                      # one bad message shouldn't stop the server
                print("whatsapp:", type(e).__name__, e, flush=True)

    def log_message(self, fmt, *args):
        pass


def main():
    load_env()
    Handler.chat = Chat(timeout=300)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Ask FlashWatch on http://127.0.0.1:{PORT} ({Handler.chat.model})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
