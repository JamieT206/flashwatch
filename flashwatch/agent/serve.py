"""Local server for Ask FlashWatch on the farm page (the page shows the chat box only when this is running).

    python -m flashwatch.agent.serve            # http://127.0.0.1:8813, needs META_API_KEY

GET /health -> {"ok": true, "model": ...}
POST /ask {"lat", "lon", "year", "date", "crop", "question", "history": [...]} -> {"answer", "tools", "cut_untraced"}
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
        if self.path == "/health":
            self._send(200, {"ok": True, "model": Handler.chat.model})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, {"error": "not found"})
            return
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            farm = Farm(float(q["lat"]), float(q["lon"]), int(q["year"]), q["date"], q.get("crop", "corn"),
                        q.get("soil", "Silt loam"), float(q.get("gpm_per_acre", 5.0)), q.get("irrigation"))
            hist = [m for m in q.get("history", [])[-8:] if m.get("role") in ("user", "assistant")]
            self._send(200, ask(farm, str(q["question"])[:500], Handler.chat, hist))
        except Exception as e:                          # the page shows the message
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

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
