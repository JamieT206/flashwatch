"""Chat client for any OpenAI-compatible endpoint, with a tool-calling loop.

Meta Model API by default; point META_API_BASE at a local server (e.g. llama.cpp or vLLM running
Muse Glimmer) to use the open-weight model through the same code. Settings, from the environment or
flashwatch.env: META_API_KEY, META_API_BASE (default https://api.meta.ai/v1), META_MODEL, and META_TIMEOUT
(seconds per call, for a slow local model).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from flashwatch.config import load_env

DEFAULT_BASE = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.1"


class Chat:
    def __init__(self, model: str | None = None, base: str | None = None, key: str | None = None,
                 temperature: float = 0.2, timeout: int = 120, post: Callable | None = None):
        load_env()
        self.model = model or os.getenv("META_MODEL", DEFAULT_MODEL)
        self.base = (base or os.getenv("META_API_BASE", DEFAULT_BASE)).rstrip("/")
        self.key = key if key is not None else os.getenv("META_API_KEY", "")
        self.temperature, self.timeout = temperature, int(os.getenv("META_TIMEOUT", timeout))
        self.post = post or self._post                    # injectable for tests
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.model, "messages": list(messages), "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        out = self.post("/chat/completions", body)
        u = out.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens"):
            self.usage[k] += int(u.get(k, 0))
        self.usage["calls"] += 1
        return out["choices"][0]["message"]

    def run(self, messages: list[dict], tools: dict[str, tuple[dict, Callable]], max_turns: int = 12):
        """Let the model call tools until it answers. tools: name -> (JSON schema, function).
        Returns (final text, messages, log of (tool name, arguments, result))."""
        specs = [{"type": "function", "function": {"name": n, **spec}} for n, (spec, _) in tools.items()]
        messages, log = list(messages), []
        for _ in range(max_turns):
            msg = self.complete(messages, specs)
            messages.append({k: v for k, v in msg.items() if v is not None})
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or "", messages, log
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                    result = tools[name][1](**args) if name in tools else {"error": f"unknown tool {name}"}
                except Exception as e:                    # the model sees the error and can retry
                    args, result = {}, {"error": f"{type(e).__name__}: {e}"}
                log.append((name, args, result))
                messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                                 "content": json.dumps(result, default=float)})
        raise RuntimeError(f"no final answer after {max_turns} turns")
