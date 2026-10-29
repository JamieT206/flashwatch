"""WhatsApp for FlashWatch: the watcher's texts and the operator's YES, on the farmer's phone.

    python -m flashwatch.agent.whatsapp watch --to 15551234567 --lat 40.87 --lon -97.99 --year 2012
    python -m flashwatch.agent.whatsapp operator --to 15551234567 --year 2012 --date 07-03
    python -m flashwatch.agent.whatsapp inbox --from 15551234567 --text YES     # a reply, as the webhook gets it

Sends through Meta's WhatsApp Cloud API (POST graph.facebook.com/<version>/<phone number id>/messages).
Without WHATSAPP_TOKEN and WHATSAPP_PHONE_ID in flashwatch.env it runs dry: every message goes to
data/whatsapp/outbox.jsonl instead, so the whole loop can be tested with no account.

- Watcher texts (agent/watch.py) go out as they are made. WhatsApp lets a business open a
  conversation only with an approved template: set WHATSAPP_TEMPLATE to one with a single {{1}} body
  field and the text goes in it. Within 24 hours of the farmer's last message, plain text is fine.
- The operator's morning plan (agent/operator.py) goes out with YES and NO buttons and waits in
  data/whatsapp/pending.json. YES (the button, or typed) sends it through the farm's connector
  (connect/), exactly as `operator --approve` does; NO drops it. A plan only holds for its own day.
- Anything else the farmer writes is a question for Ask FlashWatch (agent/chat.py), answered from the
  farm's own numbers. A voice note is transcribed with Muse Voice Transcribe and saved as a field
  report that the operator reads next morning (agent/voice.py; needs ffmpeg to turn WhatsApp's
  audio into WAV).
- Webhook: agent/serve.py answers Meta's GET check (WHATSAPP_VERIFY_TOKEN) and takes POST /whatsapp,
  checking X-Hub-Signature-256 with WHATSAPP_APP_SECRET. Meta needs a public HTTPS address, so put a
  tunnel in front of the local server, e.g. `cloudflared tunnel --url http://127.0.0.1:8813`.
Phone numbers stay in data/whatsapp/ (not committed) and are masked in anything printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from flashwatch.config import DATA, load_env

DIR = DATA / "whatsapp"
GRAPH = "https://graph.facebook.com"
YES, NO = {"yes", "y", "ok", "go", "start"}, {"no", "n", "stop", "cancel"}


def mask(phone: str) -> str:
    return f"...{phone[-4:]}" if len(phone) > 4 else "..."


def _load(name: str) -> dict:
    f = DIR / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _save(name: str, obj: dict):
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


class WhatsApp:
    """Cloud API sender; dry (writes the outbox) when the token or phone number id is missing."""

    def __init__(self, token: str | None = None, phone_id: str | None = None, post=None, outbox: Path | None = None):
        load_env()
        self.token = token if token is not None else os.getenv("WHATSAPP_TOKEN", "")
        self.phone_id = phone_id if phone_id is not None else os.getenv("WHATSAPP_PHONE_ID", "")
        self.version = os.getenv("WHATSAPP_API_VERSION", "v23.0")
        self.template = os.getenv("WHATSAPP_TEMPLATE", "")
        self.dry = not (self.token and self.phone_id)
        self.post, self.outbox = post or self._post, outbox or DIR / "outbox.jsonl"

    def _post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)

    def _send(self, to: str, body: dict) -> dict:
        body = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to, **body}
        if self.dry:
            self.outbox.parent.mkdir(parents=True, exist_ok=True)
            with open(self.outbox, "a", encoding="utf-8") as f:
                f.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"), **body}) + "\n")
            return {"dry_run": True}
        return self.post(f"{GRAPH}/{self.version}/{self.phone_id}/messages", body)

    def text(self, to: str, text: str) -> dict:
        return self._send(to, {"type": "text", "text": {"body": text[:4096]}})

    def alert(self, to: str, text: str) -> dict:
        """A message FlashWatch starts: in the approved template when there is one."""
        if not self.template:
            return self.text(to, text)
        flat = " ".join(text.split())[:1024]               # template fields can't hold line breaks
        return self._send(to, {"type": "template", "template": {
            "name": self.template, "language": {"code": os.getenv("WHATSAPP_TEMPLATE_LANG", "en_US")},
            "components": [{"type": "body", "parameters": [{"type": "text", "text": flat}]}]}})

    def yes_no(self, to: str, text: str, plan: str) -> dict:
        return self._send(to, {"type": "interactive", "interactive": {
            "type": "button", "body": {"text": text[:1024]},
            "action": {"buttons": [{"type": "reply", "reply": {"id": f"yes:{plan}", "title": "YES"}},
                                   {"type": "reply", "reply": {"id": f"no:{plan}", "title": "NO"}}]}}})

    def media(self, media_id: str) -> bytes:
        """A voice note's audio: the Graph API gives a short-lived URL that needs the same token."""
        hdr = {"Authorization": f"Bearer {self.token}"}
        with urllib.request.urlopen(urllib.request.Request(f"{GRAPH}/{self.version}/{media_id}", headers=hdr), timeout=20) as r:
            url = json.load(r)["url"]
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
            return r.read()


# --- webhook -------------------------------------------------------------------------------------

def verify(params: dict, token: str | None = None) -> str | None:
    """Meta's subscription check: echo hub.challenge if the verify token matches."""
    token = token if token is not None else os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    ok = params.get("hub.mode") == "subscribe" and token and hmac.compare_digest(params.get("hub.verify_token", ""), token)
    return params.get("hub.challenge") if ok else None


def signed(body: bytes, header: str, secret: str | None = None) -> bool:
    secret = secret if secret is not None else os.getenv("WHATSAPP_APP_SECRET", "")
    if not secret or not header.startswith("sha256="):
        return False
    return hmac.compare_digest(header[7:], hmac.new(secret.encode(), body, hashlib.sha256).hexdigest())


def incoming(payload: dict) -> list[dict]:
    """The messages in one webhook call: {from, type, text, button, media}. Delivery receipts are skipped."""
    out = []
    for e in payload.get("entry", []):
        for ch in e.get("changes", []):
            for m in ch.get("value", {}).get("messages", []):
                msg, t = {"from": m.get("from", ""), "type": m.get("type")}, m.get("type")
                if t == "text":
                    msg["text"] = m["text"].get("body", "")
                elif t == "interactive":
                    br = m["interactive"].get("button_reply", {})
                    msg["button"], msg["text"] = br.get("id", ""), br.get("title", "")
                elif t == "button":                       # quick reply on a template
                    msg["text"] = m["button"].get("text", "")
                elif t == "audio":
                    msg["media"] = m["audio"].get("id", "")
                out.append(msg)
    return out


# --- what FlashWatch does with a message ------------------------------------------------------------

def register(phone: str, farm: dict):
    farms = _load("farms.json")
    farms[phone] = farm
    _save("farms.json", farms)


def send_plan(wa: WhatsApp, phone: str, m, sms: str, farm_path: str) -> dict:
    """The operator's morning plan, with YES/NO buttons; the commands wait in pending.json."""
    plan = m.accepted and hashlib.sha1("".join(c.id for c in m.accepted).encode()).hexdigest()[:8]
    if not plan:
        return wa.alert(phone, sms)
    pending = _load("pending.json")
    pending[phone] = {"plan": plan, "date": m.date, "farm": farm_path, "commands": [c.to_dict() for c in m.accepted]}
    _save("pending.json", pending)
    return wa.yes_no(phone, sms, plan)


def decide(phone: str, yes: bool, plan: str | None = None, today: str | None = None) -> str:
    """The farmer's YES or NO on the waiting plan. Returns the reply text."""
    from flashwatch.agent.operator import connector_for, load_farm
    from flashwatch.connect.base import Command
    pending = _load("pending.json")
    p = pending.get(phone)
    if not p or (plan and plan != p["plan"]):
        return "There's no plan waiting for an answer."
    if today and today != p["date"]:
        pending.pop(phone)
        _save("pending.json", pending)
        return f"That plan was for {p['date']} and has lapsed. Tomorrow's will come in the morning."
    pending.pop(phone)
    _save("pending.json", pending)
    if not yes:
        return "OK, nothing started. Reply with a question any time."
    farm = load_farm(p["farm"])
    conn = connector_for(farm, p["date"])
    done = []
    for c in p["commands"]:
        cmd = Command(c["pivot"], c["action"], c["depth_in"], c["reason"], c["date"], "approved", f"WhatsApp {mask(phone)}")
        cmd = conn.send(cmd)
        done.append(f"{cmd.pivot} {cmd.action} {cmd.depth_in:g} in: {cmd.state}" if cmd.action == "start"
                    else f"{cmd.pivot} {cmd.action}: {cmd.state}")
    log = _load("approvals.json")
    log.setdefault(p["date"], []).append({"phone": mask(phone), "plan": p["plan"], "result": done})
    _save("approvals.json", log)
    return "Done. " + "; ".join(done) + "."


def _wav(audio: bytes) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "note.ogg").write_bytes(audio)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(tmp / "note.ogg"), "-ac", "1", "-ar", "16000",
                    "-sample_fmt", "s16", str(tmp / "note.wav")], check=True)
    return tmp / "note.wav"


def handle(msg: dict, wa: WhatsApp, chat=None, today: str | None = None) -> str:
    """One incoming message -> the reply (also sent)."""
    phone, farm = msg["from"], _load("farms.json").get(msg["from"])
    words = (msg.get("text") or "").strip().lower().rstrip(".!")
    button = msg.get("button", "")
    if button.startswith(("yes:", "no:")):
        reply = decide(phone, button.startswith("yes:"), button.split(":", 1)[1], today)
    elif words in YES | NO:
        reply = decide(phone, words in YES, today=today)
    elif not farm:
        reply = "This number isn't set up with a farm yet. Sign up on the FlashWatch farm page."
    elif msg.get("type") == "audio":
        reply = voice_report(msg, wa, farm, chat)
    elif words:
        reply = question(msg["text"], farm, chat)
    else:
        reply = "FlashWatch reads text and voice notes."
    wa.text(phone, reply)
    return reply


def question(text: str, farm: dict, chat) -> str:
    from flashwatch.agent.chat import Farm, ask
    if chat is None:
        return "Questions need the FlashWatch agent running; it will answer when it's back."
    f = Farm(farm["lat"], farm["lon"], farm["year"], farm["date"], farm.get("crop", "corn"),
             farm.get("soil", "Silt loam"), float(farm.get("gpm", 5.0)))
    if farm.get("farm"):                                  # "N2" is a pivot here, not nitrogen
        from flashwatch.agent.operator import load_farm
        names = ", ".join(load_farm(farm["farm"])["pivots"])
        text = f"(My pivots are called {names}; the numbers you have are for my farm as a whole.) {text}"
    return ask(f, text[:600], chat)["answer"][:4000]


def voice_report(msg: dict, wa: WhatsApp, farm: dict, chat) -> str:
    from flashwatch.agent.operator import load_farm
    from flashwatch.agent.voice import save, structure, transcribe
    if chat is None or wa.dry:
        return "Voice notes need the FlashWatch agent running with WhatsApp connected."
    try:
        said = transcribe(_wav(wa.media(msg["media"])))
    except Exception as e:                                # tell the farmer, don't lose the turn
        return f"Couldn't read that voice note ({type(e).__name__}). A text works too."
    rep = structure(said, load_farm(farm["farm"]) if farm.get("farm") else {"pivots": {}}, chat)
    save(rep, farm["year"], farm["date"], said)
    return f"Heard: \"{said[:300]}\". Saved as a field report; tomorrow's plan will take it into account."


# --- command line --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("watch", help="send the watcher's texts for one farm and season")
    w.add_argument("--to", required=True)
    w.add_argument("--lat", type=float, required=True)
    w.add_argument("--lon", type=float, required=True)
    w.add_argument("--year", type=int, default=2012)
    w.add_argument("--crop", default="corn")
    w.add_argument("--watering", default="irrigated", choices=["irrigated", "rain-fed"])
    w.add_argument("--soil", default="Silt loam")
    w.add_argument("--gpm", type=float, default=5.0)
    o = sp.add_parser("operator", help="send one morning's pivot plan with YES/NO buttons")
    o.add_argument("--to", required=True)
    o.add_argument("--year", type=int, default=2012)
    o.add_argument("--date", required=True, help="MM-DD")
    o.add_argument("--farm", default="")
    o.add_argument("--llm", action="store_true", help="Muse Spark plans and writes (otherwise rules)")
    i = sp.add_parser("inbox", help="handle one incoming message, as the webhook would")
    i.add_argument("--from", dest="sender", required=True)
    i.add_argument("--text", default="")
    i.add_argument("--button", default="")
    i.add_argument("--llm", action="store_true", help="answer questions with Muse Spark")
    a = ap.parse_args()
    wa = WhatsApp()
    if a.cmd == "watch":
        from flashwatch.agent.watch import follow
        sub = {"name": "your farm", "lat": a.lat, "lon": a.lon, "crop": a.crop, "watering": a.watering,
               "soil": a.soil, "gpm": a.gpm if a.watering == "irrigated" else 0}
        alerts, n, _ = follow(sub, a.year)
        for x in alerts:
            wa.alert(a.to, f"{x['date']}: {x['text']}")
            print(f"{x['date']} -> {mask(a.to)}: {x['text']}")
        last = alerts[-1]["date"][5:] if alerts else "08-31"
        register(a.to, {"lat": a.lat, "lon": a.lon, "year": a.year, "date": last, "crop": a.crop, "soil": a.soil, "gpm": a.gpm})
        print(f"{len(alerts)} texts over {n} forecast dates", "(dry run: data/whatsapp/outbox.jsonl)" if wa.dry else "")
    elif a.cmd == "operator":
        from flashwatch.agent.farm import FarmData
        from flashwatch.agent.operator import FARM, Morning, approval_line, connector_for, load_farm, plan_with_agent, rule_note, rule_plan
        path = a.farm or str(FARM)
        farm = load_farm(path)
        m = Morning(FarmData(a.year), farm, a.date, connector_for(farm, f"{a.year}-{a.date}"))
        if a.llm:
            from flashwatch.agent.llm import Chat
            msg = plan_with_agent(m, Chat())
        else:
            m.propose([c.__dict__ for c in rule_plan(m.ov, farm, m.date)])
            msg = rule_note(m)
        sms = (msg.get("sms") or "").strip() + approval_line(m)
        send_plan(wa, a.to, m, sms, path)
        piv = next(iter(farm["pivots"].values()))
        register(a.to, {"lat": piv.get("lat", farm.get("lat")), "lon": piv.get("lon", farm.get("lon")), "year": a.year,
                        "date": a.date, "crop": piv.get("crop", "corn"), "soil": piv.get("soil", "Silt loam"),
                        "gpm": piv.get("gpm_per_acre", 5.0), "farm": path})
        print(f"-> {mask(a.to)}: {sms}", "(dry run)" if wa.dry else "")
    else:
        chat = None
        if a.llm:
            from flashwatch.agent.llm import Chat
            chat = Chat(timeout=300)
        reply = handle({"from": a.sender, "type": "interactive" if a.button else "text", "text": a.text, "button": a.button}, wa, chat)
        print(f"{mask(a.sender)} said {a.text or a.button!r} -> {reply}")


if __name__ == "__main__":
    main()
