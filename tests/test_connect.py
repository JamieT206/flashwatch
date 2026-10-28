"""Equipment connectors, the command guard and the operator's agent loop (DESIGN.md O10)."""

import http.server
import json
import threading

import pytest

from flashwatch.agent import operator
from flashwatch.connect import make
from flashwatch.connect.base import Command, PivotStatus
from flashwatch.connect.guard import check
from flashwatch.connect.webhook import sign

FARM = {"name": "t", "wells": {"W": {"gpm": 1100, "pivots": ["A", "B"]}},
        "pivots": {"A": {"acres": 130, "gpm_per_acre": 4.5, "crop": "corn"},
                   "B": {"acres": 125, "gpm_per_acre": 5.0, "crop": "corn"},
                   "C": {"acres": 60, "gpm_per_acre": 4.0, "crop": "corn", "allocation_in_left": 0.5}}}
GO = {k: {"verdict": "start the pivot today"} for k in "ABC"}
OFF = {k: PivotStatus(k, False) for k in "ABC"}


def cmd(p, action="start", depth=0.75, date="2012-07-03"):
    return Command(p, action, depth, "", date)


def test_guard_well_limit_takes_commands_in_order():
    ok, bad = check([cmd("B"), cmd("A")], FARM, GO, OFF)
    assert [c.pivot for c in ok] == ["B"] and bad[0][0].pivot == "A" and "well W" in bad[0][1]


def test_guard_refuses_without_advice_out_of_season_and_bad_depth():
    advice = {**GO, "A": {"verdict": "not yet: wait for the next forecast"}}
    assert "not a start" in check([cmd("A")], FARM, advice, OFF)[1][0][1]
    assert "season" in check([cmd("B", date="2012-09-10")], FARM, GO, OFF)[1][0][1]
    assert "outside" in check([cmd("B", depth=3)], FARM, GO, OFF)[1][0][1]
    assert "allocation" in check([cmd("C", depth=0.75)], FARM, GO, OFF)[1][0][1]
    assert check([cmd("C", depth=0.5)], FARM, GO, OFF)[0]
    assert "already running" in check([cmd("B")], FARM, GO, {**OFF, "B": PivotStatus("B", True)})[1][0][1]
    assert "not running" in check([cmd("B", "stop")], FARM, GO, OFF)[1][0][1]
    assert "no pivot" in check([cmd("Z")], FARM, GO, OFF)[1][0][1]


def test_guard_counts_pivots_already_running_on_the_well():
    running = {**OFF, "A": PivotStatus("A", True)}
    assert "well" in check([cmd("B")], FARM, GO, running)[1][0][1]
    # stopping A first frees the well for B
    ok, bad = check([cmd("A", "stop"), cmd("B")], FARM, GO, running)
    assert len(ok) == 2 and not bad


def test_simulated_needs_approval_and_records_water():
    sim = make("simulated", pivots={"A": {"gpm_per_acre": 5.0}}, today="2012-07-03")
    c = cmd("A")
    with pytest.raises(PermissionError):
        sim.send(c)
    c.state, c.approved_by = "approved", "farmer"
    assert sim.send(c).state == "sent" and sim.status("A").running
    sim.advance(2)
    assert len(sim.applied("A", "2012-07-01")) == 2
    assert sim.status("A").applied_in_7d == pytest.approx(2 * 5 * 24 * 60 / 27154 * 0.85, abs=0.01)


def test_simulated_offline_pivot_is_refused_and_fails():
    sim = make("simulated", pivots={"A": {"gpm_per_acre": 5.0}, "B": {"gpm_per_acre": 5.0}}, offline=("A",))
    st = {k: sim.status(k) for k in ("A", "B")}
    assert "offline" in check([cmd("A")], FARM, GO, st)[1][0][1]


def test_manual_and_leaf():
    man = make("manual", pivots={"A": {"make": "lindsay"}})
    c = cmd("A")
    c.state, c.approved_by = "approved", "farmer"
    assert "FieldNET" in man.send(c).log[-1]
    rows = {"/irrigation-equipment": [{"id": "p1", "type": "PIVOT"}, {"id": "s1", "type": "SENSOR"}],
            "/irrigation/applied-irrigation": [{"startTime": "2026-07-01T00:00:00Z", "depth": {"value": 12.7, "unit": "mm"}}]}
    leaf = make("leaf", user_id="u", token="t", get=lambda path, params=None: rows[path])
    assert leaf.pivots() == ["p1"]
    assert leaf.applied("p1", "2026-06-01") == [{"pivot": "p1", "date": "2026-07-01", "depth_in": pytest.approx(0.5)}]
    with pytest.raises(PermissionError):
        leaf.send(c)


def test_webhook_signs_the_body():
    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.update(body=body, sig=self.headers["X-FlashWatch-Signature"])
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    hook = make("webhook", url=f"http://127.0.0.1:{srv.server_port}/", pivots=["A"], secret="s3cret")
    c = cmd("A")
    c.state, c.approved_by = "approved", "farmer"
    assert hook.send(c).state == "sent"
    assert got["sig"] == sign(got["body"], "s3cret") and json.loads(got["body"])["pivot"] == "A"
    srv.server_close()


class FakeChat:
    """Plays a model: reads the overview, proposes all three starts, then answers."""
    model = "fake"

    def __init__(self, text):
        self.text, self.usage = text, {}

    def run(self, msgs, tools):
        ov = tools["farm_overview"][1]()
        res = tools["propose_commands"][1]([{"pivot": k, "action": "start", "depth_in": 0.5} for k in ov["suggested_order"]])
        return self.text, msgs, [("farm_overview", {}, ov), ("propose_commands", {}, res)]


def morning():
    m = object.__new__(operator.Morning)
    m.farm, m.date, m.md = FARM, "2012-07-03", "07-03"
    m.conn = make("simulated", pivots={k: {"gpm_per_acre": p["gpm_per_acre"]} for k, p in FARM["pivots"].items()})
    m.ov = {"pivots": {k: {"advice": {"verdict": "start the pivot today",
                                      "value_of_starting_today_dollars_whole_pivot": v},
                           "running": False, "water_allocation_left_in": None}
                       for k, v in zip("ABC", (300, 500, 100))}}
    m.ov["suggested_order"] = operator.ranked(m.ov["pivots"])
    m.accepted, m.rejected = [], []
    return m


def test_operator_agent_goes_through_the_guard_and_needs_approval():
    m = morning()
    assert m.ov["suggested_order"] == ["B", "A", "C"]
    out = operator.plan_with_agent(m, FakeChat('{"sms": "Start B and C: worth 500 dollars", "note": "A waits for the well."}'))
    assert [c.pivot for c in m.accepted] == ["B", "C"] and m.rejected[0][0].pivot == "A"
    assert out["cut_untraced"] == [] and operator.approval_line(m) == " Reply YES to start B, C."
    assert all(not s.running for s in map(m.conn.status, "ABC"))     # nothing moves before YES
    sent = operator.approve(m, "farmer")
    assert [s["state"] for s in sent] == ["sent", "sent"] and m.conn.status("B").running


def test_operator_cuts_untraced_numbers():
    m = morning()
    out = operator.plan_with_agent(m, FakeChat('{"sms": "Start B. Saves 912 dollars.", "note": "ok"}'), max_redrafts=1)
    assert out["cut_untraced"] == ["912"] and "912" not in out["sms"]
