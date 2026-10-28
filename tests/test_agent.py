import json

from flashwatch.agent.llm import Chat
from flashwatch.agent.validate import drop_sentences, unsupported


def test_validator_accepts_traced_numbers_and_flags_the_rest():
    tools = [{"risk_week2": 0.123, "normal": 0.031, "cells": 412, "note": "D1 since 2012-06-19"}]
    ok = "Week 2 risk is 12% (12.3% exactly), about 4 times the normal 3.1%, across 412 cells in 2012."
    assert unsupported(ok, tools) == []                    # "4 times" is a small count: exempt
    assert unsupported("Risk is 18% this week.", tools) == ["18%"]
    assert unsupported("Rain 3.1 below normal over the next 28 days.", [{"z": -3.1}]) == []   # sign, span
    assert unsupported("Rain 3.2 below normal.", [{"z": -3.1}]) == ["3.2"]
    assert unsupported("It covers 500 cells.", tools) == ["500"]
    assert drop_sentences("Risk is 12%. It covers 500 cells. Watch it.", ["500"]) == "Risk is 12%. Watch it."


def test_tool_loop_runs_tools_then_returns_the_answer():
    replies = iter([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "risk", "arguments": json.dumps({"place": "Ames"})}}]},
        {"role": "assistant", "content": "Risk near Ames is 12%."},
    ])
    sent = []

    def post(path, body):
        sent.append(body)
        return {"choices": [{"message": next(replies)}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    tools = {"risk": ({"description": "risk at a place", "parameters": {"type": "object", "properties":
                       {"place": {"type": "string"}}}}, lambda place: {"place": place, "risk": 0.12})}
    text, messages, log = Chat(key="x", post=post).run([{"role": "user", "content": "Ames?"}], tools)
    assert text == "Risk near Ames is 12%."
    assert log == [("risk", {"place": "Ames"}, {"place": "Ames", "risk": 0.12})]
    assert sent[1]["messages"][-1]["role"] == "tool" and sent[0]["tools"][0]["function"]["name"] == "risk"
    assert unsupported(text, [r for _, _, r in log]) == []


def fake_pack(n=400, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    iy, ix = np.divmod(np.arange(n), 20)
    d = pd.DataFrame({"iy": iy, "ix": ix, "lat": 49 - iy * 0.25, "lon": -104 + ix * 0.25,
                      "rz_pct": rng.uniform(20, 90, n), "usdm_cat": rng.integers(-1, 3, n),
                      "in_drought": rng.random(n) < 0.2, "f_tp_w12_z": rng.normal(size=n)})
    for lead in ("week1", "week2", "week3", "week4", "any28"):
        d[f"normal_{lead}"] = 0.04
        d[f"p_{lead}"] = np.where(d.in_drought, np.nan, rng.uniform(0, 0.2, n))
    for r in (1, 2, 3):
        d[f"driver{r}"] = rng.choice(["rz_pct", "f_tp_w12_z", "hot_days14"], n)
        d[f"driver{r}_push"] = rng.uniform(0, 1, n) / r
        d[f"driver{r}_value"] = rng.normal(size=n)
    return d


def test_tools_return_traceable_numbers():
    import json
    from flashwatch.agent.tools import Tools
    t = Tools(fake_pack())
    a = t.area_summary(46.5, -101.5, 60)
    assert a["land_cells"] > 0 and 0 <= a["onset_risk"]["any28"]["chance_pct"] <= 20
    assert a["top_drivers"][0]["meaning"]
    hs = t.hotspots(3)
    assert len(hs) >= 1 and hs[0]["times_normal"] >= hs[-1]["times_normal"]
    assert t.area_summary(30.0, -120.0)["error"]
    json.dumps([a, hs, t.overview()])                  # all JSON-ready
    for name, (spec, fn) in t.specs().items():
        assert spec["description"] and callable(fn)


def test_briefing_redrafts_untraced_numbers(tmp_path, monkeypatch):
    from flashwatch.agent import brief, pack
    monkeypatch.setattr(pack, "OUT", tmp_path)
    fake_pack().to_parquet(pack.path("2012-05-03"))
    script = iter([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function":
            {"name": "overview", "arguments": "{}"}}]},
        {"role": "assistant", "content": "Risk this week: 37% chance of onset."},       # 37% is invented
        {"role": "assistant", "content": "Risk this week: elevated, about normal for May."},
    ])

    def post(path, body):
        return {"choices": [{"message": next(script)}]}

    out = brief.write("2012-05-03", "Test County", 46.5, -101.5, chat=Chat(key="x", post=post))
    assert out["drafts"][0]["untraced"] == ["37%"]
    assert out["briefing"].startswith("Risk this week: elevated") and out["cut_untraced"] == []
    assert out["tool_calls"][0]["tool"] == "overview"


def test_map_check_compares_what_the_model_sees_with_the_tools(tmp_path, monkeypatch):
    from flashwatch.agent import brief, pack
    monkeypatch.setattr(pack, "OUT", tmp_path)
    fake_pack().to_parquet(pack.path("2012-05-03"))
    sent = []

    def post(path, body):
        sent.append(body)
        return {"choices": [{"message": {"role": "assistant", "content": '{"areas": ["Bismarck", "Chicago"]}'}}]}

    out = brief.map_check("2012-05-03", chat=Chat(key="x", post=post))
    assert out["model_saw"] == ["Bismarck", "Chicago"] and "Bismarck" in out["tool_hotspots"]
    assert out["agree"] == ["Bismarck"]
    assert sent[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_glimmer_scores():
    from flashwatch.eval.glimmer_eval import direction, rouge_l, score, states_named
    assert states_named("Drought expanded in Iowa and western KS.") == {"IA", "KS"}
    assert direction("D1 expanded and conditions deteriorated") == 1 and direction("rains improved things") == -1
    assert rouge_l("a b c d", "a b c d") == 1.0 and rouge_l("a b", "c d") == 0.0
    inp = {"regions": {"Midwest": {"rain_percent_of_normal": 40, "D1_or_worse_change_pct": 3.0}}}
    s = score("Rain was 40% of normal; drought expanded in Iowa. 77 counties.", "Drought expanded across Iowa.", inp)
    assert s["untraced_figures"] == ["77"] and s["direction"] == 1.0 and s["states"] == 1.0


def test_ask_flashwatch_cuts_figures_not_in_tool_results():
    from flashwatch.agent.chat import ask

    class FakeFarm:
        year, md, lat, lon, crop = 2012, "07-03", 40.87, -97.99, "corn"

        def tools(self):
            return {}

    class FakeChat:
        def __init__(self):
            self.calls = 0

        def run(self, msgs, tools):
            self.calls += 1
            # The tool said 66% full; the reply invents "$45 an acre" both times.
            log = [("water_account", {}, {"root_zone_share_full_pct": 66})]
            return "Your root zone is 66% full. Starting now is worth $45 an acre.", msgs, log

    chat = FakeChat()
    r = ask(FakeFarm(), "How full is my soil?", chat)
    assert chat.calls == 2                                   # one redraft asked for
    assert "66%" in r["answer"] and "$45" not in r["answer"]
    assert r["cut_untraced"]
