import json

from flashwatch.agent.llm import Chat
from flashwatch.agent.validate import drop_sentences, unsupported


def test_validator_accepts_traced_numbers_and_flags_the_rest():
    tools = [{"risk_week2": 0.123, "normal": 0.031, "cells": 412, "note": "D1 since 2012-06-19"}]
    ok = "Week 2 risk is 12% (12.3% exactly), about 4 times the normal 3.1%, across 412 cells in 2012."
    assert unsupported(ok, tools) == []                    # "4 times" is a small count: exempt
    assert unsupported("Risk is 18% this week.", tools) == ["18%"]
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
