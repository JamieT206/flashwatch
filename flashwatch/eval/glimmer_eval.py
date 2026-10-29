"""Score narrative writers against the Drought Monitor's real text (DESIGN.md G2).

    python -m flashwatch.eval.glimmer_eval --n 6                         # Muse Spark, zero-shot baseline
    META_API_BASE=http://localhost:8000/v1 META_MODEL=glimmer-lora python -m flashwatch.eval.glimmer_eval --n 40
    META_API_BASE=http://127.0.0.1:8080/v1 META_MODEL=muse-glimmer-30B python -m flashwatch.eval.glimmer_eval --n 12 --shots 2

Validation weeks (2020-2021, data/glimmer/val.jsonl). For each week the writer gets the same input
JSON as in training and writes the High Plains / Midwest narrative. Scores, per week then averaged:
- untraced: share of the figures in the draft that no input number supports (agent/validate.py);
  lower is better, and the main safety measure;
- states: F1 of the states named (two-letter codes or names) against the real narrative;
- direction: does the draft say drought got worse / better / held, as the Monitor's D1+ change says?
- rouge_l: word-level longest-common-subsequence F1 with the real text (style and content overlap).
--shots k puts k real weeks from the training years (same month, latest years first) before the
week to write, as earlier turns of the chat: in-context examples of the style, no training.
Writes data/glimmer/eval_<model>[_<k>shot].json with every draft.
"""

from __future__ import annotations

import argparse
import json
import re

from flashwatch.agent.validate import numbers_in, unsupported
from flashwatch.config import DATA

STATES = {"ND": "north dakota", "SD": "south dakota", "NE": "nebraska", "KS": "kansas", "CO": "colorado",
          "WY": "wyoming", "MT": "montana", "MN": "minnesota", "IA": "iowa", "MO": "missouri", "WI": "wisconsin",
          "IL": "illinois", "IN": "indiana", "OH": "ohio", "MI": "michigan", "KY": "kentucky"}
WORSE = re.compile(r"\b(deteriorat\w*|worsen\w*|expan\w*|intensif\w*|degrad\w*|introduc\w*|added)\b", re.I)
BETTER = re.compile(r"\b(improv\w*|contract\w*|remov\w*|eas\w*|reduc\w*|shrank|recede\w*)\b", re.I)


def states_named(text: str) -> set:
    t = text.lower()
    out = {k for k, v in STATES.items() if v in t}
    out |= {k for k in STATES if re.search(rf"\b{k}\b", text)}
    # "South Dakota" also contains "dakota": keep both only when written out
    return out


def direction(text: str) -> int:
    w, b = len(WORSE.findall(text)), len(BETTER.findall(text))
    return 0 if w == b else (1 if w > b else -1)


def rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(x), lcs / len(y)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def f1(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    tp = len(pred & true)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(true))


def score(draft: str, real: str, inp: dict) -> dict:
    nums = numbers_in(draft)
    bad = unsupported(draft, [inp])
    d1 = inp["regions"]
    change = sum(r.get("D1_or_worse_change_pct", 0) for r in d1.values())
    true_dir = 0 if abs(change) < 1 else (1 if change > 0 else -1)
    return {"untraced": len(bad) / max(len(nums), 1), "untraced_figures": bad,
            "states": f1(states_named(draft), states_named(real)),
            "direction": float(direction(draft) == true_dir), "rouge_l": rouge_l(draft, real)}


def examples(train: list[dict], date: str, k: int) -> list[dict]:
    """k training weeks from the same month as `date`, latest first, as user/assistant turns."""
    same = sorted((r for r in train if r["date"][5:7] == date[5:7]), key=lambda r: r["date"], reverse=True)
    picked, years = [], set()
    for r in same:                                      # one week per year, so the examples differ
        if r["date"][:4] not in years:
            picked.append(r)
            years.add(r["date"][:4])
        if len(picked) == k:
            break
    return [m for r in reversed(picked) for m in r["messages"][1:]]


def main():
    from flashwatch.agent.llm import Chat
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="validation weeks to score (spread through the set)")
    ap.add_argument("--season-only", action="store_true", help="only weeks from April to September")
    ap.add_argument("--shots", type=int, default=0, help="real training-year weeks shown first as examples")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per week (a local model can be slow)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(DATA / "glimmer" / "val.jsonl", encoding="utf-8")]
    train = [json.loads(l) for l in open(DATA / "glimmer" / "train.jsonl", encoding="utf-8")] if args.shots else []
    if args.season_only:
        rows = [r for r in rows if "04" <= r["date"][5:7] <= "09"]
    rows = rows[:: max(1, len(rows) // args.n)][: args.n]
    chat = Chat(timeout=args.timeout)
    out = []
    for r in rows:
        sys_msg, user, real = r["messages"]
        try:
            draft = chat.complete([sys_msg, *examples(train, r["date"], args.shots), user]).get("content") or ""
        except Exception as e:                            # one slow week shouldn't lose the rest
            print(f"{r['date']}: skipped ({type(e).__name__})")
            continue
        s = score(draft, real["content"], json.loads(user["content"]))
        out.append({"date": r["date"], **s, "draft": draft, "real": real["content"]})
        print(f"{r['date']}: untraced {s['untraced']:.2f} states {s['states']:.2f} direction {s['direction']:.0f} "
              f"rouge-L {s['rouge_l']:.2f}")
    mean = {k: round(sum(o[k] for o in out) / len(out), 3) for k in ("untraced", "states", "direction", "rouge_l")}
    print("mean", mean, "tokens", chat.usage)
    f = DATA / "glimmer" / f"eval_{chat.model}{f'_{args.shots}shot' if args.shots else ''}.json"
    f.write_text(json.dumps({"model": chat.model, "shots": args.shots, "mean": mean, "weeks": out, "usage": chat.usage}, indent=1))


if __name__ == "__main__":
    main()
