"""Day-by-day irrigation backtest (DESIGN.md O9): what would a farmer following FlashWatch have
gained, net of false alarms, against the usual practice?

    python -m flashwatch.decide.backtest --oof data/oof/<file>.parquet --model M1

For each grid cell and development year, from June 1 to August 31, on every forecast date (every
2 days): if no early start is already running, the engine (decide/engine.py) chooses between starting
the pivot today and deciding again at the next date. A start is followed by an onset within 14 days
(credited with the avoided loss for that much notice, minus the extra pumping) or by none (charged
the false-alarm cost, and the pivot goes off after `review_days`). Onsets nobody anticipated are
handled the usual way, worth 0 by definition. Onset dates come from the 0-40 cm labels.

Two usual practices (James, Oct 28):
- careful: starts the pivot `reaction_days` (2) after stress begins (sensor or close watching);
- calendar: starts on a fixed date (July 1); before it, only once the crop shows stress (7 days in).
  Onsets after the calendar date find the pivot already on, so FlashWatch adds nothing then.

Assumption, stated in the report: an onset in ERA5-Land's (rain-fed) root-zone soil moisture marks a
dry spell that would stress an irrigated crop if its pivot were off. The "perfect" row acts with the
best notice before every onset and never falsely: the ceiling.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from flashwatch.config import TEST_YEARS
from flashwatch.decide import curve, load
from flashwatch.decide.engine import daily, two_way

SEASON = ("06-01", "08-31")
HORIZON = 14


def onset_days(depth="0-40") -> pd.DataFrame:
    from flashwatch.eval.lead import onsets_table
    return onsets_table(depth)


def ev_tables(c, p1, p2, scenarios, step):
    """Vectorised engine: expected value of acting now and of deciding again `step` days later."""
    net = c.net(np.arange(1, HORIZON + 1))
    p = np.stack([daily({"week1": a, "week2": b}, HORIZON) for a, b in zip(p1, p2)])
    now = p @ net - (1 - p.sum(1)) * c.cost_noevent
    rest, net_r = p[:, step:], net[:HORIZON - step]
    wait = np.zeros(len(p))
    for w, m in scenarios:
        q = np.clip(rest * m, 0, 1)
        wait += w * np.maximum(q @ net_r - (1 - q.sum(1)) * c.cost_noevent, 0)
    return now, wait


def run(oof: pd.DataFrame, ons: pd.DataFrame, model: str, action: dict, baseline: str = "careful",
        spread: float = 0.5, calendar: str = "07-01", late_reaction: int = 7, season=SEASON) -> pd.DataFrame:
    """One row per (cell, year): value ($/acre, net of pumping and false alarms), crop value saved
    (gross, $/acre), starts, hits, false alarms, onsets in the window."""
    a = {**action, "params": dict(action["params"])}
    cal = None
    if baseline == "calendar":
        a["params"]["reaction_days"] = late_reaction
    c = curve(a)
    review = int(a["params"].get("review_days", 7))
    once = bool(a.get("once"))          # a one-off decision (fungicide, side-dress): act at most once a season
    d = oof[pd.DatetimeIndex(oof.issue).strftime("%m-%d").to_series(index=oof.index).between(*season)]
    d = d[~d.year.isin(TEST_YEARS)].sort_values(["iy", "ix", "issue"])
    now, wait = ev_tables(c, d[f"{model}_week1"].to_numpy(), d[f"{model}_week2"].to_numpy(),
                          two_way(spread), 2)
    act = (now > 0) & (now >= wait)
    last = (pd.Timestamp(f"2001-{season[1]}") + pd.Timedelta(days=HORIZON)).strftime("%m-%d")
    on = ons[ons.date.dt.strftime("%m-%d").between(season[0], last)]
    on_by = {k: np.sort(g.date.to_numpy()) for k, g in on.groupby([on.iy, on.ix, on.date.dt.year])}
    out = []
    issue = pd.DatetimeIndex(d.issue).to_numpy()
    keys = d[["iy", "ix", "year"]].to_numpy()
    starts = np.r_[0, np.nonzero((np.diff(keys, axis=0) != 0).any(1))[0] + 1, len(d)]
    best = c.net(np.arange(1, HORIZON + 1)).max()
    for s, e in zip(starts[:-1], starts[1:]):
        iy, ix, yr = keys[s]
        onsets = on_by.get((iy, ix, yr), np.array([], "datetime64[ns]"))
        if baseline == "calendar":
            cal = np.datetime64(f"{yr}-{calendar}")
            onsets = onsets[onsets < cal]
        value, saved, n_act, hits, false = 0.0, 0.0, 0, 0, 0
        blocked = np.datetime64("1900-01-01")
        for i in range(s, e):
            t = issue[i]
            if t < blocked or not act[i] or (cal is not None and t >= cal):
                continue
            n_act += 1
            nxt = onsets[(onsets > t) & (onsets <= t + np.timedelta64(HORIZON, "D"))]
            if len(nxt):
                k = int((nxt[0] - t) / np.timedelta64(1, "D"))
                value += float(c.net(k))
                saved += float(np.interp(k, c.notice, c.avoided))
                hits += 1
                blocked = nxt[0] + np.timedelta64(30, "D") if not once else np.datetime64("2100-01-01")
            else:
                value -= c.cost_noevent
                false += 1
                blocked = t + np.timedelta64(review, "D") if not once else np.datetime64("2100-01-01")
        n_on = len(onsets[onsets >= np.datetime64(f"{yr}-{season[0]}")])
        out.append({"iy": iy, "ix": ix, "year": yr, "value": value, "saved": saved, "starts": n_act, "hits": hits,
                    "false_alarms": false, "onsets": n_on, "perfect": n_on * best})
    return pd.DataFrame(out)


def summary(bt: pd.DataFrame) -> dict:
    per_year = bt.groupby("year")[["value", "perfect"]].mean()
    return {"value_per_acre_year": round(bt.value.mean(), 2),
            "perfect_per_acre_year": round(bt.perfect.mean(), 2),
            "share_of_perfect": round(bt.value.sum() / bt.perfect.sum(), 3) if bt.perfect.sum() else np.nan,
            "starts_per_year": round(bt.starts.mean(), 2), "hit_rate": round(bt.hits.sum() / max(bt.starts.sum(), 1), 3),
            "onsets_caught": round(bt.hits.sum() / max(bt.onsets.sum(), 1), 3),
            "worst_year": round(per_year.value.min(), 2), "best_year": round(per_year.value.max(), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--model", default="M1")
    ap.add_argument("--crop", default="corn", choices=["corn", "soybeans"])
    ap.add_argument("--out", help="write per cell-year results (parquet)")
    args = ap.parse_args()
    oof = pd.read_parquet(args.oof)
    ons = onset_days()
    action = load()[f"irrigation_{args.crop}" if f"irrigation_{args.crop}" in load() else "irrigation"]
    rows = []
    for base in ("careful", "calendar"):
        bt = run(oof, ons, args.model, action, base)
        rows.append({"baseline": base, **summary(bt)})
        if args.out:
            bt.assign(baseline=base).to_parquet(args.out.replace(".parquet", f"_{base}.parquet"))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
