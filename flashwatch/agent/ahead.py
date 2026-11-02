"""A week ahead of the Drought Monitor (agents page): what FlashWatch said on an issue date, next to what the
Drought Monitor said about the same places one week later, and how often the Monitor's map then got worse on
the farms FlashWatch flagged.

    python -m flashwatch.agent.ahead --oof data/oof/<m2full>.parquet --dates 2012-06-19 2012-07-03 2012-07-17

FlashWatch's side is its regional outlook, written by Muse Spark from FlashWatch's own numbers on the issue date
(out-of-fold M2: each season predicted by models that never saw it), with every figure checked against them
(agent/validate.py). The Monitor's side is its weekly narrative for the map valid seven days after the issue date
(data/raw/usdm_text), the sentences naming the states FlashWatch named, and that map's change by state.
The scores use every held-out season in the OOF file (June 1 - Sep 15 issue dates): a farm is flagged when its
week-1 risk is in the riskiest tenth that day (as on the big-picture page), and the Monitor counts as worse when
its category at the farm is higher 1, 2 or 3 weekly maps after the one public on the issue date.
Writes data/agent/ahead.json; site/agents.js shows it (site.agents copies it into site/data/agents.json).
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from flashwatch.agent.llm import Chat
from flashwatch.agent.pack import OUT
from flashwatch.agent.validate import drop_sentences, unsupported
from flashwatch.config import RAW, YEARS
from flashwatch.obs import LAT, LON

STATES = {"ND": "North Dakota", "SD": "South Dakota", "NE": "Nebraska", "KS": "Kansas", "OK": "Oklahoma", "TX": "Texas",
          "MN": "Minnesota", "IA": "Iowa", "MO": "Missouri", "AR": "Arkansas", "WI": "Wisconsin", "IL": "Illinois",
          "IN": "Indiana", "MI": "Michigan", "OH": "Ohio", "KY": "Kentucky", "TN": "Tennessee", "MS": "Mississippi",
          "AL": "Alabama", "GA": "Georgia", "WV": "West Virginia", "PA": "Pennsylvania", "NY": "New York",
          "VA": "Virginia", "NC": "North Carolina", "CO": "Colorado", "WY": "Wyoming", "MT": "Montana", "NM": "New Mexico"}
SYSTEM = """You are FlashWatch, a flash-drought early-warning service. A flash drought is a rapid drop in root-zone
soil moisture (from the 40th to below the 20th percentile within 20 days). You are given FlashWatch's numbers for
this week, by state: the percentage of that state's farms at high risk (in the riskiest tenth of the region's farms
that day), the state's average chance of a
flash drought starting in the next 7 days, and the normal chance for the date. Write this week's regional outlook in
at most 90 words, as plain prose (no headings or lists): where flash drought is most likely to start in the next 7
days (name the states, and the part of the state when one is given), how that compares with normal, and where the
risk is low.
Rules: every number you write must appear in the data; round percentages to whole numbers; do not compute new
numbers; do not use outside knowledge of past events or years; this is a forecast, so say "likely", not "will"."""


def cell_states(cells: pd.DataFrame) -> pd.Series:
    """State of each (iy, ix): the nearest county centre within 0.35 degrees (site meta), else None."""
    from flashwatch.site.build import counties
    c = pd.DataFrame(counties(), columns=["fips", "name", "st", "lat", "lon"])
    la, lo = LAT[cells.iy.to_numpy()], LON[cells.ix.to_numpy()]
    d = np.hypot(la[:, None] - c.lat.to_numpy()[None], lo[:, None] - c.lon.to_numpy()[None])
    return pd.Series(np.where(d.min(1) < 0.35, c.st.to_numpy()[d.argmin(1)], None), index=cells.index)


def flag(oof: pd.DataFrame) -> pd.Series:
    return oof.M2_week1 >= oof.groupby("issue").M2_week1.transform(lambda s: s.quantile(0.9))


def scores(oof: pd.DataFrame, udates, ucat) -> dict:
    md = pd.Series(pd.DatetimeIndex(oof.issue).strftime("%m-%d"), index=oof.index)
    o = oof[md.between("06-01", "09-15")]
    f = flag(o).to_numpy()
    k0 = udates.searchsorted(pd.DatetimeIndex(o.issue) - pd.Timedelta(days=2), side="right") - 1
    iy, ix = o.iy.to_numpy(), o.ix.to_numpy()
    now = ucat[k0, iy, ix]
    out = {"years": sorted(int(y) for y in o.year.unique()), "farm_days": int(len(o)), "weeks": []}
    for w in (1, 2, 3):
        later = ucat[np.minimum(k0 + w, len(udates) - 1), iy, ix]
        worse = later > now
        out["weeks"].append({"weeks": w, "flagged": float(worse[f].mean()), "others": float(worse[~f].mean())})
    return out


def compass(dy: float, dx: float) -> str:
    ns = "northern" if dy > 0.4 else "southern" if dy < -0.4 else ""
    ew = "eastern" if dx > 0.6 else "western" if dx < -0.6 else ""
    return (ns[:-2] + ew if ns and ew else ns or ew) or ""


def summary(day: pd.DataFrame, st: pd.Series, now_cat: np.ndarray) -> dict:
    """FlashWatch's numbers by state for one issue date: the model's input to the outlook (and its check)."""
    d = day.assign(st=st.reindex(day.index).to_numpy(), flag=flag(day).to_numpy()).dropna(subset=["st"])
    rows = []
    for s, g in d.groupby("st"):
        if len(g) < 15 or s not in STATES:
            continue
        f = g[g.flag]
        part = compass(LAT[f.iy].mean() - LAT[g.iy].mean(), LON[f.ix].mean() - LON[g.ix].mean()) if len(f) >= 3 else ""
        rows.append({"state": STATES[s], "code": s, "pct_of_this_states_farms_at_high_risk": round(100 * g.flag.mean()),
                     "chance_next_7_days_pct": round(100 * g.M2_week1.mean(), 1),
                     "normal_pct": round(100 * g.B0_week1.mean(), 1),
                     "times_normal": round(g.M2_week1.mean() / g.B0_week1.mean(), 1),
                     "where_flagged": part or ("across the state" if g.flag.mean() > 0.3 else ""),
                     "already_in_drought_monitor_d1_pct": round(100 * float((now_cat[g.iy, g.ix] >= 1).mean()))})
    rows.sort(key=lambda r: -r["pct_of_this_states_farms_at_high_risk"])
    return {"states": rows}


def narrative(day: pd.Timestamp, codes: set[str], k: int = 4) -> list[str]:
    """Up to k sentences of the Monitor's narrative (for the map valid on `day`) that name the states in `codes`,
    in the narrative's order (its regional headings change over the years, so sentences, not sections)."""
    f = RAW / "usdm_text" / f"{day:%Y%m%d}.json"
    if not f.exists():
        return []
    names = [STATES[c] for c in codes]
    out = []
    for sec in json.loads(f.read_text(encoding="utf-8"))["sections"]:
        if sec["heading"].lower().startswith(("looking ahead", "hawaii")):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", sec["text"]):
            if any(n in sent for n in names) and sent not in out:
                out.append(sent.strip())
    return out[:k]


def example(oof, st, udates, ucat, issue: pd.Timestamp, chat: Chat) -> dict:
    day = oof[oof.issue == issue]
    k0 = udates.searchsorted(issue - pd.Timedelta(days=2), side="right") - 1
    data = summary(day, st, ucat[k0])
    ask = f"FlashWatch's numbers this week:\n{json.dumps(data, default=float)}\nWrite the outlook."
    text, messages, _ = chat.run([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}], {})
    for _ in range(2):
        bad = unsupported(text, [data])
        if not bad:
            break
        text, messages, _ = chat.run(messages + [{"role": "user", "content":
                                     f"These figures are not in the data: {', '.join(bad)}. Rewrite using only the data's numbers."}], {})
    bad = unsupported(text, [data])
    text = drop_sentences(text, bad) if bad else text
    named = {r["code"] for r in data["states"] if r["state"] in text}
    later = issue + pd.Timedelta(days=7)
    k1 = udates.searchsorted(later, side="right") - 1
    stc = st.dropna()
    worse = []
    for c in sorted(named):
        cells = stc[stc == c].index
        g = day.loc[day.index.intersection(cells)]
        if len(g):
            a, b = ucat[k0, g.iy, g.ix], ucat[k1, g.iy, g.ix]
            worse.append({"state": STATES[c], "worse_pct": round(100 * float((b > a).mean())),
                          "d1_before_pct": round(100 * float((a >= 1).mean())), "d1_after_pct": round(100 * float((b >= 1).mean()))})
    return {"issue": f"{issue:%Y-%m-%d}", "monitor_date": f"{udates[k1]:%Y-%m-%d}", "monitor_before": f"{udates[k0]:%Y-%m-%d}",
            "model": chat.model, "outlook": text, "cut_untraced": bad, "numbers": data,
            "monitor": narrative(udates[k1], named), "named_states_then": worse}


def main():
    from flashwatch.labels.crosscheck import usdm_cube
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--dates", nargs="+", required=True)
    a = ap.parse_args()
    oof = pd.read_parquet(a.oof, columns=["year", "issue", "iy", "ix", "M2_week1", "B0_week1"])
    oof["issue"] = pd.DatetimeIndex(oof.issue)
    udates, ucat = usdm_cube(YEARS)
    cells = oof[["iy", "ix"]].drop_duplicates()
    st = cell_states(cells)
    st.index = pd.MultiIndex.from_frame(cells)
    oof = oof.set_index(["iy", "ix"], drop=False)
    out = {"scores": scores(oof.reset_index(drop=True), udates, ucat), "examples": []}
    chat = Chat()
    for d in a.dates:
        ex = example(oof, st, udates, ucat, pd.Timestamp(d), chat)
        out["examples"].append(ex)
        print(f"{d}: {ex['outlook']}\n  then: {ex['named_states_then']}\n", flush=True)
    out["usage"] = chat.usage
    (OUT / "ahead.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out["scores"]))


if __name__ == "__main__":
    main()
