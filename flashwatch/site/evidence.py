"""Evidence for the region page (site/region.html): skill by lead, the watch list, economic value by
cost/loss ratio and the lead over the Drought Monitor, from out-of-fold predictions only.

    python -m flashwatch.site.evidence --m1 <oof> --m2 <oof> [--perfect <oof>]

Writes site/data/evidence.json. M2 is scored on the ECMWF start dates downloaded so far (spring).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from flashwatch.eval.cv import report
from flashwatch.site.build import SITE

ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def value_curve(y, p, alphas=ALPHAS):
    s = y.mean()
    out = []
    for a in alphas:
        act = p >= a
        e_f = a * act.mean() + (y * ~act).mean()
        e_c = min(a, s)
        out.append(round(float((e_c - e_f) / (e_c - a * s)), 3))
    return out


def watch_list(d, m, lead, top=0.10):
    y = d[lead].to_numpy()
    r = d[f"{m}_{lead}"].groupby(d.issue).rank(ascending=False, pct=True).to_numpy()
    flag = r <= top
    return {"caught": round(float(y[flag].sum() / y.sum()), 3), "hit_rate": round(float(y[flag].mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True, help="full-season OOF with B0 and M1")
    ap.add_argument("--m2", required=True, help="OOF with B0, M1 and M2 on forecast dates")
    ap.add_argument("--perfect", help="perfect-weather OOF (diagnostic ceiling)")
    args = ap.parse_args()
    ev = {"leads": ["week1", "week2", "any28"], "skill": {}, "value": {}, "watch": {}, "alphas": ALPHAS}
    m2 = pd.read_parquet(args.m2)
    ev["m2_dates"] = int(pd.DatetimeIndex(m2.issue).strftime("%m-%d").nunique())
    for name, f, models in (("all dates", args.m1, ["M1"]), ("forecast dates", args.m2, ["M1", "M2"]),
                            ("perfect weather", args.perfect, ["M2"])):
        if not f:
            continue
        d = m2 if f == args.m2 else pd.read_parquet(f)
        r = report(d, models, n_boot=300, labels=ev["leads"])
        ev["skill"][name] = r[["lead", "model", "bss", "bss_lo", "bss_hi"]].round(4).to_dict("records")
        for m in models:
            ev["value"][f"{name}|{m}"] = {lead: value_curve(d[lead].to_numpy(float), d[f"{m}_{lead}"].to_numpy())
                                          for lead in ("week1", "week2")}
            ev["watch"][f"{name}|{m}"] = {lead: watch_list(d, m, lead) for lead in ("week1", "week2")}
    # Lead over the Drought Monitor (DESIGN.md O0), recomputed.
    from flashwatch.config import YEARS
    from flashwatch.eval.lead import PERSIST, onsets_table, summarise, usdm_response
    from flashwatch.features.asof import LATENCY_DAYS
    from flashwatch.labels.crosscheck import usdm_cube
    dates, cat = usdm_cube(YEARS)
    ons = onsets_table()
    ons = ons[ons.date.dt.year >= dates[0].year + 1]
    worse, d1, base = usdm_response(pd.DatetimeIndex(ons.date), ons.iy, ons.ix, dates, cat)
    lat = pd.Timedelta(days=LATENCY_DAYS["era5land"])
    seen = pd.DatetimeIndex(ons.date) + lat
    ev["monitor"] = {"crossing_vs_worsening": summarise(seen, worse),
                     "crossing_vs_d1": summarise(seen, d1, base < 1),
                     "confirmed_vs_d1": summarise(seen + pd.Timedelta(days=PERSIST), d1, base < 1),
                     "onsets": int(len(ons))}
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / "evidence.json").write_text(json.dumps(ev, default=float))
    print(json.dumps({k: ev[k] for k in ("skill", "watch")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
