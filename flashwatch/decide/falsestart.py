"""What a false start really costs (Jamie, Oct 31): tie it to the field's water, net of real losses.

    python -m flashwatch.decide.falsestart --oof data/oof/<file>.parquet [--soil "Sandy loam" --gpm 4.5]

A week of pumping ahead of a dry spell that never comes is not all wasted. Water that fits in the root
zone is stored, and the crop uses it later in place of water the farmer would have pumped anyway. It is
lost only to:
- spray evaporation and drift on the way down: already in `efficiency` (0.85; measured 2-9% for spray
  nozzles under a full canopy, more in wind);
- extra evaporation from a wetted canopy and soil: EVAP_IN a day the pivot runs (0.6 mm, Bushland
  lysimeters under full corn canopy; more before the canopy closes);
- drainage below the roots when rain lands on a profile the early water had already filled;
- water still in the soil when the season ends (no credit: the crop is done with it).
The last two depend on the weather that follows, so they are measured, not assumed: a pivot-irrigated
field managed the usual way ("careful": pivot on `reaction_days` after stress begins, off once rain or
the pivot brings the deficit back to REFILL_IN, leaving room for rain) runs through each development
year's real daily rain and reference ET (ERA5-Land, every forecast cell), with and without a week's early
start on each forecast date. The cost of the early start is the extra water pumped over the rest of the
season, at the pumping cost (any crop stress it happens to spare is not credited).

The engine needs the cost before the weather is known: `table` averages the extra pumping over the dates
no onset followed, by month and by the field's deficit that morning (config/falsestart.json, all
development years; the backtest refits it leaving out the year it scores). The extra pumping comes out
nearly the same on a silt loam at 5 gpm/acre and a sandy loam at 4.5: what matters is the room in the
soil, in inches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flashwatch.config import CV_YEARS, DATA, TEST_YEARS
from flashwatch.decide import irrigation, load
from flashwatch.decide.backtest import HORIZON, SEASON
from flashwatch.decide.waterbook import kc

EVAP_IN = 0.6 / 25.4      # extra evaporation from wetted canopy and soil, inches per pivot day
REFILL_IN = 1.0           # the usual pivot goes off once the deficit is back to this: room for one rain
SEASON_END = "09-30"      # pumping and storage counted to here; water left in the soil after it is lost
CONFIG = Path(__file__).resolve().parents[1] / "config" / "falsestart.json"
S2S_DATES = Path(__file__).resolve().parents[1] / "config" / "s2s_dates.json"
MONTHS = [6, 7, 8, 9]
DR_EDGES = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]       # root-zone deficit bins, inches
DR_KNOTS = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.75, 2.25, 2.75, 3.5]
SOILS = {"Sand": 1.0, "Sandy loam": 1.4, "Loam": 1.8, "Silt loam": 2.0, "Clay loam": 1.9}   # engine.js


def season_runs(rain, eto, days, f: irrigation.Field, starts: np.ndarray, review: int, reaction: int,
                evap: float = EVAP_IN) -> dict:
    """One year at a set of cells. rain, eto: (day, cell) inches; days: month-days; starts: day indexes of
    the forecast dates. Runs the usual practice once (no early start) and once per start date with the
    pivot forced on for `review` days from it. Returns the deficit on each start date (D, cell) and, over
    the rest of the season, the extra water pumped (in), the crop shortfall spared (in) and the water the
    early week put in the root zone (in), each (D, cell)."""
    taw, raw, cap, eff = f.taw, f.p * f.taw, f.capacity, f.efficiency
    md = np.array(days)
    k = np.array([kc("corn", d) for d in md])
    allowed = (md >= SEASON[0]) & (md <= SEASON[1])
    end = int(np.searchsorted(md, SEASON_END, side="right"))
    n_c, D = rain.shape[1], len(starts)
    # Row 0: the usual practice; rows 1..D: an early start on each date (each joins the run on its date).
    dr = np.zeros((D + 1, n_c))
    on = np.zeros((D + 1, n_c), bool)
    run = np.zeros((D + 1, n_c), int)
    pumped = np.zeros((D + 1, n_c))
    short = np.zeros((D + 1, n_c))
    stored = np.zeros((D, n_c))
    dr_at = np.zeros((D, n_c))
    st = np.r_[-1, starts]
    for t in range(end):
        joining = np.nonzero(st == t)[0]
        if len(joining):                 # each branch starting today takes the usual run's state
            for arr in (dr, on, run, pumped, short):
                arr[joining] = arr[0]
            dr_at[joining - 1] = dr[0]
        forced = np.zeros(D + 1, bool)
        forced[1:] = (t >= starts) & (t < starts + review)
        go = (on & allowed[t]) | forced[:, None]
        add = np.where(go, np.minimum(cap, dr), 0.0)
        early = forced[:, None] & ~(on & allowed[t])
        stored += np.where(early, add, 0.0)[1:]
        dr = dr - add + np.where(go, evap, 0.0)
        pumped += add / eff
        etc = k[t] * eto[t]
        ks = np.where(dr <= raw, 1.0, np.clip((taw - dr) / (taw - raw), 0, None))
        short += etc * (1 - ks)
        dr = np.clip(dr + ks * etc - rain[t], 0, taw)
        run = np.where(dr > raw, run + 1, 0)
        on = (on | (run >= reaction)) & (dr > REFILL_IN)  # the usual rule, early week or not
    return {"dr": dr_at, "extra_pumped": pumped[1:] - pumped[0], "spared": short[0] - short[1:], "stored": stored}


def measure(cells: np.ndarray, years, f: irrigation.Field, review: int = 7, reaction: int = 2) -> pd.DataFrame:
    """The early week's extra pumping on every forecast date in the season, at every cell and year, with
    the deficit that morning and whether an onset followed within HORIZON days."""
    from flashwatch.decide.backtest import onset_days
    from flashwatch.site.build import WATER_DAYS, daily_water
    md = [d[5:] for d in json.loads(S2S_DATES.read_text())["model_version_dates"]]
    md = [d for d in md if SEASON[0] <= d <= SEASON[1]]
    out = []
    for y in years:
        rain, eto = (a.astype(float) / 100 for a in daily_water([y])[y])
        days = pd.date_range(f"{y}-{WATER_DAYS[0]}", f"{y}-{WATER_DAYS[1]}").strftime("%m-%d").tolist()
        starts = np.array([days.index(d) for d in md])
        r = season_runs(rain[:, cells[:, 0], cells[:, 1]], eto[:, cells[:, 0], cells[:, 1]], days, f, starts,
                        review, reaction)
        di, ci = np.meshgrid(np.arange(len(md)), np.arange(len(cells)), indexing="ij")
        out.append(pd.DataFrame({"year": y, "issue": pd.to_datetime([f"{y}-{md[i]}" for i in di.ravel()]),
                                 "iy": cells[ci.ravel(), 0], "ix": cells[ci.ravel(), 1],
                                 **{k: v.ravel() for k, v in r.items()}}))
    m = pd.concat(out, ignore_index=True)
    ons = onset_days()
    hit = m[["year", "issue", "iy", "ix"]].merge(ons, on=["iy", "ix"])
    hit = hit[(hit.date > hit.issue) & (hit.date <= hit.issue + pd.Timedelta(days=HORIZON))]
    key = ["year", "issue", "iy", "ix"]
    m["onset"] = m.set_index(key).index.isin(hit.set_index(key).index)
    return m


def table(m: pd.DataFrame) -> dict:
    """Mean extra pumping (inches) after a false start, by month and by the deficit that morning, from the
    dates no onset followed. Month-and-deficit cells with few dates fall back on the month."""
    q = m[~m.onset]
    month = q.issue.dt.month
    b = np.digitize(q.dr, DR_EDGES[1:-1])
    by_month = q.groupby(month).extra_pumped.mean().reindex(MONTHS).fillna(q.extra_pumped.mean())
    grid = q.groupby([month, b]).extra_pumped.agg(["mean", "size"])
    rows = [[round(float(grid.loc[(mo, j), "mean"]) if (mo, j) in grid.index and grid.loc[(mo, j), "size"] >= 200
                   else float(by_month[mo]), 3) for j in range(len(DR_EDGES) - 1)] for mo in MONTHS]
    return {"months": MONTHS, "dr_in": DR_KNOTS, "extra_in": rows,
            "month_in": {str(mo): round(float(by_month[mo]), 3) for mo in MONTHS},
            "all_in": round(float(q.extra_pumped.mean()), 3)}


def cost(pump_cost_per_in: float, month: int | None = None, dr: float | None = None, tab: dict | None = None) -> float:
    """Expected cost of a false start, $/acre: the table's extra pumping at the field's pumping cost. With
    the month (1-12) and, when the water account knows it, the root-zone deficit (in). Mirrored in
    site/assets/engine.js (FW.falseStart)."""
    tab = tab or TABLE()
    if month is None or str(month) not in tab["month_in"]:
        return tab["all_in"] * pump_cost_per_in
    if dr is None:
        return tab["month_in"][str(month)] * pump_cost_per_in
    row = tab["extra_in"][tab["months"].index(month)]
    return float(np.interp(dr, tab["dr_in"], row)) * pump_cost_per_in


def backtest_costs(pump_cost_per_in: float, water: bool = True, rows=None) -> pd.DataFrame | None:
    """Per forecast row: the false-start cost expected that morning, from the table refitted without the
    row's year (by month, and by the deficit when `water`), and the cost on the real weather that
    followed. None until falsestart has been run for the default field."""
    path = DATA / "decide" / "falsestart_silt_loam_5.parquet"
    if rows is None and not path.exists():
        return None
    m = pd.read_parquet(path) if rows is None else rows
    month = m.issue.dt.month.to_numpy()
    expected = np.zeros(len(m))
    for y in m.year.unique():
        tab, at = table(m[m.year != y]), (m.year == y).to_numpy()
        for mo in MONTHS:
            k = at & (month == mo)
            expected[k] = (np.interp(m.dr.to_numpy()[k], tab["dr_in"], tab["extra_in"][MONTHS.index(mo)])
                           if water else tab["month_in"][str(mo)])
    return pd.DataFrame({"year": m.year, "issue": m.issue, "iy": m.iy, "ix": m.ix,
                         "expected": expected * pump_cost_per_in, "actual": m.extra_pumped * pump_cost_per_in})


def TABLE() -> dict:  # noqa: N802
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="the cells to run (those the backtest scores)")
    ap.add_argument("--soil", default="Silt loam", choices=list(SOILS))
    ap.add_argument("--gpm", type=float, default=5.0)
    ap.add_argument("--write", action="store_true", help="write config/falsestart.json from this run")
    args = ap.parse_args()
    p = {**load()["irrigation"]["params"], "awc_in_per_ft": SOILS[args.soil], "gpm_per_acre": args.gpm}
    f = irrigation.field(**p)
    cells = pd.read_parquet(args.oof, columns=["iy", "ix"]).drop_duplicates().to_numpy()
    years = [y for y in CV_YEARS if y not in TEST_YEARS]
    m = measure(cells, years, f, int(p["review_days"]), int(p["reaction_days"]))
    tab = table(m)
    q = m[~m.onset]
    tag = f"{args.soil.replace(' ', '_').lower()}_{args.gpm:g}"
    summary = {"soil": args.soil, "gpm": args.gpm, "taw_in": f.taw, "capacity_in_day": round(f.capacity, 3),
               "years": years, "rows": len(m), "no_onset_rows": len(q),
               "old_cost": round(int(p["review_days"]) * f.capacity / f.efficiency * f.pump_cost_per_in * 0.6, 2),
               "mean_cost_no_onset": round(q.extra_pumped.mean() * f.pump_cost_per_in, 2),
               "stored_in": round(q.stored.mean(), 3), "lost_share_of_stored": round(q.extra_pumped.sum() * f.efficiency / q.stored.sum(), 3),
               "deficit_median_in": round(q.dr.median(), 2), **tab}
    print(json.dumps(summary, indent=1))
    (DATA / "decide").mkdir(exist_ok=True)
    m.to_parquet(DATA / "decide" / f"falsestart_{tag}.parquet")
    (DATA / "decide" / f"falsestart_{tag}.json").write_text(json.dumps(summary, indent=1))
    if args.write:
        CONFIG.write_text(json.dumps({"_source": f"python -m flashwatch.decide.falsestart (ERA5-Land {years[0]}-{years[-1]}, "
                                      f"{len(cells)} cells, {args.soil}, {args.gpm:g} gpm/acre); inches pumped", **tab}, indent=1) + "\n")


if __name__ == "__main__":
    main()
