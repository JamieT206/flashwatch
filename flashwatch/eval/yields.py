"""Do flash-drought onsets cost crops? County yields against onset exposure (DESIGN.md O4, L5).

    python -m flashwatch.ingest.nass yields      # once: NASS county survey yields 2005-2021
    python -m flashwatch.eval.yields

For each county and development year (2005-2021; test years untouched):
- exposure: the share of the county's land cells (within 40 km of its centroid) where a flash
  drought began (0-40 cm onset) between June 1 and August 31, the farm tool's season;
- Drought Monitor: whether at least half those cells reached D1 or worse in any week, June-September;
- yield shortfall: yield against the county's own linear trend, fitted on the *other* years, in percent.

Reported for corn and soybeans, all / irrigated / non-irrigated: mean shortfall by exposure group,
the slope of shortfall on exposure, both raw and after removing each year's average across counties
(so a drought year hitting everyone doesn't carry the result), with 90% year-bootstrap ranges.
The Monitor split answers the O0 caveat: do the onsets the Monitor never flags still cost yield?
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from flashwatch.config import DATA, RAW, TEST_YEARS
from flashwatch.eval.lead import onsets_table
from flashwatch.obs import LAT, LON

SEASON = ("06-01", "08-31")
RADIUS_KM = 40
OUT = DATA / "eval" / "yields.json"


def county_cells(cent: pd.DataFrame, land: np.ndarray, radius=RADIUS_KM) -> dict[str, np.ndarray]:
    """fips -> flat indices of land cells within radius of the centroid."""
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    out = {}
    for f, a, o in cent[["fips", "lat", "lon"]].itertuples(index=False, name=None):
        d = np.hypot(la - a, (lo - o) * np.cos(np.radians(a))) * 111
        k = np.flatnonzero((d <= radius) & land)
        if len(k):
            out[f] = k
    return out


def exposure(ons: pd.DataFrame, cells: dict, years) -> pd.DataFrame:
    md = ons.date.dt.strftime("%m-%d")
    s = ons[(md >= SEASON[0]) & (md <= SEASON[1])]
    shape = (len(LAT), len(LON))
    rows = []
    for y in years:
        hit = np.zeros(shape, bool).ravel()
        e = s[s.date.dt.year == y]
        hit[np.ravel_multi_index((e.iy.to_numpy(), e.ix.to_numpy()), shape)] = True
        rows += [(f, y, hit[k].mean()) for f, k in cells.items()]
    return pd.DataFrame(rows, columns=["fips", "year", "exposure"])


def monitor(cells: dict, years) -> pd.DataFrame:
    from flashwatch.labels.crosscheck import usdm_cube
    dates, cat = usdm_cube(list(range(2005, 2025)))
    flat = cat.reshape(len(dates), -1)
    rows = []
    for y in years:
        w = (dates.year == y) & (dates.month >= 6) & (dates.month <= 9)
        d1 = flat[w] >= 1
        rows += [(f, y, bool((d1[:, k].mean(axis=1) >= 0.5).any())) for f, k in cells.items()]
    return pd.DataFrame(rows, columns=["fips", "year", "monitor_d1"])


def shortfall(yld: pd.DataFrame, min_years=10) -> pd.DataFrame:
    """Yield vs the county's linear trend fitted on its other years, in percent (negative = below)."""
    out = []
    for _, g in yld.groupby(["fips", "crop", "practice"]):
        if len(g) < min_years:
            continue
        x, v = g.year.to_numpy(float), g.yield_bu.to_numpy(float)
        dev = np.empty(len(g))
        for i in range(len(g)):
            m = np.arange(len(g)) != i
            b = np.polyfit(x[m], v[m], 1)
            pred = np.polyval(b, x[i])
            dev[i] = 100 * (v[i] - pred) / pred
        out.append(g.assign(dev_pct=dev))
    return pd.concat(out, ignore_index=True)


def _stats(d: pd.DataFrame, col: str) -> dict:
    groups = {"none": d.exposure == 0, "some (under half)": (d.exposure > 0) & (d.exposure < 0.5),
              "most (half or more)": d.exposure >= 0.5}
    out = {f"mean_{k}": round(float(d.loc[m, col].mean()), 2) if m.any() else None for k, m in groups.items()}
    out.update({f"n_{k}": int(m.sum()) for k, m in groups.items()})
    out["slope_per_full_exposure"] = round(float(np.polyfit(d.exposure, d[col], 1)[0]), 2) if d.exposure.std() > 0 else None
    big = d.exposure >= 0.5
    for flag, name in ((True, "flagged"), (False, "not_flagged")):
        m = big & (d.monitor_d1 == flag)
        out[f"most_exposed_monitor_{name}_mean"] = round(float(d.loc[m, col].mean()), 2) if m.any() else None
        out[f"most_exposed_monitor_{name}_n"] = int(m.sum())
    return out


def summarise(d: pd.DataFrame, n_boot=1000, seed=0) -> dict:
    d = d.copy()
    d["dev_within_year"] = d.dev_pct - d.groupby("year").dev_pct.transform("mean")
    res = {}
    for col in ("dev_pct", "dev_within_year"):
        s = _stats(d, col)
        rng, years = np.random.default_rng(seed), d.year.unique()
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(years, len(years))
            b = pd.concat([d[d.year == y] for y in pick])
            boots.append(_stats(b, col))
        for k in ("slope_per_full_exposure", "mean_most (half or more)", "most_exposed_monitor_not_flagged_mean"):
            vals = [b[k] for b in boots if b[k] is not None]
            if vals:
                s[k + "_90"] = [round(float(np.percentile(vals, 5)), 2), round(float(np.percentile(vals, 95)), 2)]
        res["raw" if col == "dev_pct" else "within_year"] = s
    return res


def run(n_boot=1000) -> dict:
    from flashwatch.ingest.nass import centroids
    from flashwatch.obs import land_mask, open_obs
    years = [y for y in range(2005, 2022) if y not in TEST_YEARS]
    land = land_mask(open_obs())
    cent = centroids()
    cent = cent[(cent.lat.between(36, 49)) & (cent.lon.between(-104, -80))]
    cells = county_cells(cent, land)
    yld = pd.read_csv(RAW / "nass" / "yields_2005_2021.csv", dtype={"fips": str})
    yld = yld[yld.fips.isin(cells) & yld.year.isin(years)]
    d = (shortfall(yld).merge(exposure(onsets_table(), cells, years), on=["fips", "year"])
         .merge(monitor(cells, years), on=["fips", "year"]))
    out = {"years": years, "season": SEASON, "radius_km": RADIUS_KM, "results": {}}
    for (crop, practice), g in d.groupby(["crop", "practice"]):
        out["results"][f"{crop}|{practice}"] = {"county_years": len(g), "counties": g.fips.nunique(),
                                                 **summarise(g, n_boot)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    out = run(ap.parse_args().boot)
    from flashwatch.site.build import SITE
    for f in (OUT, SITE / "data" / "yields.json"):       # the region page reads the second copy
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, indent=1))
    for k, r in out["results"].items():
        for kind in ("raw", "within_year"):
            s = r[kind]
            print(f"{k:24s} {kind:11s} n={r['county_years']:5d}  none {s['mean_none']:+6.1f}%  some {s['mean_some (under half)']}"
                  f"  most {s['mean_most (half or more)']} {s.get('mean_most (half or more)_90')}"
                  f"  slope {s['slope_per_full_exposure']} {s.get('slope_per_full_exposure_90')}"
                  f"  | most+Monitor {s['most_exposed_monitor_flagged_mean']} (n {s['most_exposed_monitor_flagged_n']})"
                  f"  most, no Monitor {s['most_exposed_monitor_not_flagged_mean']} (n {s['most_exposed_monitor_not_flagged_n']})"
                  f" {s.get('most_exposed_monitor_not_flagged_mean_90')}")


if __name__ == "__main__":
    main()
