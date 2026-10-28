"""Weekly data pack for the analyst agent: data/agent/pack_<issue>.parquet (DESIGN.md G1).

    python -m flashwatch.agent.pack --issue 2012-06-19 [--model M2]

For one issue date and every land cell: the forecast chance of an onset in weeks 1-4 and within
28 days, the "normal" chance for that place and date (B0), the three features pushing the 28-day
risk up the most (this week's conditions; place and season features excluded) (LightGBM per-prediction contributions), the current soil-moisture percentile,
whether the cell is already in drought, the latest published Drought Monitor category and, for
M2, the ECMWF weather outlook. Models are trained on the CV years minus the issue's own year, so a
replay never sees its own year (light settings, every other training cell). Test years are refused
until the model freeze (they are scored once).
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

from flashwatch.config import CV_YEARS, DATA, TEST_YEARS, YEARS
from flashwatch.obs import LAT, LON

OUT = DATA / "agent"
LEADS = ["week1", "week2", "week3", "week4", "any28"]
# Features that describe the place or the season, not this week: left out of the drivers.
NOT_DRIVERS = ("aridity", "gs_precip", "doy_sin", "doy_cos")
OUTLOOK = [f"f_{v}_{w}_z" for v in ("tp", "t2m", "vpd") for w in ("w1", "w2", "w12")] + ["f_hotdry_w12_p"]


def build(issue: pd.Timestamp, model: str = "M2", depth: str = "0-40") -> pd.DataFrame:
    from flashwatch.cube import load_cube
    from flashwatch.eval.cv import fold_table, model_issue_dates
    from flashwatch.features.asof import as_of
    from flashwatch.features.build import FeatureBuilder
    from flashwatch.ingest import indices, nino
    from flashwatch.labels.crosscheck import usdm_cube
    from flashwatch.models.baselines import LIGHT, MODELS

    year = issue.year
    if year in TEST_YEARS:
        raise SystemExit(f"{year} is a test year: no packs before the model freeze")
    issues = model_issue_dates(YEARS)
    if issue not in issues:
        raise SystemExit(f"{issue.date()} is not a forecast issue date; nearest: "
                         f"{issues[np.abs(issues - issue).argmin()].date()}")
    cube = load_cube(depth=depth)
    fb = FeatureBuilder(cube, issues, YEARS, nino=nino.load(), **indices.load_all())
    ff = None
    if model == "M2":
        import xarray as xr
        from flashwatch.features.forecast import ForecastFeatures
        from flashwatch.forecast import FCST_STORE
        ff = ForecastFeatures(xr.open_zarr(FCST_STORE).load(), issues)
    land = cube.land
    cube = dataclasses.replace(cube, **{f: None for f in ("rzsm", "sfsm", "t2m", "vpd", "precip", "evap", "pet", "ssrd")})
    ref = [y for y in CV_YEARS if y != year]
    df = fold_table(cube, fb, ref, ref + [year], ff=ff)
    train = df[(df.year != year) & (df.iy % 2 == 0) & (df.ix % 2 == 0)]
    test = df[df.issue == issue]
    if model == "M2" and test.filter(like="f_tp_w1_z").isna().all().all():
        raise SystemExit(f"no ECMWF forecast for {issue.date()} yet")
    out = test[["iy", "ix", "rz_pct"] + [c for c in OUTLOOK if c in test]].copy()
    for lead in LEADS:
        out[f"normal_{lead}"] = MODELS["B0"]().fit(train, lead).predict(test)
        m = MODELS[model](**LIGHT).fit(train, lead)
        out[f"p_{lead}"] = m.predict(test)
        if lead == "any28":
            c = m.contributions(test).drop(columns=["bias", *[n for n in NOT_DRIVERS if n in m.cols]])
            top = np.argsort(-c.to_numpy(), axis=1)[:, :3]
            for r in range(3):
                out[f"driver{r + 1}"] = np.array(c.columns)[top[:, r]]
                out[f"driver{r + 1}_push"] = np.take_along_axis(c.to_numpy(), top[:, r:r + 1], 1)[:, 0]
                out[f"driver{r + 1}_value"] = [test[n].iloc[i] for i, n in enumerate(out[f"driver{r + 1}"])]
    # Land cells with no row are below the 20th percentile already: in drought, no onset possible.
    iy, ix = np.nonzero(land)
    cells = pd.DataFrame({"iy": iy, "ix": ix})
    out = cells.merge(out, on=["iy", "ix"], how="left")
    out["in_drought"] = out.p_any28.isna()
    out["lat"], out["lon"] = LAT[out.iy], LON[out.ix]
    udates, ucat = usdm_cube(list(range(2004, 2025)))
    j = udates.searchsorted(as_of("usdm", issue), side="right") - 1
    out["usdm_cat"] = ucat[j][out.iy, out.ix] if j >= 0 else -1
    out.attrs.update(issue=str(issue.date()), model=model, depth=depth)
    return out


def path(issue) -> "Path":
    return OUT / f"pack_{pd.Timestamp(issue).date()}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--model", default="M2", choices=["M1", "M2"])
    ap.add_argument("--depth", default="0-40")
    args = ap.parse_args()
    issue = pd.Timestamp(args.issue)
    pack = build(issue, args.model, args.depth)
    OUT.mkdir(parents=True, exist_ok=True)
    pack["model"] = args.model
    pack.to_parquet(path(issue))
    print(f"wrote {path(issue)}: {pack.p_any28.notna().sum()} cells with a forecast, {pack.in_drought.sum()} in drought")


if __name__ == "__main__":
    main()
