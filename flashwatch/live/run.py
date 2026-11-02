"""The live feed: today's flash-drought forecast for the farm page's Today view.

    python -m flashwatch.live.run train              # once: B0, M1 and the live M2 on every archive year (2005-2024)
    python -m flashwatch.live.run today              # daily: update the data, forecast, write the site's live year
    python -m flashwatch.live.run today --publish    # ... and publish the site (flashwatch.site.publish)

Each day: ERA5-Land through the latest day Earth Engine has (L, usually 5-10 days back), the Drought Monitor
maps and the climate indices. The forecast is issued for L + 5 days, the ERA5-Land latency the models were
trained with, so it never uses less observed data than in training; it is placed on the site's forecast dates
(every other day), so the issue date is the latest of those on or before L + 5.

In season (issue dates Apr 1 - Oct 15, as in training) the ECMWF open-data ensemble of the issue date is
fetched (live/opendata.py) and the live M2 gives the forecast. The live M2 is M2 without the week 3-4 forecast
features, which the open data (15 days) can't supply; its cross-validated skill is in data/logs/cv_m2w12.log
and docs/writeup.md. Out of season the Today view shows July 3 of the latest season, with a note saying so.

The live year's other dates use M1 (soil and weather so far, no forecast), as the replay years use the
land-only model where no ECMWF forecast was downloaded. Writes site/data/tiles/<year>/ and site/data/live.json;
the replay years' files are not touched. Models are trained on 2005-2024 with climatology from the same years;
the frozen evaluation (data/final/) is unaffected.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from flashwatch.config import DATA, YEARS

LIVE = DATA / "live"
MODEL_FILE = LIVE / "models.joblib"
OBS = LIVE / "obs.zarr"
LEADS = ("week1", "week2")
SEASON = ("04-01", "10-15")
SHOW_OFFSEASON = "07-03"
DEPTH = "0-40"
# Forecast features the open data can't give (weeks 3-4); the live M2 is trained without them.
DROP = tuple(f"f_{v}_w{w}" for v in ("tp", "t2m", "vpd", "sm100", "sm100chg", "dryhot", "hotdry") for w in (3, 4))


def _builder(cube, issues, years):
    from flashwatch.features.build import FeatureBuilder
    from flashwatch.ingest import indices, nino
    return FeatureBuilder(cube, issues, years, nino=nino.load(), **indices.load_all())


def train(thin: int = 2):
    """B0, M1 and the live M2 for each lead, on every archive year (every other cell, as in the evaluation)."""
    import joblib
    import xarray as xr

    from flashwatch.cube import load_cube
    from flashwatch.eval.cv import fold_table, model_issue_dates
    from flashwatch.features.forecast import ForecastFeatures
    from flashwatch.forecast import FCST_STORE
    from flashwatch.models.baselines import MODELS
    cube = load_cube(depth=DEPTH)
    issues = model_issue_dates(YEARS)
    fb = _builder(cube, issues, YEARS)
    ff = ForecastFeatures(xr.open_zarr(FCST_STORE).load(), issues)
    df = fold_table(cube, fb, YEARS, YEARS, ff=ff)
    del fb, ff, cube
    df = df[(df.iy % thin == 0) & (df.ix % thin == 0)]
    df = df.drop(columns=[c for c in df.columns if c.startswith(DROP)])
    models = {}
    for lead in LEADS:
        for m in ("B0", "M1", "M2"):
            models[(m, lead)] = MODELS[m]().fit(df, lead)
            print(f"trained {m} {lead} on {len(df)} rows", flush=True)
    LIVE.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "years": YEARS, "drop": DROP, "trained": pd.Timestamp.now().isoformat()},
                MODEL_FILE)
    print(f"models in {MODEL_FILE}")


def site_dates() -> list[str]:
    from flashwatch.site.build import SITE
    return json.loads((SITE / "data" / "meta.json").read_text())["dates"]


def plan(last: pd.Timestamp, today: date) -> dict:
    """Which year and date the Today view shows. last: the latest ERA5-Land day."""
    from flashwatch.features.asof import LATENCY_DAYS
    ready = last + pd.Timedelta(days=LATENCY_DAYS["era5land"])
    md = ready.strftime("%m-%d")
    dates = site_dates()
    if SEASON[0] <= md <= SEASON[1]:
        show = max(d for d in dates if d <= md)
        return {"year": ready.year, "date": show, "offseason": False}
    year = ready.year if md > SEASON[1] else ready.year - 1
    return {"year": year, "date": SHOW_OFFSEASON, "offseason": True}


def forecast_year(year: int, upto: str, ens_dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions for the site's cells and dates in `year` up to `upto`, onsets). The live obs store must
    already be built (OBS)."""
    import joblib
    import xarray as xr

    from flashwatch import obs
    from flashwatch.cube import load_cube
    from flashwatch.eval.cv import LOW, fold_table
    from flashwatch.forecast import FCST_STORE
    from flashwatch.labels.onset import onset_days
    from flashwatch.live import opendata
    from flashwatch.live.features import live_features
    from flashwatch.site.build import SITE

    saved = joblib.load(MODEL_FILE)
    models, ref = saved["models"], saved["years"]
    obs.OBS_STORE = OBS
    cube = load_cube(years=list(range(ref[0], year + 1)), depth=DEPTH)
    md = [d for d in site_dates() if d <= upto]
    issues = pd.DatetimeIndex([f"{year}-{d}" for d in md])
    fb = _builder(cube, issues, list(range(ref[0], year + 1)))   # pool: percentiles for the live year too; climatology from ref (fold)
    df = fold_table(cube, fb, ref, [year])
    meta = json.loads((SITE / "data" / "meta.json").read_text())
    cells = pd.DataFrame(meta["cells"], columns=["iy", "ix"])
    df = df.merge(cells, on=["iy", "ix"])
    fcst = xr.open_zarr(FCST_STORE).load()
    for lead in LEADS:
        df[f"B0_{lead}"] = models[("B0", lead)].predict(df)
        df[f"M1_{lead}"] = models[("M1", lead)].predict(df)
        df[f"FW_{lead}"] = df[f"M1_{lead}"]
    df["fc"] = False
    for d in ens_dates:
        rows = (df.issue == d).to_numpy()
        if not rows.any():
            continue
        f = live_features(fcst, opendata.load(d), d, ref)
        part = df[rows].copy()
        for k, grid in f.items():
            part[k] = grid[part.iy.to_numpy(), part.ix.to_numpy()]
        for lead in LEADS:
            df.loc[rows, f"M2_{lead}"] = models[("M2", lead)].predict(part)
            df.loc[rows, f"FW_{lead}"] = df.loc[rows, f"M2_{lead}"]
        df.loc[rows, "fc"] = True
    # Onsets in the live year (the farm page marks them), from the same percentiles as the features.
    P = fb.rz_counts.fold(ref)
    P[:, ~cube.land] = np.nan
    t, y, x = np.nonzero(onset_days(P, high=40, low=LOW, mask=cube.masked))
    ons = pd.DataFrame({"date": cube.time[t], "iy": y, "ix": x})
    return df, ons[ons.date.dt.year == year]


def today(publish: bool = False, now: date | None = None):
    from flashwatch import obs
    from flashwatch.config import OBS_YEARS, load_env
    from flashwatch.ingest import indices, nino
    from flashwatch.live import era5, opendata, usdm
    from flashwatch.site.build import SITE, build
    load_env()
    now = now or date.today()
    last = era5.update(sorted({now.year - 1, now.year}))
    for f in (nino.main, indices.main):
        try:
            f()
        except Exception as e:  # noqa: BLE001 - last week's index file still works
            print(f"index refresh failed, using the saved file: {e}")
    udates, ucat = usdm.update(date(OBS_YEARS[-1], 1, 1))
    p = plan(last, now)
    year, show = p["year"], p["date"]
    issue = pd.Timestamp(f"{year}-{show}")
    old = json.loads((SITE / "data" / "live.json").read_text()) if (SITE / "data" / "live.json").exists() else {}
    if p["offseason"] and old.get("offseason") and (old.get("year"), old.get("date")) == (year, show):
        # Out of season the view doesn't change: just the date it is shown on.
        (SITE / "data" / "live.json").write_text(json.dumps({**old, "today": now.isoformat(),
                                                             "era5_through": last.strftime("%Y-%m-%d")}, indent=1))
        print(f"off season: still showing {year}-{show}")
        return _publish(publish)
    if not opendata.available(issue):
        raise SystemExit(f"no ECMWF open-data run on the mirror for {issue:%Y-%m-%d}")
    opendata.download(issue)
    ens = sorted(pd.Timestamp(f.stem) for f in opendata.OUT.glob(f"{year}-*.npz"))
    obs.build(years=list(range(OBS_YEARS[0], max(year, last.year) + 1)), store=OBS)
    upto = show if not p["offseason"] else SEASON[1]
    df, ons = forecast_year(year, upto, [d for d in ens if d.strftime("%m-%d") <= upto])
    obs.OBS_STORE = OBS                       # frozen days and the water account read the live store
    from flashwatch.labels.crosscheck import usdm_cube
    a_dates, a_cat = usdm_cube(YEARS)          # the archive's maps stop in 2024; the live ones follow
    keep = a_dates < udates[0]
    ud = (a_dates[keep].append(udates), np.concatenate([a_cat[keep], ucat]))
    meta = json.loads((SITE / "data" / "meta.json").read_text())
    build(df, "FW", md=meta["dates"], cells=pd.DataFrame(meta["cells"], columns=["iy", "ix"]), ons=ons,
          usdm=ud, write_meta=False)
    live = {**p, "today": now.isoformat(), "issued": issue.strftime("%Y-%m-%d"),
            "era5_through": last.strftime("%Y-%m-%d"), "forecast_dates": [d.strftime("%m-%d") for d in ens],
            "model": "M2, weeks 1-2 forecast inputs (ECMWF open data); M1 on dates without a forecast"}
    (SITE / "data" / "live.json").write_text(json.dumps(live, indent=1))
    print(f"live: {live}")
    _publish(publish)


def _publish(publish: bool):
    if publish:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "flashwatch.site.publish"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "today"])
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--date", help="run as if today were this date (YYYY-MM-DD)")
    a = ap.parse_args()
    if a.step == "train":
        train()
    else:
        today(a.publish, pd.Timestamp(a.date).date() if a.date else None)


if __name__ == "__main__":
    main()
