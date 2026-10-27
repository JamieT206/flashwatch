"""Large-scale climate indices to data/raw/indices/ (DESIGN.md F6).

    python -m flashwatch.ingest indices

- GISTEMP v4 global land-ocean temperature anomaly, monthly (NASA GISS): "state of climate change".
  Month m is treated as available 15 days after it ends (LATENCY_DAYS["gistemp"]).
- MJO, daily: NOAA PSL's real-time OLR MJO index (ROMI, CPC OLR), which uses past data only
  (the standard OMI uses a centred filter, i.e. future data, so it would leak). PC1, PC2,
  amplitude, plus the phase angle. Available 2 days after the valid date.
Both are small and revised over time, so every run refreshes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.config import RAW
from flashwatch.ingest.common import get_logger, http_download, retry

OUT = RAW / "indices"
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
MJO_URL = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
log = get_logger("indices")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in (("gistemp.csv", GISTEMP_URL), ("romi.txt", MJO_URL)):
        target = OUT / name
        target.unlink(missing_ok=True)
        retry(lambda: http_download(url, target, log), log, name, tries=3, wait=10)
    log.info("done: GISTEMP to %s, MJO to %s", load_gistemp().index[-1].strftime("%Y-%m"),
             load_mjo().index[-1].date())


def load_gistemp(path=None) -> pd.Series:
    """Monthly global temperature anomaly (deg C vs 1951-1980), indexed by month start."""
    df = pd.read_csv(path or OUT / "gistemp.csv", skiprows=1, na_values="***")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    s = df.melt(id_vars="Year", value_vars=months, var_name="m", value_name="v").dropna()
    s.index = pd.to_datetime(s.Year.astype(str) + "-" + s.m, format="%Y-%b")
    return s.v.astype(float).sort_index()


def load_mjo(path=None) -> pd.DataFrame:
    """Daily ROMI pc1, pc2, amplitude and phase angle (radians); missing days dropped."""
    df = pd.read_csv(path or OUT / "romi.txt", sep=r"\s+", header=None,
                     names=["y", "m", "d", "h", "pc1", "pc2", "amplitude"])
    df.index = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
    df = df[["pc1", "pc2", "amplitude"]]
    df = df[(df.abs() < 900).all(axis=1)].copy()
    df["angle"] = np.arctan2(df.pc2, df.pc1)
    return df


def load_all() -> dict:
    """Keyword arguments for FeatureBuilder: whichever indices have been downloaded."""
    out = {}
    if (OUT / "gistemp.csv").exists():
        out["gistemp"] = load_gistemp()
    if (OUT / "romi.txt").exists():
        out["mjo"] = load_mjo()
    return out


def gistemp_12m(series: pd.Series) -> pd.Series:
    """Trailing 12-month mean, indexed by the last month (needs all 12 months)."""
    return series.rolling(12, min_periods=12).mean().dropna()


def daily_as_of(df: pd.DataFrame, source: str, issues) -> np.ndarray:
    """Rows of a daily table as of each issue date: the last valid day <= as_of(source, issue)."""
    from flashwatch.features.asof import as_of
    cut = pd.DatetimeIndex([as_of(source, d) for d in issues])
    pos = df.index.searchsorted(cut, side="right") - 1
    out = df.to_numpy(float)[np.maximum(pos, 0)]
    out[pos < 0] = np.nan
    return out
