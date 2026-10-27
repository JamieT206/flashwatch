"""Features per (issue date, cell), computed as of the issue date minus source latency.

Fold-independent work (window series, calendar sums, percentile counts) happens once in
FeatureBuilder(); build(ref_years) then returns one fold's features in seconds. Every read of
the issue year's own data goes through as_of(); the leakage test perturbs everything after
the cutoff (and every non-reference year) and requires identical features.

Not yet included (need extra downloads): soil water capacity, cropland fraction,
and the ECMWF forecast features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.cube import Cube
from flashwatch.features.asof import as_of, monthly_as_of
from flashwatch.features.climatology import YearSums, trailing_mean
from flashwatch.labels.percentile import PercentileCounts, calendar_day

# (feature stem, cube field, window days, sign): sign flips ERA5's negative-upward fluxes.
WINDOWS = [("t2m", "t2m", 14, 1), ("t2m", "t2m", 30, 1), ("vpd", "vpd", 14, 1), ("vpd", "vpd", 30, 1),
           ("pet", "pet", 14, -1), ("pet", "pet", 30, -1), ("ssrd", "ssrd", 14, 1), ("ssrd", "ssrd", 30, 1),
           ("precip", "precip", 14, 1), ("precip", "precip", 30, 1), ("precip", "precip", 60, 1),
           ("rzsm", "rzsm", 5, 1)]
ESI_WINDOWS = (14, 30)
SINCE_HIGH_MAX = 60           # days-since-P>=40 is capped here
SINCE_RAIN_MAX = 90           # dry-spell features are capped here
WET_DAY = {2: 0.002, 10: 0.010}   # mm threshold -> m/day (ERA5-Land units)
HOT_K = 298.15                # daily mean above 25 C
TO_THRESHOLD_MAX = 60         # days-to-threshold cap (also used when not declining)
NBR_RADIUS = 3                # F5 neighbourhood: 7 x 7 cells (~175 km)


# Directional boxes (DESIGN.md F5, James): rows (dy) and columns (dx) relative to the cell.
# Row index increases southwards (lat descending), column index eastwards. 1-6 cells (~25-150 km)
# along the direction, +/-2 cells across it.
DIRECTIONS = {"W": ((-2, 2), (-6, -1)), "E": ((-2, 2), (1, 6)), "N": ((-6, -1), (-2, 2)), "S": ((1, 6), (-2, 2))}


def box_mean(a: np.ndarray, dy: tuple[int, int], dx: tuple[int, int]) -> np.ndarray:
    """Mean of a over the cells at row offsets dy[0]..dy[1] and column offsets dx[0]..dx[1]
    from each cell (last two axes), ignoring NaN and cells off the grid."""
    ny, nx = a.shape[-2:]
    pad = 6
    big = np.full(a.shape[:-2] + (ny + 2 * pad, nx + 2 * pad), np.nan, np.float64)
    big[..., pad:pad + ny, pad:pad + nx] = a
    num = np.zeros(a.shape, np.float64)
    den = np.zeros(a.shape, np.float64)
    for oy in range(dy[0], dy[1] + 1):
        for ox in range(dx[0], dx[1] + 1):
            v = big[..., pad + oy:pad + oy + ny, pad + ox:pad + ox + nx]
            ok = np.isfinite(v)
            num += np.where(ok, v, 0)
            den += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def nbr_mean(a: np.ndarray, r: int = NBR_RADIUS) -> np.ndarray:
    """Mean over the (2r+1)^2 neighbourhood in the last two axes, ignoring NaN (incl. the cell)."""
    from scipy.ndimage import uniform_filter
    ok = np.isfinite(a)
    size = (1,) * (a.ndim - 2) + (2 * r + 1, 2 * r + 1)
    num = uniform_filter(np.where(ok, a, 0).astype(np.float64), size=size, mode="constant")
    den = uniform_filter(ok.astype(np.float64), size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9, num / den, np.nan)


def days_since(event: np.ndarray, cap: int) -> np.ndarray:
    """Days since the last True along axis 0 (0 on the day itself), capped; past values only."""
    out = np.empty(event.shape, np.float32)
    run = np.full(event.shape[1:], cap, np.float32)
    for t in range(len(event)):
        run = np.where(event[t], 0, np.minimum(run + 1, cap))
        out[t] = run
    return out
PCT_SEASON = ("01-20", "12-15")


def season_days(time: pd.DatetimeIndex, start=PCT_SEASON[0], end=PCT_SEASON[1]) -> np.ndarray:
    md = time.strftime("%m-%d")
    return (md >= start) & (md <= end)


class FeatureBuilder:
    def __init__(self, cube: Cube, issues: pd.DatetimeIndex, pool_years, rz_counts=None, nino=None,
                 gistemp=None, mjo=None, usdm=None):
        """nino: monthly Nino 3.4 series (ingest.nino.load()); gistemp: monthly global temperature
        anomaly (ingest.indices.load_gistemp()); mjo: daily MJO table (ingest.indices.load_mjo()).
        usdm: (map dates, categories (map, lat, lon), -1 none .. 4 = D4), labels.crosscheck.usdm_cube().
        Each feature is omitted if its input is None."""
        from flashwatch.ingest.indices import daily_as_of, gistemp_12m
        self.issues = pd.DatetimeIndex(issues)
        self.nino = None if nino is None else np.array([monthly_as_of(nino, "nino", d) for d in self.issues])
        self.gmst12 = None if gistemp is None else np.array(
            [monthly_as_of(gistemp_12m(gistemp), "gistemp", d) for d in self.issues])
        self.mjo = None if mjo is None else dict(zip(mjo.columns, daily_as_of(mjo, "mjo", self.issues).T))
        self.usdm = None
        if usdm is not None:
            # F9: latest Drought Monitor map published by the issue date, and the one 4 weeks before.
            dates, cat = pd.DatetimeIndex(usdm[0]), np.asarray(usdm[1], np.float32)
            cut = pd.DatetimeIndex([as_of("usdm", d) for d in self.issues])
            i = dates.searchsorted(cut, side="right") - 1
            now = np.where((i >= 0)[:, None, None], cat[np.maximum(i, 0)], np.nan)
            then = np.where((i >= 4)[:, None, None], cat[np.maximum(i - 4, 0)], np.nan)
            self.usdm = {"usdm_cat": now, "usdm_chg4w": now - then,
                         "nbr_usdm_d1": nbr_mean(np.where(np.isnan(now), np.nan, (now >= 1).astype(np.float32)))}
        time = cube.time
        cutoff = pd.DatetimeIndex([as_of("era5land", d) for d in self.issues])
        self.pos = time.get_indexer(cutoff)
        if (self.pos < 0).any():
            raise ValueError("an issue date's as-of cutoff is outside the cube")
        cal = calendar_day(cutoff)
        self.cal_days, self.k_idx = np.unique(cal, return_inverse=True)
        self.own_year = self.issues.year.to_numpy()
        self.pool_years = list(pool_years)

        # One window series at a time: keep only its cutoff values and calendar sums.
        self.values, self.sums = {}, {}

        def add(name, R):
            self.values[name] = R[self.pos]
            self.sums[name] = YearSums(R, time, self.cal_days, self.pool_years)

        for stem, fieldname, n, sign in WINDOWS:
            add(f"{stem}{n}", sign * trailing_mean(getattr(cube, fieldname), n))
        for n in ESI_WINDOWS:
            e, p = trailing_mean(cube.evap, n), trailing_mean(cube.pet, n)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(np.abs(p) > 1e-5, e / p, np.nan)
            del e, p
            r = np.clip(r, 0, 1.5).astype(np.float32)
            add(f"esi{n}", r)
            self.values[f"esi{n}_prev14"] = r[self.pos - 14]      # F2 rapid change, as of the cutoff

        # Static climatologies: annual precipitation / PET (aridity) and Apr-Sep precipitation.
        yr = time.year.to_numpy()
        gs = (time.month >= 4) & (time.month <= 9)
        self.annual_p = {y: np.nansum(cube.precip[yr == y], 0) for y in self.pool_years}
        self.annual_pet = {y: -np.nansum(cube.pet[yr == y], 0) for y in self.pool_years}
        self.gs_p = {y: np.nansum(cube.precip[(yr == y) & gs], 0) for y in self.pool_years}

        # Dry spells and hot days: running counts over the daily series, read at the cutoff.
        precip = np.nan_to_num(cube.precip)
        self.dry = {mm: days_since(precip > m, SINCE_RAIN_MAX)[self.pos] for mm, m in WET_DAY.items()}
        hot = np.nan_to_num(cube.t2m) > HOT_K
        self.hot14 = trailing_mean(hot.astype(np.float32), 14)[self.pos] * 14
        del precip, hot

        days = season_days(time)
        self.rz_counts = rz_counts or PercentileCounts(trailing_mean(cube.rzsm, 5), time, self.pool_years, days)
        self.sf_counts = PercentileCounts(trailing_mean(cube.sfsm, 5), time, self.pool_years, days)
        # F1 deep storage: 0-100 cm percentile when the target depth is shallower (load_cube puts it in extra).
        deep = cube.extra.get("deep_sm")
        self.deep_counts = None if deep is None else PercentileCounts(trailing_mean(deep, 5), time,
                                                                     self.pool_years, days)
        self.shape = cube.shape

    def build(self, ref_years, P_rz: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Features (n_issue, lat, lon) under climatology from ref_years (each issue's own year
        excluded). P_rz: this fold's root-zone percentiles, if already computed for labels."""
        ref = list(ref_years)
        P_rz = self.rz_counts.fold(ref) if P_rz is None else P_rz
        P_sf = self.sf_counts.fold(ref)
        f = {}
        p = self.pos
        f["rz_pct"] = P_rz[p]
        for k in (5, 10, 20):
            f[f"rz_pct_chg{k}"] = P_rz[p] - P_rz[p - k]
        hist = np.stack([P_rz[p - i] for i in range(SINCE_HIGH_MAX + 1)])          # (61, n, ...)
        with np.errstate(invalid="ignore"):
            high = hist >= 40
        f["rz_days_since_p40"] = np.where(high.any(0), high.argmax(0), SINCE_HIGH_MAX + 1).astype(np.float32)
        f["sf_pct"] = P_sf[p]
        f["sf_pct_chg5"] = P_sf[p] - P_sf[p - 5]
        # F1 soil moisture dynamics (DESIGN.md): acceleration, time to the 20th percentile at the
        # current 10-day rate, top-down drying, and the recent high point (onset condition 2).
        f["rz_pct_acc5"] = (P_rz[p] - P_rz[p - 5]) - (P_rz[p - 5] - P_rz[p - 10])
        rate = (P_rz[p - 10] - P_rz[p]) / 10                                   # points per day, >0 = drying
        with np.errstate(invalid="ignore", divide="ignore"):
            ttt = np.where(rate > 0, (P_rz[p] - 20) / rate, TO_THRESHOLD_MAX)
        f["rz_days_to_p20"] = np.where(np.isnan(rate), np.nan, np.clip(ttt, 0, TO_THRESHOLD_MAX))
        f["sf_minus_rz_pct"] = P_sf[p] - P_rz[p]
        f["rz_pct_max15"] = np.max(np.stack([P_rz[p - i] for i in range(16)]), 0)
        # F3 dry spells, F2 hot days
        for mm, v in self.dry.items():
            f[f"dry_days_{mm}mm"] = v
        f["hot_days14"] = self.hot14
        # F5 spatial context: the 7 x 7 neighbourhood's state, incl. cells already in drought
        # (which are not rows themselves). "Newly dry": crossed below 20 in the last 14 days.
        P0, P14 = P_rz[p], P_rz[p - 14]
        f["nbr_rz_pct"] = nbr_mean(P0)
        f["nbr_rz_pct_chg10"] = nbr_mean(P0 - P_rz[p - 10])
        with np.errstate(invalid="ignore"):
            dry = np.where(np.isnan(P0), np.nan, (P0 <= 20).astype(np.float32))
            newly = np.where(np.isnan(P0) | np.isnan(P14), np.nan, ((P0 <= 20) & (P14 > 20)).astype(np.float32))
        f["nbr_share_dry"] = nbr_mean(dry)
        f["nbr_share_newly_dry"] = nbr_mean(newly)
        # Directional: does drought upwind (west) matter more than downwind (east)?
        for d, (dy, dx) in DIRECTIONS.items():
            f[f"nbr{d}_rz_pct"] = box_mean(P0, dy, dx)
            f[f"nbr{d}_share_dry"] = box_mean(dry, dy, dx)
        for name, ys in self.sums.items():
            f[f"{name}_z"] = ys.zscore(self.values[name], self.k_idx, self.own_year, ref)
        for n in ESI_WINDOWS:
            f[f"esi{n}"] = self.values[f"esi{n}"]
            # F2 rapid change: how far evaporation has fallen behind demand over two weeks
            f[f"esi{n}_chg14"] = self.values[f"esi{n}"] - self.values[f"esi{n}_prev14"]
        # F2 drying pressure: air dryness anomaly weighted by how dry the soil already is
        f["vpd14_x_dryness"] = f["vpd14_z"] * (100 - P_rz[p]) / 100
        if self.deep_counts is not None:
            P_deep = self.deep_counts.fold(ref)
            f["deep_pct"] = P_deep[p]
            f["deep_pct_chg10"] = P_deep[p] - P_deep[p - 10]
        doy = self.issues.day_of_year.to_numpy()
        shape = (len(p),) + self.shape
        f["doy_sin"] = np.broadcast_to(np.sin(2 * np.pi * doy / 365.25)[:, None, None], shape)
        f["doy_cos"] = np.broadcast_to(np.cos(2 * np.pi * doy / 365.25)[:, None, None], shape)
        # Static climatologies over the reference years minus the issue's own year: a site
        # mean that included the issue year's full totals would leak its future rainfall.
        W = (np.isin(self.pool_years, ref)[None, :]
             & (np.asarray(self.pool_years)[None, :] != self.own_year[:, None])).astype(np.float64)
        W /= W.sum(1, keepdims=True)

        def site_mean(per_year):
            return np.einsum("nj,jyx->nyx", W, np.stack([per_year[y] for y in self.pool_years]))

        with np.errstate(invalid="ignore", divide="ignore"):
            f["aridity"] = site_mean(self.annual_p) / site_mean(self.annual_pet)
        f["gs_precip"] = site_mean(self.gs_p)
        if self.nino is not None:
            f["nino34"] = np.broadcast_to(self.nino[:, None, None], shape)
        if self.usdm is not None:
            f.update(self.usdm)
        if self.gmst12 is not None:
            f["gmst12"] = np.broadcast_to(self.gmst12[:, None, None], shape)
        if self.mjo is not None:
            for k, v in self.mjo.items():
                f[f"mjo_{k}"] = np.broadcast_to(v[:, None, None], shape)
        # F10: calendar year as a number (trend); the year-as-category diagnostic is separate.
        f["year_num"] = np.broadcast_to(self.issues.year.to_numpy(np.float32)[:, None, None], shape)
        return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}
