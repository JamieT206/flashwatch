"""B0's smoothing and shrinkage, chosen per lead by year-grouped CV on the dev years (QUESTIONS.md Q6a).

    python -m flashwatch.models.tune_b0 --depth 0-40     # needs python -m flashwatch.labels --depth 0-40

Onsets cluster by year, so the effective sample behind a cell's rate is a handful of years and the
original B0 (+/-2 weeks, 3x3 cells, k=20) was far too extreme (reliability slope ~0.4). Grid search
over the time half-width tw, spatial radius sr and pseudo-row count k, minimising pooled Brier over
4 year-groups of 2005-2021, using the reference labels (climatology from all years, own year
excluded). Fitting works on per-group (week of year, cell) counts, so each grid point is cheap.
Writes config/b0_params.json (for the current target depth, recorded in the file), which Climatology reads.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from flashwatch.config import CV_YEARS, DATA
from flashwatch.models.baselines import B0_FILE, b0_rate

GRID = {"tw": [1, 2, 4, 8, 13], "sr": [0, 1, 2, 4, 8], "k": [20, 500, 5000, 50000, 200000, 1000000]}
LABELS = ["week1", "week2", "week3", "week4", "any28"]


def group_counts(z, label: str, groups) -> list[tuple[np.ndarray, np.ndarray]]:
    issue = pd.DatetimeIndex(z["issue"])
    woy = ((issue.day_of_year.to_numpy() - 1) // 7).clip(0, 52)
    el, y = z["eligible"].astype(bool), z[label].astype(float)
    out = []
    for g in groups:
        pos = np.zeros((53,) + el.shape[1:])
        cnt = np.zeros_like(pos)
        for i in np.nonzero(np.isin(issue.year, g))[0]:
            cnt[woy[i]] += el[i]
            pos[woy[i]] += y[i] * el[i]
        out.append((pos, cnt))
    return out


def brier_and_slope(counts, k, tw, sr, slope=False):
    """Pooled held-out Brier over the groups; optionally the reliability (logistic) slope."""
    se, n, agg = 0.0, 0.0, []
    for j, (pos, cnt) in enumerate(counts):
        tp = sum(c[0] for i, c in enumerate(counts) if i != j)
        tc = sum(c[1] for i, c in enumerate(counts) if i != j)
        r = b0_rate(tp, tc, k, tw, sr)
        se += (cnt * r ** 2 - 2 * pos * r + pos).sum()
        n += cnt.sum()
        if slope:
            m = cnt > 0
            agg.append((r[m], pos[m], cnt[m]))
    if not slope:
        return se / n, None
    from sklearn.linear_model import LogisticRegression
    r = np.concatenate([a[0] for a in agg]).clip(1e-6, 1 - 1e-6)
    p_, c_ = np.concatenate([a[1] for a in agg]), np.concatenate([a[2] for a in agg])
    x = np.log(r / (1 - r))[:, None]
    X, Y, W = np.vstack([x, x]), np.r_[np.ones(len(x)), np.zeros(len(x))], np.r_[p_, c_ - p_]
    lr = LogisticRegression(C=1e6).fit(X, Y, sample_weight=W)
    return se / n, float(lr.coef_[0, 0])


def main():
    import argparse

    from flashwatch.labels.__main__ import DEPTHS, label_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="0-40", choices=list(DEPTHS))
    depth = ap.parse_args().depth
    z = np.load(DATA / "labels" / label_file(40, 20, depth), allow_pickle=True)
    years = np.array(CV_YEARS)
    groups = np.array_split(np.random.default_rng(0).permutation(years), 4)
    tuned = {"depth": depth}
    for label in LABELS:
        counts = group_counts(z, label, groups)
        base_n = sum(c[1].sum() for c in counts)
        rate = sum(c[0].sum() for c in counts) / base_n
        flat = rate * (1 - rate)                           # Brier of the constant base rate (approx.)
        default, d_slope = brier_and_slope(counts, 20, 2, 1, slope=True)
        res = [(brier_and_slope(counts, k, tw, sr)[0], k, tw, sr)
               for tw, sr, k in itertools.product(GRID["tw"], GRID["sr"], GRID["k"])]
        b, k, tw, sr = min(res)
        _, slope = brier_and_slope(counts, k, tw, sr, slope=True)
        tuned[label] = {"k": k, "tw": tw, "sr": sr, "brier": b, "default_brier": default,
                        "slope": slope, "default_slope": d_slope}
        edge = [n for n, v in (("tw", tw), ("sr", sr), ("k", k)) if v in (GRID[n][0], GRID[n][-1])]
        print(f"{label}: tw={tw} sr={sr} k={k}  Brier {b:.6f} (default {default:.6f}, flat {flat:.6f}); "
              f"BSS vs default {1 - b / default:+.4f}; slope {slope:.2f} (default {d_slope:.2f})"
              + (f"  [at grid edge: {', '.join(edge)}]" if edge else ""), flush=True)
    B0_FILE.write_text(json.dumps(tuned, indent=1))
    print(f"wrote {B0_FILE}")


if __name__ == "__main__":
    main()
