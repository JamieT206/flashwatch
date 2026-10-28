"""Scores of fixed blends of models from OOF files (DESIGN.md L1: M1 + the mean of several M3s).

    python -m flashwatch.eval.blend B0_M1_M3_M3b_week1_week2_fast_d0-40_m3seeds \
        B0_M3c_M3d_week1_week2_fast_d0-40_m3seeds2 --base M1 --blend M1 M3+M3b+M3c+M3d

The OOF files are joined on their rows (they must share held-out years, thinning and depth). Each
--blend term is a model, or several joined by "+" (their mean in log-odds); the blend is the
equal-weight log-odds mean of the terms. Weights are fixed, not fitted, so the scores are still
out-of-fold. Prints each model alone, every growing mean of the "+" term, and the blend: Brier
skill against B0 and paired against --base, each with a year-bootstrap range, and the reliability
slope (1 is calibrated; below 1, too spread out). --recal adds the blend after a leave-one-year-out
Platt map.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from flashwatch.config import DATA
from flashwatch.eval.metrics import brier, pr_auc, reliability, year_bootstrap_bss

KEYS = ["year", "issue", "iy", "ix"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mean_prob(cols: list[np.ndarray]) -> np.ndarray:
    return 1 / (1 + np.exp(-np.mean([logit(c) for c in cols], 0)))


def recalibrate(years: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Leave-one-year-out Platt map: each year's probabilities are mapped with a fit on the other
    years' out-of-fold predictions and outcomes, so the result is still out-of-fold."""
    from sklearn.linear_model import LogisticRegression
    z, out = logit(p)[:, None], np.empty_like(p)
    for yr in np.unique(years):
        m = years == yr
        out[m] = LogisticRegression(C=1e3).fit(z[~m], y[~m].astype(int)).predict_proba(z[m])[:, 1]
    return out


def join(stems: list[str]) -> pd.DataFrame:
    out = None
    for s in stems:
        df = pd.read_parquet(DATA / "oof" / f"{s}.parquet")
        if out is None:
            out = df
            continue
        new = [c for c in df.columns if c not in out.columns]
        n = len(out)
        out = out.merge(df[KEYS + new], on=KEYS)
        if len(out) != n:
            raise SystemExit(f"{s}: only {len(out)} of {n} rows match; not comparable")
    return out


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--base", default="M1")
    ap.add_argument("--blend", nargs="+", required=True, help='terms, e.g. M1 "M3+M3b"')
    ap.add_argument("--leads", nargs="+", default=["week1", "week2"])
    ap.add_argument("--recal", action="store_true", help="also score the blend after a leave-one-year-out Platt map")
    a = ap.parse_args(args)
    df = join(a.stems)
    rows = []
    for lead in a.leads:
        y, b0, base = df[lead].to_numpy(float), df[f"B0_{lead}"].to_numpy(), df[f"{a.base}_{lead}"].to_numpy()
        cand = {}
        for term in a.blend:
            parts = term.split("+")
            for k in range(1, len(parts) + 1):
                cand["+".join(parts[:k]) if k > 1 else parts[0]] = mean_prob([df[f"{m}_{lead}"].to_numpy() for m in parts[:k]])
        cand[" & ".join(a.blend)] = mean_prob([cand[t] for t in a.blend])
        if a.recal:
            cand[" & ".join(a.blend) + " recal"] = recalibrate(df.year.to_numpy(), y, cand[" & ".join(a.blend)])
        for name, p in cand.items():
            v, lo, hi = year_bootstrap_bss(df.year, y, p, b0)
            dv, dlo, dhi = year_bootstrap_bss(df.year, y, p, base)
            rows.append({"lead": lead, "model": name, "brier": brier(y, p), "bss": v, "bss_range": f"{lo:.3f} to {hi:.3f}",
                         f"vs_{a.base}": dv, "vs_range": f"{dlo:.3f} to {dhi:.3f}", "slope": reliability(y, p)[1],
                         "pr_auc": pr_auc(y, p)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
