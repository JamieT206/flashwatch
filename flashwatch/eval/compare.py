"""Side-by-side scores of design variants from their OOF files (the --fast iteration loop).

    python -m flashwatch.eval.compare base=B0_B1_M1_week1_week2_any28_fast cf4=B0_M1_week1_week2_any28_fast_cf4

Each argument is name=<file stem in data/oof/> (the first is the baseline). Per lead: M1's Brier,
BSS vs B0 with a 95% year-bootstrap CI, reliability slope, PR-AUC, and the paired BSS of each
variant against the baseline's M1 with its year-bootstrap CI. Variants must share rows
(same held-out years, thinning and target depth).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from flashwatch.config import DATA
from flashwatch.eval.metrics import brier, pr_auc, reliability, year_bootstrap_bss

KEYS = ["year", "issue", "iy", "ix"]


def main(args=None):
    args = args or sys.argv[1:]
    named = [a.split("=", 1) for a in args]
    frames = {n: pd.read_parquet(DATA / "oof" / f"{stem}.parquet") for n, stem in named}
    base_name = named[0][0]
    base = frames[base_name]
    rows = []
    for lead in ("week1", "week2", "week3", "week4", "any28"):
        if f"M1_{lead}" not in base:
            continue
        for name, df in frames.items():
            if f"M1_{lead}" not in df:
                continue
            j = base[KEYS + [lead, f"M1_{lead}", f"B0_{lead}"]].merge(
                df[KEYS + [f"M1_{lead}"]], on=KEYS, suffixes=("_base", ""))
            if len(j) != len(base):
                raise SystemExit(f"{name}: {len(j)} rows match the baseline's {len(base)}; not comparable")
            y, p, b0, pb = (j[lead].to_numpy(float), j[f"M1_{lead}"].to_numpy(), j[f"B0_{lead}"].to_numpy(),
                            j[f"M1_{lead}_base"].to_numpy())
            v, lo, hi = year_bootstrap_bss(j.year, y, p, b0)
            row = {"lead": lead, "variant": name, "brier": brier(y, p), "bss_vs_B0": v,
                   "ci": f"[{lo:.4f}, {hi:.4f}]", "slope": reliability(y, p)[1], "pr_auc": pr_auc(y, p)}
            if name != base_name:
                dv, dlo, dhi = year_bootstrap_bss(j.year, y, p, pb)
                row["bss_vs_base"], row["ci_vs_base"] = dv, f"[{dlo:.4f}, {dhi:.4f}]"
            rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
