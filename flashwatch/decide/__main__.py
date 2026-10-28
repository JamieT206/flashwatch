from __future__ import annotations

import argparse

from flashwatch.decide import curve, irrigation, load
from flashwatch.decide.engine import daily, decide, two_way


def main():
    ap = argparse.ArgumentParser(description="Act now or wait? (DESIGN.md O)")
    ap.add_argument("--week1", type=float, required=True, help="chance of onset in week 1 (0-1)")
    ap.add_argument("--week2", type=float, default=0.0, help="chance of onset in week 2 (0-1)")
    ap.add_argument("--action", nargs="+", help="actions to show (default: all)")
    ap.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE", help="override settings")
    ap.add_argument("--spread", type=float, default=0.6,
                    help="placeholder forecast revision: next update's risk x(1+spread) or lower, 50/50")
    ap.add_argument("--step", type=int, default=3, help="days to the next decision")
    args = ap.parse_args()
    over = {k: float(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load(overrides=over)
    p = daily({"week1": args.week1, "week2": args.week2})
    for name, a in cfg.items():
        if args.action and name not in args.action:
            continue
        c = curve(a)
        d = decide(c, p, two_way(args.spread), args.step)
        print(f"\n{a['label']} ($ per {a['per']})")
        if a["kind"] == "irrigation":
            ic = irrigation.curve(irrigation.field(**a["params"]), 14)
            print(f"  stress begins on day {ic.attrs['stress_day']} of a dry spell; pivot delivers "
                  f"{ic.attrs['capacity_in_day']:.2f} in/day vs crop use {a['params']['et_peak']:.2f}")
            print("  " + ic[["notice_days", "yield_loss_pct", "avoided", "extra_cost"]].round(1)
                  .to_string(index=False).replace("\n", "\n  "))
        verdict = "ACT NOW" if d.act_now else ("NOT YET" if d.ev_wait > 0 else "NO")
        print(f"  {verdict}: acting today {d.ev_now:+.1f}, deciding in {args.step} days {d.ev_wait:+.1f} "
              f"(break-even chance of onset in 14 days: {100 * d.breakeven:.0f}%)")
        for w, m, ev in d.scenarios:
            print(f"    if risk moves x{m:.2f} (chance {w:.0%}): acting then {ev:+.1f}")


if __name__ == "__main__":
    main()
