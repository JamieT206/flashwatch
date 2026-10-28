"""Values to check the browser engine (site/assets/engine.js) against the Python one.

    python -m flashwatch.site.parity

Paste the snippet it prints into the browser console on the farm page; the lines must match.
"""

from flashwatch.decide import curve, load
from flashwatch.decide.engine import daily, decide, two_way

CASES = [({}, (0.28, 0.15)), ({}, (0.04, 0.10)),
         ({"gpm_per_acre": 4.0, "awc_in_per_ft": 1.4}, (0.28, 0.15)),
         ({"gpm_per_acre": 4.0, "awc_in_per_ft": 1.4}, (0.04, 0.10))]

JS = """const out = [];
for (const [over, p] of %s) {
  const c = FW.curve({...FW.DEFAULTS.corn, ...over}, 21);
  const d = FW.decide(c, FW.daily(p[0], p[1]), FW.twoWay(0.6), 2);
  out.push([d.evNow, d.evWait, d.breakeven, c.costNoEvent, c.rows[7].avoided].map(v => v.toFixed(4)).join(' '));
}
out.join('\n')"""


def main():
    import json
    for over, p in CASES:
        c = curve(load(overrides=over)["irrigation"])
        d = decide(c, daily({"week1": p[0], "week2": p[1]}), two_way(0.6), 2)
        print(" ".join(f"{v:.4f}" for v in (d.ev_now, d.ev_wait, d.breakeven, c.cost_noevent, float(c.avoided[7]))))
    print("\n" + JS % json.dumps([[o, list(p)] for o, p in CASES]))


if __name__ == "__main__":
    main()
