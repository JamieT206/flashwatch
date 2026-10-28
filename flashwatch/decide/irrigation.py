"""How much a few more days of notice are worth to an irrigator (DESIGN.md O1a).

A daily root-zone water balance in the FAO-56 style through a dry spell: the profile starts full,
the crop uses `et_peak` inches a day, no rain falls. Once depletion passes the readily available
water (p x total available water) the crop is stressed and uses less than it needs; the shortfall
costs yield through the FAO-33 yield response factor Ky. A centre pivot adds at most `capacity`
inches a day and cannot fill the profile beyond full.

The usual practice without FlashWatch ("reactive") starts the pivot `reaction_days` after stress
begins (a soil-moisture sensor or crop symptoms). FlashWatch can start it k days *before* stress
begins. Starting early banks water while the profile still has room; once the pivot can't keep up
with the crop, that bank is what carries it through. Extra notice stops helping once the pivot is on
before the profile has drained at all, and helps little where the pivot can match peak demand.

All defaults are typical published values for irrigated corn on a silt loam in Nebraska, to be
replaced with the farm's own (config/actions.json says where each comes from).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

GPM_PER_ACRE_TO_IN_PER_DAY = 24 * 60 / 27154      # 1 gpm/acre = 0.053 in/day (27,154 gal = 1 acre-inch)


@dataclass
class Field:
    awc_in_per_ft: float = 2.0       # plant-available water, inches per foot of soil (silt loam ~2.0, sand ~1.0)
    root_ft: float = 3.0             # managed root zone depth
    p: float = 0.55                  # FAO-56 depletion fraction before stress (maize)
    et_peak: float = 0.30            # crop water use in a hot, dry spell, inches/day
    gpm_per_acre: float = 5.0        # pivot capacity
    efficiency: float = 0.85         # share of pumped water that reaches the root zone
    ky: float = 1.5                  # FAO-33 yield response factor (maize around flowering; 1.25 whole season)
    season_et: float = 24.0          # seasonal crop water use, inches
    yield_bu: float = 220.0          # expected yield, bushels/acre
    price: float = 4.50              # $/bushel
    pump_cost_per_in: float = 10.0   # $ per acre-inch pumped (energy + wear; varies widely)
    spell_days: int = 35             # length of the dry spell simulated
    reaction_days: int = 2           # reactive start: days after stress begins
    dr0: float = 0.0                 # root-zone deficit when the dry spell begins, inches (0 = full; the water account)

    @property
    def taw(self) -> float:
        return self.awc_in_per_ft * self.root_ft

    @property
    def capacity(self) -> float:
        """Inches a day reaching the root zone."""
        return self.gpm_per_acre * GPM_PER_ACRE_TO_IN_PER_DAY * self.efficiency


def simulate(f: Field, start: int | None) -> dict:
    """Dry spell from the field's current deficit (full unless f.dr0); pivot on from day `start` (None =
    never). Returns the crop's water shortfall (in), water pumped (in) and the depletion each day."""
    taw, raw = f.taw, f.p * f.taw
    dr, shortfall, pumped = min(f.dr0, taw), 0.0, 0.0
    trace = []
    for t in range(f.spell_days):
        if start is not None and t >= start:
            add = min(f.capacity, dr)                    # can't fill beyond full
            dr -= add
            pumped += add / f.efficiency
        ks = 1.0 if dr <= raw else max(0.0, (taw - dr) / (taw - raw))
        eta = ks * f.et_peak
        dr = min(taw, dr + eta)
        shortfall += f.et_peak - eta
        trace.append(dr)
    return {"shortfall_in": shortfall, "pumped_in": pumped, "depletion": np.array(trace)}


def stress_day(f: Field) -> int:
    """First day an unirrigated crop is stressed (depletion above p x TAW): day 0 of 'onset'."""
    d = simulate(f, None)["depletion"]
    over = np.nonzero(d > f.p * f.taw)[0]
    return int(over[0]) if len(over) else f.spell_days


def curve(f: Field, max_notice: int = 21) -> pd.DataFrame:
    """Per days of notice k (pivot on k days before stress begins): yield loss, the loss avoided
    and the extra pumping cost (net of water left in the soil) against the reactive start, $/acre."""
    s0 = stress_day(f)
    value = f.yield_bu * f.price

    def run(start):
        # Water still in the soil at the end of the spell is credited at the pumping cost: an early
        # start that leaves the profile fuller hasn't spent that water, only stored it.
        r = simulate(f, max(0, start))
        loss = f.ky * r["shortfall_in"] / f.season_et * value
        net_pumped = r["pumped_in"] - (f.taw - r["depletion"][-1]) / f.efficiency
        return loss, net_pumped * f.pump_cost_per_in, r["shortfall_in"]

    base_loss, base_pump, _ = run(s0 + f.reaction_days)
    rows = []
    for k in range(0, max_notice + 1):
        loss, pump, short = run(s0 - k)
        rows.append({"notice_days": k, "start_day": max(0, s0 - k), "shortfall_in": round(short, 2),
                     "yield_loss_pct": round(100 * loss / value, 1), "avoided": base_loss - loss,
                     "extra_cost": pump - base_pump})
    out = pd.DataFrame(rows)
    out.attrs.update(stress_day=s0, reactive_loss=base_loss, capacity_in_day=f.capacity, taw_in=f.taw)
    return out


def false_alarm_cost(f: Field, days: int) -> float:
    """Pumping for `days` when no dry spell comes (rain refills the profile, the water is lost)."""
    return days * f.capacity / f.efficiency * f.pump_cost_per_in


def field(**kw) -> Field:
    known = {k: v for k, v in kw.items() if k in {x.name for x in dataclasses.fields(Field)}}
    return Field(**known)
