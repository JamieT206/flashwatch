"""Act now, or wait for the next forecast? (DESIGN.md O1, O2)

An action is described by how much it is worth *by days of notice*: the loss it avoids if an onset
comes k days after acting, its extra cost in that case, and its cost if no onset comes (a false
alarm). FlashWatch gives the chance of an onset on each coming day (weekly chances spread evenly over
their days). The baseline is the farmer's usual practice without FlashWatch, worth 0 by definition.

Waiting `step` days buys a fresher forecast but loses notice, and onsets in the meantime are handled
the usual way. The fresher forecast is described by scenarios: (weight, risk multiplier) pairs whose
weighted mean multiplier is 1, e.g. from groups of ECMWF ensemble members (DESIGN.md O2). Acting now
is right when its expected value is positive and beats the expected value of deciding again later.
Without effectiveness falling with notice, waiting would always win; without information arriving,
acting early would always win. The answer sits between the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Curve:
    notice: np.ndarray            # days of notice, increasing (0 = acting on the day of onset)
    avoided: np.ndarray           # $ of loss avoided if the onset comes after that much notice
    cost_event: np.ndarray        # $ extra cost of acting, in that case
    cost_noevent: float           # $ cost of acting when no onset comes within the forecast range

    def net(self, k) -> np.ndarray:
        k = np.asarray(k, float)
        return np.interp(k, self.notice, self.avoided) - np.interp(k, self.notice, self.cost_event)


def daily(p_week: dict, days: int = 14) -> np.ndarray:
    """Chance of onset on each of the next `days` days from weekly chances {"week1": p, ...}."""
    out = np.zeros(days)
    for w in range(1, days // 7 + 1):
        out[7 * (w - 1):7 * w] = p_week.get(f"week{w}", 0.0) / 7
    s = out.sum()
    return out / s if s > 1 else out           # weekly chances can overlap; never above 1 in total


def ev_act(c: Curve, p_day: np.ndarray) -> float:
    """Expected $ of acting today, against the usual practice."""
    p_day = np.asarray(p_day, float)
    notice = np.arange(1, len(p_day) + 1)       # an onset on day d gives d days of notice
    return float((p_day * c.net(notice)).sum() - (1 - p_day.sum()) * c.cost_noevent)


@dataclass
class Decision:
    act_now: bool
    ev_now: float
    ev_wait: float
    breakeven: float                            # chance of onset (next 14 days) at which acting today breaks even
    scenarios: list = field(default_factory=list)   # (weight, multiplier, ev of acting at the next decision)


def decide(c: Curve, p_day, scenarios=((1.0, 1.0),), step: int = 3) -> Decision:
    p_day = np.asarray(p_day, float)
    now = ev_act(c, p_day)
    rest = p_day[step:]
    per = [(w, m, ev_act(c, np.clip(rest * m, 0, 1))) for w, m in scenarios]
    wait = sum(w * max(ev, 0.0) for w, m, ev in per)
    # Break-even for acting today, keeping the shape of p_day.
    shape = p_day / p_day.sum() if p_day.sum() > 0 else np.full(len(p_day), 1 / len(p_day))
    gain = float((shape * c.net(np.arange(1, len(p_day) + 1))).sum())
    be = c.cost_noevent / (gain + c.cost_noevent) if gain + c.cost_noevent > 0 else np.nan
    return Decision(act_now=now > 0 and now >= wait, ev_now=now, ev_wait=wait, breakeven=be, scenarios=per)


def two_way(spread: float, weight_up: float = 0.5) -> list[tuple[float, float]]:
    """Placeholder scenarios until the ensemble groups are wired in: risk goes up by (1 + spread)
    with probability weight_up, otherwise down so the mean multiplier stays 1."""
    up = 1 + spread
    down = (1 - weight_up * up) / (1 - weight_up)
    return [(weight_up, up), (1 - weight_up, max(down, 0.0))]


def decide_members(c: Curve, p1m, p2m, lam: float = 0.5, step: int = 3) -> Decision:
    """Like decide(), with the next forecast's possible outcomes taken from the ensemble members:
    member j's risk, pulled toward the ensemble mean by (1 - lam) because the next forecast resolves
    only part of the spread. scenarios: (weight, member index, ev of acting at the next decision)."""
    p1m, p2m = np.asarray(p1m, float), np.asarray(p2m, float)
    p1, p2 = p1m.mean(), p2m.mean()
    base = decide(c, daily({"week1": p1, "week2": p2}), ((1.0, 1.0),), step)
    per = []
    for j in range(len(p1m)):
        q = daily({"week1": p1 + lam * (p1m[j] - p1), "week2": p2 + lam * (p2m[j] - p2)})
        per.append((1 / len(p1m), j, ev_act(c, q[step:])))
    wait = sum(w * max(ev, 0.0) for w, _, ev in per)
    return Decision(act_now=base.ev_now > 0 and base.ev_now >= wait, ev_now=base.ev_now, ev_wait=wait,
                    breakeven=base.breakeven, scenarios=per)
