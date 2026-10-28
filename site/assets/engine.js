// FlashWatch decision engine in the browser: a line-by-line port of flashwatch/decide/irrigation.py
// and flashwatch/decide/engine.py, so the farm's own settings recompute instantly.
// Keep the two in step: python -m flashwatch.site.parity prints the values and a console snippet to compare.
"use strict";

const FW = (() => {
  const GPM_TO_IN_DAY = 24 * 60 / 27154;
  const HORIZON = 14;

  const DEFAULTS = {
    corn: { awc_in_per_ft: 2.0, root_ft: 3.0, p: 0.55, et_peak: 0.30, gpm_per_acre: 5.0, efficiency: 0.85,
            ky: 1.5, season_et: 24.0, yield_bu: 220, price: 4.5, pump_cost_per_in: 10, spell_days: 35,
            reaction_days: 2, false_alarm_waste: 0.6, review_days: 7 },
    soybeans: { awc_in_per_ft: 2.0, root_ft: 3.0, p: 0.50, et_peak: 0.28, gpm_per_acre: 5.0, efficiency: 0.85,
                ky: 0.85, season_et: 22.0, yield_bu: 70, price: 11.0, pump_cost_per_in: 10, spell_days: 35,
                reaction_days: 2, false_alarm_waste: 0.6, review_days: 7 },
  };
  const SOILS = { "Sand": 1.0, "Sandy loam": 1.4, "Loam": 1.8, "Silt loam": 2.0, "Clay loam": 1.9 };

  const taw = f => f.awc_in_per_ft * f.root_ft;
  const capacity = f => f.gpm_per_acre * GPM_TO_IN_DAY * f.efficiency;

  function simulate(f, start) {
    const T = taw(f), R = f.p * T, cap = capacity(f);
    let dr = 0, shortfall = 0, pumped = 0;
    const depletion = [];
    for (let t = 0; t < f.spell_days; t++) {
      if (start !== null && t >= start) {
        const add = Math.min(cap, dr);
        dr -= add;
        pumped += add / f.efficiency;
      }
      const ks = dr <= R ? 1 : Math.max(0, (T - dr) / (T - R));
      const eta = ks * f.et_peak;
      dr = Math.min(T, dr + eta);
      shortfall += f.et_peak - eta;
      depletion.push(dr);
    }
    return { shortfall, pumped, depletion };
  }

  function stressDay(f) {
    const d = simulate(f, null).depletion;
    const i = d.findIndex(x => x > f.p * taw(f));
    return i < 0 ? f.spell_days : i;
  }

  // Avoided loss and extra pumping ($/acre) by days of notice, against the usual start.
  function curve(f, maxNotice = 21) {
    const s0 = stressDay(f), value = f.yield_bu * f.price;
    const run = start => {
      const r = simulate(f, Math.max(0, start));
      const loss = f.ky * r.shortfall / f.season_et * value;
      const netPumped = r.pumped - (taw(f) - r.depletion[r.depletion.length - 1]) / f.efficiency;
      return { loss, pump: netPumped * f.pump_cost_per_in, short: r.shortfall };
    };
    const base = run(s0 + f.reaction_days);
    const rows = [];
    for (let k = 0; k <= maxNotice; k++) {
      const r = run(s0 - k);
      rows.push({ notice: k, yieldLossPct: 100 * r.loss / value, avoided: base.loss - r.loss,
                  extraCost: r.pump - base.pump });
    }
    return { rows, stressDay: s0, reactiveLoss: base.loss, capacity: capacity(f), taw: taw(f),
             costNoEvent: f.review_days * capacity(f) / f.efficiency * f.pump_cost_per_in * f.false_alarm_waste };
  }

  function interp(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let i = 1;
    while (xs[i] < x) i++;
    const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
    return ys[i - 1] + t * (ys[i] - ys[i - 1]);
  }

  function netAt(c, k) {
    const xs = c.rows.map(r => r.notice);
    return interp(k, xs, c.rows.map(r => r.avoided)) - interp(k, xs, c.rows.map(r => r.extraCost));
  }

  function daily(p1, p2, days = HORIZON) {
    const out = new Array(days).fill(0);
    for (let d = 0; d < days; d++) out[d] = (d < 7 ? p1 : p2) / 7;
    const s = out.reduce((a, b) => a + b, 0);
    return s > 1 ? out.map(v => v / s) : out;
  }

  function evAct(c, pDay) {
    let gain = 0, tot = 0;
    pDay.forEach((p, i) => { gain += p * netAt(c, i + 1); tot += p; });
    return gain - (1 - tot) * c.costNoEvent;
  }

  function twoWay(spread, weightUp = 0.5) {
    const up = 1 + spread, down = (1 - weightUp * up) / (1 - weightUp);
    return [[weightUp, up], [1 - weightUp, Math.max(down, 0)]];
  }

  // Act today, or decide again in `step` days? Also: how high would the risk have to climb by then
  // for starting then to pay (the "what we're waiting for" number).
  function decide(c, pDay, scenarios = twoWay(0.5), step = 2) {
    const now = evAct(c, pDay);
    const rest = pDay.slice(step);
    const per = scenarios.map(([w, m]) => [w, m, evAct(c, rest.map(p => Math.min(1, p * m)))]);
    const wait = per.reduce((a, [w, , ev]) => a + w * Math.max(ev, 0), 0);
    const tot = pDay.reduce((a, b) => a + b, 0);
    const shape = tot > 0 ? pDay.map(p => p / tot) : pDay.map(() => 1 / pDay.length);
    let g = 0;
    shape.forEach((s, i) => { g += s * netAt(c, i + 1); });
    const breakeven = g + c.costNoEvent > 0 ? c.costNoEvent / (g + c.costNoEvent) : NaN;
    // Multiplier on today's risk profile at which acting on a forecast like that breaks even
    // (mirrors engine.risk_threshold): the fallback trigger.
    let lo = 0, hi = 50, mStar = NaN;
    if (evAct(c, pDay.map(p => Math.min(1, p * hi))) > 0) {
      for (let i = 0; i < 40; i++) {
        const mid = (lo + hi) / 2;
        if (evAct(c, pDay.map(p => Math.min(1, p * mid))) > 0) hi = mid; else lo = mid;
      }
      mStar = hi;
    }
    return { actNow: now > 0 && now >= wait, evNow: now, evWait: wait, breakeven, scenarios: per, mStar };
  }

  // The season played day by day for one cell (mirrors decide/backtest.py).
  function season(c, dates, p1, p2, onsets, opts = {}) {
    const { baseline = "careful", calendar = "07-01", first = "06-01", last = "08-31", spread = 0.5,
            review = 7, year = 2012, once = false } = opts;   // once: a one-off decision, act at most once
    const day = s => Date.UTC(year, +s.slice(0, 2) - 1, +s.slice(3, 5)) / 864e5;
    let ons = onsets.map(day);
    const cal = baseline === "calendar" ? day(calendar) : null;
    if (cal !== null) ons = ons.filter(d => d < cal);
    const events = [];
    let value = 0, blocked = -Infinity;
    dates.forEach((md, i) => {
      if (md < first || md > last || p1[i] === null || p2[i] === null) return;
      const t = day(md);
      if (t < blocked || (cal !== null && t >= cal)) return;
      const d = decide(c, daily(p1[i], p2[i]), twoWay(spread), 2);
      if (!d.actNow) return;
      const nxt = ons.find(o => o > t && o <= t + HORIZON);
      if (nxt !== undefined) {
        const k = nxt - t, v = netAt(c, k);
        value += v;
        events.push({ date: md, kind: "hit", notice: k, value: v });
        blocked = once ? Infinity : nxt + 30;
      } else {
        value -= c.costNoEvent;
        events.push({ date: md, kind: "false", value: -c.costNoEvent });
        blocked = once ? Infinity : t + review;
      }
    });
    const inSeason = ons.filter(o => o >= day(first)).length;
    return { value, events, onsets: inSeason, hits: events.filter(e => e.kind === "hit").length,
             falseAlarms: events.filter(e => e.kind === "false").length };
  }

  // The ensemble members as the next forecast's possible outcomes (mirrors decide_members), and the
  // weather split that tells the "start then" members from the rest (mirrors decide/trigger.py).
  function decideMembers(c, p1m, p2m, lam = 0.5, step = 2) {
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    const p1 = mean(p1m), p2 = mean(p2m);
    const base = decide(c, daily(p1, p2), [[1, 1]], step);
    const per = p1m.map((_, j) => {
      const q = daily(p1 + lam * (p1m[j] - p1), p2 + lam * (p2m[j] - p2));
      return [1 / p1m.length, j, evAct(c, q.slice(step))];
    });
    const wait = per.reduce((a, [w, , ev]) => a + w * Math.max(ev, 0), 0);
    return { actNow: base.evNow > 0 && base.evNow >= wait, evNow: base.evNow, evWait: wait, breakeven: base.breakeven, scenarios: per };
  }

  const TRIGGER_VARS = { rain_in: ["less than", "in of rain over the coming week", -1, 1],
                         temp_f: ["an average temperature above", "°F over the coming week", 1, 0] };

  function bestSplit(x, act, sign) {
    const xs = [...new Set(x)].sort((a, b) => a - b);
    let best = [NaN, 0];
    for (let i = 0; i + 1 < xs.length; i++) {
      const t = (xs[i] + xs[i + 1]) / 2;
      const agree = x.filter((v, j) => (sign * (v - t) > 0) === act[j]).length / x.length;
      if (agree > best[1]) best = [t, agree];
    }
    return best;
  }

  function trigger(c, p1m, p2m, weather, lam = 0.5, step = 2, minAgree = 0.9) {
    const d = decideMembers(c, p1m, p2m, lam, step);
    const actNext = d.scenarios.map(([, , ev]) => ev > 0);
    const out = { ...d, membersTipping: actNext.filter(Boolean).length, members: actNext.length, trigger: null };
    if (d.actNow || out.membersTipping === 0 || out.membersTipping === out.members) return out;
    let best = null;
    for (const [v, [pre, unit, sign, dec]] of Object.entries(TRIGGER_VARS)) {
      if (!weather[v]) continue;
      const [t, agree] = bestSplit(weather[v], actNext, sign);
      if (isFinite(t) && (!best || agree > best.agree)) best = { v, t, agree, text: `${pre} ${t.toFixed(dec)} ${unit}` };
    }
    if (best && best.agree >= minAgree) out.trigger = best;
    return out;
  }

  return { DEFAULTS, SOILS, simulate, stressDay, curve, netAt, daily, evAct, twoWay, decide, season, capacity,
           decideMembers, trigger };
})();
