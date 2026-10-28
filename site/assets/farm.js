// FlashWatch farm page: one location, one season, one decision a day.
"use strict";

const S = {
  meta: null, defaults: null, tileCache: {}, cell: null, rec: null, year: 2012, i: 0,
  lat: 40.87, lon: -97.99,            // York County, Nebraska: centre-pivot country
  over: {}, farm: null, cty: null, mode: "today", live: null, allYears: {},
  map: null, marker: null, mapMode: "street", droughtLayer: null, regionRecs: {},
};
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (p, d = 0) => p === null || p === undefined || !isFinite(p) ? "-" : (100 * p).toFixed(d) + "%";
const usd = (v, d = 0) => (v < 0 ? "-$" : "+$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const USDM = { "-1": "None", 0: "D0 abnormally dry", 1: "D1 moderate drought", 2: "D2 severe drought",
               3: "D3 extreme drought", 4: "D4 exceptional drought" };
const SEASON = ["06-01", "08-31"];
const DEMO_TODAY = { year: 2012, date: "07-03", demo: true };

// ---------- the farm: local starting values, the farmer's own changes on top ----------

// [key, label, type, options/step, shown for: "all" | "irrigated"]
const FARM_FIELDS = [
  ["livestock", "Livestock on the farm", "select", ["no", "yes"], "all"],
  ["crop", "Crop", "select", ["corn", "soybeans"], "all"],
  ["acres", "Acres", "number", 1, "all"],
  ["gpm_per_acre", "Pivot capacity (gpm per acre)", "number", 0.1, "irrigated"],
  ["soil", "Soil", "select", Object.keys(FW.SOILS), "all"],
  ["yield_bu", "Expected yield (bu/acre)", "number", 1, "all"],
  ["price", "Expected price ($/bu)", "number", 0.05, "all"],
  ["pump_cost_per_in", "Pumping cost ($ per acre-inch)", "number", 0.5, "irrigated"],
  ["reaction_days", "Usual start (days after stress shows)", "number", 1, "irrigated"],
  ["spell_days", "Dry spell to plan for (days)", "number", 1, "irrigated"],
  ["calendar", "Calendar start date (MM-DD)", "text", null, "irrigated"],
];
// Local starting values and where each comes from (site/data/defaults.json, flashwatch/site/defaults.py).
function localValues() {
  const crop = S.over.crop || "corn", watering = S.over.watering || "pivot-irrigated";
  const D = S.defaults || {}, st = S.cty && D.states ? D.states[S.cty[2]] : null;
  const cy = S.cty && D.counties ? D.counties[S.cty[0]] : null;
  const fx = k => D.fixed && D.fixed[k] ? D.fixed[k] : null;
  const prac = watering === "rain-fed" ? "rainfed" : "irrigated";
  const yr = D.yield_years ? `${D.yield_years[0]}-${String(D.yield_years[1]).slice(2)}` : "";
  const v = { ...FW.DEFAULTS[crop], ...RF.DEFAULTS, crop, watering, livestock: "no", soil: "Silt loam", spell_days: 21, calendar: "07-01",
              acres: watering === "rain-fed" ? (crop === "corn" ? 300 : 300) : 130 };
  const src = { ...RF.SOURCES, acres: watering === "rain-fed" ? "Average corn farm in the region (2022 Census)" : "One quarter-section pivot", gpm_per_acre: "Typical Nebraska pivot", soil: "Pick your soil",
                spell_days: "Average flash drought (county yields)", reaction_days: "Typical practice",
                calendar: "A fixed start some farms use", hay_tons: "Example herd", hay_carry: "Estimate",
                watering: "", crop: "", livestock: "Shows hay, feed and silage decisions" };
  const set = (k, val, s) => { if (val !== null && val !== undefined && isFinite(val)) { v[k] = val; src[k] = s; } };
  if (cy && cy[crop]) set("yield_bu", cy[crop][prac], `USDA, ${S.cty[1]} ${S.cty[2]}, ${yr} average`);
  if (st) {
    set("price", crop === "corn" ? st.corn_price : st.soy_price, `USDA, ${S.cty[2]} ${D.price_year} average`);
    set("hay_price", st.hay_price, `USDA, ${S.cty[2]} ${D.price_year} average`);
    set("hay_rise", st.hay_rise, `USDA, ${S.cty[2]} rise in 2012`);
  }
  for (const k of ["n_price", "pump_cost_per_in"]) if (fx(k)) set(k, fx(k).value, fx(k).source);
  for (const k of Object.keys(v)) if (!(k in src)) src[k] = "Typical value";
  return { v, src };
}

function farm() { return { ...localValues().v, ...S.over }; }
function loadOver() {
  try { S.over = JSON.parse(localStorage.getItem("fw.over") || "{}") || {}; } catch (e) { S.over = {}; }
}
function saveOver() { try { localStorage.setItem("fw.over", JSON.stringify(S.over)); } catch (e) { /* ignore */ } }
const irrigated = () => S.farm.watering !== "rain-fed";
function params() { return { ...S.farm, awc_in_per_ft: FW.SOILS[S.farm.soil] ?? 2.0 }; }

// ---------- drought decisions other than irrigation (site/assets/rainfed.js) ----------

function decisions() {
  return irrigated() ? RF.forFarm(S.farm, S.farm.livestock === "yes").filter(a => a.key === "hay")
                     : RF.forFarm(S.farm, S.farm.livestock === "yes");
}
// One decision on one date: act / wait / calm, outside its window ("off"), or already in drought.
function decisionAt(a, i, r = S.rec) {
  const md = S.meta.dates[i], p1 = r.p1[i], p2 = r.p2[i], c = RF.curve(a, S.farm);
  const be = c.cost > 0 ? c.cost / (c.gain + c.cost) : 0;
  if (p1 === null || p2 === null) {
    const dry = r.rz[i] === null && !(r.fz || []).includes(i);
    return { c, be, chance: null, state: !RF.inWindow(a, md) ? "off" : dry ? "drought" : "none" };
  }
  const d = FW.decide(c, FW.daily(p1, p2), FW.twoWay(0.5), 2), chance = Math.min(1, p1 + p2);
  if (!RF.inWindow(a, md)) return { c, d, be, chance, state: "off" };
  return { c, d, be, chance, state: d.actNow ? "act" : d.evWait > 0 ? "wait" : "calm" };
}
function decisionSeason(a, r, year, upTo = null) {
  const c = RF.curve(a, S.farm), dts = S.meta.dates;
  const cut = x => upTo === null ? x : x.map((v, i) => i <= upTo ? v : null);
  const ons = upTo === null ? r.onsets : r.onsets.filter(o => o <= dts[upTo]);
  return FW.season(c, dts, cut(r.p1), cut(r.p2), ons, { year, first: a.window[0], last: a.window[1], once: a.key !== "hay" });
}

// ---------- geography ----------

function cellLatLon([iy, ix]) { const m = S.meta; return [m.lat0 - iy * m.res, m.lon0 + ix * m.res]; }
function nearestCell(lat, lon) {
  let best = null, bd = Infinity;
  for (const c of S.meta.cells) {
    const [a, o] = cellLatLon(c);
    const d = Math.hypot(a - lat, (o - lon) * Math.cos(lat * Math.PI / 180));
    if (d < bd) { bd = d; best = c; }
  }
  return bd * 111 < 60 ? best : null;
}
function nearestCounty(lat, lon) {
  let best = null, bd = Infinity;
  for (const c of S.meta.counties || []) {
    const d = Math.hypot(c[3] - lat, (c[4] - lon) * Math.cos(lat * Math.PI / 180));
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}
async function tileFile(year, ty, tx) {
  const key = `${year}/${ty}_${tx}`;
  if (!S.tileCache[key]) S.tileCache[key] = fetch(`data/tiles/${key}.json`).then(r => r.ok ? r.json() : []).catch(() => []);
  return S.tileCache[key];
}
async function loadTile(year, cell) {
  const m = S.meta, recs = await tileFile(year, Math.floor(cell[0] / m.thin / m.tile), Math.floor(cell[1] / m.thin / m.tile));
  return recs.find(r => r.iy === cell[0] && r.ix === cell[1]);
}

// ---------- dates and modes ----------

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("lat")) S.lat = +h.get("lat");
  if (h.has("lon")) S.lon = +h.get("lon");
  if (h.has("year")) S.year = +h.get("year");
  if (h.has("date")) S.wantDate = h.get("date");
  S.mode = h.get("mode") === "today" ? "today" : (h.has("year") || h.has("date") || h.get("mode") === "replay") ? "replay" : "today";
}
function writeHash() {
  const md = S.meta.dates[S.i];
  const tail = S.mode === "today" ? "mode=today" : `mode=replay&year=${S.year}&date=${md}`;
  history.replaceState(null, "", `#lat=${S.lat.toFixed(2)}&lon=${S.lon.toFixed(2)}&${tail}`);
}
function fmtDate(md, year = S.year) {
  return new Date(Date.UTC(year, +md.slice(0, 2) - 1, +md.slice(3, 5)))
    .toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}
function dateOf(md) { return new Date(Date.UTC(S.year, +md.slice(0, 2) - 1, +md.slice(3, 5))); }
const inSeason = md => md >= SEASON[0] && md <= SEASON[1];

async function loadLive() {
  try {
    const r = await fetch(`data/live.json?t=${Date.now()}`);
    if (r.ok) return await r.json();
  } catch (e) { /* no feed */ }
  return DEMO_TODAY;
}
function todayIndex() {
  const d = S.meta.dates;
  let i = 0;
  while (i + 1 < d.length && d[i + 1] <= S.live.date) i++;
  return i;
}

// Today mode follows the feed: re-read it after local midnight and whenever the tab comes back.
let lastDay = new Date().toDateString();
function scheduleRefresh() {
  const now = new Date(), next = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1, 0, 5);
  setTimeout(async () => { await refreshToday(); scheduleRefresh(); }, next - now);
}
async function refreshToday() {
  lastDay = new Date().toDateString();
  if (S.mode !== "today") return;
  S.live = await loadLive();
  S.year = S.live.year;
  S.rec = await loadTile(S.year, S.cell);
  S.i = todayIndex();
  renderAll();
}
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && new Date().toDateString() !== lastDay) refreshToday();
});

async function setMode(mode) {
  S.mode = mode;
  $("modeToday").setAttribute("aria-pressed", mode === "today");
  $("modeReplay").setAttribute("aria-pressed", mode === "replay");
  $("yearBox").hidden = mode === "today";
  if (mode === "today") {
    S.live = await loadLive();
    S.year = S.live.year;
    S.i = todayIndex();
  }
  $("year").value = S.year;
  if (S.cell) { S.rec = await loadTile(S.year, S.cell); renderAll(); }
}

// ---------- advice for one date ----------

function adviceAt(i, c, full = false) {
  const r = S.rec, md = S.meta.dates[i], p1 = r.p1[i], p2 = r.p2[i];
  if (p1 === null || p2 === null) {
    if ((r.fz || []).includes(i)) return { cls: "frozen", big: "No forecast: frozen ground", sub: "",
      why: "Snow or frozen soil. Forecasts start once the ground thaws." };
    if (r.rz[i] === null) return { cls: "drought", big: "In drought now", sub: "Your soil is already in drought",
      why: `Soil moisture is below the 20th percentile for the date, so there is no flash drought left to warn about.
            ${irrigated() ? "This is when irrigation matters most: keep up with the crop, with the pivot running full time in hot weather."
              : "The drought decisions below show what to do now."} Warnings start again when the soil recovers.` };
    return { cls: "none", big: "No forecast for this date", sub: "", why: "" };
  }
  if (!irrigated()) return rainfedAdvice(i, full);
  if (!inSeason(md)) return { cls: "off", big: "Outside the irrigation season", sub: "",
    why: "Irrigation advice runs June to August. The risk numbers still apply." };
  // With ensemble members for this date, the next forecast's outcomes come from them, and the
  // trigger is named in weather; otherwise a placeholder spread and a risk threshold.
  const mem = r.mp1 && r.mp1[i] ? FW.trigger(c, r.mp1[i], r.mp2[i], { rain_in: r.mr[i], temp_f: r.mt[i] }) : null;
  const d = mem || FW.decide(c, FW.daily(p1, p2), FW.twoWay(0.5), 2);
  const a = { d, ev: true, cls: d.actNow ? "act" : d.evWait > 0 ? "wait" : "calm" };
  if (!full) return a;
  const per = S.farm.acres, total = v => `${usd(v * per)} on ${per} acres`;
  const nextRisk = (d.mStar ?? NaN) * p1;
  const nextLine = v => `if FlashWatch's next forecast (in 2 days) puts the chance of a flash drought in the next 7 days at
    <strong>${pct(Math.min(v, 1))}</strong> or more (today: ${pct(p1)})`;
  if (a.cls === "act") {
    const k = 4, perDay = FW.netAt(c, k) - FW.netAt(c, k - 1);
    return { ...a, big: "Start extra irrigation today", sub: "Flash-drought risk high",
      why: `A flash drought is <strong>${(p1 / (r.n1[i] || 1e-9)).toFixed(1)} times</strong> as likely as normal this week.
            Run the pivot full time: it puts down ${c.capacity.toFixed(2)} in a day, and the crop uses up to ${S.farm.et_peak.toFixed(2)} in a heat wave.
            Keep going until about an inch of rain falls or the risk drops back to normal. On average: ${total(d.evNow)}.`,
      trigger: `<strong>Why not wait?</strong> Each day of warning is worth about $${perDay.toFixed(0)} an acre on your field right now.` };
  }
  if (a.cls === "wait") {
    return { ...a, big: "Not yet: keep checking forecasts", sub: "Flash-drought risk rising",
      why: d.evNow > 0
        ? `Starting today pays a little (${usd(d.evNow, 1)} an acre). Waiting two days for a fresher forecast pays more: you lose a little warning but may avoid pumping for nothing.`
        : `Starting today would cost more than it saves (${usd(d.evNow, 1)} an acre), but the next forecast could change that.`,
      trigger: mem && mem.trigger
        ? `<strong>What to watch for:</strong> if FlashWatch's next forecast shows <strong>${mem.trigger.text}</strong>, start the pivot then.
           ${mem.membersTipping} of today's ${mem.members} ECMWF forecast runs look like that.`
        : mem ? `<strong>What to watch for:</strong> ${mem.membersTipping} of today's ${mem.members} ECMWF forecast runs would make starting pay, but no single rain or heat level separates them.`
        : isFinite(nextRisk) && nextRisk < 1 ? `<strong>What would change it:</strong> ${nextLine(nextRisk)}, start the pivot then. You'll see it here and in the morning text.` : "" };
  }
  return { ...a, big: "Irrigate as usual", sub: "Flash-drought risk low",
    why: `The chance of a flash drought this week is ${pct(p1)} (normal for the date: ${pct(r.n1[i])}). Extra irrigation would cost more in pumping than it saves.`,
    trigger: isFinite(nextRisk) && nextRisk < 1 ? `<strong>What would change it:</strong> ${nextLine(nextRisk)}.` : "" };
}

function rainfedAdvice(i, full) {
  const all = decisions().map(a => ({ a, s: decisionAt(a, i) }));
  const live = all.filter(o => o.s.state !== "off");
  const now = live.filter(o => o.s.state === "act" || o.s.state === "drought"), wait = live.filter(o => o.s.state === "wait");
  const cls = now.length ? "act" : wait.length ? "wait" : live.length ? "calm" : "off";
  if (!full) return { cls };
  const r = S.rec, md = S.meta.dates[i];
  const next = all.filter(o => o.a.window[0] > md).sort((x, y) => x.a.window[0] < y.a.window[0] ? -1 : 1)[0];
  const risk = r.p1[i] === null ? "" : `The chance of a flash drought this week is ${pct(r.p1[i])} (normal for the date: ${pct(r.n1[i])}). `;
  if (!live.length) return { cls: "off", big: "No drought decisions due", sub: next ? `Next: ${next.a.title.toLowerCase()}, from ${fmtShort(next.a.window[0])}` : "",
    why: risk + "Each decision below shows when it's made and what it's worth.", trigger: "" };
  return { cls, big: now.length ? now.map(o => o.a.act).join(" · ") : wait.length ? "Not yet: keep checking forecasts" : "Keep your plans",
    sub: cls === "calm" ? "Flash-drought risk low" : cls === "act" ? (now.some(o => o.s.state === "drought") ? "Already in drought" : "Flash-drought risk high") : "Flash-drought risk rising",
    why: risk + "Details and the numbers behind each decision are below.", trigger: "" };
}
const fmtShort = md => new Date(Date.UTC(2001, +md.slice(0, 2) - 1, +md.slice(3, 5))).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

// ---------- rendering ----------

function renderAll() {
  if (!S.rec) return;
  S.farm = farm();
  const c = FW.curve(params(), 21);
  renderNote();
  renderOps();
  renderDecisions();
  renderVerdict(c);
  renderStrip(c);
  renderTiles();
  renderSeason(c);
  renderValue(c);
  renderNotice(c);
  renderMapState();
  $("wIrr").setAttribute("aria-pressed", irrigated());
  $("wRain").setAttribute("aria-pressed", !irrigated());
  const demo = S.mode === "today" && S.live && S.live.demo;
  $("banner").hidden = !(demo || S.mode === "replay");
  $("banner").innerHTML = demo
    ? `<strong>Demo.</strong> "Today" is ${fmtDate(S.live.date, S.live.year)}, replayed from the archive: the live feed isn't connected yet.`
    : `Replaying the ${S.year} season, one forecast at a time.`;
  writeHash();
}

function renderVerdict(c) {
  const a = adviceAt(S.i, c, true);
  $("verdict").className = "verdict " + a.cls;
  $("vQ").firstChild.textContent = irrigated() ? "Extra irrigation today? " : "Drought action today? ";
  $("vBig").textContent = a.big;
  $("vSub").textContent = a.sub || "";
  $("vWhy").innerHTML = a.why;
  const ev = a.ev && irrigated();
  $("evBox").hidden = !ev; $("evNote").hidden = !ev;
  if (ev) {
    $("evNow").innerHTML = `${usd(a.d.evNow, 1)} <small>an acre</small>`;
    $("evWait").innerHTML = `${usd(a.d.evWait, 1)} <small>an acre</small>`;
  }
  $("trigger").innerHTML = a.trigger || "";
  $("trigger").hidden = !a.trigger;
}

// The date bar: before June, each forecast date of the irrigation season, after August; coloured by
// the advice on that date. In Today mode it shows the season so far and can't be moved.
function stops() {
  const d = S.meta.dates, out = [];
  const pre = d.map((m, i) => [m, i]).filter(([m]) => m < SEASON[0]), post = d.map((m, i) => [m, i]).filter(([m]) => m > SEASON[1]);
  if (pre.length) out.push({ kind: "pre", i: pre[pre.length - 1][1], has: i => d[i] < SEASON[0] });
  d.forEach((m, i) => { if (inSeason(m)) out.push({ kind: "day", i, has: j => j === i }); });
  if (post.length) out.push({ kind: "post", i: post[0][1], has: i => d[i] > SEASON[1] });
  return out;
}
function renderStrip(c) {
  const st = stops(), el = $("strip"), live = S.mode === "today";
  const cur = st.findIndex(s => s.has(S.i));
  el.classList.toggle("locked", live);
  el.setAttribute("aria-disabled", live);
  el.setAttribute("aria-valuemin", 0); el.setAttribute("aria-valuemax", st.length - 1); el.setAttribute("aria-valuenow", cur);
  el.setAttribute("aria-valuetext", fmtDate(S.meta.dates[S.i]));
  el.innerHTML = st.map((s, k) => {
    const future = live && s.i > S.i && !s.has(S.i);
    const cls = s.kind === "day" && !future ? adviceAt(s.i, c).cls : s.kind === "day" ? "future" : "off";
    const label = s.kind === "pre" ? "Before June" : s.kind === "post" ? "After August" : fmtDate(S.meta.dates[s.i]);
    const md = S.meta.dates[s.i], prev = k > 0 ? S.meta.dates[st[k - 1].i] : "";
    const month = s.kind === "day" && (st[k - 1].kind !== "day" || md.slice(0, 2) !== prev.slice(0, 2))
      ? `<span class="mlab">${dateOf(md).toLocaleDateString("en-US", { month: "short", timeZone: "UTC" })}</span>` : "";
    return `<div class="seg-${cls}${s.kind !== "day" ? " end" : ""}${k === cur ? " on" : ""}" data-s="${k}" title="${label}">${s.kind === "pre" ? "Before June" : s.kind === "post" ? "After Aug" : ""}${month}</div>`;
  }).join("");
  $("stripLegend").innerHTML = `<span class="d">${fmtDate(S.meta.dates[S.i])}</span>` +
    (irrigated() ? `<span class="keys"><b class="k-act"></b>Start extra irrigation <b class="k-wait"></b>Not yet <b class="k-calm"></b>As usual <b class="k-drought"></b>In drought</span>`
                 : `<span class="keys"><b class="k-act"></b>Act <b class="k-wait"></b>Not yet <b class="k-calm"></b>Keep your plans <b class="k-drought"></b>In drought</span>`);
  if (!live) el.querySelectorAll("[data-s]").forEach(div => { div.onclick = () => setDate(st[+div.dataset.s].i); });
}

function renderTiles() {
  const r = S.rec, i = S.i;
  const ratio = (p, n) => p === null || !n ? "" : `${(p / n).toFixed(1)} times normal (${pct(n, 1)})`;
  $("t1").textContent = pct(r.p1[i], 1);
  $("t1x").textContent = r.p1[i] === null ? (r.rz[i] === null ? "already in drought" : "") : ratio(r.p1[i], r.n1[i]);
  $("t1x").className = "x" + (r.p1[i] !== null && r.p1[i] < 1.5 * r.n1[i] ? " low" : "");
  $("t2").textContent = pct(r.p2[i], 1);
  $("t2x").textContent = r.p2[i] === null ? "" : ratio(r.p2[i], r.n2[i]);
  $("t2x").className = "x" + (r.p2[i] !== null && r.p2[i] < 1.5 * r.n2[i] ? " low" : "");
  $("t3").textContent = r.rz[i] === null ? "below 20" : `${r.rz[i].toFixed(0)}`;
  $("t3x").textContent = r.rz[i] === null ? "percentile: in drought" : "percentile for the date (below 20 is drought)";
  $("t4").textContent = USDM[r.usdm[i]] ?? "None";
}

// Contiguous runs of dates where test(i) holds, as [first, last] index pairs.
function runs(n, test) {
  const out = [];
  for (let i = 0; i < n; i++) if (test(i)) { let j = i; while (j + 1 < n && test(j + 1)) j++; out.push([i, j]); i = j; }
  return out;
}

// A line coloured one way above `at` and another below, with the "bad" side shaded.
function splitLine(svg, pts, x, y, at, top, bottom, cAbove, cBelow, id) {
  const line = d3.line().defined(p => p[1] !== null).x(p => x(p[0])).y(p => y(p[1]));
  const yv = y(at);
  const clip = (name, y0, h) => svg.append("clipPath").attr("id", id + name).append("rect").attr("x", 0).attr("width", 1e4).attr("y", y0).attr("height", h);
  clip("a", top, yv - top); clip("b", yv, bottom - yv);
  svg.append("path").attr("d", line(pts)).attr("fill", "none").attr("stroke", cAbove).attr("stroke-width", 2.2).attr("clip-path", `url(#${id}a)`);
  svg.append("path").attr("d", line(pts)).attr("fill", "none").attr("stroke", cBelow).attr("stroke-width", 2.6).attr("clip-path", `url(#${id}b)`);
}

// In Today mode the season so far: nothing after today is shown.
function sofar(r) {
  if (S.mode !== "today") return r;
  const cut = a => a.map((v, i) => i <= S.i ? v : null);
  return { ...r, p1: cut(r.p1), p2: cut(r.p2), n1: r.n1, rz: cut(r.rz), usdm: r.usdm.map((u, i) => i <= S.i ? u : -1),
           onsets: r.onsets.filter(o => o <= S.meta.dates[S.i]), fz: (r.fz || []).filter(i => i <= S.i), future: true };
}

function renderSeason(c) {
  const r = sofar(S.rec), dates = S.meta.dates.map(dateOf), n = dates.length;
  const W = 1100, H = 210, m = { l: 48, r: 12, t: 10, b: 24 };
  const x = d3.scaleUtc().domain(d3.extent(dates)).range([m.l, W - m.r]);
  S.x = x;
  const ymax = Math.max(0.08, d3.max(r.p1.concat(r.n1).filter(v => v !== null)) * 1.15);
  const y = d3.scaleLinear().domain([0, ymax]).range([H - m.b, m.t]).nice();
  const svg = d3.select("#seasonChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const band = (a, b, fill, op, label, h0 = m.t, h1 = H - m.b, s = svg) => {
    const x0 = x(dates[a]) - (a > 0 ? (x(dates[a]) - x(dates[a - 1])) / 2 : 0);
    const x1 = x(dates[b]) + (b < n - 1 ? (x(dates[b + 1]) - x(dates[b])) / 2 : 0);
    s.append("rect").attr("x", x0).attr("width", x1 - x0).attr("y", h0).attr("height", h1 - h0).attr("fill", fill).attr("opacity", op);
    if (label && x1 - x0 > 60) s.append("text").attr("x", (x0 + x1) / 2).attr("y", h0 + 14).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", css("--muted")).text(label);
  };
  svg.append("rect").attr("x", x(dateOf(SEASON[0]))).attr("width", x(dateOf(SEASON[1])) - x(dateOf(SEASON[0])))
    .attr("y", m.t).attr("height", H - m.t - m.b).attr("fill", css("--calm-bg")).attr("opacity", .5);
  // Gaps: no forecast because the soil is already in drought, or the ground is frozen.
  const fz = new Set(r.fz || []);
  const past = i => !r.future || i <= S.i;
  const dr = runs(n, i => past(i) && r.p1[i] === null && r.rz[i] === null && !fz.has(i));
  const fr = runs(n, i => r.p1[i] === null && fz.has(i));
  for (const [a, b] of dr) band(a, b, css("--drought"), 0.16, "in drought");
  for (const [a, b] of fr) band(a, b, css("--normal"), 0.2, "frozen");
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7).tickFormat(d3.utcFormat("%b %-d")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")));
  const line = key => d3.line().defined(d => r[key][d] !== null).x(d => x(dates[d])).y(d => y(r[key][d]));
  const idx = d3.range(n);
  svg.append("path").attr("d", line("n1")(idx)).attr("fill", "none").attr("stroke", css("--normal")).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
  svg.append("path").attr("d", line("p1")(idx)).attr("fill", "none").attr("stroke", css("--risk")).attr("stroke-width", 2.2);
  const shown = r.onsets;
  for (const o of shown) svg.append("path").attr("d", `M${x(dateOf(o))},${H - m.b} l-6,-10 l12,0 z`).attr("fill", css("--drought"));
  const ss = seasonValue(c, r);
  for (const e of ss.events) {
    const di = S.meta.dates.indexOf(e.date);
    svg.append("circle").attr("cx", x(dates[di])).attr("cy", y(r.p1[di] ?? 0)).attr("r", 6)
      .attr("fill", e.kind === "hit" ? css("--act") : css("--panel")).attr("stroke", css("--act")).attr("stroke-width", 2);
  }
  cursor(svg, x(dates[S.i]), m.t, H - m.b);
  clicker(svg, x, dates, m, W, H);
  $("riskLegend").innerHTML = [`<span><i style="background:var(--risk)"></i>FlashWatch forecast</span>`,
    `<span><i style="background:var(--normal)"></i>Normal for the date</span>`,
    shown.length ? `<span>&#9650; Flash drought began</span>` : "",
    irrigated() && ss.events.length ? `<span>&#9679; "Start extra irrigation" (hollow: no dry spell came)</span>` : "",
    dr.length ? `<span><i style="background:var(--drought);opacity:.3;height:10px"></i>Already in drought: no forecast needed</span>` : "",
    fr.length ? `<span><i style="background:var(--normal);opacity:.4;height:10px"></i>Frozen ground: no forecast</span>` : ""].join("");

  // Soil moisture: the drought zone shaded, the line turning purple inside it.
  const H2 = 150, y2 = d3.scaleLinear().domain([0, 100]).range([H2 - 40, m.t]);
  const s2 = d3.select("#soilChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H2}`);
  s2.append("rect").attr("x", m.l).attr("width", W - m.l - m.r).attr("y", y2(20)).attr("height", y2(0) - y2(20)).attr("fill", css("--drought")).attr("opacity", 0.12);
  s2.append("text").attr("x", W - m.r - 4).attr("y", y2(0) - 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", css("--drought")).text("drought (below 20)");
  s2.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y2).tickValues([0, 20, 50, 80, 100]));
  const rz = r.rz.map((v, k) => [dates[k], fz.has(k) || !past(k) ? null : v === null ? 12 : v]);   // no row = already below 20
  splitLine(s2, rz, x, y2, 20, m.t, y2(0), css("--soil"), css("--drought"), "sm");
  const anyDm = r.usdm.some(u => u >= 1);
  if (anyDm) {
    const bw = (W - m.l - m.r) / n + 1;
    r.usdm.forEach((u, k) => {
      if (u >= 1) s2.append("rect").attr("x", x(dates[k])).attr("width", bw).attr("y", H2 - 30).attr("height", 10)
        .attr("fill", css("--drought")).attr("opacity", 0.35 + 0.15 * u);
    });
  }
  s2.append("text").attr("x", m.l).attr("y", H2 - 6).attr("fill", css("--muted")).attr("font-size", 11)
    .text(anyDm ? "Drought Monitor" : "No D1 or worse drought on the Drought Monitor this season");
  cursor(s2, x(dates[S.i]), m.t, H2 - 18);
  clicker(s2, x, dates, m, W, H2);
  $("soilLegend").innerHTML = `<span><i style="background:var(--soil)"></i>Soil moisture (0-40 cm), percentile for the date</span>` +
    (anyDm ? `<span><i style="background:var(--drought);height:10px"></i>Drought Monitor D1 or worse</span>` : "");
  renderTally(ss, c);
}

function cursor(svg, cx, y0, y1) {
  svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", y0).attr("y2", y1).attr("stroke", css("--ink")).attr("stroke-width", 1).attr("opacity", .5);
}
function clicker(svg, x, dates, m, W, H) {
  if (S.mode === "today") return;
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b)
    .attr("fill", "transparent").style("cursor", "pointer")
    .on("click", ev => { const [px] = d3.pointer(ev); setDate(d3.leastIndex(dates, d => Math.abs(x(d) - px))); });
}

// Value of extra irrigation today, date by date through the irrigation season.
function renderValue(c) {
  $("valueBlock").hidden = !irrigated();
  if (!irrigated()) return;
  const r = S.rec, dates = S.meta.dates.map(dateOf), x = S.x;
  const W = 1100, H = 150, m = { l: 48, r: 12, t: 10, b: 24 };
  const last = S.mode === "today" ? S.i : dates.length - 1;
  const pts = S.meta.dates.map((md, i) => {
    if (!inSeason(md) || i > last || r.p1[i] === null || r.p2[i] === null) return [dates[i], null];
    return [dates[i], adviceAt(i, c).d.evNow];
  });
  const vals = pts.map(p => p[1]).filter(v => v !== null);
  const y = d3.scaleLinear().domain([Math.min(-5, d3.min(vals) ?? -5), Math.max(5, d3.max(vals) ?? 5)]).range([H - m.b, m.t]).nice();
  const svg = d3.select("#valueChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("rect").attr("x", m.l).attr("width", W - m.l - m.r).attr("y", m.t).attr("height", y(0) - m.t).attr("fill", css("--act-bg"));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7).tickFormat(d3.utcFormat("%b %-d")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(v => (v < 0 ? "-$" : "$") + Math.abs(v)));
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", css("--muted"));
  splitLine(svg, pts, x, y, 0, m.t, H - m.b, css("--act"), css("--normal"), "vl");
  cursor(svg, x(dates[S.i]), m.t, H - m.b);
  clicker(svg, x, dates, m, W, H);
}

function seasonValue(c, r, year = S.year, upTo = S.mode === "today" ? S.i : null) {
  const d = S.meta.dates;
  const cut = (a) => upTo === null ? a : a.map((v, i) => i <= upTo ? v : null);
  const ons = upTo === null ? r.onsets : r.onsets.filter(o => o <= d[upTo]);
  return FW.season(c, d, cut(r.p1), cut(r.p2), ons, { year, baseline: "careful" });
}

async function renderTally(ss, c) {
  const acres = S.farm.acres, today = S.mode === "today";
  if (!irrigated()) { await renderRainfedTally(); return; }
  const cal = FW.season(FW.curve({ ...params(), reaction_days: 7 }, 21), S.meta.dates, S.rec.p1, S.rec.p2, S.rec.onsets,
                        { year: S.year, baseline: "calendar", calendar: S.farm.calendar });
  const row = (title, v, sub, big = true) => `<div class="row"><div class="note">${title}</div><div class="v ${v >= 0 ? "pos" : "neg"}${big ? "" : " small"}">${usd(v)} an acre</div><div class="note">${sub}</div></div>`;
  const many = await multiYear(c);
  $("seasonTally").innerHTML =
    row(today ? "This season so far" : `The ${S.year} season`, ss.value,
        `${usd(ss.value * acres)} on ${acres} acres. Caught ${ss.hits} of ${ss.onsets} dry spells early; ${ss.falseAlarms} false alarm${ss.falseAlarms === 1 ? "" : "s"}.`) +
    (many ? row(`Average of ${many.n} past seasons here`, many.mean, `${usd(many.mean * acres)} a year on ${acres} acres. Best year ${usd(many.best)} an acre (${many.bestYear}).`) : "") +
    (today ? "" : `<div class="row wide"><div class="note">Compared with starting 2 days after stress shows. Against a fixed ${S.farm.calendar} start instead: <strong>${usd(cal.value)} an acre</strong>.</div></div>`);
  $("events").innerHTML = ss.events.length ? ss.events.map(e => e.kind === "hit"
    ? `<li>${fmtDate(e.date)}: start extra irrigation. A flash drought began ${e.notice} day${e.notice === 1 ? "" : "s"} later (${usd(e.value)} an acre).</li>`
    : `<li>${fmtDate(e.date)}: start extra irrigation. No dry spell came (${usd(e.value)} an acre of pumping).</li>`).join("")
    : `<li>No extra irrigation ${today ? "so far" : "this summer"}.${ss.onsets ? ` ${ss.onsets} dry spell${ss.onsets > 1 ? "s" : ""} came without a warning.` : ""}</li>`;
}

// Rain-fed season value: the field decisions per acre, and hay for the farm, this season and on average.
async function renderRainfedTally() {
  const list = decisions(), acres = S.farm.acres, today = S.mode === "today";
  const field = list.filter(a => a.per !== "for the farm"), hay = list.find(a => a.key === "hay");
  const total = (r, y, upTo = null) => {
    const f = d3.sum(field, a => decisionSeason(a, r, y, upTo).value);
    return { field: f, hay: hay ? decisionSeason(hay, r, y, upTo).value : 0 };
  };
  const now = total(S.rec, S.year, today ? S.i : null);
  const ys = S.meta.years, recs = await Promise.all(ys.map(y => loadTile(y, S.cell)));
  const all = recs.map((r, k) => r ? total(r, ys[k]) : null).filter(Boolean);
  const mean = { field: d3.mean(all, v => v.field), hay: d3.mean(all, v => v.hay) };
  const row = (title, v) => `<div class="row"><div class="note">${title}</div><div class="v ${v.field >= 0 ? "pos" : "neg"}">${usd(v.field)} an acre</div>
    <div class="note">${usd(v.field * acres)} on ${acres} acres${hay ? `, plus ${usd(v.hay)} from buying hay early` : ""}.</div></div>`;
  $("seasonTally").innerHTML = row(today ? "This season so far" : `The ${S.year} season`, now) +
    (all.length ? row(`Average of ${all.length} past seasons here`, mean) : "") +
    `<div class="row wide"><div class="note">From the decisions below (${list.map(a => a.title.toLowerCase()).join(", ")}), each made at most once a season, compared with your usual plan.</div></div>`;
  $("events").innerHTML = "";
}

async function multiYear(c) {
  const ys = S.meta.years;
  const recs = await Promise.all(ys.map(y => loadTile(y, S.cell)));
  const v = recs.map((r, k) => r ? FW.season(c, S.meta.dates, r.p1, r.p2, r.onsets, { year: ys[k], baseline: "careful" }).value : null);
  const ok = v.map((x, k) => [x, ys[k]]).filter(([x]) => x !== null);
  if (!ok.length) return null;
  const best = ok.reduce((a, b) => b[0] > a[0] ? b : a);
  return { n: ok.length, mean: d3.mean(ok, d => d[0]), best: best[0], bestYear: best[1] };
}

function renderNotice(c) {
  $("noticeCard").hidden = !irrigated();
  if (!irrigated()) return;
  const W = 520, H = 220, m = { l: 44, r: 12, t: 10, b: 30 };
  const rows = c.rows.slice(0, 15);
  const x = d3.scaleLinear().domain([0, 14]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, d3.max(rows, r => Math.max(r.avoided, r.extraCost)) * 1.15 || 1]).range([H - m.b, m.t]).nice();
  const svg = d3.select("#noticeChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(v => "$" + v));
  svg.append("text").attr("x", W - m.r).attr("y", H - 2).attr("text-anchor", "end").attr("fill", css("--muted")).attr("font-size", 11).text("days of warning ($ an acre)");
  const ln = key => d3.line().x(r => x(r.notice)).y(r => y(r[key]));
  svg.append("path").attr("d", ln("avoided")(rows)).attr("fill", "none").attr("stroke", css("--calm")).attr("stroke-width", 2.5);
  svg.append("path").attr("d", ln("extraCost")(rows)).attr("fill", "none").attr("stroke", css("--act")).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
  const last = rows[rows.length - 1];
  svg.append("text").attr("x", x(14)).attr("y", y(last.avoided) - 6).attr("text-anchor", "end").attr("fill", css("--calm")).attr("font-size", 12).text("crop loss avoided");
  svg.append("text").attr("x", x(14)).attr("y", y(last.extraCost) - 6).attr("text-anchor", "end").attr("fill", css("--act")).attr("font-size", 12).text("extra pumping");
  const sat = rows.findIndex(r => r.avoided >= last.avoided - 0.5);
  $("noticeNote").innerHTML = `Your pivot puts down ${c.capacity.toFixed(2)} in a day; the crop uses ${S.farm.et_peak.toFixed(2)} in a day in a heat wave.
    With no rain, stress starts ${c.stressDay} days into a dry spell. ${sat > 0 ? `Warning helps up to about <strong>${sat} days</strong>. After that, the soil is already full when the pivot starts.` : "Warning barely matters: your pivot keeps up with the crop if started on time."}`;
}

// ---------- decision cards: nitrogen, fungicide, silage, hay ----------

function renderDecisions() {
  const list = decisions(), sec = $("decisions");
  sec.hidden = !list.length;
  // Rain-fed: the decisions are the main event, straight under the advice. Irrigated: hay comes after the season chart.
  const anchor = irrigated() ? $("seasonCard") : $("top2");
  if (anchor.nextElementSibling !== sec) anchor.after(sec);
  $("decTitle").firstChild.textContent = irrigated() ? "Other decisions " : "Drought decisions ";
  const lv = localValues();
  $("decCards").innerHTML = list.map(a => `
    <div class="dcard" id="dc-${a.key}">
      <div class="dhead"><h4>${a.title}</h4><span class="note">${a.when}</span></div>
      <div class="dbody">
        <div class="dleft">
          <div class="dpill"></div>
          <div class="dgauge"></div>
          <div class="dout"><div><b class="dgain"></b><span>if a flash drought comes</span></div><div><b class="dcost"></b><span>if it doesn't</span></div></div>
          <p class="dwhy"></p>
          <div class="dseason"></div>
        </div>
        <div class="dright">
          <div class="dchart"></div>
          <div class="dparams">${a.params.map(([k, label, step, def]) => {
            const max = Math.max(def * 3, step * 10), v = S.farm[k];
            return `<label><span>${label} <b data-v="${k}">${v}</b></span>
              <input type="range" data-p="${k}" min="0" max="${max}" step="${step}" value="${v}">
              <span class="src${k in S.over ? " over" : ""}">${k in S.over ? `Your value (local: ${lv.v[k]}) <button class="linkish" data-reset="${k}">reset</button>` : (lv.src[k] || "")}</span></label>`;
          }).join("")}</div>
        </div>
      </div>
    </div>`).join("");
  for (const a of list) {
    updateCard(a);
    const el = $(`dc-${a.key}`);
    el.querySelectorAll("[data-p]").forEach(inp => {
      inp.addEventListener("input", () => {
        S.over[inp.dataset.p] = +inp.value; S.farm[inp.dataset.p] = +inp.value;
        el.querySelector(`[data-v="${inp.dataset.p}"]`).textContent = inp.value;
        updateCard(a);
      });
      inp.addEventListener("change", () => { saveOver(); renderAll(); });
    });
    el.querySelectorAll("[data-reset]").forEach(b => b.addEventListener("click", ev => {
      ev.preventDefault(); delete S.over[b.dataset.reset]; saveOver(); renderAll();
    }));
  }
}

function updateCard(a) {
  const el = $(`dc-${a.key}`), s = decisionAt(a, S.i), farmWide = a.per === "for the farm";
  const money = v => farmWide ? usd(v) : `${usd(v, 0)} <small>an acre</small>`;
  const pill = { act: [a.act, "act"], drought: [`${a.act}: already in drought`, "act"], wait: ["Not yet: decide at the next forecast", "wait"],
                 calm: [a.calm, "calm"], off: [`Not due yet: ${a.when.toLowerCase()}`, "off"], none: ["No forecast for this date", "off"] }[s.state];
  if (s.state === "off" && S.meta.dates[S.i] > a.window[1]) pill[0] = "Decided for this season";
  el.querySelector(".dpill").className = `dpill ${pill[1]}`;
  el.querySelector(".dpill").textContent = pill[0];
  el.querySelector(".dgain").innerHTML = money(s.c.gain);
  el.querySelector(".dcost").innerHTML = money(-s.c.cost);
  el.querySelector(".dwhy").innerHTML = a.explain(S.farm);
  // Gauge: today's 14-day chance against the chance at which acting pays.
  const top = Math.min(1, Math.max(0.3, s.be * 1.6, (s.chance || 0) * 1.25)), at = v => `${(100 * Math.min(v, top) / top).toFixed(1)}%`;
  el.querySelector(".dgauge").innerHTML = `
    <div class="gtrack"><div class="gkeep" style="width:${at(s.be)}"></div>
      <div class="gbe" style="left:${at(s.be)}"><span>pays above ${pct(s.be)}</span></div>
      ${s.chance === null ? "" : `<div class="gnow" style="left:${at(s.chance)}"><span>14-day chance today: ${pct(s.chance)}</span></div>`}</div>
    <div class="glab"><span>${a.calm}</span><span>${a.act}</span></div>`;
  const upTo = S.mode === "today" ? S.i : null, sv = decisionSeason(a, S.rec, S.year, upTo);
  const acres = S.farm.acres;
  el.querySelector(".dseason").innerHTML = `<strong>${S.mode === "today" ? "This season so far" : `The ${S.year} season`}: ${usd(sv.value)}${farmWide ? "" : " an acre"}</strong>
    ${farmWide ? "" : `(${usd(sv.value * acres)} on ${acres} acres)`}. ${sv.events.length ? sv.events.map(e => `${fmtShort(e.date)}: ${e.kind === "hit" ? `acted, flash drought ${e.notice} days later` : "acted, no flash drought came"}`).join("; ") + "."
      : "FlashWatch didn't call it this season."}`;
  decisionChart(a, el.querySelector(".dchart"), s.be);
}

function decisionChart(a, box, be) {
  const r = sofar(S.rec), dates = S.meta.dates.map(dateOf);
  const W = 520, H = 170, m = { l: 40, r: 10, t: 12, b: 22 };
  const x = d3.scaleUtc().domain(d3.extent(dates)).range([m.l, W - m.r]);
  const ch = r.p1.map((p, i) => p === null || r.p2[i] === null ? null : Math.min(1, p + r.p2[i]));
  const y = d3.scaleLinear().domain([0, Math.max(0.2, be * 1.4, (d3.max(ch.filter(v => v !== null)) || 0) * 1.1)]).range([H - m.b, m.t]).nice();
  const svg = d3.select(box).html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const w0 = x(dateOf(a.window[0])), w1 = x(dateOf(a.window[1]));
  svg.append("rect").attr("x", w0).attr("width", w1 - w0).attr("y", m.t).attr("height", H - m.t - m.b).attr("fill", css("--calm-bg"));
  svg.append("text").attr("x", w0 + 4).attr("y", m.t + 11).attr("font-size", 10).attr("fill", css("--muted")).text("decision window");
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.utcFormat("%b")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")));
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(be)).attr("y2", y(be)).attr("stroke", css("--act")).attr("stroke-dasharray", "5 3");
  svg.append("text").attr("x", W - m.r).attr("y", y(be) - 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", css("--act")).text("pays above this");
  svg.append("path").attr("d", d3.line().defined(v => v !== null).x((v, i) => x(dates[i])).y(v => y(v))(ch))
    .attr("fill", "none").attr("stroke", css("--risk")).attr("stroke-width", 2);
  ch.forEach((v, i) => {
    if (v === null || !RF.inWindow(a, S.meta.dates[i])) return;
    if (decisionAt(a, i, r).state === "act") svg.append("circle").attr("cx", x(dates[i])).attr("cy", y(v)).attr("r", 3.5).attr("fill", css("--act"));
  });
  for (const o of r.onsets) svg.append("path").attr("d", `M${x(dateOf(o))},${H - m.b} l-5,-8 l10,0 z`).attr("fill", css("--drought"));
  cursor(svg, x(dates[S.i]), m.t, H - m.b);
}

async function renderNote() {
  const md = S.meta.dates[S.i], key = `${S.year}_${S.cell[0]}_${S.cell[1]}_${md}`;
  let n = null;
  try { const r = await fetch(`data/notes/${key}.json`); if (r.ok) n = await r.json(); } catch (e) { /* none */ }
  $("noteCard").hidden = !n;
  if (n) { $("sms").textContent = n.sms || ""; $("note").textContent = n.note || ""; }
}

async function renderOps() {
  const md = S.meta.dates[S.i];
  let o = null;
  try { const r = await fetch(`data/ops/${S.year}_${md}.json`); if (r.ok) o = await r.json(); } catch (e) { /* none */ }
  const show = irrigated() && o && o.place && o.place === $("place").textContent;
  $("opsCard").hidden = !show;
  if (!show) return;
  $("opsSms").textContent = o.sms;
  const acc = Object.fromEntries(o.accepted.map(c => [c.pivot, c]));
  const ref = Object.fromEntries(o.refused.map(c => [c.pivot, c]));
  const rows = Object.entries(o.overview.pivots).map(([k, p]) => {
    const a = p.advice, v = a.value_of_starting_today_dollars_whole_pivot;
    const plan = acc[k] ? `<strong>${acc[k].action} ${acc[k].action === "start" ? acc[k].depth_in + " in" : ""}</strong>`
      : ref[k] ? `held: ${ref[k].why}` : p.running ? "keep running"
      : (a.verdict || "").startsWith("start") ? "held (see note)" : "-";
    return `<tr><td><strong>${k}</strong></td><td>${p.crop}, ${p.acres} ac, ${p.soil.toLowerCase()}</td><td>${a.verdict ?? "-"}</td>
      <td>${v === undefined ? "-" : usd(v)}</td><td>${p.running ? "on" : "off"}</td><td>${plan}</td></tr>`;
  }).join("");
  $("opsTable").innerHTML = `<table><thead><tr><th>Pivot</th><th>Field</th><th>Advice</th><th>Start today, whole pivot</th><th>Now</th><th>Plan</th></tr></thead><tbody>${rows}</tbody></table>`;
  const reps = Object.entries(o.overview.pivots).filter(([, p]) => p.field_report);
  const sev = ["looks fine", "mild signs", "stress showing", "severe"];
  $("opsNote").innerHTML = (reps.length ? reps.map(([k, p]) => `<div class="note" style="margin-bottom:6px">Voice note from the farmer, ${p.field_report.date.slice(5)} (Muse Voice Transcribe), on ${k}:
    "${p.field_report.farmer_said}" (${sev[p.field_report.severity] || ""})</div>`).join("") : "") + `<div>${o.note}</div>`;
  $("opsState").textContent = o.sent.length ? `Approved by ${o.approved_by}; sent: ${o.sent.map(c => `${c.pivot} ${c.state}`).join(", ")}.` : "Waiting for approval.";
}

// ---------- the map: street map or drought map, coverage outline, search ----------

function initMap() {
  if (typeof L === "undefined") { $("map").textContent = "Map unavailable."; return; }
  const m = L.map("map", { scrollWheelZoom: false, zoomSnap: 0.5 }).setView([S.lat, S.lon], 6);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 17, attribution: "&copy; OpenStreetMap contributors" }).addTo(m);
  L.rectangle([[36, -104], [49, -80]], { color: css("--ink"), weight: 2, dashArray: "6 6", fill: false, interactive: false }).addTo(m);
  const icon = L.divIcon({ className: "farm-pin", iconSize: [16, 16] });
  S.marker = L.marker([S.lat, S.lon], { icon, draggable: true, keyboard: false }).addTo(m);
  S.marker.on("dragend", () => { const p = S.marker.getLatLng(); setLocation(p.lat, p.lng); });
  m.on("click", e => setLocation(e.latlng.lat, e.latlng.lng));
  S.map = m;
  const cov = S.defaults && S.defaults.coverage ? (S.defaults.coverage.total / 1e6).toFixed(1) : null;
  S.covText = cov ? `FlashWatch coverage: ${cov} million acres of corn and soybeans` : "FlashWatch coverage";
  renderMapLegend();
}

function renderMapLegend() {
  const items = [`<span><i class="dash"></i>${S.covText}</span>`];
  if (S.mapMode === "drought") {
    const color = riskColor();
    items.push(...[1, 2, 4, 8].map(v => `<span><i style="background:${color(Math.log2(v))};height:10px"></i>${v === 1 ? "normal or below" : v + "x normal"}</span>`),
      `<span><i style="background:${css("--drought")};opacity:.5;height:10px"></i>in drought</span>`);
  }
  $("mapLegend").innerHTML = items.join("");
}
const riskColor = () => d3.scaleSequential(d3.interpolateYlOrRd).domain([0, Math.log2(8)]);

async function renderMapState() {
  if (!S.map) return;
  S.marker.setLatLng([S.lat, S.lon]);
  if (!S.map.getBounds().pad(-0.1).contains([S.lat, S.lon])) S.map.panTo([S.lat, S.lon]);
  $("map").classList.toggle("drought", S.mapMode === "drought");
  if (S.droughtLayer) { S.droughtLayer.remove(); S.droughtLayer = null; }
  renderMapLegend();
  if (S.mapMode !== "drought") return;
  const m = S.meta, i = S.i, year = S.year;
  if (!S.regionRecs[year]) {
    const keys = [...new Set(m.cells.map(([iy, ix]) => `${Math.floor(iy / m.thin / m.tile)}_${Math.floor(ix / m.thin / m.tile)}`))];
    S.regionRecs[year] = (await Promise.all(keys.map(k => { const [ty, tx] = k.split("_"); return tileFile(year, ty, tx); }))).flat();
  }
  if (S.mapMode !== "drought" || S.i !== i || S.year !== year) return;
  const color = riskColor(), step = m.res * m.thin / 2, layer = L.layerGroup(), rnd = L.canvas();
  for (const r of S.regionRecs[year]) {
    const lat = m.lat0 - r.iy * m.res, lon = m.lon0 + r.ix * m.res;
    let fill, op = 0.75;
    if (r.p1[i] === null) { if ((r.fz || []).includes(i)) continue; fill = css("--drought"); op = 0.45; }
    else fill = color(Math.max(0, Math.log2(r.p1[i] / Math.max(r.n1[i], 1e-4))));
    L.rectangle([[lat - step, lon - step], [lat + step, lon + step]], { renderer: rnd, stroke: false, fillColor: fill, fillOpacity: op, interactive: false }).addTo(layer);
  }
  S.droughtLayer = layer.addTo(S.map);
}

async function search(q) {
  const ll = q.match(/^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (ll) { setLocation(+ll[1], +ll[2], true); return; }
  $("searchNote").textContent = "Searching...";
  try {
    const r = await fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&countrycodes=us&limit=1&q=${encodeURIComponent(q)}`);
    const j = await r.json();
    if (!j.length) { $("searchNote").textContent = "Not found. Try a town and state, or click the map."; return; }
    $("searchNote").textContent = `Found: ${j[0].display_name}`;
    setLocation(+j[0].lat, +j[0].lon, true);
  } catch (e) { $("searchNote").textContent = "Search unavailable. Click the map instead."; }
}

// ---------- state changes ----------

async function setLocation(lat, lon, zoom = false) {
  S.lat = lat; S.lon = lon;
  if (zoom && S.map) S.map.setView([lat, lon], 10);
  const cell = nearestCell(lat, lon);
  if (!cell) {
    $("place").textContent = "Outside FlashWatch's area";
    $("placeSub").textContent = "FlashWatch covers the Corn Belt and northern Plains (inside the dashed line).";
    if (S.marker) S.marker.setLatLng([lat, lon]);
    return;
  }
  S.cell = cell;
  S.cty = nearestCounty(lat, lon);
  $("place").textContent = S.cty ? `${S.cty[1]}, ${S.cty[2]}` : `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
  const [clat, clon] = cellLatLon(cell);
  $("placeSub").textContent = `${lat.toFixed(2)}, ${lon.toFixed(2)}. Forecast for the ${(S.meta.res * S.meta.thin * 111).toFixed(0)} km square around ${clat.toFixed(2)}, ${clon.toFixed(2)}.`;
  S.rec = await loadTile(S.year, cell);
  renderSettings();
  renderAll();
}

function setDate(i) {
  if (S.mode === "today") return;
  S.i = Math.max(0, Math.min(S.meta.dates.length - 1, i));
  renderAll();
}

function fieldHtml(k, label, type, opt, v, src, isOver) {
  const input = type === "select"
    ? `<select data-k="${k}">${opt.map(o => `<option${v[k] === o ? " selected" : ""}>${o}</option>`).join("")}</select>`
    : `<input data-k="${k}" type="${type}" ${type === "number" ? `step="${opt}"` : ""} value="${v[k]}">`;
  const lv = localValues();
  const tag = isOver ? `<span class="src over">Your value (local: ${lv.v[k]}) <button class="linkish" data-reset="${k}">reset</button></span>`
    : src ? `<span class="src">${src}</span>` : "";
  return `<label>${label}${input}${tag}</label>`;
}

function renderSettings() {
  const lv = localValues(), v = farm();
  $("settings").innerHTML = FARM_FIELDS.filter(f => f[4] === "all" || v.watering !== "rain-fed")
    .map(([k, label, type, opt]) => fieldHtml(k, label, type, opt, v, lv.src[k], k in S.over && !["crop", "watering"].includes(k))).join("");
  document.querySelectorAll("[data-k]").forEach(inp => inp.addEventListener("change", () => {
    const k = inp.dataset.k;
    S.over[k] = inp.type === "number" ? +inp.value : inp.value;
    if (k === "crop" || k === "watering") delete S.over.yield_bu, delete S.over.price;
    saveOver(); renderSettings(); renderAll();
  }));
  document.querySelectorAll("[data-reset]").forEach(b => b.addEventListener("click", ev => {
    ev.preventDefault(); delete S.over[b.dataset.reset]; saveOver(); renderSettings(); renderAll();
  }));
}

async function init() {
  [S.meta, S.defaults] = await Promise.all([fetch("data/meta.json").then(r => r.json()),
                                           fetch("data/defaults.json").then(r => r.ok ? r.json() : null).catch(() => null)]);
  loadOver();
  readHash();
  if (!S.meta.years.includes(S.year)) S.year = S.meta.years.includes(2012) ? 2012 : S.meta.years[0];
  $("year").innerHTML = S.meta.years.map(y => `<option${y === S.year ? " selected" : ""}>${y}</option>`).join("");
  S.i = Math.max(0, S.meta.dates.indexOf(S.wantDate || "07-03"));
  const setWatering = w => { S.over.watering = w; delete S.over.acres; delete S.over.yield_bu; saveOver(); renderSettings(); renderAll(); };
  $("wIrr").onclick = () => setWatering("pivot-irrigated");
  $("wRain").onclick = () => setWatering("rain-fed");
  $("modeToday").onclick = () => setMode("today");
  $("modeReplay").onclick = () => setMode("replay");
  $("year").onchange = async () => { S.year = +$("year").value; S.rec = await loadTile(S.year, S.cell); renderAll(); };
  $("strip").addEventListener("keydown", ev => {
    if (S.mode === "today" || !["ArrowLeft", "ArrowRight"].includes(ev.key)) return;
    const st = stops(), k = st.findIndex(s => s.has(S.i)) + (ev.key === "ArrowRight" ? 1 : -1);
    if (k >= 0 && k < st.length) setDate(st[k].i);
    ev.preventDefault();
  });
  $("search").onsubmit = ev => { ev.preventDefault(); if ($("q").value.trim()) search($("q").value); };
  $("mapStreet").onclick = () => { S.mapMode = "street"; $("mapStreet").setAttribute("aria-pressed", true); $("mapDrought").setAttribute("aria-pressed", false); renderMapState(); };
  $("mapDrought").onclick = () => { S.mapMode = "drought"; $("mapDrought").setAttribute("aria-pressed", true); $("mapStreet").setAttribute("aria-pressed", false); renderMapState(); };
  $("resetAll").onclick = () => { const keep = pick(S.over, ["crop", "watering", "livestock"]); S.over = keep; saveOver(); renderSettings(); renderAll(); };
  initMap();
  await setMode(S.mode);
  await setLocation(S.lat, S.lon);
  scheduleRefresh();
}
const pick = (o, ks) => Object.fromEntries(ks.filter(k => k in o).map(k => [k, o[k]]));

init();
