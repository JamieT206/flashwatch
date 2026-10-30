// FlashWatch region page: value to a typical farm, forecast quality, a regional risk map, the lead
// over the Drought Monitor.
"use strict";

const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const R = { meta: null, year: 2012, i: 0, recs: null, states: null, roll: null, prices: [], ev: null, defaults: null };
const usd0 = v => (v < -0.5 ? "-$" : "$") + Math.abs(Math.round(v)).toLocaleString("en-US");
const usd2 = v => (v < -0.005 ? "-$" : "$") + Math.abs(v).toFixed(2);
const pct1 = v => (100 * v).toFixed(1) + "%";
const PRICES = [3.5, 4.5, 5.5, 6.5, 7.5];

// ---------- a typical farm ----------

function perAcre(scopes, year) {
  // Several decisions on the same acres add up (rain-fed corn: nitrogen and fungicide).
  const one = s => {
    if (!s) return null;
    const r = year === undefined ? s.region : s.region_by_year.find(x => x.year === year);
    return r && r.acres ? r.dollars / r.acres : null;
  };
  const v = [].concat(scopes).map(one);
  return v.some(x => x === null) ? null : v.reduce((a, b) => a + b, 0);
}

function renderFarms() {
  const base = $("base").value, s = R.roll && R.roll.scopes[base];
  if (!s) { $("farmTable").textContent = "Not computed yet."; return; }
  const fs = (R.defaults && R.defaults.farm_sizes) || {};
  const round = v => Math.round(v / 50) * 50;
  // One flash drought with a week's warning: the farm page's water balance (21-day dry spell), or
  // for rain-fed fields the fungicide pass a dry spell makes unnecessary (site/assets/rainfed.js).
  const week = crop => { const c = FW.curve({ ...FW.DEFAULTS[crop], spell_days: 21 }, 21); return FW.netAt(c, 7); };
  const rf = key => { const a = RF.ACTIONS.find(x => x.key === key); return RF.curve(a, { ...RF.DEFAULTS, price: key === "fungicide_soy" ? 10 : 4.3 }).gain; };
  const cols = [
    { name: "Irrigated corn farm", acres: round(fs.corn_irrigated || 350), scope: s.corn, event: week("corn") },
    { name: "Irrigated soybean farm", acres: round(fs.soybeans_irrigated || 250), scope: s.soybeans, event: week("soybeans") },
    { name: "Rain-fed corn farm", acres: round(fs.corn_all || 300), scope: [s.rainfed_nitrogen, s.rainfed_fungicide_corn], event: rf("fungicide_corn") },
    { name: "Rain-fed soybean farm", acres: round(fs.soybeans_all || 300), scope: s.rainfed_fungicide_soy, event: rf("fungicide_soy") },
  ];
  const cell = (perA, acres) => perA === null ? "<td>-</td>" : `<td><strong>${usd0(perA * acres)}</strong><br><span class="note">${usd2(perA)} an acre</span></td>`;
  const rows = [
    ["A flash drought, with a week's warning", c => cell(c.event, c.acres)],
    ["2012, a flash-drought year", c => cell(perAcre(c.scope, 2012), c.acres)],
    [`Average year (${R.roll.years[0]}-${R.roll.years[R.roll.years.length - 1]})`, c => cell(perAcre(c.scope), c.acres)],
  ];
  $("farmTable").innerHTML = `<table class="farms"><thead><tr><th></th>${cols.map(c => `<th>${c.name}<br><span class="note">${c.acres} acres</span></th>`).join("")}</tr></thead>
    <tbody>${rows.map(([label, f]) => `<tr><td>${label}</td>${cols.map(f).join("")}</tr>`).join("")}</tbody></table>`;
  const tot = k => (s[k] && s[k].region ? s[k].region.dollars : 0);
  const irr = tot("corn") + tot("soybeans"), rain = tot("rainfed_nitrogen") + tot("rainfed_fungicide_corn") + tot("rainfed_fungicide_soy");
  $("regionTotals").innerHTML = `
    <div><div class="bignum">$${(rain / 1e6).toFixed(0)}M</div><div class="note">a year across ${(R.defaults ? R.defaults.coverage.corn_rainfed + R.defaults.coverage.soybeans_rainfed : 130e6) / 1e6 | 0} million rain-fed acres in the coverage area (nitrogen and fungicide decisions)</div></div>
    <div><div class="bignum">$${(irr / 1e6).toFixed(0)}M</div><div class="note">a year across ${(R.defaults ? R.defaults.coverage.corn_irrigated + R.defaults.coverage.soybeans_irrigated : 13e6) / 1e6 | 0} million irrigated acres (extra irrigation)</div></div>`;
}

// ---------- forecast quality: how often a flagged farm gets a flash drought ----------

function renderFlags() {
  const f = R.ev && R.ev.flagged;
  if (!f) { $("flagKpis").textContent = "Not computed yet."; return; }
  const w1 = f.M2.week1;
  $("flagKpis").innerHTML = `
    <div><div class="bignum">${w1.lift.toFixed(1)}x</div><div class="note">the normal chance of a flash drought in the next 7 days, on the farms FlashWatch flags</div></div>
    <div><div class="bignum">${Math.round(100 * w1.caught)}%</div><div class="note">of the flash droughts that begin in the next 7 days are on flagged farms</div></div>`;
  const W = 520, H = 220, m = { l: 40, r: 10, t: 16, b: 40 };
  const leads = ["week1", "week2"], names = { week1: "Next 7 days", week2: "Days 8-14" };
  const groups = [["base", "Normal chance", css("--normal")], ["H1", "Flagged by a hot-and-dry rule of thumb", css("--wait")], ["M2", "Flagged by FlashWatch", css("--risk")]];
  const val = (g, lead) => g === "base" ? f.M2[lead].base : f[g][lead].hit_rate;
  const x0 = d3.scaleBand().domain(leads).range([m.l, W - m.r]).padding(0.25);
  const x1 = d3.scaleBand().domain(groups.map(g => g[0])).range([0, x0.bandwidth()]).padding(0.08);
  const y = d3.scaleLinear().domain([0, Math.max(0.3, d3.max(leads, l => val("M2", l)) * 1.15)]).range([H - m.b, m.t]);
  const svg = d3.select("#flagChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x0).tickFormat(l => names[l]));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  for (const lead of leads) for (const [g, , col] of groups) {
    const v = val(g, lead), bx = x0(lead) + x1(g);
    svg.append("rect").attr("x", bx).attr("width", x1.bandwidth()).attr("y", y(v)).attr("height", y(0) - y(v)).attr("fill", col);
    svg.append("text").attr("x", bx + x1.bandwidth() / 2).attr("y", y(v) - 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", css("--ink")).text(pct1(v));
  }
  $("flagNote").innerHTML = groups.map(([, label, col]) => `<span style="display:inline-block;width:10px;height:10px;background:${col}"></span> ${label}`).join(" &nbsp; ") +
    `<br>Chance that a flash drought begins, on all farm-days and on the riskiest tenth by each method. Scored on ${R.ev.m2_dates} spring forecast dates
     in six past seasons; summer forecast dates are still downloading.`;
}

// ---------- value by corn price ----------

function renderPrice() {
  const pts = R.prices.filter(p => p.v !== null);
  if (!pts.length) { $("priceChart").textContent = "Not computed yet."; return; }
  const W = 520, H = 220, m = { l: 48, r: 14, t: 14, b: 36 };
  const x = d3.scaleLinear().domain([3, 8]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, d3.max(pts, p => p.v) * 1.2]).range([H - m.b, m.t]).nice();
  const svg = d3.select("#priceChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(v => "$" + v));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(v => "$" + v));
  svg.append("text").attr("x", W - m.r).attr("y", H - 2).attr("text-anchor", "end").attr("font-size", 11).attr("fill", css("--muted")).text("corn price ($ a bushel)");
  svg.append("text").attr("x", m.l + 4).attr("y", m.t + 2).attr("font-size", 11).attr("fill", css("--muted")).text("$ an acre a year");
  const now = R.defaults && R.defaults.states && R.defaults.states.NE ? R.defaults.states.NE.corn_price : null;
  if (now) {
    svg.append("line").attr("x1", x(now)).attr("x2", x(now)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", css("--line")).attr("stroke-dasharray", "3 3");
    svg.append("text").attr("x", x(now) + 4).attr("y", m.t + 16).attr("font-size", 11).attr("fill", css("--muted")).text(`${R.defaults.price_year} Nebraska price, $${now.toFixed(2)}`);
  }
  svg.append("path").attr("d", d3.line().x(p => x(p.price)).y(p => y(p.v))(pts)).attr("fill", "none").attr("stroke", css("--calm")).attr("stroke-width", 2.5);
  for (const p of pts) {
    svg.append("circle").attr("cx", x(p.price)).attr("cy", y(p.v)).attr("r", 4).attr("fill", css("--calm"));
    svg.append("text").attr("x", x(p.price)).attr("y", y(p.v) - 8).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", css("--ink")).text(usd2(p.v));
  }
}

// ---------- risk map ----------

function tileKeys() {
  const m = R.meta, keys = new Set();
  for (const [iy, ix] of m.cells) keys.add(`${Math.floor(iy / m.thin / m.tile)}_${Math.floor(ix / m.thin / m.tile)}`);
  return [...keys];
}
async function loadYear(y) {
  const files = await Promise.all(tileKeys().map(k => fetch(`data/tiles/${y}/${k}.json`).then(r => r.ok ? r.json() : [])));
  R.recs = files.flat();
}
function renderMap() {
  const m = R.meta, i = R.i, W = 960, H = 520;
  const md = m.dates[i];
  $("dateLabel").textContent = new Date(Date.UTC(R.year, +md.slice(0, 2) - 1, +md.slice(3, 5)))
    .toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  const proj = d3.geoMercator().fitExtent([[4, 4], [W - 4, H - 4]], { type: "MultiPoint", coordinates: [[-104, 36], [-80, 36], [-80, 49], [-104, 49]] });
  const svg = d3.select("#rmap").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const color = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, Math.log2(8)]);
  const step = m.res * m.thin;
  const day = s => Date.UTC(R.year, +s.slice(0, 2) - 1, +s.slice(3, 5)) / 864e5;
  const t0 = day(md);
  for (const r of R.recs) {
    const lat = m.lat0 - r.iy * m.res, lon = m.lon0 + r.ix * m.res;
    const [x0, y0] = proj([lon - step / 2, lat + step / 2]), [x1, y1] = proj([lon + step / 2, lat - step / 2]);
    const frozen = (r.fz || []).includes(i);
    let fill;
    if (r.p1[i] === null) fill = frozen ? css("--off-bg") : r.rz[i] === null ? css("--drought") : css("--line");
    else fill = color(Math.max(0, Math.log2(r.p1[i] / Math.max(r.n1[i], 1e-4))));
    svg.append("rect").attr("x", x0).attr("y", y0).attr("width", x1 - x0 + 0.5).attr("height", y1 - y0 + 0.5)
      .attr("fill", fill).attr("opacity", r.p1[i] === null && r.rz[i] === null && !frozen ? 0.45 : 1).style("cursor", "pointer")
      .on("click", () => { location.href = `index.html#lat=${lat.toFixed(2)}&lon=${lon.toFixed(2)}&mode=replay&year=${R.year}&date=${md}`; });
    if (r.onsets.some(o => { const d = day(o); return d > t0 && d <= t0 + 7; })) {
      svg.append("circle").attr("cx", (x0 + x1) / 2).attr("cy", (y0 + y1) / 2).attr("r", 2.6).attr("fill", css("--ink"));
    }
  }
  const [cx0, cy0] = proj([-104, 49]), [cx1, cy1] = proj([-80, 36]);
  svg.append("clipPath").attr("id", "box").append("rect").attr("x", cx0).attr("y", cy0).attr("width", cx1 - cx0).attr("height", cy1 - cy0);
  if (R.states) svg.append("g").attr("clip-path", "url(#box)").selectAll("path").data(R.states.features).join("path").attr("d", d3.geoPath(proj))
    .attr("fill", "none").attr("stroke", css("--ink")).attr("stroke-opacity", 0.35).attr("stroke-width", 0.8);
  $("mapLegend").innerHTML = [1, 2, 4, 8].map(v => `<span><i style="background:${color(Math.log2(v))};height:10px;width:16px"></i>${v === 1 ? "normal or below" : v + "x normal"}</span>`).join("") +
    `<span><i style="background:${css("--drought")};opacity:.45;height:10px;width:16px"></i>already in drought</span><span>&#9679; a flash drought began within 7 days</span>`;
}

// ---------- ahead of the Drought Monitor ----------

function renderMonitor() {
  const mo = R.ev.monitor;
  const card = (t, s, sub) => `<div class="row"><div class="note">${t}</div><div class="v">${s}</div><div class="note">${sub}</div></div>`;
  $("leadDays").textContent = `${mo.crossing_vs_d1.median_days_ahead.toFixed(0)} days`;
  $("monitor").innerHTML =
    card("Soil drops into drought, compared with the Monitor's first D1 (moderate drought)", `${mo.crossing_vs_d1.median_days_ahead.toFixed(0)} days earlier`,
         `median; FlashWatch first ${(100 * mo.crossing_vs_d1.share_flashwatch_first).toFixed(0)}% of the time`) +
    card("Compared with the Monitor's first change of any kind", `${mo.crossing_vs_worsening.median_days_ahead.toFixed(0)} days earlier`,
         `median; FlashWatch first ${(100 * mo.crossing_vs_worsening.share_flashwatch_first).toFixed(0)}% of the time`);
  $("monitorNote").textContent = `Across ${mo.onsets.toLocaleString()} flash droughts, 2006-2021, counted from the day each data set is public.
    The forecasts add one to two weeks on top. The Monitor never marks about half of these flash droughts; on average those cost no yield, because rain came in time.`;
}

async function init() {
  const get = (u, fallback = null) => fetch(u).then(r => r.ok ? r.json() : fallback).catch(() => fallback);
  [R.meta, R.ev, R.roll, R.defaults] = await Promise.all([get("data/meta.json"), get("data/evidence.json"),
    get("data/rollup_blur07_cal.json"), get("data/defaults.json")]);
  R.prices = await Promise.all(PRICES.map(async p => {
    const r = await get(`data/rollup_price${String(p).replace(".", "")}.json`);
    const h = r && r.headline.find(x => x.baseline === "careful");
    return { price: p, v: h ? h.per_acre_corn : null };
  }));
  try {
    const us = await fetch("https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json").then(r => r.json());
    R.states = topojson.feature(us, us.objects.states);
  } catch (e) { /* outlines optional */ }
  $("year").innerHTML = R.meta.years.map(y => `<option${y === R.year ? " selected" : ""}>${y}</option>`).join("");
  $("date").max = R.meta.dates.length - 1;
  R.i = R.meta.dates.indexOf("07-03");
  $("date").value = R.i;
  $("base").onchange = renderFarms;
  $("year").onchange = async () => { R.year = +$("year").value; await loadYear(R.year); renderMap(); };
  const setI = i => { R.i = Math.max(0, Math.min(R.meta.dates.length - 1, i)); $("date").value = R.i; renderMap(); };
  $("date").oninput = () => setI(+$("date").value);
  $("prev").onclick = () => setI(R.i - 1);
  $("next").onclick = () => setI(R.i + 1);
  renderFarms();
  renderPrice();
  if (R.ev) { renderFlags(); renderMonitor(); }
  await loadYear(R.year);
  renderMap();
}

init();
