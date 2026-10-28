// FlashWatch farm page: one location, one season, one decision a day.
"use strict";

const S = {
  meta: null, tileCache: {}, cell: null, rec: null, year: 2012, i: 0,
  lat: 40.87, lon: -97.99,            // York County, Nebraska: centre-pivot country
  farm: null, states: null,
};
const $ = id => document.getElementById(id);
const pct = (p, d = 0) => p === null || p === undefined || !isFinite(p) ? "-" : (100 * p).toFixed(d) + "%";
const usd = (v, d = 0) => (v < 0 ? "-$" : "+$") + Math.abs(v).toFixed(d);
const USDM = { "-1": "None", 0: "D0 abnormally dry", 1: "D1 moderate drought", 2: "D2 severe drought",
               3: "D3 extreme drought", 4: "D4 exceptional drought" };
const SEASON = ["06-01", "08-31"];

const FARM_FIELDS = [
  ["crop", "Crop", "select", ["corn", "soybeans"]],
  ["acres", "Irrigated acres", "number", 1],
  ["gpm_per_acre", "Pivot capacity (gpm per acre)", "number", 0.1],
  ["soil", "Soil", "select", Object.keys(FW.SOILS)],
  ["yield_bu", "Expected yield (bu/acre)", "number", 1],
  ["price", "Expected price ($/bu)", "number", 0.05],
  ["pump_cost_per_in", "Pumping cost ($ per acre-inch)", "number", 0.5],
  ["reaction_days", "Usually start the pivot (days after stress shows)", "number", 1],
  ["baseline", "Compare against", "select", ["careful", "calendar"]],
  ["calendar", "Calendar start date (MM-DD)", "text"],
];

function defaultFarm(crop = "corn") {
  return { ...FW.DEFAULTS[crop], crop, acres: 130, soil: "Silt loam", baseline: "careful", calendar: "07-01" };
}

function loadFarm() {
  try {
    const s = JSON.parse(localStorage.getItem("fw.farm") || "null");
    if (s && s.crop) return { ...defaultFarm(s.crop), ...s };
  } catch (e) { /* storage blocked: defaults */ }
  return defaultFarm();
}
function saveFarm() { try { localStorage.setItem("fw.farm", JSON.stringify(S.farm)); } catch (e) { /* ignore */ } }

function params() {
  return { ...S.farm, awc_in_per_ft: FW.SOILS[S.farm.soil] ?? 2.0 };
}

function cellLatLon([iy, ix]) {
  const m = S.meta;
  return [m.lat0 - iy * m.res, m.lon0 + ix * m.res];
}

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

async function loadTile(year, cell) {
  const m = S.meta, ty = Math.floor(cell[0] / m.thin / m.tile), tx = Math.floor(cell[1] / m.thin / m.tile);
  const key = `${year}/${ty}_${tx}`;
  if (!S.tileCache[key]) S.tileCache[key] = fetch(`data/tiles/${key}.json`).then(r => r.json());
  const recs = await S.tileCache[key];
  return recs.find(r => r.iy === cell[0] && r.ix === cell[1]);
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("lat")) S.lat = +h.get("lat");
  if (h.has("lon")) S.lon = +h.get("lon");
  if (h.has("year")) S.year = +h.get("year");
  if (h.has("date")) S.wantDate = h.get("date");
}
function writeHash() {
  const md = S.meta.dates[S.i];
  history.replaceState(null, "", `#lat=${S.lat.toFixed(2)}&lon=${S.lon.toFixed(2)}&year=${S.year}&date=${md}`);
}

function fmtDate(md, year = S.year) {
  return new Date(Date.UTC(year, +md.slice(0, 2) - 1, +md.slice(3, 5)))
    .toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

// ---------- rendering ----------

async function renderNote() {
  const md = S.meta.dates[S.i], key = `${S.year}_${S.cell[0]}_${S.cell[1]}_${md}`;
  let n = null;
  try { const r = await fetch(`data/notes/${key}.json`); if (r.ok) n = await r.json(); } catch (e) { /* none */ }
  $("noteCard").hidden = !n;
  if (n) { $("sms").textContent = n.sms || ""; $("note").textContent = n.note || ""; }
}

function renderAll() {
  if (!S.rec) return;
  renderNote();
  const c = FW.curve(params(), 21);
  renderVerdict(c);
  renderTiles();
  renderSeason(c);
  renderNotice(c);
  renderPicker();
  writeHash();
}

function renderVerdict(c) {
  const r = S.rec, i = S.i, md = S.meta.dates[i];
  const card = $("verdict");
  $("dateLabel").textContent = fmtDate(md);
  const p1 = r.p1[i], p2 = r.p2[i];
  const per = S.farm.acres;
  const set = (cls, big, why) => { card.className = "card verdict " + cls; $("vBig").textContent = big; $("vWhy").innerHTML = why; };
  if (p1 === null || p2 === null) {
    set("calm", "Irrigate as you normally would",
      r.rz[i] === null ? "Your soil is already below the drought line (or frozen), so there is no flash drought left to warn about." : "No forecast for this date.");
    $("evNow").textContent = "-"; $("evWait").textContent = "-"; $("trigger").textContent = "";
    return;
  }
  // With ensemble members for this date, the next forecast's outcomes come from them, and the
  // trigger is named in weather; otherwise a placeholder spread and a risk threshold.
  const mem = r.mp1 && r.mp1[i] ? FW.trigger(c, r.mp1[i], r.mp2[i], { rain_in: r.mr[i], temp_f: r.mt[i] }) : null;
  const d = mem || FW.decide(c, FW.daily(p1, p2), FW.twoWay(0.5), 2);
  $("evNow").textContent = `${usd(d.evNow, 1)} /acre`;
  $("evWait").textContent = `${usd(d.evWait, 1)} /acre`;
  const inSeason = md >= SEASON[0] && md <= SEASON[1];
  const total = v => `${usd(v * per)} for ${per} acres`;
  if (!inSeason) {
    set("calm", "Outside the irrigation season",
      `FlashWatch's irrigation advice runs June to August, when the crop's water use peaks. The risk numbers still apply.`);
    $("trigger").textContent = "";
    return;
  }
  const nextRisk = (d.mStar ?? NaN) * p1;
  if (d.actNow) {
    const k = 4;
    const perDay = FW.netAt(c, k) - FW.netAt(c, k - 1);
    set("act", "Start the pivot today",
      `A flash drought is <strong>${(p1 / (r.n1[i] || 1e-9)).toFixed(1)}x</strong> as likely as normal this week. Starting now banks water while the soil still has room: ${total(d.evNow)} on average.`);
    $("trigger").innerHTML = `<strong>Why not wait?</strong> Each day of warning is worth about $${perDay.toFixed(0)}/acre on your field right now, more than the next forecast is likely to tell you.`;
  } else if (d.evWait > 0) {
    set("wait", "Not yet: watch the next forecast",
      d.evNow > 0
        ? `Starting today is worth a little (${usd(d.evNow, 1)}/acre), but waiting two days for a fresher forecast is worth more: you lose a little warning, but you avoid pumping for a dry spell that may not come.`
        : `Starting today would cost more than it is likely to save (${usd(d.evNow, 1)}/acre), but the risk is high enough that the next forecast could tip it.`);
    $("trigger").innerHTML = mem && mem.trigger
      ? `<strong>What we're waiting for:</strong> if the next forecast (in 2 days) shows <strong>${mem.trigger.text}</strong> at your farm, start the pivot then.
         ${mem.membersTipping} of ${mem.members} of today's forecast scenarios look like that.`
      : mem ? `<strong>What we're waiting for:</strong> ${mem.membersTipping} of ${mem.members} of today's forecast scenarios would make starting pay at the next forecast; no single rain or heat threshold separates them.`
      : isFinite(nextRisk) && nextRisk < 1
      ? `<strong>What would change it:</strong> if the next forecast (in 2 days) shows at least a <strong>${pct(Math.min(nextRisk, 1))}</strong> chance of a flash drought in the coming week (today: ${pct(p1)}), start the pivot then.`
      : "";
  } else {
    set("calm", "No need to start early",
      `The chance of a flash drought this week is ${pct(p1)} (normal: ${pct(r.n1[i])}). Starting early wouldn't pay for its pumping.`);
    $("trigger").innerHTML = isFinite(nextRisk) && nextRisk < 1
      ? `<strong>What would change it:</strong> a forecast showing at least a <strong>${pct(nextRisk)}</strong> chance this week (today: ${pct(p1)}).`
      : "";
  }
}

function renderTiles() {
  const r = S.rec, i = S.i;
  const ratio = (p, n) => p === null || !n ? "" : `${(p / n).toFixed(1)}x normal (${pct(n, 1)})`;
  $("t1").textContent = pct(r.p1[i], 1);
  $("t1x").textContent = ratio(r.p1[i], r.n1[i]);
  $("t1x").className = "x" + (r.p1[i] !== null && r.p1[i] < 1.5 * r.n1[i] ? " low" : "");
  $("t2").textContent = pct(r.p2[i], 1);
  $("t2x").textContent = ratio(r.p2[i], r.n2[i]);
  $("t2x").className = "x" + (r.p2[i] !== null && r.p2[i] < 1.5 * r.n2[i] ? " low" : "");
  $("t3").textContent = r.rz[i] === null ? "below 20th" : `${r.rz[i].toFixed(0)}th`;
  $("t3x").textContent = r.rz[i] === null ? "percentile: already in drought" : "percentile for the date (20th = drought line)";
  $("t4").textContent = USDM[r.usdm[i]] ?? "None";
}

function dateOf(md) { return new Date(Date.UTC(S.year, +md.slice(0, 2) - 1, +md.slice(3, 5))); }

function renderSeason(c) {
  const r = S.rec, dates = S.meta.dates.map(dateOf);
  const W = 1100, H = 230, m = { l: 44, r: 12, t: 10, b: 24 };
  const x = d3.scaleUtc().domain(d3.extent(dates)).range([m.l, W - m.r]);
  const ymax = Math.max(0.08, d3.max(r.p1.concat(r.n1).filter(v => v !== null)) * 1.15);
  const y = d3.scaleLinear().domain([0, ymax]).range([H - m.b, m.t]).nice();
  const svg = d3.select("#seasonChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v);
  svg.append("rect").attr("x", x(dateOf(SEASON[0]))).attr("width", x(dateOf(SEASON[1])) - x(dateOf(SEASON[0])))
    .attr("y", m.t).attr("height", H - m.t - m.b).attr("fill", css("--calm-bg")).attr("opacity", .55);
  svg.append("text").attr("x", x(dateOf(SEASON[0])) + 6).attr("y", m.t + 14).attr("fill", css("--muted"))
    .attr("font-size", 11).text("irrigation season");
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7).tickFormat(d3.utcFormat("%b %-d")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")));
  const line = key => d3.line().defined(d => r[key][d] !== null).x(d => x(dates[d])).y(d => y(r[key][d]));
  const idx = d3.range(dates.length);
  svg.append("path").attr("d", line("n1")(idx)).attr("fill", "none").attr("stroke", css("--normal")).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
  svg.append("path").attr("d", line("p1")(idx)).attr("fill", "none").attr("stroke", css("--risk")).attr("stroke-width", 2.2);
  for (const o of r.onsets) {
    const xo = x(dateOf(o));
    svg.append("path").attr("d", `M${xo},${H - m.b} l-6,-10 l12,0 z`).attr("fill", css("--drought"));
  }
  const ss = FW.season(c, S.meta.dates, r.p1, r.p2, r.onsets, { year: S.year, baseline: "careful" });
  for (const e of ss.events) {
    const di = S.meta.dates.indexOf(e.date);
    svg.append("circle").attr("cx", x(dates[di])).attr("cy", y(r.p1[di] ?? 0)).attr("r", 6)
      .attr("fill", e.kind === "hit" ? css("--act") : css("--panel")).attr("stroke", css("--act")).attr("stroke-width", 2);
  }
  const cx = x(dates[S.i]);
  svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", css("--ink")).attr("stroke-width", 1).attr("opacity", .5);
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b)
    .attr("fill", "transparent").style("cursor", "pointer")
    .on("click", ev => { const [px] = d3.pointer(ev); setDate(d3.leastIndex(dates, d => Math.abs(x(d) - px))); });

  // Soil moisture and the Drought Monitor underneath.
  const H2 = 130, y2 = d3.scaleLinear().domain([0, 100]).range([H2 - 34, m.t]);
  const s2 = d3.select("#soilChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H2}`);
  s2.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y2).tickValues([20, 40, 80]));
  for (const v of [20, 40]) s2.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y2(v)).attr("y2", y2(v)).attr("stroke", css("--line")).attr("stroke-dasharray", v === 20 ? "0" : "3 3");
  const rz = r.rz.map(v => v === null ? 15 : v);   // no row = already below the line
  s2.append("path").attr("d", d3.line().x((d, k) => x(dates[k])).y(d => y2(d))(rz)).attr("fill", "none").attr("stroke", css("--soil")).attr("stroke-width", 2);
  const bw = (W - m.l - m.r) / dates.length + 1;
  r.usdm.forEach((u, k) => {
    if (u >= 1) s2.append("rect").attr("x", x(dates[k])).attr("width", bw).attr("y", H2 - 26).attr("height", 10)
      .attr("fill", css("--drought")).attr("opacity", 0.35 + 0.15 * u);
  });
  s2.append("text").attr("x", m.l).attr("y", H2 - 4).attr("fill", css("--muted")).attr("font-size", 11).text("Drought Monitor");
  s2.append("line").attr("x1", cx).attr("x2", cx).attr("y1", m.t).attr("y2", H2 - 16).attr("stroke", css("--ink")).attr("opacity", .5);

  // Season tally under both usual practices.
  const cal = FW.season(FW.curve({ ...params(), reaction_days: 7 }, 21), S.meta.dates, r.p1, r.p2, r.onsets,
                        { year: S.year, baseline: "calendar", calendar: S.farm.calendar });
  const acres = S.farm.acres;
  const row = (title, s) => `<div class="row"><div class="note">${title}</div><div class="v ${s.value >= 0 ? "pos" : "neg"}">${usd(s.value)}/acre</div>
    <div class="note">${usd(s.value * acres)} on ${acres} acres; ${s.hits} of ${s.onsets} dry spells caught early, ${s.falseAlarms} false alarm${s.falseAlarms === 1 ? "" : "s"}</div></div>`;
  $("tally").innerHTML = row("vs a careful irrigator (starts 2 days after stress)", ss) +
                         row(`vs a calendar irrigator (starts ${S.farm.calendar})`, cal);
  $("events").innerHTML = ss.events.length ? ss.events.map(e => e.kind === "hit"
    ? `<li>${fmtDate(e.date)}: start the pivot; a flash drought began ${e.notice} days later (${usd(e.value)}/acre)</li>`
    : `<li>${fmtDate(e.date)}: start the pivot; no dry spell came (${usd(e.value)}/acre of pumping)</li>`).join("")
    : `<li>FlashWatch never said "start early" this summer${ss.onsets ? `; ${ss.onsets} dry spell${ss.onsets > 1 ? "s" : ""} went unforecast` : ""}.</li>`;
}

function renderNotice(c) {
  const W = 520, H = 220, m = { l: 44, r: 12, t: 10, b: 30 };
  const rows = c.rows.slice(0, 15);
  const x = d3.scaleLinear().domain([0, 14]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, d3.max(rows, r => Math.max(r.avoided, r.extraCost)) * 1.15 || 1]).range([H - m.b, m.t]).nice();
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v);
  const svg = d3.select("#noticeChart").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(v => "$" + v));
  svg.append("text").attr("x", W - m.r).attr("y", H - 2).attr("text-anchor", "end").attr("fill", css("--muted")).attr("font-size", 11).text("days of warning before the soil runs dry");
  const ln = key => d3.line().x(r => x(r.notice)).y(r => y(r[key]));
  svg.append("path").attr("d", ln("avoided")(rows)).attr("fill", "none").attr("stroke", css("--calm")).attr("stroke-width", 2.5);
  svg.append("path").attr("d", ln("extraCost")(rows)).attr("fill", "none").attr("stroke", css("--act")).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
  const last = rows[rows.length - 1];
  svg.append("text").attr("x", x(14)).attr("y", y(last.avoided) - 6).attr("text-anchor", "end").attr("fill", css("--calm")).attr("font-size", 12).text("crop loss avoided");
  svg.append("text").attr("x", x(14)).attr("y", y(last.extraCost) - 6).attr("text-anchor", "end").attr("fill", css("--act")).attr("font-size", 12).text("extra pumping");
  const sat = rows.findIndex(r => r.avoided >= last.avoided - 0.5);
  $("noticeNote").innerHTML = `Your pivot puts down ${c.capacity.toFixed(2)} in/day; the crop uses ${S.farm.et_peak.toFixed(2)} in/day in a heat wave.
    Without rain the crop is stressed ${c.stressDay} days into a dry spell. Warning ${sat > 0 ? `helps up to about <strong>${sat} days</strong>` : "barely matters"}: ${sat > 0 ? "after that the soil is already full when the pivot starts." : "your pivot can keep up with the crop if started on time."}`;
}

function renderPicker() {
  if (!S.states) return;
  const W = 520, H = 260;
  const box = { type: "MultiPoint", coordinates: [[-104, 36], [-80, 36], [-80, 49], [-104, 49]] };
  const proj = d3.geoMercator().fitExtent([[4, 4], [W - 4, H - 4]], box);
  const svg = d3.select("#picker").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").selectAll("path").data(S.states.features).join("path").attr("class", "state").attr("d", d3.geoPath(proj));
  const [fx, fy] = proj([S.lon, S.lat]);
  svg.append("circle").attr("class", "farm").attr("cx", fx).attr("cy", fy).attr("r", 7);
  svg.on("click", ev => {
    const [px, py] = d3.pointer(ev);
    const [lon, lat] = proj.invert([px, py]);
    setLocation(lat, lon);
  });
}

// ---------- state changes ----------

async function setLocation(lat, lon) {
  S.lat = lat; S.lon = lon;
  $("lat").value = lat.toFixed(2); $("lon").value = lon.toFixed(2);
  const cell = nearestCell(lat, lon);
  if (!cell) {
    $("place").textContent = "Outside FlashWatch's area";
    $("placeSub").textContent = "FlashWatch covers 36-49 N, 104-80 W (the Corn Belt and northern Plains).";
    return;
  }
  S.cell = cell;
  const cty = nearestCounty(lat, lon);
  $("place").textContent = cty ? `${cty[1]}, ${cty[2]}` : `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
  const [clat, clon] = cellLatLon(cell);
  $("placeSub").textContent = `Your farm at ${lat.toFixed(2)}, ${lon.toFixed(2)}; forecast for the ${(S.meta.res * S.meta.thin * 111).toFixed(0)} km grid square centred ${clat.toFixed(2)}, ${clon.toFixed(2)}`;
  S.rec = await loadTile(S.year, cell);
  renderAll();
}

function setDate(i) {
  S.i = Math.max(0, Math.min(S.meta.dates.length - 1, i));
  $("date").value = S.i;
  renderAll();
}

function renderSettings() {
  const el = $("settings");
  el.innerHTML = FARM_FIELDS.map(([k, label, type, opt]) => type === "select"
    ? `<label>${label}<select data-k="${k}">${opt.map(o => `<option${S.farm[k] === o ? " selected" : ""}>${o}</option>`).join("")}</select></label>`
    : `<label>${label}<input data-k="${k}" type="${type}" ${type === "number" ? `step="${opt}"` : ""} value="${S.farm[k]}"></label>`).join("");
  el.querySelectorAll("[data-k]").forEach(inp => inp.addEventListener("change", () => {
    const k = inp.dataset.k;
    if (k === "crop") { S.farm = { ...defaultFarm(inp.value), acres: S.farm.acres, soil: S.farm.soil }; renderSettings(); }
    else S.farm[k] = inp.type === "number" ? +inp.value : inp.value;
    saveFarm();
    renderAll();
  }));
}

async function init() {
  S.meta = await fetch("data/meta.json").then(r => r.json());
  S.farm = loadFarm();
  readHash();
  if (!S.meta.years.includes(S.year)) S.year = S.meta.years.includes(2012) ? 2012 : S.meta.years[0];
  $("year").innerHTML = S.meta.years.map(y => `<option${y === S.year ? " selected" : ""}>${y}</option>`).join("");
  $("date").max = S.meta.dates.length - 1;
  S.i = Math.max(0, S.meta.dates.indexOf(S.wantDate || "07-03"));
  $("date").value = S.i;
  $("modelNote").textContent = S.meta.note ? `Forecasts on this page: ${S.meta.note}.` : "";
  $("replayNote").textContent = `Replay of the ${S.year} season`;
  renderSettings();
  $("go").onclick = () => setLocation(+$("lat").value, +$("lon").value);
  $("year").onchange = async () => { S.year = +$("year").value; $("replayNote").textContent = `Replay of the ${S.year} season`; S.rec = await loadTile(S.year, S.cell); renderAll(); };
  $("date").oninput = () => setDate(+$("date").value);
  $("prev").onclick = () => setDate(S.i - 1);
  $("next").onclick = () => setDate(S.i + 1);
  try {
    const us = await fetch("https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json").then(r => r.json());
    S.states = topojson.feature(us, us.objects.states);
  } catch (e) { /* map optional */ }
  await setLocation(S.lat, S.lon);
}

init();
