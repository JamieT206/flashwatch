// FlashWatch region page: the roll-up, a regional risk map and the evidence.
"use strict";

const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const R = { meta: null, year: 2012, i: 0, recs: null, states: null, roll: {}, ev: null };

const money = v => {
  const a = Math.abs(v), s = v < 0 ? "-$" : "$";
  return a >= 1e9 ? `${s}${(a / 1e9).toFixed(2)} billion` : a >= 1e6 ? `${s}${(a / 1e6).toFixed(0)} million` : `${s}${(a / 1e3).toFixed(0)}k`;
};
const people = v => v >= 1e6 ? `${(v / 1e6).toFixed(1)} million` : `${(v / 1e3).toFixed(0)}k`;
const VARIANTS = [["m1", "Land-only forecasts (today's model, measured)"],
                  ["blur07", "Weather forecasts of ECMWF quality (estimate)"],
                  ["perfect", "Perfect weather forecasts (ceiling)"]];

function headline(tag, base) {
  const r = R.roll[tag];
  return r ? r.headline.find(h => h.baseline === base) : null;
}

function renderRoll() {
  const base = $("base").value;
  const main = headline("blur07", base) || headline("m1", base);
  if (!main) { $("kpis").textContent = "Roll-up not computed yet."; return; }
  $("kpis").innerHTML = `
    <div><div class="bignum">${money(main.us_irrigated_dollars)}</div><div class="note">a year, net of extra pumping and false alarms</div></div>
    <div><div class="bignum">${people(main.us_people_fed)}</div><div class="note">people's yearly calories in grain saved from drought</div></div>
    <div><div class="bignum">${(100 * main.us_share_of_retail_loss_by_weight).toFixed(1)}%</div><div class="note">of what US grocery stores throw away, by weight</div></div>`;
  const rows = VARIANTS.filter(([t]) => headline(t, base)).map(([t, label]) => {
    const h = headline(t, base);
    return `<tr><td>${label}</td><td>${money(h.region_irrigated_dollars)}</td><td>${money(h.us_irrigated_dollars)}</td>
      <td>$${h.per_acre_corn.toFixed(2)} / $${h.per_acre_soy.toFixed(2)}</td><td>${people(h.us_people_fed)}</td><td>${money(h.rainfed_nitrogen_dollars)}</td></tr>`;
  }).join("");
  $("rollTable").innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:13.5px">
    <thead><tr style="text-align:left;color:var(--muted)"><th>Forecasts used</th><th>Study region, irrigated</th><th>All US irrigated corn + soy</th>
    <th>Per acre a year (corn / soy)</th><th>People fed (US)</th><th>Rain-fed corn, nitrogen (placeholder costs)</th></tr></thead><tbody>${rows}</tbody></table>`;
  $("rollNote").innerHTML = `The headline uses the middle row: the full-season ECMWF forecasts are still downloading, so it is estimated from
    weather blurred to ECMWF's measured accuracy. Most of the value is in corn; soybeans are less sensitive to a dry spell and worth less per acre.
    Year to year it swings widely: drought years carry most of it.`;
}

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
  const svg = d3.select("#map").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const color = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, Math.log2(8)]);
  const step = m.res * m.thin;
  const day = (s) => Date.UTC(R.year, +s.slice(0, 2) - 1, +s.slice(3, 5)) / 864e5;
  const t0 = day(md);
  for (const r of R.recs) {
    const lat = m.lat0 - r.iy * m.res, lon = m.lon0 + r.ix * m.res;
    const [x0, y0] = proj([lon - step / 2, lat + step / 2]), [x1, y1] = proj([lon + step / 2, lat - step / 2]);
    let fill;
    if (r.p1[i] === null) fill = r.rz[i] === null ? css("--drought") : css("--line");
    else fill = color(Math.max(0, Math.log2(r.p1[i] / Math.max(r.n1[i], 1e-4))));
    svg.append("rect").attr("x", x0).attr("y", y0).attr("width", x1 - x0 + 0.5).attr("height", y1 - y0 + 0.5)
      .attr("fill", fill).attr("opacity", r.p1[i] === null && r.rz[i] === null ? 0.45 : 1).style("cursor", "pointer")
      .on("click", () => { location.href = `index.html#lat=${lat.toFixed(2)}&lon=${lon.toFixed(2)}&year=${R.year}&date=${md}`; });
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

function renderSkill() {
  const ev = R.ev, W = 520, H = 230, m = { l: 40, r: 10, t: 10, b: 40 };
  const groups = [["all dates", "M1", "Land only", css("--normal")], ["forecast dates", "M2", "With ECMWF forecasts", css("--risk")],
                  ["perfect weather", "M2", "Perfect weather (ceiling)", css("--line")]];
  const leads = ["week1", "week2", "any28"], names = { week1: "Next 7 days", week2: "Days 8-14", any28: "Within 28 days" };
  const val = (g, mod, lead) => (ev.skill[g] || []).find(r => r.model === mod && r.lead === lead);
  const x0 = d3.scaleBand().domain(leads).range([m.l, W - m.r]).padding(0.2);
  const x1 = d3.scaleBand().domain(groups.map(g => g[2])).range([0, x0.bandwidth()]).padding(0.08);
  const y = d3.scaleLinear().domain([0, 0.5]).range([H - m.b, m.t]);
  const svg = d3.select("#skill").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x0).tickFormat(l => names[l]));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  for (const lead of leads) for (const [g, mod, label, col] of groups) {
    const v = val(g, mod, lead); if (!v) continue;
    const bx = x0(lead) + x1(label);
    svg.append("rect").attr("x", bx).attr("width", x1.bandwidth()).attr("y", y(Math.max(0, v.bss))).attr("height", y(0) - y(Math.max(0, v.bss))).attr("fill", col).attr("rx", 2);
    svg.append("text").attr("x", bx + x1.bandwidth() / 2).attr("y", y(Math.max(0, v.bss)) - 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", css("--muted")).text((100 * v.bss).toFixed(0) + "%");
  }
  $("skillNote").innerHTML = groups.map(([, , label, col]) => `<span style="display:inline-block;width:10px;height:10px;background:${col};border-radius:2px"></span> ${label}`).join(" &nbsp; ") +
    `<br>"With ECMWF forecasts" is scored on the ${ev.m2_dates} spring start dates downloaded so far. Week 1 is where the forecasts pay; beyond two weeks no weather forecast can see the dry spells.`;
}

function renderValue() {
  const ev = R.ev, W = 520, H = 230, m = { l: 40, r: 10, t: 10, b: 34 };
  const x = d3.scaleLog().domain([0.02, 0.5]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([-0.1, 0.8]).range([H - m.b, m.t]);
  const svg = d3.select("#value").html("").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(ev.alphas).tickFormat(d3.format("~g")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("x", W - m.r).attr("y", H - 2).attr("text-anchor", "end").attr("font-size", 11).attr("fill", css("--muted")).text("cost of acting / loss it avoids");
  svg.append("rect").attr("x", x(0.1)).attr("width", x(0.2) - x(0.1)).attr("y", m.t).attr("height", H - m.t - m.b).attr("fill", css("--calm-bg"));
  svg.append("text").attr("x", x(0.1) + 4).attr("y", m.t + 12).attr("font-size", 10).attr("fill", css("--muted")).text("early pivot start");
  const series = [["forecast dates|M2", "week1", css("--risk"), "Week 1, with ECMWF"], ["forecast dates|M1", "week1", css("--normal"), "Week 1, land only"],
                  ["forecast dates|M2", "week2", css("--wait"), "Week 2, with ECMWF"]];
  for (const [k, lead, col, label] of series) {
    const v = ev.value[k]?.[lead]; if (!v) continue;
    svg.append("path").attr("d", d3.line().x((d, j) => x(ev.alphas[j])).y(d => y(d))(v)).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2.2);
    svg.append("text").attr("x", x(ev.alphas[1]) + 4).attr("y", y(v[1]) + (label.includes("land") ? 14 : -5)).attr("font-size", 11).attr("fill", col).text(label);
  }
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", css("--line"));
}

function renderMonitor() {
  const mo = R.ev.monitor;
  const card = (t, s, sub) => `<div class="row"><div class="note">${t}</div><div class="v">${s}</div><div class="note">${sub}</div></div>`;
  $("monitor").innerHTML =
    card("Soil crosses the drought line vs the Monitor's first D1", `${mo.crossing_vs_d1.median_days_ahead.toFixed(0)} days earlier`, `median; FlashWatch first ${(100 * mo.crossing_vs_d1.share_flashwatch_first).toFixed(0)}% of the time`) +
    card("vs the Monitor's first worsening of any kind", `${mo.crossing_vs_worsening.median_days_ahead.toFixed(0)} days earlier`, `median; first ${(100 * mo.crossing_vs_worsening.share_flashwatch_first).toFixed(0)}% of the time`);
  $("monitorNote").textContent = `Over ${mo.onsets.toLocaleString()} flash droughts, 2006-2021, counting each from the day the data are public. The week-1 forecast adds about a week more. The Monitor never reacts to about half of them; whether those cost crops is still open.`;
}

function renderWatch() {
  const w = R.ev.watch;
  const card = (t, k) => w[k] ? `<div class="row"><div class="note">${t}</div><div class="v">${(100 * w[k].week1.caught).toFixed(0)}% caught</div><div class="note">1 in ${Math.round(1 / w[k].week1.hit_rate)} flagged squares had one</div></div>` : "";
  $("watch").innerHTML = card("Land only, next 7 days", "forecast dates|M1") + card("With ECMWF forecasts, next 7 days", "forecast dates|M2");
}

async function init() {
  R.meta = await fetch("data/meta.json").then(r => r.json());
  R.ev = await fetch("data/evidence.json").then(r => r.json()).catch(() => null);
  for (const [t] of VARIANTS) R.roll[t] = await fetch(`data/rollup_${t}.json`).then(r => r.ok ? r.json() : null).catch(() => null);
  try {
    const us = await fetch("https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json").then(r => r.json());
    R.states = topojson.feature(us, us.objects.states);
  } catch (e) { /* outlines optional */ }
  $("year").innerHTML = R.meta.years.map(y => `<option${y === R.year ? " selected" : ""}>${y}</option>`).join("");
  $("date").max = R.meta.dates.length - 1;
  R.i = R.meta.dates.indexOf("07-03");
  $("date").value = R.i;
  $("base").onchange = renderRoll;
  $("year").onchange = async () => { R.year = +$("year").value; await loadYear(R.year); renderMap(); };
  const setI = i => { R.i = Math.max(0, Math.min(R.meta.dates.length - 1, i)); $("date").value = R.i; renderMap(); };
  $("date").oninput = () => setI(+$("date").value);
  $("prev").onclick = () => setI(R.i - 1);
  $("next").onclick = () => setI(R.i + 1);
  renderRoll();
  await loadYear(R.year);
  renderMap();
  if (R.ev) { renderSkill(); renderValue(); renderMonitor(); renderWatch(); }
}

init();
