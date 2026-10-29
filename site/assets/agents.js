// FlashWatch agents page: real runs of each agent, from site/data/agents.json (flashwatch/site/agents.py).
"use strict";

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function pivots(p) {
  if (!p) { $("aPivots").hidden = true; return; }
  $("pivImg").innerHTML = `<img src="data/pivots/${p.stem}.png" alt="Aerial photo with the pivots SAM 3.1 found circled" class="aimg">
    <div class="note">USDA aerial photo, ${(p.half_m * 2 / 1000).toFixed(1)} km across, near ${p.lat}, ${p.lon}. Red: full circles.
      Orange: part and half circles, found by a second question to SAM 3.1.</div>`;
  const tot = p.pivots.reduce((a, q) => a + q.acres, 0), part = p.pivots.filter(q => q.shape).length;
  $("pivList").innerHTML = `<div class="step"><b>Asked</b> "${esc(p.phrase)}"${p.phrase_partial ? `, then "${esc(p.phrase_partial)}"` : ""}</div>
    <div class="step"><b>Found</b> ${p.pivots.length} pivots${part ? ` (${part} part or half circles)` : ""}, ${tot.toLocaleString()} irrigated acres</div>
    <table><thead><tr><th>Pivot</th><th>Acres</th><th>Shape</th><th>Across</th><th>Centre</th></tr></thead><tbody>
    ${p.pivots.map((q, k) => `<tr><td>P${k + 1}</td><td>${q.acres}${q.acres_estimated ? "*" : ""}</td><td>${esc(q.shape || "full circle")}</td><td>${q.diameter_m} m</td><td>${q.lat}, ${q.lon}</td></tr>`).join("")}</tbody></table>
    <p class="note">The farm file this makes goes straight to the operator agent below. Crop, soil and pivot capacity start as guesses for the farmer
      to correct. *Acres estimated from the box (about 3/4 of the circle for a part circle, half for a half circle): the farmer confirms them.</p>`;
}

function operator(o) {
  if (!o) { $("aOperator").hidden = true; return; }
  const rows = Object.entries(o.pivots).map(([k, p]) => `<tr><td><strong>${k}</strong></td><td>${p.crop}, ${p.acres} ac</td>
    <td>${p.root_zone ? `${Math.round(100 * p.root_zone.share_full)}% full, ${p.root_zone.days_until_stress_if_no_rain} days to stress` : "-"}</td>
    <td>${esc(p.verdict)}</td><td>${p.allocation_left_in ?? "-"}</td><td>${p.farmer ? `"${esc(p.farmer.slice(0, 60))}..."` : ""}</td></tr>`).join("");
  const step = (n, title, body) => `<div class="tstep"><div class="tn">${n}</div><div><div class="tt">${title}</div>${body}</div></div>`;
  $("opTrace").innerHTML =
    step(1, `<code>farm_overview</code>: what the agent sees, ${o.date}`, `<div style="overflow-x:auto"><table><thead><tr><th>Pivot</th><th>Field</th>
      <th>Root zone</th><th>FlashWatch advice</th><th>Allocation left (in)</th><th>Farmer's voice note</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">Wells: ${Object.entries(o.wells || {}).map(([w, s]) => `${w} gives ${s.gives_gpm} gpm to ${s.pivots.join(" and ")}`).join("; ")}.</p>`) +
    step(2, "<code>propose_commands</code>: the farm rules check each one", `<ul class="events">${o.accepted.map(c => `<li class="ok">Accepted: ${c.action} ${c.pivot}${c.action === "start" ? `, ${c.depth_in} in` : ""}</li>`).join("")}
      ${o.refused.map(c => `<li class="no">Refused: ${c.action} ${c.pivot}: ${esc(c.why)}</li>`).join("")}</ul>`) +
    step(3, "As a text message (140 characters)", `<div class="sms-ask">${esc(o.sms)}</div>
      <p class="note">${o.drafts} draft${o.drafts === 1 ? "" : "s"}; ${o.cut_untraced && o.cut_untraced.length ? `figures the checker couldn't trace, removed: ${o.cut_untraced.join(", ")}` : "every figure traced to a tool result"}. Model: ${esc(o.model)}.</p>`) +
    step(4, "On WhatsApp: the farmer taps YES, the connector sends it", `<div class="wa">
        <div class="wa-in">${esc(o.note)} ${esc((o.sms.match(/Reply YES.*$/) || [""])[0])}<div class="wa-btns"><span>YES</span><span>NO</span></div></div>
        <div class="wa-out">YES</div>
        <div class="wa-in">Done. ${o.sent.map(c => `${c.pivot} ${c.action}${c.action === "start" ? ` ${c.depth_in} in` : ""}: ${c.state}`).join("; ")}.</div></div>
      <p class="note">Two days later the as-applied water shows up in each pivot's root zone and the agent plans again. The same thread takes
        questions and voice notes. WhatsApp Cloud API, tested on a real phone on Meta's test number: plan delivered,
        YES tapped, commands sent and the confirmation back within about a minute.</p>`);
}

function voice(v) {
  if (!v) { $("aVoice").hidden = true; return; }
  $("voice").innerHTML = `<div><div class="tt">Transcript, ${v.date}</div><blockquote>${esc(v.transcript)}</blockquote></div>
    <div><div class="tt">What Muse Spark took from it</div><table><thead><tr><th>Pivot</th><th>Crop</th><th>Stress</th><th>Signs</th></tr></thead><tbody>
    ${v.reports.map(r => `<tr><td>${esc(r.pivot)}</td><td>${esc(r.crop)}</td><td>${["fine", "mild", "showing", "severe"][r.severity] || r.severity}</td>
      <td>${esc((r.symptoms || []).join(", "))}</td></tr>`).join("")}</tbody></table></div>`;
}

function brief(b) {
  if (!b) { $("aBrief").hidden = true; return; }
  $("brief").innerHTML = `<div><div class="tt">${esc(b.place)}, ${b.issue}</div><div class="briefing">${esc(b.briefing).replace(/\n/g, "<br>")}</div></div>
    <div><div class="tt">Tools it called</div><ul class="events">${b.tool_calls.map(c => `<li><code>${esc(c.tool)}</code> ${esc(JSON.stringify(c.args))}</li>`).join("")}</ul>
    <p class="note">${b.drafts} draft${b.drafts === 1 ? "" : "s"}; ${b.cut_untraced && b.cut_untraced.length ? `untraced figures removed: ${b.cut_untraced.join(", ")}` : "every figure traced to a tool result"}. Model: ${esc(b.model)}.</p></div>`;
}

function glimmer(g) {
  if (!g) { $("aGlimmer").hidden = true; return; }
  const s = g.stats || {}, ex = g.example;
  const name = r => `${r.local ? "Muse Glimmer 30B, run locally" : "Muse Spark, API"}${r.shots ? `, shown ${r.shots} past weeks` : ""}`;
  const rows = g.rows.map(r => `<tr><td>${name(r)}</td><td>${Math.round(100 * r.rouge_l)}%</td><td>${Math.round(100 * r.states)}%</td>
    <td>${Math.round(100 * r.untraced)}%</td><td>${r.words}</td></tr>`).join("");
  $("glimmer").innerHTML = `<div class="kpis three">
      <div><div class="bignum">${s.train ?? "-"}</div><div class="note">weeks of real Drought Monitor narrative to learn from (2005-2019)</div></div>
      <div><div class="bignum">${s.val ?? "-"}</div><div class="note">held-out weeks to score against (2020-2021)</div></div>
      <div><div class="bignum">17 GB</div><div class="note">Glimmer, 4-bit, running on one desktop graphics card (12 GB) with no API</div></div></div>
    <div style="overflow-x:auto;margin-top:16px"><table><thead><tr><th>Writer, from the week's numbers</th><th>Wording shared with the real text</th>
      <th>States named right</th><th>Figures not in the input</th><th>Words</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="note">Scored on the ${g.weeks} held-out weeks every writer finished. "Shown 2 past weeks": two real narratives from earlier years
      come first in the prompt as examples; no training.</p>
    <div class="agrid" style="margin-top:16px"><div><div class="tt">The Drought Monitor, week of ${ex.date}</div><blockquote>${esc(ex.real)}</blockquote></div>
      <div><div class="tt">${name({local: ex.model.includes("glimmer"), shots: ex.shots})}, same week</div><blockquote>${esc(ex.draft)}</blockquote></div></div>
    <p class="note">On their own, both models write a list of numbers. Shown two past weeks, Glimmer picks up the Monitor's way of writing
      but starts borrowing details it wasn't given, such as a temperature range; the number checker catches these. The fine-tune is to
      learn the style from all ${s.train ?? 749} weeks while keeping every figure to the input. Status: running locally; fine-tune not yet run.</p>`
    + offline(g.offline);
}

function offline(o) {
  if (!o) return "";
  const gl = o.asks["muse-glimmer-30B"] || [], sp = o.asks["muse-spark-1.1"] || [];
  const rows = gl.map((a, i) => `<tr><td>${esc(a.q)}</td><td>${esc(a.answer)}<div class="note">${a.seconds} s; tools: ${a.tools.join(", ")}${a.cut.length ? `; checker cut: ${a.cut.join(", ")}` : ""}</div></td>
    <td>${sp[i] ? `${esc(sp[i].answer)}<div class="note">${sp[i].seconds} s; tools: ${sp[i].tools.join(", ")}${sp[i].cut.length ? `; checker cut: ${sp[i].cut.join(", ")}` : ""}</div>` : "-"}</td></tr>`).join("");
  return `<h4 style="margin-top:24px">The same agents, offline</h4>
    <p>Glimmer can also run FlashWatch's agents with no internet: the same tools, farm rules and number checker as Muse Spark.
      For the demo farm on 3 July 2012 it planned the same morning as Spark (${o.operator.accepted.map(c => `start ${c.pivot} at ${c.depth_in} in`).join(", ")}),
      with every figure in its first draft traced, in about 8 minutes on a 12 GB graphics card.</p>
    <div class="sms-ask">${esc(o.operator.sms)}</div>
    <div style="overflow-x:auto;margin-top:12px"><table class="asks"><thead><tr><th>Farmer asks</th><th>Muse Glimmer 30B, run locally</th><th>Muse Spark, API</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="note">Glimmer answers correctly but more slowly, and is less thorough: on the first question it didn't look at the field's water account,
      and on the third it left out week 2. Spark stays the default; Glimmer is the fallback where there's no connection, or no data should leave the farm.</p>`;
}

function then(o, warn) {
  if (!o) return "";
  const txt = o.flash_drought_within_14_days ? `a flash drought began within 2 weeks${warn ? "" : " anyway"}.` : "no flash drought in the next 2 weeks.";
  return `<span class="then ${o.right ? "right" : "wrong"}">${o.right ? "Right" : "Wrong"}: ${txt}</span>`;
}

function watch(w) {
  if (!w) { $("aWatch").hidden = true; return; }
  $("watch").innerHTML = w.farms.map(f => `<div><div class="tt">${esc(f.name)}: ${f.watering} ${f.crop}</div>
    <div class="note">${f.alerts.length} texts from ${f.dates_checked} forecasts, June to August ${w.year}</div>
    <ul class="texts">${f.alerts.map(a => `<li class="${a.level}"><span class="when">${new Date(a.date + "T12:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span>${esc(a.text)}${a.by !== "template" ? ` <span class="by">${esc(a.by)}</span>` : ""}${then(a.outcome, a.level !== "info")}</li>`).join("")}</ul></div>`).join("");
}

const KIND = { "irrigated:start": "Start the pivot today", "irrigated:watch": "Risk rising: not yet", "irrigated:ease": "Risk has eased",
  "rain-fed:act": "Act now on a rain-fed decision" };

function scorecard(c) {
  if (!c) { $("scorecard").hidden = true; return; }
  const yrs = c.years.length > 1 ? `${c.years[0]}-${c.years[c.years.length - 1]} (${c.years.length} seasons)` : c.years[0];
  $("scorecard").innerHTML = `<div class="tt">How often were the texts right?</div>
    <p class="note">The same watcher on ${c.farms} farms spread across the region, irrigated and rain-fed corn, every summer in ${yrs}.
      Each season was forecast by a model that never saw it. A flash drought followed ${c.normal_pct}% of all summer forecast dates
      within 2 weeks (soil not already in drought): that's the normal chance to beat.</p>
    <table><thead><tr><th>Text</th><th>Sent</th><th>Flash drought within 2 weeks</th><th>Times normal</th></tr></thead><tbody>
    ${c.kinds.map(k => `<tr><td>${KIND[k.watering + ":" + k.kind] || k.kind}</td><td>${k.sent.toLocaleString()}</td>
      <td>${k.followed_by_flash_drought_pct}%</td><td>${k.times_normal ?? "-"}</td></tr>`).join("")}</tbody></table>
    <p class="note">"Risk has eased" means lower than it was, not gone. ${c.onsets_warned_pct}% of the ${c.onsets.toLocaleString()} summer flash droughts
      at these farms had a warning text in the 2 weeks before: the rest came with little sign in the forecast, or while the farm was already being told to water.</p>`;
}

fetch("data/agents.json").then(r => r.json()).then(d => {
  pivots(d.pivots); operator(d.operator); watch(d.alerts); scorecard(d.scorecard); voice(d.voice); brief(d.brief); glimmer(d.glimmer);
}).catch(() => { document.querySelector(".intro").insertAdjacentHTML("afterend", "<p class='note'>Agent runs not built yet (python -m flashwatch.site.agents).</p>"); });
