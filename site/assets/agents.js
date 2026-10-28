// FlashWatch agents page: real runs of each agent, from site/data/agents.json (flashwatch/site/agents.py).
"use strict";

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function pivots(p) {
  if (!p) { $("aPivots").hidden = true; return; }
  $("pivImg").innerHTML = `<img src="data/pivots/${p.stem}.png" alt="Aerial photo with the pivots SAM 3.1 found circled" class="aimg">
    <div class="note">USDA aerial photo, ${(p.half_m * 2 / 1000).toFixed(1)} km across, near ${p.lat}, ${p.lon}. Red: pivots SAM 3.1 found.</div>`;
  const tot = p.pivots.reduce((a, q) => a + q.acres, 0);
  $("pivList").innerHTML = `<div class="step"><b>Asked</b> "${esc(p.phrase)}"</div>
    <div class="step"><b>Found</b> ${p.pivots.length} pivots, ${tot.toLocaleString()} irrigated acres</div>
    <table><thead><tr><th>Pivot</th><th>Acres</th><th>Across</th><th>Centre</th></tr></thead><tbody>
    ${p.pivots.map((q, k) => `<tr><td>P${k + 1}</td><td>${q.acres}</td><td>${q.diameter_m} m</td><td>${q.lat}, ${q.lon}</td></tr>`).join("")}</tbody></table>
    <p class="note">The farm file this makes goes straight to the operator agent below. Crop, soil and pivot capacity start as guesses for the farmer
      to correct. Partly planted circles (a corner cut out for a farmyard) aren't picked up yet.</p>`;
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
    step(3, "The text to the farmer", `<div class="sms-ask">${esc(o.sms)}</div><p>${esc(o.note)}</p>
      <p class="note">${o.drafts} draft${o.drafts === 1 ? "" : "s"}; ${o.cut_untraced && o.cut_untraced.length ? `figures the checker couldn't trace, removed: ${o.cut_untraced.join(", ")}` : "every figure traced to a tool result"}. Model: ${esc(o.model)}.</p>`) +
    step(4, "Farmer replies YES; the connector sends it", `<ul class="events">${o.sent.map(c => `<li class="ok">${c.pivot}: ${c.action} ${c.depth_in ? c.depth_in + " in" : ""} (${c.state})</li>`).join("")}</ul>
      <p class="note">Approved by ${esc(o.approved_by)}. Two days later the as-applied water shows up in each pivot's root zone and the agent plans again.</p>`);
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
  const s = g.stats || {};
  $("glimmer").innerHTML = `<div class="kpis three">
      <div><div class="bignum">${s.train ?? "-"}</div><div class="note">weeks of real Drought Monitor narrative to learn from (2005-2019)</div></div>
      <div><div class="bignum">${s.val ?? "-"}</div><div class="note">held-out weeks to score against (2020-2021)</div></div>
      <div><div class="bignum">${Math.round(100 * g.mean.untraced)}%</div><div class="note">of figures made up, in the starting point (Muse Spark, no fine-tune)</div></div></div>
    <div class="agrid" style="margin-top:16px"><div><div class="tt">The Drought Monitor, week of ${g.week.date}</div><blockquote>${esc(g.week.real)}</blockquote></div>
      <div><div class="tt">Starting point: ${esc(g.model)}, from the same week's numbers</div><blockquote>${esc(g.week.draft)}</blockquote></div></div>
    <p class="note">Without fine-tuning, the model gets the facts and the states right but writes a list of numbers, not the Monitor's prose
      (word overlap with the real text: ${Math.round(100 * g.mean.rouge_l)}%). The fine-tune is to teach the style while keeping every figure traceable.
      Status: training data ready; fine-tune scheduled.</p>`;
}

function watch(w) {
  if (!w) { $("aWatch").hidden = true; return; }
  $("watch").innerHTML = w.farms.map(f => `<div><div class="tt">${esc(f.name)}: ${f.watering} ${f.crop}</div>
    <div class="note">${f.alerts.length} texts from ${f.dates_checked} forecasts, June to August ${w.year}</div>
    <ul class="texts">${f.alerts.map(a => `<li class="${a.level}"><span class="when">${new Date(a.date + "T12:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span>${esc(a.text)}${a.by !== "template" ? ` <span class="by">${esc(a.by)}</span>` : ""}</li>`).join("")}</ul></div>`).join("");
}

fetch("data/agents.json").then(r => r.json()).then(d => {
  pivots(d.pivots); operator(d.operator); watch(d.alerts); voice(d.voice); brief(d.brief); glimmer(d.glimmer);
}).catch(() => { document.querySelector(".intro").insertAdjacentHTML("afterend", "<p class='note'>Agent runs not built yet (python -m flashwatch.site.agents).</p>"); });
