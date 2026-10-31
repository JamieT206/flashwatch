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

// A WhatsApp-style phone. Messages: {day}, {in, btns, tag}, {out}, {voice: seconds} (from the farmer).
function phone(msgs, sub = "") {
  const t = m => m.time ? `<span class="ph-time">${m.time}</span>` : "";
  const body = msgs.map(m => m.day ? `<div class="ph-day">${m.day}</div>`
    : m.voice ? `<div class="ph-out ph-voice"><span class="ph-play">&#9654;</span><span class="ph-wave"></span>0:${String(m.voice).padStart(2, "0")}${t(m)}</div>`
    : m.out ? `<div class="ph-out">${m.out}${t(m)}</div>`
    : `<div class="ph-in">${m.in}${t(m)}${m.btns ? `<div class="ph-btns">${m.btns.map(b => `<span>${b}</span>`).join("")}</div>` : ""}</div>${m.tag || ""}`).join("");
  return `<div class="phone"><div class="ph-top"><span class="ph-av"></span><div><b>FlashWatch</b><small>${sub || "business account"}</small></div></div>
    <div class="ph-chat">${body}</div></div>`;
}
const day = d => new Date(d + "T12:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" });

function operator(o, ask, v) {
  if (!o) { $("aOperator").hidden = true; return; }
  const rows = Object.entries(o.pivots).map(([k, p]) => `<tr><td><strong>${k}</strong></td><td>${p.crop}, ${p.acres} ac</td>
    <td>${p.root_zone ? `${Math.round(100 * p.root_zone.share_full)}% full, ${p.root_zone.days_until_stress_if_no_rain} days to stress` : "-"}</td>
    <td>${esc(p.verdict)}</td><td>${p.allocation_left_in ?? "-"}</td></tr>`).join("");
  const step = (n, title, body) => `<div class="tstep"><div class="tn">${n}</div><div><div class="tt">${title}</div>${body}</div></div>`;
  $("opTrace").innerHTML =
    step(1, `What FlashWatch sees on the farm, ${day(o.date)}`, `<div style="overflow-x:auto"><table><thead><tr><th>Pivot</th><th>Field</th>
      <th>Root zone</th><th>FlashWatch advice</th><th>Allocation left (in)</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">Wells: ${Object.entries(o.wells || {}).map(([w, s]) => `${w} gives ${s.gives_gpm} gpm to ${s.pivots.join(" and ")}`).join("; ")}.</p>`) +
    step(2, "The farm's rules check the plan", `<ul class="events">${o.accepted.map(c => `<li class="ok">Accepted: ${c.action} ${c.pivot}${c.action === "start" ? `, ${c.depth_in} in` : ""}</li>`).join("")}
      ${o.refused.map(c => `<li class="no">Refused: ${c.action} ${c.pivot}: ${esc(c.why)}</li>`).join("")}</ul>`) +
    step(3, "The farmer approves on WhatsApp, and the pivot starts", `<p class="note">The pivots here are simulated.</p>`);
  const q = (ask || []).find(a => /unusual/.test(a.q)) || (ask || [])[0];
  $("opPhone").innerHTML = phone([
    { day: day(o.date) },
    ...(v ? [{ voice: 17, time: "6:12 AM" },
             { in: `Heard: "${esc(v.transcript.slice(0, 140))}..." Saved as a field report; the morning plan takes it into account.`, time: "6:12 AM" }] : []),
    { in: esc(o.sms), time: "6:30 AM", btns: ["YES", "NO"] },
    { out: "YES", time: "6:41 AM" },
    { in: `Done. ${o.sent.map(c => `${c.pivot} ${c.action}${c.action === "start" ? ` ${c.depth_in} in` : ""}: ${c.state}`).join("; ")}.`, time: "6:41 AM" },
    ...(q ? [{ out: esc(q.q), time: "7:05 AM" }, { in: esc(q.answer).replace(/\n+/g, "<br>"), time: "7:05 AM" }] : []),
  ]);
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
  const name = r => `${r.tuned ? "Muse Glimmer 30B, fine-tuned" : r.local ? "Muse Glimmer 30B, run locally" : "Muse Spark, API"}${r.shots ? `, shown ${r.shots} past weeks` : ""}`;
  const rows = g.rows.map(r => `<tr><td>${name(r)}</td><td>${Math.round(100 * r.rouge_l)}%</td><td>${Math.round(100 * r.states)}%</td>
    <td>${Math.round(100 * r.untraced)}%</td><td>${r.words}</td></tr>`).join("");
  $("glimmer").innerHTML = `<div class="kpis three">
      <div><div class="bignum">${s.train ?? "-"}</div><div class="note">weeks of real Drought Monitor narrative to learn from (2005-2019)</div></div>
      <div><div class="bignum">${s.val ?? "-"}</div><div class="note">held-out weeks to score against (2020-2021)</div></div>
      <div><div class="bignum">1 hour</div><div class="note">to fine-tune Glimmer (a LoRA adapter) on one cloud GPU</div></div></div>
    <div style="overflow-x:auto;margin-top:16px"><table><thead><tr><th>Writer, from the week's numbers</th><th>Wording shared with the real text</th>
      <th>States named right</th><th>Figures not in the input</th><th>Words</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="tt" style="margin-top:20px">Both written about the week of ${day(ex.date)}</div>
    <div class="agrid"><div><div class="tt">${name({local: ex.model.includes("glimmer"), tuned: ex.model.includes("lora"), shots: ex.shots})}</div><blockquote>${esc(ex.draft)}</blockquote></div>
      <div><div class="tt">The Drought Monitor</div><blockquote>${esc(ex.real)}</blockquote></div></div>` + nomap(g.nomap, name);
}

// The same writers given last week's Drought Monitor map instead of this week's, so the summary can come out first.
function nomap(n, name) {
  if (!n) return "";
  const rows = n.rows.map(r => `<tr><td>${name(r)}</td><td>${Math.round(100 * r.direction)}%</td><td>${Math.round(100 * r.rouge_l)}%</td>
    <td>${Math.round(100 * r.states)}%</td><td>${Math.round(100 * r.untraced)}%</td></tr>`).join("");
  return `<div class="tt" style="margin-top:24px">Before the Monitor's map is out</div>
    <p class="note">The same task with last week's Drought Monitor map instead of this week's, so the summary can be written as soon as the
      week's weather data is in. The writer now has to judge whether drought spread or eased. ${n.weeks} held-out weeks.</p>
    <p class="note">Glimmer, fine-tuned again on these inputs, keeps the Monitor's style but drifts to places it wasn't given (Texas and
      Oklahoma, which older narratives often covered), and neither writer calls the direction better than half the time.
      So FlashWatch's summary still waits for the Monitor's map.</p>
    <div style="overflow-x:auto"><table><thead><tr><th>Writer, from the week's numbers and last week's map</th><th>Got right whether drought spread or eased</th>
      <th>Wording shared with the real text</th><th>States named right</th><th>Figures not in the input</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function then(o, warn) {
  if (!o) return "";
  const txt = o.flash_drought_within_14_days ? `a flash drought began within 2 weeks${warn ? "" : " anyway"}` : "no flash drought in the next 2 weeks";
  return `<div class="ph-tag ${o.right ? "right" : "wrong"}">${o.right ? "Right" : "Wrong"}: ${txt}</div>`;
}

// The value of a recommendation, for texts written before the watcher's templates carried it.
function worth(a) {
  const v = a.facts && a.facts.value_of_starting_today_dollars_per_acre;
  if (v === undefined || v === null || /\$/.test(a.text)) return "";
  return `<br><b>Starting today: ${v < 0 ? "-" : "+"}$${Math.abs(v).toFixed(1)} an acre on average.</b>`;
}

function watch(w) {
  if (!w) { $("aWatch").hidden = true; return; }
  const f = w.farms.reduce((a, b) => b.alerts.filter(x => x.outcome).length > a.alerts.filter(x => x.outcome).length ? b : a);
  const msgs = [];
  for (const a of f.alerts) {
    const d = day(a.date);
    if (!msgs.length || msgs[msgs.length - 1].d !== d) msgs.push({ day: d, d });
    msgs.push({ in: esc(a.text) + worth(a), time: "6:30 AM", tag: then(a.outcome, a.level !== "info") });
  }
  $("watch").innerHTML = phone(msgs, `${esc(f.name)}: ${f.watering} ${f.crop}, ${w.year}`) +
    `<p class="note">${f.alerts.length} messages from ${f.dates_checked} forecasts, June to August ${w.year}.</p>`;
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
  pivots(d.pivots); operator(d.operator, d.ask, d.voice); watch(d.alerts); scorecard(d.scorecard); voice(d.voice); brief(d.brief); glimmer(d.glimmer);
}).catch(() => { document.querySelector(".intro").insertAdjacentHTML("afterend", "<p class='note'>Agent runs not built yet (python -m flashwatch.site.agents).</p>"); });
