# FlashWatch: handover

Last updated **Tue Oct 27, 2026 (Sydney morning) / Mon Oct 26 ~21:00 UTC**, at the move from a
Claude Code cloud session to Claude Code on James's Windows PC. Replaces the Oct 26 version.
Read sections 0-5 before touching anything. The repo is **public**: never commit secrets,
credentials, account names, raw ECMWF data or anything from `data/`.

---

## 0. If you are an AI assistant reading this

- You are helping **James Twiss** (Sydney, UTC+11), an expert in predictive analytics and ML.
  Skip basic explanations; surface status and decisions clearly.
- 10-day hackathon. **Hard deadline: Nov 4, 9:00 PM US Eastern = Nov 5 02:00 UTC = Nov 5 1:00 PM
  Sydney.** Plan backwards with at least 3 hours of buffer.
- The design is settled (`FlashWatch Design Spec.pdf` + section 6). Don't redesign; raise any
  deviation with James first. Open questions live in **`QUESTIONS.md`**; James answers there or in chat.
- Work on branch **`claude/practical-planck-eyczp0`** (all code is there; `main` still holds only
  the original scoping commit). Commit and push regularly; James reviews from work via GitHub.
- Credentials are in `flashwatch.env` (or `.env`) in the repo root, gitignored. `load_env()` reads
  either. Never print them.
- Pre-registered expectations (section 6.6) must be reported honestly, including nulls.
- You are now on James's machine: you can see `data/`, run the pipeline and read logs directly.
  The previous cloud session could not; everything in `flashwatch/` was tested only on synthetic
  data (section 4.3).

---

## 1. The project in one paragraph

**FlashWatch** forecasts the *onset* of flash drought 1-4 weeks ahead on a 0.25 degree grid over the
US Corn Belt and Plains, and turns the forecasts into county advisories using a Meta-model agent
that checks its own track record. The forecasting core is classical ML (LightGBM on ERA5-Land
land state plus ECMWF sub-seasonal reforecasts). The agent explains and verifies; it never
forecasts. Headline claim to test: calibrated onset probabilities that beat climatology,
persistence and raw ECMWF on held-out years, at least at 1-2 week leads. The demo replays the
2023 Midwest flash drought week by week against the US Drought Monitor.

- Live spec (Claude Doc, editable, the master copy): https://claude.ai/code/artifact/8f6f3a21-ca1c-446c-a32f-253272029e7c
- `FlashWatch Design Spec.pdf`: Oct 17 snapshot. **Out of date on test years** (says 2023-2025; now 2022-2024, see D1).

---

## 2. Hackathon facts (official rules read: `docs/hackathon-rules.pdf`)

| Item | Detail |
| --- | --- |
| Event | Meta Global AI Developer Hackathon (Cerebral Valley), virtual. James accepted, Indie track |
| Dates | Kickoff Oct 26 12:00 PM EDT. Workshops and Discord office hours Oct 27 - Nov 4. **Submit by Nov 4, 9 PM ET** at https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit. Judging Nov 4-8, finals Nov 9 |
| **Judging** | **Impact 30%** (real-world potential, fits a theme) - **Use of Meta AI tooling 25%** ("how creatively... beyond a basic integration... surfaced unexpected capabilities") - **Technical depth & execution 20%** - **Demo 25%** (working, impressive, holds up live, video quality) |
| Themes | "Build for Your World" and "Push the Frontier". FlashWatch is Push the Frontier (understand and anticipate a physical system) |
| Indie prizes | $150k / $100k / $75k / $50k / $25k; bonus prizes up to $50k total |
| Submission rules | Original work, owned by the entrant; outside technical help allowed if the entrant owns the result; open source allowed under its licences. Entrant grants organisers a perpetual licence to use the submission "for any purpose" (see QUESTIONS Q2 re ECMWF CC BY-NC). **No stated video length, repo requirement or format**; aim for ~3 min video, public repo, live demo page |
| Meta models | Muse Spark series, Muse Wonder, Muse Image, SAM 3.1 via Meta Model API (OpenAI-compatible, base `https://api.meta.ai/v1`, model ids like `muse-spark-1.1`); open weights **Muse Glimmer 30B** (Apache 2.0, HF `meta-models/Muse-Glimmer-30B`, GGUF builds exist); Muse Code |
| Credits | $150 Meta API credits: **James has applied and not yet heard back**. Partner credits (Modal $100, Baseten $100, Fireworks $25) for select participants: not yet claimed |
| Meta API from Australia | Untested (no key yet). The cloud sandbox got HTTP 403 from `api.meta.ai` even with network open, so test from James's PC |

---

## 3. Accounts and access

All accounts exist and were tested Oct 17 (old PC) and Oct 26 (this PC). Variables live in
`flashwatch.env`: `EARTHDATA_USERNAME/PASSWORD`, `EE_PROJECT`, `CDSAPI_URL/CDSAPI_KEY` (also used for the
ECMWF Data Store), `NASS_API_KEY`, `ECMWF_API_*` (old Web API; retired for S2S), later `META_API_KEY`.

| Service | Status on this PC (Oct 26) |
| --- | --- |
| Earth Engine | `earthengine authenticate` done; `ee.Initialize(project=EE_PROJECT)` works (printed 150 bands for an ERA5-Land image) |
| ECMWF Data Store (ECDS) | Downloads running (section 4.2). **Caps queued requests per user per dataset**: "Maximum number queued requests for this dataset has been reached. Please configure your scripts to request data sequentially" |
| Drought Monitor, HydroShare (SMVI), NOAA PSL (Nino 3.4) | Public; working |
| NASA Earthdata | Credentials in env; not used yet (SMAP is a stretch goal). **Rotate the Earthdata password** at some point: it was shared in an earlier chat and the old handover (still in git history) named the account |
| GitHub | Claude GitHub App connected; the repo is public |

---

## 4. Where things stand

### 4.1 Decisions

**D1 resolved (Oct 26): study period 2005-2024 on one ECMWF cycle.** Forecast inputs are the
Cy49r1 reforecasts attached to the **2025 model-version dates**. Verified from the ECDS constraints
table: 101 dates from Apr 1 to Oct 15, 2025 (every other day, list in
`flashwatch/config/s2s_dates.json`), each with control + 10 perturbed members, hindcast years
**2005-2024**, all needed variables to day 46. **CV years 2005-2021 (leave-one-year-out), test years
2022-2024** (opened once, after the model freeze). 2012 and 2023 case studies kept; 2025 dropped.
ERA5-Land is pulled for 2004-2024 (2004 for spin-up only).

Locked from the spec (change only with James): region 36-49N, 104-80W, 0.25 degree, 52 x 96 cells;
issue dates Apr 1 - Oct 15 (the 101 model-version month-days in each hindcast year); leads week 1-4
plus "any onset in days 1-28"; eligibility root-zone percentile > 20 at issue; LightGBM per lead,
plain log loss, no downsampling or class weights, monotone constraints, isotonic calibration;
LOYO CV, test opened once, BSS vs climatology, year-block bootstrap CIs; agent explains and
verifies only.

Defaults taken by the cloud session are listed at the bottom of `QUESTIONS.md` (trailing 5-day
mean, own-year-excluded percentiles, masks, monotone list, etc.).

### 4.2a Status at Oct 28, 07:50 Sydney (local session)

- ERA5-Land (232), USDM (1,096), SMVI and Nino 3.4 complete. `obs.zarr` and labels built (see QUESTIONS for base rates).
- Tests pass on Windows (29). eccodes decodes on Windows; `forecast build` works on real files.
- **S2S was the bottleneck**: ECDS job timestamps show each request waits 1.5-3 h in the queue and
  runs in 3-8 min, so 1 date per request meant ~3 h per file (15 of 202 in 26 h). Fixes:
  requests now carry only what features use (days 1-28; t2m, d2m, sm100, tp), `FW_S2S_PACK`
  packs several same-month dates into one request (auto-halved on ECDS "cost limits exceeded":
  ~3 daily or ~5-6 accum dates fit) and the reply is split into the usual per-date files, and
  `FW_S2S_WORKERS=3` (the queue accepted 3 of our jobs at once). Running detached:
  `FW_S2S_PACK=16 FW_S2S_WORKERS=3 python -m flashwatch.ingest s2s`. ~50 requests, so done around Oct 30 if the queue holds.
  Three stale jobs from the old process sit in the ECDS queue (could not cancel them from here); harmless.

### 4.2 Downloads (running on this PC since ~05:45 Sydney, Oct 27)

Started with `python -m flashwatch.ingest all`. James was asked to stop it (Ctrl+C), `git pull` and
restart after two fixes (S2S one request at a time; ERA5-Land monthly calls); **confirm the restart
happened** (the S2S log should show "ECDS queue full, waiting 2 min" lines rather than giving up,
and ERA5-Land logs no "falling back to monthly" lines). Everything is resumable: re-running skips finished files and retries
failures. Check with `python -m flashwatch.ingest status`; logs in `data/logs/`.

| Source | Output | Expected | Last known (05:46 Sydney) | Notes |
| --- | --- | --- | --- | --- |
| ECMWF reforecasts | `data/raw/s2s/<date>_{daily,accum}.grib` | 202 files | 0 files, queue-cap errors (fixed by restart) | One request at a time; ~3 min each so ~10 h. Coarse-to-fine order (every 4th date first). `_style.json` records whether the form- or MARS-style request keys worked (`probe OK with ...` in `s2s.log`) |
| ERA5-Land (Earth Engine) | `data/raw/era5land/<var>_<year>.npz` + `land_frac.npz` | 232 files | 9 | Now monthly calls: a 365-band `NUMPY_NDARRAY` reply has an ~18 KB header that numpy 2.x refuses (max 10 KB) |
| Drought Monitor | `data/raw/usdm/USDM_YYYYMMDD_M.zip` | 1,096 | 209 | Tuesdays 2004-2024 |
| SMVI | `data/raw/smvi/` | README, LONLAT, `SMVI_GLDAS_Events.tar.gz` (the events), `FD_Events_NC.tar.gz` | 4 files | Old runs also started the ~3 GB "Composites" archives: atmospheric composites, **not needed**; delete them if present |
| Nino 3.4 | `data/raw/nino/nina34.anom.data` | 1 | not run | `python -m flashwatch.ingest nino` (not in the running `all`) |

**First thing to do on the local machine:** run `python -m flashwatch.ingest status`, read
`data/logs/*.log` for `ERROR` lines, confirm the S2S probe line, and re-run any source with failures.

### 4.3 Code (branch `claude/practical-planck-eyczp0`)

Written in the cloud session and **tested only on synthetic data** (29 pytest tests pass on Linux,
Python 3.13; not yet run on Windows). Includes a full-size synthetic run of the baseline CV
(17 folds x ~6.4M training rows) that completed in 39 min.

```
flashwatch/
  config/__init__.py     region, grid, YEARS/CV_YEARS/TEST_YEARS, paths (FW_DATA overrides data/), load_env()
  config/s2s_dates.json  the 101 verified ECMWF model-version dates
  ingest/                __main__ (all|status|<source>), s2s, era5land, usdm, smvi, nino, common (retry, resumable HTTP)
  obs.py                 python -m flashwatch.obs: npz -> data/obs.zarr (time, lat, lon) + rzsm (layers 1-3, 7/21/72 weights) + land_frac
  cube.py                in-memory Cube of the fields labels/features need (VPD via Tetens); load_cube()
  forecast.py            python -m flashwatch.forecast [inspect <grib>|build]: GRIB -> data/fcst.zarr weekly ensemble fields
  labels/percentile.py   percentiles() (direct) and PercentileCounts (computed once, exact per-fold percentiles by summing counts)
  labels/onset.py        onset rule, lead-week labels, snow/frozen mask
  labels/episodes.py     8-connected, +/-7-day union-find episodes
  labels/__main__.py     python -m flashwatch.labels [--variants]: reference labels, episodes, base rates -> data/labels/
  labels/crosscheck.py   python -m flashwatch.labels.crosscheck: known events, SMVI overlap vs chance, USDM worsening
  features/asof.py       as_of(source, issue) and LATENCY_DAYS (era5land 5, smap 3, s2s 0, usdm 2, nino 7); monthly_as_of
  features/climatology.py YearSums: calendar-window sums per year, per-fold z-scores without re-reading data
  features/build.py      FeatureBuilder: ~30 observed features; build(ref_years) per fold
  features/forecast.py   ForecastFeatures: per-week ensemble z, spread ratio, P(lower tercile) -> 0.25 deg (bilinear)
  models/baselines.py    B0 smoothed climatology, B1 logistic on state; MODELS registry (B0, B1, M1, M2)
  models/gbm.py          M1/M2 LightGBM, MONOTONE map, inner year-group isotonic calibration
  eval/metrics.py        Brier, BSS, reliability (intercept/slope), PR-AUC, hit/FAR, year bootstraps
  eval/cv.py             python -m flashwatch.eval.cv --models B0 B1 [M1 M2] [--leads ...]: LOYO CV, OOF parquet, score table
tests/                   conftest (synthetic AR(1) cube), asof, labels, features (leakage), cv, gbm, forecast, crosscheck
```

Not built yet: B2 (raw ECMWF), SMAP variant, SHAP drivers, the agent (`agent/`), replay/demo
(`demo/`), report generator, `fw` CLI (use `python -m ...` for now), vegetated-land mask (section 5),
static soil water capacity and cropland features, the test-year scoring script.

### 4.4 Open questions for James (full text in `QUESTIONS.md`)

| # | Question | Default until answered | Needed by |
| --- | --- | --- | --- |
| Q1 | Glimmer narrative LoRA off the cut list; cut SMAP first | **Resolved: yes** | done |
| Q2 | ECMWF CC BY-NC vs the submission licence | Publish code, results, attributed figures; no ECMWF-derived data files | before submission |
| Q3 | Tune once on dev years vs fully nested | **Resolved: tune once per lead on dev years** (`flashwatch.models.tune`) | done |
| Q21 | WhatsApp: real sends for the video (Meta developer app, test number) | **Resolved: live, tested on James's phone** | done |
| Q4 | Compute budget; XGBoost on GPU? | **Resolved: stay on LightGBM; light training + fast cycles (`eval.cv --fast`) while iterating design, heavy training last** | done |

### 4.5 Things learned the hard way (cloud session, Oct 26)

- **ECDS**: one queued request at a time per dataset. The dataset's own form keys are `origin`,
  `year/month/day` (model version), `hyear/hmonth/hday`, `level_type`, `variable` (long names),
  `forecast_type` (control_forecast/perturbed_forecast), `leadtime_hour` (`"0_24"` for daily means,
  `"24"` for accumulations), `area`, `data_format`. The constraints JSON is public (linked from
  `https://ecds.ecmwf.int/api/catalogue/v1/collections/s2s-reforecasts`).
- **ecCodes names GRIB2 daily means `avg_2t`** (paramId 228004), not `2t`; `forecast.py` maps names.
  Accumulations (tp, ssrd) run from step 0 and are differenced. **Run `python -m flashwatch.forecast
  inspect <file>` on a real download before `build`** and check the keys (`dataDate` should be the
  hindcast date, `number` 0 for control, 1-10 perturbed). Whether the Windows `eccodes` wheel
  decodes is untested; fallback is WSL.
- **numpy 2.x refuses `.npy` headers > 10 KB**: Earth Engine `NUMPY_NDARRAY` replies with hundreds
  of bands hit this.
- **Memory**: the first full-size synthetic CV was OOM-killed at 14 GB. After fixes: ~6 GB after
  building the feature caches, ~7 GB between folds, peak 11 GB (the main peak, the row-copying
  bootstrap, has since been replaced by `year_bootstrap_bss`). Needs >= 16 GB free; close other apps.
- **Runtime**: baseline CV 39 min on 4 cores. One LightGBM fit on 6.4M rows x 31 features, 400 rounds:
  93 s on 4 cores, so M1 CV is many hours (Q4).
- **The leakage tests work**: they caught static climatologies (aridity, growing-season rain) that
  included the issue year's own totals. Any new feature must pass `tests/test_features.py`; add new
  feature families to the perturbation tests.
- **SMVI**: events are in `SMVI_GLDAS_Events.tar.gz` (per-year CSVs, rows = GLDAS 0.25 degree cells,
  latitude-major from (-179.875, -59.875); columns `fstdate1..6`, ISO dates). GLDAS cell centres
  coincide with our grid. SMVI is sparse for 2017 (452 onset cell-events on our grid vs 5,213 in 2012),
  so a weak 2017 match is expected.
- **USDM shapefiles** are EPSG:4326 with a `DM` column 0-4.
- **Nino 3.4** (NOAA PSL ERSST v6) is revised over time; hindcasts use slightly better values than were
  available in real time. Note in the model card.

---

## 5. What to do next (in order)

**Status Oct 28 evening (Sydney):** steps 1-5 of the old list are done (downloads except S2S, Windows
setup, obs.zarr, labels, cross-checks, baselines). The working plan now lives in **`DESIGN.md`**
(decide-now vs open-until-burst, planned tests P1-P9, feature backlog, visual plan); read it next.
Current defaults: target 0-40 cm (T1), crossfit calibration with 4 practice models and a Platt map
(A7), global index and directional features excluded (P9, F5). Iterate with
`python -m flashwatch.eval.cv --models B0 B1 M1 --leads week1 week2 any28 --fast --tag <name>` and
`python -m flashwatch.eval.compare a=<oof stem> b=<oof stem>` (~8 min per run). James's rule: light
training while the design moves; heavy burst ~Nov 2 Sydney (DESIGN.md B).

1. **S2S download**: packed and trimmed (section 4.2a). Check it is alive every session
   (`python -m flashwatch.ingest status`; it once died silently). Then `python -m flashwatch.forecast build`.
2. Feature backlog (DESIGN.md F), each on the fast loop against the current baseline. Drought Monitor
   features (`--usdm`) on trial.
3. Planned tests P1-P9 (DESIGN.md P).
4. M2 once fcst.zarr is complete; B2 (raw ECMWF) not built.
5. Visual story (DESIGN.md V), built first on dev-year OOF predictions.
6. Agent (DESIGN.md G): weekly alert emails, number-checked. Muse Spark works live (G1 test, Oct 28).
6a. **Decision dashboard (DESIGN.md O; agreed Oct 28, Q8):** cost/loss engine with effectiveness by
   days of notice, "not yet, but if ..." triggers from ensemble members, a per-farm page, agent alerts.
   Irrigation first, then hay and feed.
6b. **Equipment and farm software (DESIGN.md O10; James, Oct 28 night):** `flashwatch/connect`
   (simulated, webhook, Leaf read-only, manual) + guard + Muse operator agent
   (`agent/operator.py`), shown as "Your pivots" on the farm page. Q10: autonomy and accounts.

6c. **Overnight Oct 28-29 (James asleep):**
   - site restyle (Public Sans, square, plainer text);
   - pivot operator agent + connectors (O10);
   - county-yield check (O4: rain-fed corn -20%, irrigated -2%, roll-up calibrated to ~$23M, Q11);
   - hay/feed and nitrogen on the farm page;
   - voice field reports (G3, Muse Voice Transcribe works at `/asr/transcribe`);
   - Glimmer data prep: `ingest.usdm_text` (narratives, slow: ~3/min, running), `agent.glimmer_data`
     -> data/glimmer/{train,val}.jsonl, `eval.glimmer_eval` (scores + Spark baseline).
   S2S at 16 complete dates (Oct 29): fcst.zarr rebuilt, M2 re-run (`m2fc16`: week 1 16.8%, week 2
   3.8%; members `m2mem16` slightly worse, DESIGN P11), evidence.json rebuilt. Next rebuild when
   June-August start dates land (watcher: `data/logs/wait_s2s.sh` waits for the first June-August date).
   Downloader restarted Oct 29 04:22 (Q13): `FW_S2S_PACK=16 FW_S2S_WORKERS=3 FW_S2S_FIRST=6,7,8`,
   started detached with Start-Process (stdout/stderr in data/logs/s2s.*.txt). Expect June-August in
   about a day, the rest about a day later.

6d. **Oct 29 (James at work; Remote Control on):**
   - site rework from James's notes (farm page wording, Today/Replay, date bar, Leaflet map with search,
     coverage outline; big picture rewrite; Help and About pages);
   - rain-fed mode with decision cards (fungicide corn/soy, nitrogen trim, silage, hay); rain-fed roll-up
     scopes (~$171M/yr region vs ~$19M irrigated);
   - H1 rule-of-thumb baseline and pooled top-10% evidence (5.4x, 54%);
   - field water account (`decide/waterbook.py`, FAO-56 PM ETo, Kc by date; dr0 in the engine), on
     the farm page and in the operator agent;
   - agents: SAM 3.1 pivot finder (`agent/pivots.py`), Ask FlashWatch chat (`agent/chat.py`,
     `agent/serve.py`), Agents page (`site/agents.html`, `site/agents.py`); Glimmer Modal script
     (`agent/glimmer_train.py`, Q20);
   - docs: README rewrite, write-up update, `docs/video.md`.
   Open: Q14-Q20 (Q14, Q15 resolved). S2S restarted 04:22 with summer first (Q13); watcher waits for
   the first June-August date, then rebuild fcst.zarr, M2, site data (`site.build`, then `site.agents`).

6e. **Oct 29 morning (after the 9:00 usage reset):**
   - M3 seeds: four M3s averaged, blended 50/50 with M1 (`eval.blend`): week 1 6.1% vs 4.3% (paired +1.8),
     week 2 1.7% vs 1.2%; more seeds no longer help (DESIGN L1). The site now runs on this blend (`E1`);
     the old M1 site data is in `data/site_backup/m1_0924/`. Agent texts regenerated on it.
   - Watcher fixes: risk easing must hold two forecasts; drought text at most every 4 weeks.
   - Watcher scorecard (`agent/scorecard.py`): each forecast text checked against the next 14 days, on
     the Agents page with a six-season track record.

6f. **Oct 29 late morning (James on Remote Control):**
   - WhatsApp loop (`agent/whatsapp.py`, webhook in `agent/serve.py`): operator plan with YES/NO buttons,
     YES sends through the connector, questions go to Ask FlashWatch, voice notes to agent/voice.py.
     Dry run (outbox in data/whatsapp/) until James sets up a Meta developer app (Q21).
   - Glimmer run locally (James approved Oct 29): llama.cpp b11972 CUDA in data/tools/llama, official
     `meta-models/Muse-Glimmer-30B-GGUF` Q4_K_M (16.8 GB) in data/models/glimmer; `eval.glimmer_eval`
     gains `--shots` (real training-year weeks as in-context examples) and `--timeout`.
     Results (7 common held-out weeks): alone, Glimmer and Spark both ~9% wording overlap, 0% untraced;
     Glimmer 2-shot 15% overlap but 7% untraced (borrows details). Offline agents: operator on Glimmer
     (`--tag glimmer`, META_TIMEOUT=1500) made Spark's plan for 2012-07-03, first draft clean, ~8 min;
     Ask FlashWatch 3 questions (`data/logs/ask_compare.py`): right but slower/less thorough. On the
     Agents page. Server: `bash data/logs/glimmer_server.sh` (port 8080; stop it before GPU training).

6g. **Oct 29 evening (James at the PC):**
   - S2S summer dates in (31 start dates in fcst.zarr, Apr 1 - Jul 7). M2 re-run `m2fcsum`: week 1 17.4%,
     week 2 3.3%, top tenth 5.9x (DESIGN, M2 section). evidence.json rebuilt from it; README, write-up and
     video numbers updated. The farm page stays on E1 (M2 still has no late-July/August dates).
     Oct 30: 41 start dates (Apr 1 - Jul 13), re-run `m2fc41`: week 1 16.5%, week 2 3.7%, top tenth 6.2x;
     west-east gap now clear (expectation 4 holds for M2). evidence.json and docs updated from it.
   - Oct 30 morning (Jamie): the public name is **Jamie Twiss**, sole contributor (site footers, end card); no email or
     phone anywhere public. Commits from this repo use the GitHub noreply address of JamieT206. Licence: PolyForm
     Noncommercial 1.0.0 (LICENSE.md). Working notes moved to workbench/ (Jamie deletes it before the final commit);
     CLAUDE.md moved to .claude/. README now has "Get the data and keys" and flashwatch.env.example. Security audit of
     all history: clean (only the Earthdata username in two Oct 27 commits; Jamie to rotate that password). Glimmer
     LoRA launched on Modal (Jamie's account, secret `huggingface`). Outside-in review planned for Nov 3.
   - WhatsApp live (Q21): keys in flashwatch.env; WHATSAPP_TOKEN is a permanent system-user token
     (never expires; set up Oct 29). Restart serve after changing flashwatch.env (existing env vars win).
     Webhook and WABA subscription set through the Graph API (app subscriptions + subscribed_apps). Tunnel: `cloudflared tunnel --url http://127.0.0.1:8813` (address changes on
     restart; re-point with the same API call). Server now logs each webhook call (run with `python -u`).
     Tested on James's phone: plan delivered (the test number takes ~1 min), YES button and typed "yes" both
     approve the plan and get the "Done" reply. The very first send was lost before the webhook was set.
   - Later that evening: map model with summer forecasts still no gain (`m3fcsum`); M2 recalibration no gain;
     expectations 1 and 4 hold on the dev years (DESIGN). `eval.final` built for the burst: `freeze`
     records models/options and the code's git tree hash; `score` refuses if the code changed or the test
     years were already scored, trains on 2005-2021 only and writes data/final/test_scores.json.

   - Oct 30 (Jamie at work): Glimmer LoRA scored (`glimmer_train.py::score`, one A100, ~20 min): wording 0.16 vs 0.09,
     direction 0.67 vs 0.50, but states 0.51 vs 0.67 and 2 of 12 weeks with an untraced figure (DESIGN G2 result); on the
     Agents page and in the write-up. Modal: a vLLM web server tried earlier crashed and its polls started 10 GPUs (~$6
     wasted); every Modal function is now capped at one GPU with a timeout. The agent server answers tunnelled requests on
     /whatsapp only. Modal volume `flashwatch-glimmer` holds the model cache (~60 GB) and the adapter.

**Priorities (James, Oct 28): (1) anything that meaningfully improves prediction, (2) the farmer tool,
(3) modelling nice-to-haves. Nothing is cut; what misses Nov 4 continues after.**
7. Burst (Nov 2), freeze, test years once, replay, video, write-up.

---

## 6. Design summary (full detail in the PDF and live doc)

### 6.1 Target and labels
- P(d) = percentile of the trailing 5-day mean of root-zone soil moisture (0-100 cm; ERA5-Land layers
  1-3, weights 7/21/72) on day d, against the same calendar window (+/-15 days) in other years
  (own year excluded; reference years = the fold's training years).
- **Onset on day d** when: (1) P(d) <= 20 and P(d-1) > 20; (2) P >= 40 on at least one day in
  [d-20, d-5]; (3) P <= 20 for all of [d+1, d+10]. Labels may use future data; features never.
- Masks: snow depth > 1 cm or top soil layer <= 0 C on the onset day; cells under 20% vegetated land
  (interim: land fraction).
- Label for issue date tau and week w: an onset in (tau + 7(w-1), tau + 7w]. 28-day label = OR of weeks 1-4.
  Rows only for eligible cells: P(tau - 5 days) > 20.
- Episodes: onsets in the same or 8-neighbouring cells within +/-7 days.
- Sensitivity variants: thresholds 30/20 and 40/10.

### 6.2 Features (all as of issue minus latency; ERA5-Land latency 5 days)
Implemented: root-zone and surface percentiles and their 5/10/20-day changes, days since P >= 40
(capped at 60); 14/30-day z-anomalies of temperature, VPD, PET, solar radiation; 14/30/60-day
precipitation z-anomalies; ET/PET (14, 30 days, raw and z); 5-day root-zone z; day of year sin/cos;
aridity and growing-season precipitation (fold climatology, own year excluded); Nino 3.4.
Forecast (when `fcst.zarr` exists): per week 1-4, ensemble-mean z and spread ratio for precipitation,
temperature, VPD, 0-100 cm soil moisture; P(precipitation in lower tercile).
No lat/lon or cell id features. NaNs stay NaN (except B1, median-filled).

### 6.3 Model ladder
B0 climatology -> B1 logistic on state -> B2 raw ECMWF (onset rule on forecast soil moisture + logistic
recalibration; **not built**) -> **M1** LightGBM land only -> **M2** LightGBM + forecast (main) -> M3
U-Net/ConvLSTM (stretch). Key ablation M2 - M1 per lead. SHAP top-3 drivers per county feed the agent.

### 6.4 Evaluation
LOYO over 2005-2021; isotonic calibration inside each fold; metrics per lead: Brier, BSS vs B0,
reliability slope/intercept, PR-AUC with base rate, hit rate/FAR at a fixed threshold, episode lead
time, lead over the Drought Monitor's first one-category worsening; 1,000 year-bootstrap resamples;
always report the episode count. Test years 2022-2024 scored once with the frozen model.

### 6.5 Agent (Meta model)
Weekly cycle: fetch (as_of) -> frozen M2 + SHAP -> maps and county table -> [LLM] read and cross-check
vs Drought Monitor and crop condition -> [LLM] draft advisory JSON -> validate numbers (every figure
traceable to a tool output; at most 2 redrafts, then drop flagged sentences) -> publish -> score
forecasts from 28+ days earlier. Tools, all taking `as_of`: `get_forecast`, `get_drivers`, `get_map`
(PNG), `get_drought_monitor`, `get_crop_condition`, `get_scorecard`, `publish_advisory`. Advisory per
county: FIPS, name, risk level (Low/Elevated/High/Severe from fixed bands), week 1-4 probabilities,
top 3 drivers, confidence sentence, what to watch, actions (producers/extension; water managers),
tool calls relied on. `agent/llm.py` wraps any OpenAI-compatible endpoint (local Glimmer or the
Meta API by config). Replay leakage guard: relative dates, anonymised counties, no outside knowledge,
number validator. Stretch (see Q1): Glimmer LoRA on Modal writing Drought Monitor-style narratives.

### 6.6 Pre-registered expectations (written before any run)
1. M1 beats B1 at weeks 1-2.
2. M2 beats M1 at weeks 3-4 but not at week 1.
3. B2 is poorly calibrated before recalibration.
4. Skill is higher in the western Plains than in the humid east.

### 6.7 Demo and submission
Replay May 1 - Aug 31, 2023 weekly: left FlashWatch week-2 probability with alert outline; right
actual onsets + Drought Monitor changes; below the agent's advisories and scorecard; timeline strip
for focus counties; ECOSTRESS 70 m close-up if coverage allows. Second clip: 2012. Package: public
repo (README, results table, one-command report from cached data), ~3 min video, static replay
page, two-page write-up, model card.

---

## 7. Schedule (US Eastern dates; Sydney is the next day)

| Day | Date | Exit criterion | Status |
| --- | --- | --- | --- |
| 1 | Oct 26 | obs downloads, reforecasts queued, USDM + SMVI | Running |
| 2 | Oct 27 | Labels, episodes, cross-check table, base rate | Code ready; needs data |
| 3 | Oct 28 | Training tables, as-of test passing, B0-B1 in CV | Code ready (synthetic) |
| 4 | Oct 29 | M1 calibrated and scored; gate at week 2 | Code ready; compute question Q4 |
| 5 | Oct 30 | M2, B2, ablation, **freeze**, test years once | M2 code ready; B2 and test scoring not built |
| 6 | Oct 31 | SMAP variant (stretch), SHAP per county, maps | Not started |
| 7 | Nov 1 | Agent tools, schema, validator, one cycle | Not started |
| 8 | Nov 2 | Cross-checks and self-scoring; 2023 replay | Not started |
| 9 | Nov 3 | Demo site, case studies, write-up draft, Glimmer fine-tune | Not started |
| 10 | Nov 4 | Video, README, model card; **submit by 9 PM ET** | |

Cut order if behind: SMAP variant, M3, then Glimmer fine-tune (Q1). **Never cut**: calibrated
out-of-fold evaluation, the as-of leakage test, the 2023 replay.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| ECMWF downloads slow (sequential, ~10 h) | Certain | Coarse-to-fine order; M1 needs no forecasts; GEFSv12 fallback |
| GRIB decoding fails on Windows | Medium | `forecast inspect` first; WSL fallback |
| CV compute too slow for M1+M2 (+ tuning) | High | Q3/Q4: headline leads first, fewer inner fits, XGBoost on GPU |
| RAM (~11 GB peak) | Medium | Close other apps; ask James for RAM |
| Little skill at weeks 3-4 | High | Pre-registered; report with CIs; demo uses weeks 1-2 |
| Few independent episodes | Medium | Report counts; year bootstrap |
| Labels sensitive to thresholds | Medium | 30/20 and 40/10 variants; SMVI, USDM triangulation |
| Meta API key not working from Australia / credits not granted | Medium | Develop the agent on local Glimmer through the same client |
| LLM already knows 2023 | Medium | Anonymised replay, validator, prompts |

---

## 9. Licences and required attribution

- **ECMWF S2S data: CC BY-NC 4.0**, non-commercial research only. Required wording: "This work is based
  on S2S data. S2S is a joint initiative of the World Weather Research Programme (WWRP) and the World
  Climate Research Programme (WCRP). The original S2S database is hosted at ECMWF as an extension of
  the TIGGE database." Cite Vitart et al., BAMS 98(1), 163-176, doi:10.1175/BAMS-D-16-0017.1. Also
  "(c) [year] ECMWF", source (ECMWF Data Store), CC BY-NC 4.0 link, ECMWF's no-liability disclaimer, a
  note of modifications. See Q2.
- ERA5-Land: "Contains modified Copernicus Climate Change Service information [year]".
- NASA SMAP/ECOSTRESS: dataset DOIs. Earth Engine: noncommercial registration; credit Google Earth Engine.
- US Drought Monitor: credit NDMC, USDA, NOAA. SMVI: Osman et al. (2024) HydroShare doi:10.4211/hs.b6d1d853b82247e1aeaf907ac254c99e and the 2021/2022 papers.
- Nino 3.4: NOAA PSL / ERSST v6. Muse Glimmer: Apache 2.0.

---

## 10. Context worth knowing

- James's past work, used in the design rationale and write-up: a monthly interstate-conflict forecaster
  (100M+ GDELT records, strict time split; persistence inflated skill, hence the onset-only target;
  downsampling broke calibration, hence plain log loss + isotonic) and Project Justice (litigation
  outcome prediction with LLM extraction through fixed tool schemas, leakage guards, a pre-registered
  null), whose habits shape the agent's replay mode.
- Research positioning: ML subseasonal skill comes mostly from the current land state, and gains mainly
  predict moistening rather than drying (Lorenz & Otkin); hybrid DL + dynamical models help beyond
  2 weeks (Lesinger & Tian 2025, Nat. Comms); onset remains open for S2S systems (SMCast, arXiv
  2610.07060); global onset prediction to 10 days shown recently (Gou et al. 2026, GRL).

---

## 11. Files

| File | What it is |
| --- | --- |
| `HANDOVER.md` | This document |
| `CLAUDE.md` | Short standing instructions for Claude Code in this repo |
| `QUESTIONS.md` | Open questions for James and the defaults taken |
| `DESIGN.md` | Design decisions (decide now vs open until the burst), feature backlog, visual plan |
| `README.md` | Run instructions (Day 1 downloads; Day 2+ pipeline order) |
| `FlashWatch Design Spec.pdf` | Spec snapshot, Oct 17 (test years out of date) |
| `docs/hackathon-rules.pdf` | Official rules and waiver |
| `check_access.py`, `README-access-check.md`, `requirements-check.txt` | Pre-kickoff access diagnostic (`--throughput` tests GRIB decoding; never run) |
| `flashwatch/`, `tests/`, `requirements.txt`, `pyproject.toml` | The project code (section 4.3) |
| `flashwatch.env` / `.env` | Credentials, local only, gitignored |
| `data/` | All downloads and stores, local only, gitignored |
