# Open questions and decisions

Questions for James are at the top; answer inline (edit this file, or reply in chat) and
I'll pick them up. Below them are defaults I've taken; say if you disagree.
Public repo: nothing private goes here.

## Questions

### Q1. Meta tooling is 25% of the score: promote the Glimmer work up the plan? **RESOLVED Oct 28: (b)**
James: fine. The Glimmer narrative LoRA is off the cut list; SMAP is cut first.
The official rules (docs/hackathon-rules.pdf) score Impact 30%, **Use of Meta AI tooling 25%**
("how creatively... beyond a basic integration... surfaced unexpected capabilities"),
Technical depth 20%, Demo 25%. The spec makes the Meta model an explainer/verifier and puts the
Glimmer LoRA **first on the cut list**. That's sound science but under-sells a quarter of the marks.
Options, none of which touches the forecasting core:
- (a) Keep the plan; lean on the agent's multimodal map reading and self-scoring in the pitch.
- (b) **Recommended:** keep the Glimmer narrative LoRA (Modal credits) and move it off the cut list;
  cut the SMAP variant first instead. Pitch: a 30B open model fine-tuned to write Drought
  Monitor-style narratives from structured state, scored against the real USDM narratives.
- (c) Also use SAM 3.1 or Muse Image in the demo. I don't see a natural fit; I'd skip it.

### Q2. ECMWF licence vs the submission licence
S2S data is CC BY-NC (non-commercial). The rules make each entrant grant the organisers a
perpetual licence to "exploit the Submission for any purpose without restriction, including
marketing". My default: the submission is code, results tables and figures with the required
ECMWF attribution; **no ECMWF-derived data files** go in the repo or the demo bundle. If you
want more caution, the fallback is to drop B2/M2 from the public demo, or use GEFSv12 (public domain).
Your call on how much risk to carry.

### Q3. Hyperparameter tuning: fully nested, or tuned once on the dev years? **RESOLVED Oct 28: (a)**
James: tune once per lead on the dev years, in the heavy-training phase (see Q4). Implemented as `python -m flashwatch.models.tune`
(4 year-groups over 2005-2021, every other cell, 30 random-search trials, early stopping picks
the rounds; results in `flashwatch/config/gbm_params.json`, which M1/M2 read).
The spec asks for nested tuning (~30 trials per lead) inside the leave-one-year-out CV. At real
scale a fold has ~6.4M training rows; one LightGBM fit is roughly a minute on 4 cores, and each
model already needs 5 fits per lead per fold (4 inner fits for the isotonic calibration, plus
the final fit). Fully nested tuning would be 17 folds x 5 leads x 30 trials x 5 fits, i.e. days.
Options:
- (a) **Recommended:** tune once per lead with grouped-by-year CV on the dev years (on
  a 25% spatial thinning of cells), then freeze the hyperparameters for the outer LOYO CV. The
  small optimistic bias in CV scores is stated in the model card; the test years (2022-2024)
  stay fully clean, and they're the headline.
- (b) Nested, but only ~8 trials on a small search space, on the thinned data.
- (c) No tuning: the current sensible defaults (models/gbm.py PARAMS).
(Original default was (c).)

### Q4. Compute budget for M1/M2, and your PC's RAM and core count **RESOLVED Oct 28: stay on LightGBM**
Measured at real scale on a 4-core cloud VM (synthetic data of the true size: 17 folds,
~6.4M training rows per fold): the baseline CV (B0, B1) takes 39 min and peaks at ~11 GB RAM
(I've since cut the main contributor). One LightGBM fit takes 93 s, so M1's CV
(17 folds x 5 leads x 5 fits) is ~11 h here; M2 is about the same again. Your PC will be
faster (how many cores, and how much RAM?), but M1+M2 is still overnight work and leaves
little room for reruns. Options, cheapest first:
- (a) Run week 1, week 2 and any-28 first (the headline leads); weeks 3-4 after.
- (b) Calibrate with 2 inner year-groups instead of 4 (3 fits instead of 5: ~40% faster).
- (c) **Recommended if you agree to the deviation:** XGBoost `hist` on the RTX 4070 (CUDA
  wheels exist for Windows; it supports monotone constraints). Probably 10x+ faster, which
  makes tuning (Q3) affordable. The spec locks LightGBM, so this needs your OK.
Until you answer: (a) + (b) with LightGBM.
**RESOLVED Oct 28 (James): get the full pipeline running with light training and short cycle times,
iterate on design (features, targets, architecture, ensembling) while training stays light, and only
then train heavily.** `eval.cv --fast` is the iteration loop (6 held-out years incl. 2012, every
other cell, LightGBM at learning rate 0.1 x 150 rounds). The tuner (`models.tune`) is deferred to
the heavy phase.

**Earlier update Oct 28:** the PC has an i7-14700KF (20 cores, 28 threads), 32 GB RAM
and the RTX 4070. At 5x the cores of the cloud VM, M1's CV should take a few hours with LightGBM,
so I now **recommend staying on LightGBM** (no deviation) and spending the headroom on Q3 (a).
I'll confirm with a timed fit after the B0/B1 run.

### Q5. S2S download: trimmed to what the features use (FYI; default already applied)
The ECMWF queue made one-date requests ~3 h each (15 of 202 files in 26 h). To pack more dates per
request under ECDS's cost limit, new requests carry days 1-28 only, and t2m, d2m, 0-100 cm soil
moisture and precipitation only. Dropped: days 29-46, solar radiation, 0-20 cm soil moisture and
temperature, none of which any feature or B2 uses. Estimated finish ~Oct 30 (Sydney). If you
want the dropped fields for anything (e.g. the demo), say so; they'd cost about 2x the queue time.

### Q7. Add NOAA's GEFS as a second forecast model? (decide after the early M2 results)
ECMWF is the best single source for weeks 1-4 and stays primary (DESIGN.md W). The realistic upgrade
is a second, independent model: combining two usually helps at weeks 2-4, and GEFS is public domain
(which also covers the Q2 licence worry). Cost: ~450 GB of downloading (global files; we keep only a
few MB), a new ingest module, about half a day of my time, and its archive stops in 2019/20, so
2020-2024 need the operational archive. **Recommendation:** decide once we know how much real ECMWF
forecasts keep of the perfect-weather ceiling and how steeply skill rises with forecast quality
(both runs queued). **Default until you answer:** ECMWF only; no GEFS work started.

### Q8. Decision dashboard (DESIGN.md O): scope and what gives **RESOLVED Oct 28: scope (a) and (c); nothing cut**
James: agree the dashboard scope; **cut nothing** from the build plan (it may still fit, or come after
the deadline). Priorities: (1) anything that meaningfully improves prediction, (2) making FlashWatch a
genuinely useful tool for farmers, (3) the modelling nice-to-haves. Hero action: irrigation first,
then hay and feed (both available; James notes buying feed early is partly zero-sum between
neighbours). Effectiveness must vary with days of notice (DESIGN.md O1a), or the engine would always
wait for the last moment. James is asking around for farmers to road-test it.

Original question:
Your idea, fleshed out in DESIGN.md O: per-location action cards (act now / not yet / trigger) from a
cost/loss engine with user-edited costs by days of notice, triggers from the ECMWF ensemble members,
dollars via crop yield response and a price, Muse alerts and Q&A. Evidence so far: week-1 forecasts
are worth 40-60% of a perfect forecast for cheap actions (cost under a tenth of the loss avoided),
and little for expensive ones. Three choices:
(a) scope for Nov 4: engine + triggers + dashboard in replay mode + agent alerts (O8 steps 1-4),
    yield link only if time allows;
(b) what gives: G3 voice and M3 out, Glimmer (G2) smaller;
(c) alerts are drafted and shown, not sent (sending needs a messaging account and your go-ahead).
**Recommendation:** (a), (b) and (c) as written; keep pushing M2 performance in parallel, since every
dollar figure scales with it. **Default until you answer:** none of O is built until we've discussed
it, except the cost/loss engine and value curves (O8 step 1, useful either way); I keep improving M2 and the agent.

### Q9. Download the irrigation months (June-August) first?
The S2S downloader works through the months in order, so June-August, the irrigation season the farm
tool is about, arrive last (around Oct 30-31). Until then M2, the weather triggers and the dollar
figures for summer rest on the land-only model or an estimate. `FW_S2S_FIRST=6,7,8` now puts those
months first. Switching needs a restart of the running downloader, which can lose the request
currently waiting in ECMWF's queue (up to ~1.5 h of work). Restart command:
`$env:FW_S2S_FIRST="6,7,8"; python -m flashwatch.ingest s2s`, after stopping the running one.
**Recommendation:** restart with summer first, right after the current request finishes (the log
says "done _pack_..."). **Default until you answer:** leave it running as is; I won't stop it.
**RESOLVED Oct 28 night: keep running in the current order (James).**

### Q13. Restart the S2S downloader after all? (asked Oct 29, 4 am Sydney) **RESOLVED Oct 29: restarted at 04:22 with packing and June-August first (James).**
At 23:29 on Oct 28 ECMWF dropped two of our queued jobs ("job not found"). The downloader then switched
itself to one date per request for the rest of the run, about 3 times slower: 4 files in the next
4.6 hours, against about 2 an hour before. 151 of 202 files are left. At this pace June-August arrive
around Nov 1 and the rest around Nov 3, too late for a comfortable Nov 2 burst. The switch is fixed in
the code (a failed pack now only affects its own dates), but the running process still has the old
code. A restart with packing and summer first would finish June-August in about a day and everything
in about two. Cost: the 3 requests now in ECMWF's queue are lost and may block new ones for up to
~3 hours. Restart command:
`$env:FW_S2S_PACK="16"; $env:FW_S2S_WORKERS="3"; $env:FW_S2S_FIRST="6,7,8"; python -m flashwatch.ingest s2s`.
A different IP address wouldn't help: ECMWF's queue limit is per account, not per address.
**Recommendation:** restart now. **Default until you answer:** leave it running; I won't stop it.

### Q14. Rain-fed farms: give them equal billing? (Oct 29) **RESOLVED Oct 29 (James: yes). Built: Irrigated / Rain-fed toggle on the farm page; decision cards for fungicide (corn, soybeans), side-dress nitrogen, silage and hay; rain-fed scopes in the roll-up (~$171M a year across the region's rain-fed acres, against ~$19M irrigated; rough values).**
The coverage area has 130 million rain-fed acres of corn and soybeans against 13 million irrigated (2022 Census). Even the
rough nitrogen estimate (~$50M a year in the region) is larger than the irrigation one (~$19M). The farm page now has a
Watering setting (pivot-irrigated / rain-fed); rain-fed switches the advice box to drought decisions (side-dress nitrogen,
hay and feed). Candidates to add, each with 1-2 weeks' notice: skip the mid-July fungicide pass (~$25-35 an acre; dry
spells mean little disease), early silage harvest of drought-hit corn, herd and pasture decisions. Not crop insurance
(choices close in March) and not grain marketing (prices already carry the forecasts; edges into trading advice).
DESIGN L9.
**Recommendation:** yes; research cost and benefit figures for fungicide and silage, add them as decisions, and add a
rain-fed scope to the big picture. **Default until you answer:** rain-fed mode as built; fungicide and silage after the
summer M2 work.

### Q15. Track the water a farmer has already put on? (Oct 29) **RESOLVED Oct 29 (James: yes, build it, with irrigation-system and farm-software hookups as an agent case).**
You asked whether FlashWatch can credit irrigation already applied ("they irrigated 2 days, so they can stop now").
The current engine can't: it assumes the soil is full when a dry spell begins, so water put on earlier doesn't count.
Doing it properly means a day-by-day soil-water account for the field (the "checkbook method" irrigators use): rain and
crop water use from ERA5-Land, plus irrigation the farmer enters (or reads from the pivot through Leaf), with advice
based on today's actual deficit. It also gives the "how much" answer exactly (inches to put on now).
**Recommendation:** build it; it's the most useful single addition for irrigators. Changes the engine in Python and the
browser (with the parity check). **Default until you answer:** after the summer M2 rebuild and before the Nov 2 burst,
if time allows; the farm page meanwhile says "run full time until about an inch of rain or the risk drops".

### Q16. Crop water use by date (Oct 29)
Your percentile question, taken further: the engine uses one heat-wave water use (0.30 in a day for corn) all summer.
Real use and yield sensitivity peak around silking (mid-July) and are lower in early June. FAO-56 crop coefficients by
growth stage would make June advice more cautious and July advice stronger.
**Recommendation:** add it together with Q15 (same engine change). **Default:** with Q15.

### Q17. Contact form, your name, copyright (Oct 29)
A static site can't send email itself. Options: (a) a form service such as Formspree (free tier ~50 messages a month,
spam filtering, forwards to any address you choose; you create the account); (b) a plain email link to a dedicated
address (bots harvest it, so expect spam). The About page has the form built, with a hidden "leave this empty" field
that catches simple bots; it says "not connected yet" until it has an endpoint.
Copyright: your work is copyrighted automatically; a "© 2026 Your Name" line is customary but optional. The licence for
the code (e.g. MIT) is a separate choice, tied to Q2.
**Recommendation:** (a) with a new address used only for FlashWatch, and a "© 2026 [name]" footer line.
**Default until you answer:** form not connected; no name on the site. Send me the endpoint (not a password) and the name to use.

### Q18. Forecast every day, not every other day? (Oct 29)
ECMWF runs its extended-range forecast every day and ERA5-Land updates daily, so a live service can forecast daily
and compare "start today" with tomorrow's forecast. The replay is every other day because that's how the archived
forecasts we trained on are spaced.
**Recommendation:** daily when live. **Default:** footer keeps your "every other day" wording (true for the replay).

### Q19. A live feed for Today mode (Oct 29)
Today mode is built (moves on by itself at midnight, date bar locked), but shows July 3, 2012 as a labelled demo.
Going live needs current ERA5-Land (free, ~5-day delay) and current ECMWF extended-range forecasts. The free S2S archive
we use is released 3 weeks late, so live forecasts need ECMWF's real-time product (a licence, probably paid) or a free
substitute such as NOAA's GEFS (Q7).
**Recommendation:** after the hackathon; the demo label stays until then. **Default:** demo.

### Q10. Pivot control: may the agent ever act without the farmer's YES? Which accounts to pursue?
The operator agent (DESIGN.md O10) plans pivot starts and stops, and the guard checks them. Nothing
is sent without an approval. Two decisions:
(a) **Autonomy.** Options: always ask (now); auto-*stop* only (for example, allocation used up),
still asking for starts; or auto-start within limits the farmer sets ahead (such as "up to 2
pivots, only when FlashWatch says start").
(b) **Accounts.** Real connections need sign-ups: Leaf (reads FieldNET/Valley; paid, with a trial),
the John Deere developer program, and the Climate FieldView partner program (free; needs approval).
A text-reply loop needs Twilio or similar. All need your go-ahead, and none fits before Nov 4
except perhaps a Leaf trial.
**Recommendation:** (a) always ask for the hackathon, and show "auto-stop when the allocation runs
out" as a farmer setting later. (b) Demo with the simulated pivots and the signed webhook; mention
Leaf, Deere and FieldView as the path in the write-up and video. **Default until you answer:**
exactly that. No sign-ups and no messages sent.

### Q11. County yields say our irrigation dollars are about 3 times too high. Calibrate? **Applied Oct 29: your big-picture wording uses 21-day dry spells, so the farm page now defaults to 21 days too (the field is still editable). Say if you'd rather keep 35.**
New check (DESIGN.md O4, `eval/yields.py`). Irrigated corn in counties hit hard by summer flash
droughts yields only about 1.5-3.6% below trend. Our water-balance model, with its 35-day rainless
spell, has a careful irrigator losing 8.6%. A 21-day spell matches the counties (2.7%). Calibrated,
the US roll-up drops from about $99M to about **$23M a year** (ceiling $33M).

Also found: onsets the Drought Monitor never marks D1 cost no yield. Rain-fed crops lose about 20%
when most of the county has an onset.

**Recommendation:**
- Use 21-day spells everywhere: the headline, and the farm page's default (shown as an editable
  setting, "dry spell length").
- Pitch the story as "flash droughts cost rain-fed corn 20%; irrigators can win back the last 2-3%,
  about $20-30M a year", not $100M. Honest, and backed by USDA data.
- Separately, rain-fed farmers are the bigger prize. Their actions (nitrogen, marketing, crop
  insurance timing) are worth a look after the hackathon.

**Default until you answer:** the region page leads with the calibrated $23M, with a selector for the
textbook $99M. The farm page and the demo keep 35 days, so the July 3 demo reads as before.

### Q12. Should the target be "damaging" flash droughts? (after the freeze, not now)
The yield check (Q11) shows that onsets the Drought Monitor never marks D1 cost no yield on average.
Those are dry spells that broke in time. About half our onsets are like that. A farmer only cares
about the other half. A stricter label, such as "stays below the 20th percentile for 20+ days" or
"the Monitor reaches D1 within 6 weeks", would aim warnings at the ones that cost money, and could
make each warning worth more.
**Recommendation:** not before Nov 4. The spec is settled and the test years are scored on the
current label. Record it as the first post-hackathon experiment, and mention it in the write-up as
an honest limitation. **Default:** no change.

### Q6. B0 (the BSS reference) is miscalibrated: strengthen its smoothing? **RESOLVED Oct 28: (a)**
Done before any M1 score was read. `python -m flashwatch.models.tune_b0` grid-searches B0's time
half-width tw, spatial radius sr and shrinkage k per lead (4 year-groups of 2005-2021, pooled
Brier, reference labels); `config/b0_params.json` holds the result, which B0 now uses.
Result: heavy shrinkage to the regional week-of-year rate wins (week 1-2: tw=8, sr=1,
k=50,000; weeks 3-4 and any-28: tw=4, sr=0, k=5,000; optimum inside the grid). Held-out Brier
improves 1.2-3.4% over the original B0, and the reliability slope goes from 0.13-0.21 to 0.93-0.95.
Tuned B0 only just beats a flat base rate (week 1 Brier 0.014736 vs 0.014749): out of sample,
the per-cell onset climatology carries almost no information beyond seasonality. So BSS vs B0
is now close to BSS vs a seasonal base rate, i.e. an honest (harder) reference.

## Day-2 results (real data, Oct 28)
- Reference labels (40/20): 46,172 onset cell-days, 3,409 episodes (median 2 cells; 170 with 50+
  cells); 8.09M eligible cell-issue rows. Base rates: week 1 1.69%, week 2 1.61%, week 3 1.42%,
  week 4 1.30%, any-28 6.01% (spec guessed 1-3% per cell-week). Onsets by year put
  2023 (4,408) and 2012 (4,351) on top, as expected.
- Variants: 30/20 gives 73,274 onsets (week 1 2.52%); 40/10 gives 12,650 (week 1 0.43%).
- Baseline LOYO CV (B0, B1; 2005-2021, 6.85M rows; BSS vs B0 with 95% year-bootstrap CI):
  week 1 B1 BSS 0.016 (0.012, 0.021); week 2 0.001 (-0.003, 0.006); weeks 3-4 ~0; any-28 -0.001
  (-0.016, 0.015). B1 is well calibrated (reliability slope 0.95-1.01). **B0 is not**: slope
  0.34-0.49, intercept -1.3 to -2.7, i.e. its cell-by-week rates are too extreme (too little
  smoothing/shrinkage for 1-2% events), so its Brier is worse than it should be and BSS vs B0
  flatters every model. B0 still has the best PR-AUC at weeks 2-4 (the spatial pattern is real).
  See Q6.
- **First M1 (fast CV, Oct 28; tuned B0; 6 held-out years, every other cell, light LightGBM;
  6.8 min per cycle).** BSS vs B0 [95% year-bootstrap CI]: week 1 M1 0.043 [0.035, 0.048] vs B1
  0.014; week 2 M1 0.012 [0.007, 0.016] vs B1 -0.002; any-28 M1 0.025 [0.008, 0.040] vs B1 -0.012.
  PR-AUC week 1 0.085 (M1) vs 0.052 (B1) vs 0.037 (B0); base rate 1.9%. Preliminary support for
  pre-registered expectation 1 (M1 beats B1 at weeks 1-2), to be confirmed in the full CV.
  **M1 is under-confident**: reliability slope 1.5-2.0 (intercept > 0), i.e. its probabilities
  should be more spread out. Likely causes: isotonic map fitted on inner models trained on half
  the years (weaker than the final model), and light training. First design item to look at.
  Tuned B0 shows slope 2.5-2.7 here: its near-flat predictions make the slope unstable (and
  year-specific base rates vary); Brier is what matters for the reference.
- Cross-checks: SMVI overlap 13.1% vs 6.5% chance (2,812 episodes); USDM 2+ category worsening
  4.1% vs 1.5% chance (170 large episodes). Known events: 2023 one 1,910-cell episode (onset
  ~May 27, 36% of box); 2022 39%; 2017 20%; **2012 3% in the Jun-Jul window** because the
  root-zone onsets came in May (675 cell-days) and 92% of the box was already below the 20th
  percentile by Jun 1 (see the window question in chat).

## Defaults taken (Day 2 code)
- D1 resolved: 2005-2024, CV 2005-2021, test 2022-2024 (the PDF spec still says 2023-2025).
  101 issue dates/season, verified against ECDS's constraints table.
- 5-day running mean of root-zone moisture is **trailing** (days d-4..d), not centred.
- Percentiles: each value's pool is the +/-15-day calendar window in the fold's training years
  **excluding its own year** ("other years"); held-out years use all training years. Mid-rank ties.
  Feb 29 shares Feb 28's calendar day.
- Root zone 0-100 cm = layers 1-3 weighted 7/21/72.
- Onset mask: snow depth > 1 cm or top soil layer <= 0 C on the onset day.
- "Under 20% vegetated land" mask: interim rule is ERA5-Land land fraction >= 0.2. I plan to
  replace it with MODIS land cover (MCD12Q1 via Earth Engine) vegetated fraction on Day 3.
- Drought Monitor as-of latency 2 days (maps are valid Tuesday, released Thursday).

## Defaults taken (Day 3-4 code)
- Monotone constraints (models/gbm.py MONOTONE): probability may only fall with root-zone and
  surface percentiles and their changes, precipitation anomalies, ET/PET; only rise with
  temperature, VPD and PET anomalies; forecast precipitation anomaly down, its lower-tercile
  probability up, forecast temperature and VPD up, forecast soil moisture down. Days since
  P>=40, Nino 3.4, season, aridity and growing-season rain are unconstrained.
- Isotonic calibration per lead, fitted on inner out-of-fold predictions (year-groups
  inside each outer fold's training years; 2 groups pending Q4).
- Each lead's model sees all four forecast weeks (not just weeks <= lead).
- B1 fills missing inputs with training medians (the GBMs keep NaN).
- B0 smooths onset rates over +/-2 weeks of year and the 3x3 neighbourhood, shrunk to the
  regional weekly rate with 20 pseudo-rows.
- Forecast features: z-anomaly of the ensemble mean vs the reforecast climatology for the
  same start date and week (fold reference years minus the own year), spread / climatological
  SD, and the fraction of members below the climatological lower tercile (precipitation).
- Nino 3.4: NOAA PSL ERSST v6 monthly anomaly; month m is treated as available 7 days after
  it ends. ERSST values are revised over time, so hindcasts use slightly better values than
  were available in real time (noted for the model card).

## Defaults taken (Oct 28, afternoon, while James was out)
- **Great Lakes removed from the land mask** (bug fix, FYI). ERA5-Land carries 269 lake cells as
  land with a constant soil moisture of ~0.01. They can never have an onset, so the model learned
  "zero" there while B0 gave them the regional rate, which flattered every skill score. Fixed in
  `obs.land_mask` (soil moisture must vary over the record); labels, B0 tuning and the fast-loop
  baselines are being re-run, and all earlier skill numbers in DESIGN.md P9/P10 predate the fix.
- New forecast features (F8) kept in: neutral so far, physically sensible, and cheap.
- Agent packs are refused for the test years until the model freeze.

## Decisions (James, Oct 28, evening)
- **Focus: "irrigate now or not?", decided day by day.** Notice is worth little beyond about a week
  (DESIGN.md O1a), so week 1 is the core, week 2 a nice-to-have, and weeks 3-4 stay only in the
  pre-registered evaluation.
- **Dashboard centred on one farm:** enter a lat/lon and see everything for that place. Add a few
  system-wide pages for the judges. The pitch is grounded in "a real idea to help real American farmers".
- **Roll-up number** ("$X billion saved, Y more food, equivalent to Z"): compute all three scopes
  separately: (1) irrigated corn and soy in the study region, backtested; (2) the same scaled up to
  all US irrigated corn and soy (labelled as an extrapolation); (3) also rain-fed farms through other
  actions (placeholder costs).
- **Baselines for "usual practice": both,** side by side: a careful irrigator (starts 2 days after
  stress begins) and a calendar irrigator (starts on a fixed date; before it, only once the crop shows
  stress).
- **Hosting:** a static site in the repo (`site/`), previewed locally; James switches on GitHub Pages.
  Site data stays out of git until Q2 (ECMWF licence) is settled for anything forecast-derived.

## Defaults taken (Oct 28, evening)
- Agent number checker ignores the sign of a figure ("2.6 below normal" matches -2.6) and exempts
  time spans ("28 days"); agent tools now return percentages. Found in the first live Muse run.
- Lead over the Drought Monitor counts each date from when it is public (Monitor +2 days, ERA5-Land
  +5 days) and uses development years only (`eval/lead.py`).
