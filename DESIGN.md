# FlashWatch: design decisions and work plan

Working rule (James, Oct 28): get the whole pipeline running with **light training and short
cycles**, improve the design while training stays light, and only train heavily once the design
is in place. Decisions fall in two groups:

- **A. Decide now, under light training.** Things we can be confident about without much compute:
  the target, which rows go into training, the features, how we evaluate. Where possible these are
  decided on evidence about the labels and data themselves, not on which version scores best (picking
  the target or features by score would make our cross-validation numbers look better than they
  really are). Each decision records what it was based on.
- **B. Open until the late burst.** Things that may look different with more compute behind them
  and **don't change what we build**: which boosting library, tuning settings, ensembling, random
  seeds. All settled in one intensive burst at about 70% of the way through (**Nov 2, Sydney**),
  using a decision rule written down before the burst; then freeze the model and score the test
  years once. Exception: if an open question could lead us into a dead end (it changes the data
  pipeline or what we build), run a heavier test now rather than leaving it open.

Iteration loop (about 7-8 minutes per run):
`python -m flashwatch.eval.cv --models B0 M1 --leads week1 week2 any28 --fast --tag <name>` then
`python -m flashwatch.eval.compare base=<file> new=<file>`. "Fast" means 6 held-out years
(including 2012), every other grid cell, and light LightGBM training (learning rate 0.1, 150 trees).

Plain-language glossary: **Brier score** = mean squared error of the probabilities (lower is
better). **BSS** (Brier skill score) = how much better than the climatology baseline B0, e.g. 0.04
means 4% lower Brier. **Calibration slope** = whether the probabilities are spread out the right
amount: 1 is right, above 1 means too timid (should be more extreme), below 1 too bold.
**OOF** = out-of-fold, i.e. predictions for a year the model was not trained on.

---

## A. Decided, or to decide now

| # | Topic | Status | Decision / current default |
| --- | --- | --- | --- |
| A1 | Study period and splits | Decided, plus robustness checks (P1) | 2005-2024; cross-validation leaves out one year at a time over 2005-2021; 2022-2024 kept back as the test, opened once |
| A2 | Target definition | **0-40 cm is the working target** (literature standard; default since Oct 28); more checks (T1) | Onset = soil moisture percentile drops from 40+ to 20 or below within 20 days and stays at or below 20 for 10 days |
| A3 | Forecast horizons | Decided for now; test the "accuracy cliff" (P2) | Week 1, 2, 3, 4 and "any onset in the next 28 days" |
| A4 | Which rows go into training | See T2-T4 | Explained below |
| A5 | Leakage rules | Decided | Features read data only as it would have been available on the issue date; every feature family has an automatic test |
| A6 | Baseline (B0) | Decided | Climatology, smoothing tuned on the dev years (Q6) |
| A7 | Calibration method | **Decided: "crossfit" with 4 practice models** (default since Oct 28) | See below |
| A8 | Features | In progress | F-list below |
| A9 | M3 (deep-learning map model) | **Cut for now; on the "later" list (L1)** | Spatial context goes in as features instead (F5) |
| A10 | One model per horizon vs one model for all horizons | **Recommend one per horizon** | Explained below |
| A11 | Glimmer narrative fine-tune | Decided | In plan (Q1) |

**A4. Which rows go into training, in plain terms.** One row = one grid cell on one forecast issue
date. A row is included only if: (1) the cell is **not already in drought** on the issue date (soil
moisture above the 20th percentile; a cell already below it can't have an *onset*); (2) the issue
date is one of the dates the ECMWF forecasts exist for (every other day, Apr 1 - Oct 15); (3) the
cell is mostly land; and onset days with snow or frozen soil don't count as onsets.

**A7. Calibration.** Our model's probabilities are too timid: when it says 7.5% the true rate is
about 9%, and when it says 0.5% the true rate is 0.2%. Likely cause: the step that converts raw
scores into probabilities is learned from weaker practice models (each trained on half the years)
and then applied to the stronger final model. First fix tested ("crossfit": use the practice models
themselves for prediction, 4 of them) helps a little (slope 1.35-1.69, from 1.51-1.95; skill up
slightly). Results on the fast loop (0-100 cm target, same features):

| Variant | Week-1 skill vs climatology | Calibration slope (1 = right) | Change vs current, with 95% range |
| --- | --- | --- | --- |
| Current (2 practice models, then refit) | 4.3% | 1.51 | - |
| **crossfit, 4 practice models** | **4.5%** | 1.35 | +0.3% [+0.1, +0.6] |
| refit, 4 practice models | 4.4% | 1.25 | +0.1% [+0.0, +0.2] |
| no conversion step | -0.6% | 0.70 (too bold) | -5.1% |

Without the conversion step the model is far too bold and loses to climatology; with it, it is a
bit timid. crossfit gives the best scores at every horizon (clearly at week 1 and any-28), so it is
the default. The remaining timidity (slope 1.3-1.7) is noted; B3 tests more practice models in the burst.

Platt calibrator (Oct 28, 0-40 cm, no year-level features): same skill (week 1 4.35% vs 4.40%, week 2
1.3% vs 1.4%, any-28 5.2% vs 5.1%, all within noise) and less timid (slope 1.22-1.30 vs 1.33-1.41).
**Decided: Platt is the default** (Oct 28).

**A10. One model per horizon, or one for all?** Today we train five separate models: one answers
"onset in week 1?", another "onset in week 2?", and so on. The alternative is a single model with
"how many weeks ahead" as an input column: every row appears five times, once per horizon. That
shares patterns between horizons and keeps the answers consistent (e.g. never more likely in week 3
alone than in the whole 28 days), but the training table becomes five times bigger and slower, and
it can blur what is special about each horizon. It changes the shape of the training data, so it
isn't a burst item. **Decided (James, Oct 28): one per horizon**, with a check that the raw outputs
make sense together. Check (0-40 cm baseline): week-1 probability exceeds the 28-day probability in
3.5% of rows and week 2 in 2.0%, by small margins; week 1 and 28-day correlate 0.80. Not absurd; a
simple cap (no single week above the 28-day value) would tidy it if needed.

### Target questions (A2, A4)

- **T1. Soil depth.** Compared on label evidence only (no model scores):

  | Depth (cm) | Onsets (cell-days) | Weekly rate | SMVI agreement vs chance | Drought Monitor big worsening vs chance | Share of onsets in clusters of 10+ cells | Known events caught in their windows (2012 / 2017 / 2022 / 2023) |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0-100 (spec) | 46,172 | 1.7% | 2.0x | 2.7x | 0.86 | 3% / 20% / 39% / 36% |
  | 0-50 | 79,133 | 2.8% | 2.1x | 2.0x | 0.91 | 11% / 49% / 68% / 51% |
  | **0-40** | 91,841 | 3.2% | 2.1x | 4.3x | 0.93 | 17% / 54% / 76% / 59% |
  | 0-28 | 109,379 | 3.7% | 1.9x | 4.1x | 0.94 | 24% / 62% / 80% / 76% |
  | 0-7 | 127,895 | 4.0% | 1.9x | 2.9x | 0.95 | 32% / 87% / 76% / 84% |

  - 0-100 cm reacts slowly (72% of its weight is the 28-100 cm layer) and misses most documented events.
  - 0-7 cm finds the most onsets but agrees least with the independent sources (rain makes the surface noisy).
  - 0-40 cm agrees best with the independent sources and is the standard definition in the
    literature (Ford & Labosier 2017: 40th -> 20th percentile within 20 days, 0-40 cm).
  - In every depth, the Drought Monitor worsens a median 13-14 days after our onset.
  - 2012 dried out early: most onsets in the central-US box are March-May at every depth.
  - **James (Oct 28): align with the literature, so lean 0-40; also test 0-28, 0-50 and 0-100.**
    0-50 (done): between 0-40 and 0-100 on every check, and clearly weaker than 0-40 on the Drought
    Monitor check (2.0x vs 4.3x), so it doesn't change the lean. ERA5-Land's third layer spans 28-100 cm, so 0-40 and 0-50 assume moisture
    is even within it (a proxy).
  - **Crop sense-check (to do):** root depths and critical periods of the main crops (corn: deep roots,
    critical at silking in July; soybeans: pod fill in August; winter wheat: planting Sep-Oct, heading
    May). Possible later step: crop-specific targets (e.g. a deeper layer in July-August for corn, a
    shallower one in autumn for wheat planting). On the "later" list (L4) unless it turns out cheap.
    James: if root depths are similar, crop-specific mostly means different critical windows plus where
    each crop is planted. They are broadly similar (mature corn, soybean and wheat roots all reach about
    1-1.5 m, with most water drawn from the top ~60 cm), so yes. See L5 (crop impact).
- **T2. Already-dry cells.** Cells already below the 20th percentile are left out (92% of the 2012
  box was by Jun 1). Right for an onset forecast; we say so in the write-up.
- **T3. Season: go wide** (James). More off-season rows also help find false alarms. Limit: ECMWF
  forecasts only exist for Apr 1 - Oct 15 start dates, so M2 (with forecasts) stays there, but M1
  (land only) can train on a wider window (e.g. Mar 1 - Nov 30). Date is already a feature
  (day of year). Test P3: one model with date as a feature vs separate models per part of the season.
- **T4. Masks.** Test P4: drop the snow/frozen and land-fraction masks to see how much we are
  predicting rainfall shortfalls vs the soil's ability to hold water. Irrigation: ERA5-Land doesn't
  model it, so labels and features are consistent; flag in the impact story.
- **T5. Short dips don't count.** Our onset rule requires soil moisture to *stay* at or below the
  20th percentile for 10 days after the drop. So a dry spell that a storm ends within days is not
  counted as an onset, and we don't need a separate rule to remove false starts. James: fine for
  now; ideally link to actual farm outcomes later (L5).

---

## P. Planned tests (phase 2, on the fast loop unless noted)

| # | Test | Why |
| --- | --- | --- |
| P1 | Other ways of splitting years: leave out 3-year blocks; "forward" (train only on earlier years, predict the next) | Droughts follow multi-year climate patterns (El Nino / La Nina, Pacific decadal swings), so neighbouring years may not be independent; forward splits also mimic real use |
| P2 | How far ahead can we see? M1 at weeks 1-8 | Find the accuracy cliff and set the horizons there. M1 needs no extra data. M2 beyond 4 weeks would need the ECMWF days 29-46 we trimmed from the download (~2x queue time) |
| P3 | Date as a feature vs separate models per season segment | James, T3 |
| P4 | Drop masks | T4 |
| P5 | Target depth with a model (0-28 / 0-40 / 0-50 / 0-100) | Only after the label-evidence choice, as a check |
| P6 | Year effects (F10) | How much is left unexplained by year, and is it a trend or a few odd years? (Year as a category: later, L7) |
| P7 | Predict "drought day" (presence) instead of onset | James has had interesting results modelling presence vs onset. "Was yesterday a drought day" will dominate; the interesting part is what else adds skill. Second priority |
| P8 | Testing on held-out *places* within the same years | Second priority (James). Caution: flash droughts are mostly large. At 0-40 cm the median episode is 2 cells, but 72% of onset cell-days are in episodes of 100+ cells and 23% in episodes of 1,000+ (the largest covers 4,087 of 4,992 land cells). Held-out regions would have to be big blocks, or neighbouring training cells would give the answer away |
| P9 | Year-level features (El Nino, global temperature, year, MJO) | Oct 28 finding: the top 1% of week-2 predictions were 84% from 2015 (record El Nino, a value outside the training range). With 16 training years, a once-per-year number lets the model memorise years. **Result: removing them helps** (week 1 3.9% -> 4.4% skill, week 2 1.1% -> 1.4%, any-28 4.4% -> 5.1%, fast loop, 0-40 cm). MJO (daily) on its own also hurts (-0.4 points at week 1). **Decided: all four are left out of the models** (code kept) |
| P10 | Can the onset waves be predicted at all? (Oct 28, after the 2012 replay showed M1 flat through the spikes) | **Done; see below.** |

**Lake fix (Oct 28, afternoon).** ERA5-Land carries the Great Lakes as 269 "land" cells with constant soil
moisture; they can never have an onset and flattered all skill scores. Now excluded (`obs.land_mask`,
4,723 land cells). Re-run on the fast loop: B0 retunes to the regional seasonal rate (no place detail
survives); M1 skill week 1 **4.3%**, week 2 **1.2%**, any-28 **4.6%** (before: 4.4 / 1.4 / 5.4); M2 on the
10 forecast start dates (`--fc-dates`) **15.1% / 2.4% / 8.5%** against M1 1.7 / 0.4 / 1.4 on the same dates
(before: 15.6 / 2.7 / 9.7). Conclusions unchanged; numbers below this line predate the fix (slightly high).

**P10 result: the waves are weather, and they are predictable given the weather.**
- *Where the skill is.* Onsets come in waves: the burstiest 10% of issue dates hold 37% of week-2 onsets. Knowing the exact share of the *whole region* with an onset on each date would only give 5-10% skill; knowing it for each 2-degree block (about 200 km) would give 37-51%. So the waves are a few hundred km wide, the size of a weather system.
- *What M1 sees.* M1's daily regional average does rise and fall with the actual rate (correlation 0.37-0.55), but it moves only a quarter as much. Land state tells it where the ground is primed, not where next week's hot, dry spell lands.
- *Perfect-weather test* (`--oracle coarse`, eval/oracle.py; DIAGNOSTIC, reads the future on purpose). M2 is given the weather that actually happened in weeks 1-4 (precipitation, temperature, VPD anomalies), smoothed to about 1.75 degrees, i.e. roughly what the ECMWF grid could resolve with perfect skill. No future soil moisture. Fast loop, 0-40 cm:

| Lead | M1 skill (BSS) | Perfect-weather M2 skill | PR-AUC M1 -> M2 | Daily regional average tracks actual (corr) |
| --- | --- | --- | --- | --- |
| Week 1 | 4.4% | **26.7%** (24.7-28.4) | 0.11 -> 0.43 | 0.47 -> 0.89 |
| Week 2 | 1.4% | **20.0%** (17.1-23.3) | 0.06 -> 0.34 | 0.37 -> 0.82 |
| Any in 28 days | 5.4% | **42.7%** (41.0-45.2) | 0.23 -> 0.68 | 0.55 -> 0.93 |

  In 2012 the perfect-weather model catches both waves (actual week-2 rate 28% on May 3 / 19% on Aug 13; M2 13% then 26% the next issue, and 19%), where M1 stays at 3-5%.
- *Resolution barely matters.* The same test at the full 0.25 degree grid (`--oracle native`) gives week 1 28.2%, week 2 21.2%, any-28 45.0%: only 1-2 points above the 1.75 degree version. So the 1.5 degree ECMWF archive is not what limits us; forecast *accuracy* is.
- *How good are the real ECMWF forecasts?* Correlation between the ECMWF ensemble-mean weekly anomaly and what actually happened (same smoothing; 200 forecast start dates downloaded so far, land cells, 2005-2024):

| Variable | Week 1 | Week 2 | Week 3 | Week 4 |
| --- | --- | --- | --- | --- |
| Precipitation | 0.78 | 0.37 | 0.12 | 0.07 |
| Temperature | 0.97 | 0.66 | 0.22 | 0.18 |
| VPD (air dryness) | 0.90 | 0.55 | 0.32 | 0.20 |

  So week 1 is close to perfect, week 2 is about half-way, and weeks 3-4 are weak. The blurred runs below cover exactly this range.
- *Early M2 on real ECMWF forecasts* (Oct 28, `m2early`; only the 10 start dates downloaded so far, Apr 1 - May 19, so 60 of the fast loop's issue dates; M2 trained with forecasts on just those dates). Scored on those dates only, with year-bootstrap 90% ranges:

| Lead | M1 | M2 (real ECMWF) | Gain | Perfect-weather gain, same dates | Share of the ceiling captured |
| --- | --- | --- | --- | --- | --- |
| Week 1 | 2.9% | **14.3%** | +11.4 (+7.6 to +14.6) | +24.9 | about half |
| Week 2 | 0.6% | 2.0% | +1.4 (-0.5 to +2.9) | +20.0 | under a tenth |
| Any in 28 days | 2.7% | 8.4% | +5.6 (+2.2 to +9.0) | +38.2 | about a seventh |

  Week 1 is a clear win: real forecasts roughly quintuple the skill. Week 2 gains little so far, although the week-2 forecasts are half-way good (correlation 0.4-0.65). Possible reasons, to check as the download fills in: too few dates to learn from yet, or the features don't present the week-2 signal well (e.g. a combined "dry and hot" index, or the forecast soil moisture). The blurred runs below show what a forecast of that quality *should* deliver.
- *Two checks on the week-2 shortfall* (Oct 28, same 10 start dates):
  - New forecast features (F8: weeks 1+2 mean, ensemble chance of a hot-and-dry week, dry-hot index, forecast soil drying): **neutral** (week 1 14.0%, week 2 2.2%, any-28 8.7% vs 14.3 / 2.0 / 8.4).
  - Training *only* on the forecast dates (`--fc-dates`, so no rows without forecasts): real M2 15.6% / 2.7% / 9.7%; perfect weather with the same small training set 26.2% / 19.9% / 40.4%. So it isn't too little training data: perfect weather still gets 20% at week 2 from the same rows.
  - **Conclusion so far:** the week-2 limit is in the real forecasts themselves. Their *average* week-2 match is fair (0.4-0.65), but flash-drought onsets hinge on the strong hot, dry spells, and an ensemble mean two weeks out blurs exactly those (the blur test added uniform noise, which is kinder). Spring-only dates may also understate it. Re-check with the full season once downloaded.
  - Spring 2012 example (week 1): on May 11 M2 forecast 14.8% against 37.8% actual (land only 4.6%; perfect weather 29.5%). From two weeks out (May 3, week 2): M2 4.0%, actual 27.6%.
  - **Pre-registered expectation 2 (HANDOVER 6.6: "M2 beats M1 at weeks 3-4 but not at week 1") is so far wrong in the opposite direction**: the gain is largest at week 1. To be reported as such.
- *Blurred forecasts* (`--oracle-skill r`): the same test with the perfect weather mixed with large-scale noise so it matches the truth with correlation r, to map how flash-drought skill depends on forecast quality (all fast-loop dates):

| Forecast quality (correlation with what happened) | 1.0 (perfect) | 0.7 | 0.5 | 0.3 | none (M1) |
| --- | --- | --- | --- | --- | --- |
| Week 1 skill | 26.7% | 16.6% | 12.3% | 8.3% | 4.4% |
| Week 2 skill | 20.0% | 10.5% | 6.9% | 3.9% | 1.4% |
| Any in 28 days | 42.7% | 27.7% | 20.2% | 12.4% | 5.4% |

  Reading it against the real ECMWF accuracy: week 1 forecasts sit at 0.8-0.97, so M2 "should" reach roughly 17-25% at week 1 (it gets 14.3% so far); week 2 sits at 0.4-0.65, so roughly 5-9% (it gets 2.0%). So **M2 is not yet using the forecasts as well as forecasts of that quality allow**, most of all at week 2. That points at our side (training data and features), which we control, not at the forecasts. Caveats: blur noise is a simplification of real forecast errors, and early M2 is scored on spring dates only. Next: re-run M2 as dates arrive; try forecast features built the way the land features are (combined dry-and-hot index, forecast soil-moisture change, week 1+2 totals).
- *Meaning.* The target is sound and the onset waves are not noise; they are what weather forecasts exist to see. The ceiling is about 15x M1's skill at week 2. The open question is how much of it the real ECMWF forecasts keep (good at week 1, weaker at week 2, weak beyond), so M2 on real forecasts is the decisive run. The `native` (0.25 degree) variant is the absolute upper bound.

---

## B. Open until the burst (Nov 2, Sydney)

| # | Question | Candidates | Tooling ready |
| --- | --- | --- | --- |
| B1 | Boosting library | LightGBM (current), XGBoost (GPU), CatBoost | LightGBM only |
| B2 | Tuning settings | Random search per horizon on the dev years (Q3a) | `models.tune` |
| B3 | Number of practice models for calibration | 2 vs 4 (depends on A7) | yes |
| B4 | Ensembling | Several seeds averaged; blend M1 and M2; combine with B2 | No |
| B5 | Which models make the headline | M1, M2, B2 (raw ECMWF, recalibrated) | M1/M2 yes, B2 no |
| B6 | Use all grid cells or every other one | Does thinning cost skill? | `--thin` |

Burst decision rule (to finalise before Nov 2): choose by Brier score over the full 17-year
cross-validation per horizon; keep the simpler option unless the improvement is clearly real
(the 95% range across resampled years excludes zero).

---

## L. Later, if time allows

- L1. M3: deep-learning model on the map (U-Net/ConvLSTM).
- L2. Interactive map: click a location and watch two charts animate, the FlashWatch forecast
  [x] weeks out and the actual soil moisture catching up behind it.
- L3. A full website (beyond the static replay page).
- L4. Crop-specific targets (T1).
- L5. **Crop impact** (James): "X% of the corn crop is under flash-drought threat; yields likely
  down Y% on last year in A and B". Needs where each crop is planted (USDA Cropland Data Layer via
  Earth Engine, or NASS county acreage) and a simple link from drought exposure in the crop's critical
  window to yield (NASS county yields, 2005-2021; we have a NASS key). Strong for the impact score
  and for non-specialists; worth pulling forward if time allows.
- L6. More ECMWF lead time (days 29-46), downloaded two weeks at a time, to see how far ahead M2 can go (A3).
- L7. Year as a category (diagnostic, F10).

### W. Weather forecast sources (Oct 28, James: "are we using the best available forecasts?")

We use forecasts, we don't build them. What a source must have to be usable for training: past
forecasts ("reforecasts") from **one unchanged model** over many years (2005-2024 here), so the model
learns how that forecast behaves, with weeks 1-4 covered.

| Source | Strength | Fit for FlashWatch |
| --- | --- | --- |
| **ECMWF extended-range ensemble** (what we use: S2S database, 2025 model, 11 members, 20 years, 1.5 degrees) | Generally the most skilful system at weeks 1-4 in the international comparisons | The right primary source. Limits: 11 members, 1.5 degree archive, non-commercial licence (Q2), slow queue |
| **NOAA GEFSv12 reforecast** (checked on AWS Oct 28) | Second-tier but independent physics; combining two models usually beats either one at weeks 2-4 | Public domain. 2000-2019 (plus part of 2020), a start every day, 5 members to day 16 and 11 members to day 35 on Wednesdays; 0.25 degree to day 10, 0.5 after. Files are global (no cropping on the server), so about 1 GB of transfer per start date for our 4 variables and 11 members, roughly 450 GB in total; a few MB kept after cropping. 2020-2024 would come from the operational GEFS archive (same model since Sept 2020; not yet checked) |
| Other centres in the S2S database (UK Met Office, Canada, NCEP CFSv2, ...) | Extra models for a multi-model blend | Same slow ECMWF queue we're already waiting on; their past-forecast periods differ and are often shorter |
| AI weather models (ECMWF AIFS, Google GraphCast / GenCast / WeatherNext, Pangu, FuXi) | Match or beat physics models over the first 10-15 days | Mostly stop at 10-15 days, and there is no 20-year archive of their past forecasts to train and test on; making one means running them ourselves on GPUs for ~2,000 start dates. A candidate for a live product's week 1-2 later, not for this build |
| NOAA CPC official outlooks (6-10 day, 8-14 day, weeks 3-4) | Forecaster-blended, already combine ECMWF and GEFS | Coarse categories (above / near / below normal); archive length and format not yet checked |

Plan: ECMWF stays primary. Whether to add GEFS (Q7) depends on (1) how much of the perfect-weather
ceiling the real ECMWF keeps (early M2) and (2) the blurred-forecast curve, which says how much
flash-drought skill a better forecast would buy.

---

## F. Feature backlog (A8)

[x] built, [ ] idea. Every new family is covered by the automatic leakage test.

**F1. Soil moisture dynamics**
- [x] percentiles (root zone, surface); 5/10/20-day changes; days since last at 40th percentile+
- [x] acceleration (is the drop speeding up?); days until the 20th percentile at the current rate;
  surface minus root-zone percentile (drying from the top down); highest percentile in the last 15 days
- [ ] how fast this cell typically dries after rain (drydown timescale)
- [ ] the 0-100 cm percentile as a "deep storage" feature now that the target is 0-40

First batch result (F1-F3, fast loop, 0-100 cm target): week-1 skill 4.3% -> 5.5%, week 2 1.2% -> 1.5%,
any-28 2.5% -> 3.4%; all three improvements clearly real (95% range excludes zero). Ranking of risky cells
(PR-AUC, week 1) 0.085 -> 0.105.

**F2. Atmospheric demand**
- [x] 14/30-day anomalies of temperature, vapour pressure deficit (air dryness), potential evaporation, sunshine; evaporation / potential evaporation
- [x] hot days (daily mean above 25 C) in the last 14 days
- [ ] Rapid Change Index (Otkin et al.): how fast evaporation is falling behind demand over several weeks
- [ ] air dryness x soil moisture ("drying pressure"); evaporative demand percentile (EDDI-style)

**F3. Rainfall**
- [x] 14/30/60-day anomalies; days since more than 2 mm and 10 mm of rain (dry spell length)
- [ ] fraction of normal; multi-timescale standardised precipitation

**F4. Vegetation and land surface**: MODIS greenness anomaly and change; land surface temperature anomaly; growing degree days (crop stage)

**F5. Spatial context** (instead of M3)
- [x] 7x7-cell neighbourhood (~175 km): average percentile and 10-day change; share of neighbours already below the 20th percentile; share that crossed below it in the last 14 days (drought spreads)
- [x] directional boxes (James): west, east, north and south of the cell (1-6 cells along the direction,
  +/-2 across): average percentile and share already in drought. **Result: no effect** (difference 0.0
  with a tight range at every horizon) once the all-round neighbourhood is in, so left out of the models
- [ ] distance to the nearest Drought Monitor D1+ area

**F6. Large-scale drivers**
- [x] El Nino index (Nino 3.4)
- [x] MJO (a tropical pattern that moves round the globe over 30-60 days and affects weather 2-4 weeks out): NOAA PSL's real-time index (ROMI), which uses past data only
- [ ] PNA, PDO, AO
- [x] **climate-change state** (James): global mean surface temperature anomaly over the previous 12 months (NASA GISTEMP, as available on the issue date)
- [ ] Plains-wide average soil moisture anomaly

**F7. Static**: aridity, growing-season rain [x]; soil water holding capacity, cropland share, irrigated share [ ]

**F8. ECMWF forecast (M2)**: weekly forecast anomalies [x]; forecast soil moisture as a percentile (same units as the target) [ ]; forecast rainfall deficit added up over weeks [ ]

**F9. Other sources**
- [x] Drought Monitor category and its 4-week change at the cell; neighbourhood share at D1+ (`eval.cv --usdm`).
  **Result: no gain** (week 1 -0.24 points, range crossing zero; weeks 2 and 28-day flat). It is largely a
  lagged summary of the soil and rain signals the model already has. Left off by default.
- [ ] ESI and EDDI (NOAA drought indices)

**Where features stand (Oct 28, fast loop, 0-40 cm):** week-1 skill 4.4%, week 2 1.4%, any-28 5.3% vs
climatology. Gains so far came from soil-moisture dynamics, dry spells and the neighbourhood (F1-F3, F5).
The latest land-state additions (deep storage, ET/PET rapid change, VPD x dryness: +0.03 points) and the
Drought Monitor added nothing, so the land-state signal looks close to saturated. The next big lever is
weather forecasts (M2, F8), waiting on the ECMWF download.

**F10. Year (diagnostic, James)**
- [x] year as a number (does risk trend over time?)
- [ ] (later, L7) year as a category: diagnostic only. In leave-one-year-out testing the held-out year's own
  category is never seen in training, so it can't help predictions. We fit it on all dev years to
  measure how much each year differs, then look at whether those year effects trend or are just a few odd years.

---

## G. Agent (Meta model)

Needs a real job, not decoration (25% of the score is the use of Meta's tools). Current idea (James):
**weekly alert emails** drafted for county extension offices and farmers when drought risk is
rising: the risk level, why (top drivers), what to watch and practical actions. Each email is
checked number by number against the model outputs before it goes out, and the agent keeps score of
its past alerts. In the demo, emails are drafted and shown, not actually sent. Meta API credits
arrived Oct 28 ($150).

**Scoring context** (official rules, docs/hackathon-rules.pdf): Impact 30%, Use of Meta AI tooling 25%
("how creatively ... beyond a basic integration ... surfaced unexpected capabilities"), Technical depth
20%, Demo 25% (working demo, holds up live, video quality). (The Discord FAQ says "scored equally";
the signed rules give the weights above.) Gate: must fit a theme and *meaningfully* use Meta models.
Bonus prizes that fit us: Best Agentic Use-Case, Best Build on Open-Weight Models.

**Plan for the Meta models** (each does a job the forecast can't do itself):

| # | Model | Job | Why it's more than decoration |
| --- | --- | --- | --- |
| G1 | **Muse Spark** (Model API, tool calling) | **FlashWatch analyst agent.** Each week: pulls risk, the drivers behind it, the ECMWF weather outlook and the Drought Monitor status for a county or region through tools; writes the briefing and county alert emails; a checker compares every number in the draft with the tool outputs and sends it back to fix anything that doesn't match; keeps a track record of its past alerts | Grounded, self-checking agent with real tools. Multimodal twist: it also reads the rendered risk map image and must agree with the numbers (a cross-check judges can see) |
| G2 | **Muse Glimmer** (open weights, LoRA fine-tune) | **Drought Monitor-style narrative writer**: trained on the real weekly USDM narratives (public domain) paired with our structured state for that week; writes the weekly regional summary | Scored against held-out real narratives plus a fact check. Runs without an API. Targets the open-weights prize. Training on Modal or Fireworks (Oct 30 live session covers training) |
| G3 | Muse Voice Transcribe (stretch) | Field reports: a farmer or extension agent leaves a voice note ("corn curling by noon on the ridges"); transcribed, turned into a structured report by Spark, shown on the map and used by the agent | Ground truth from the people who benefit; good demo moment. Only if G1 and G2 are done |
| - | SAM 3.1, Muse Image, Muse Wonder | No natural job; skip | Forced uses would cost marks on "meaningful" |
| - | Muse Code | Fine for James to use when building the agent and site | Genuine use of Meta tooling; mention it |

Order: G1 first (core of the demo), G2 from Oct 30, G3 only if time allows.

**G1 live test (Oct 28, Muse Spark `muse-spark-1.1`, 2012-05-11 pack, Champaign County IL).** Tool
calling works. First run: the checker flagged 5 figures, but they were real values written as "2.6
below normal" (sign dropped) and "28 days", so the checker now ignores sign and exempts time spans;
the tools now return percentages, and the prompt asks for whole-number percentages and plain words for
standard deviations. Second run: passed on the first draft, in plain language for farmers (28% this
week vs 4% normally, about 7 times normal; drivers: forecast rain well below normal, very dry air, 83%
of forecast scenarios hot and dry). About 7,600 tokens per briefing. Map check: Muse named Sioux
Falls, Chicago, St. Louis; the hotspot tool's top 3 were Des Moines, Chicago, Detroit (1 of 3 agree).
Not yet looked into: the map may need clearer contrast or a bigger k.

## O. From forecast to action: the decision dashboard (proposed Oct 28, James; under discussion)

James: an abstract predictor isn't compelling; a tool that tells a farmer *what to do this week, at
their place* is. The forecasting spec doesn't change; this is a layer on top of it. Nothing here is
built yet except the evidence in O0.

### O0. Evidence so far (spring forecast dates, 6 development years, fast loop; `eval/lead.py`)

*Watch list.* Flag the riskiest 10% of land cells each issue date and count real onsets:

| | Week 1: onsets caught, hit rate | Week 2 |
| --- | --- | --- |
| Seasonal normal (B0) | 12%, 6% | 10%, 5% |
| Land only (M1) | 22%, 11% | 15%, 8% |
| With ECMWF forecasts (M2) | **42%, 21%** | 19%, 11% |

*Lead over the Drought Monitor* (2006-2021, 64,765 onsets at 0-40 cm; each date counted from when it is
public: Monitor maps 2 days after their valid date, ERA5-Land 5 days after). Monitor response = the
first map showing the cell worse than before the soil began drying (20 days before onset), and the
first showing D1 (moderate drought) or worse. Median days FlashWatch is earlier, among onsets the
Monitor does respond to within 12 weeks:

| FlashWatch signal | vs first Monitor worsening | vs first D1+ |
| --- | --- | --- |
| Soil crosses the drought line (seen 5 days later) | **8 days** (FlashWatch first 70%) | **22 days** (86%) |
| Onset confirmed (10 days below the line, seen 15 days later) | -2 days (46%) | 12 days (68%) |
| M2 week-1 watch list (spring dates) | 23 days (96%) | 45 days (99%) |
| M2 week-2 watch list (spring dates) | 28 days (99%) | 49 days (100%) |

The watch-list rows count from the forecast's issue date, so they include the forecast lead itself.
**Caveat:** the Monitor worsens in the following 12 weeks after 55% of our onsets, against 32% for
the same cells and dates in other years (D1+: 36% vs 15%). So it confirms our onsets well above
chance but stays silent on about half. Either those onsets recover before mattering, or the Monitor
misses them; crop yields (O4) can tell which.

*Economic value* (Richardson-style cost/loss test). A user takes a protective action costing C that
avoids loss L if an onset happens; the value is the share of a perfect forecast's saving over the
best fixed policy (always or never act). 0 = useless, 1 = perfect:

| Cost/loss ratio C/L | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- |
| M2 week 1 | 0.48 | **0.60** | 0.40 | 0.27 | 0.17 | 0.07 |
| M1 week 1 | 0.16 | 0.21 | 0.00 | 0.00 | 0.00 | 0.00 |
| M2 week 2 | 0.20 | 0.29 | -0.02 | -0.03 | -0.01 | 0.00 |
| M2 any 28 days | 0.01 | 0.00 | 0.16 | 0.27 | 0.24 | 0.05 |

**Meaning:** FlashWatch pays for *cheap, high-leverage* actions, where acting costs up to about a fifth
of the loss it avoids. Week-2 warnings only pay for very cheap actions. Expensive actions (C/L above
0.3) are better decided on the seasonal normal. That shapes which actions the dashboard should offer.
All spring-only so far; re-run on the full season.

### O1. Decision engine (cost/loss, by days of notice)

Each action is a row in a user-editable config (defaults are placeholders, labelled as such):

| Field | Meaning |
| --- | --- |
| `cost(lead)` | $/acre (or $/head) of acting with `lead` days of notice before onset |
| `avoided(lead)` | $/acre of loss avoided by acting, *if* an onset happens; usually falls as notice shrinks |
| `salvage` | share of the cost recovered if no drought comes (e.g. stored hay keeps) |
| `window` | the months/crop stages where the action applies; minimum and maximum notice |
| `event` | which forecast it keys on: onset in week 1, week 2 or any 28 days |

Given FlashWatch's chance p of the event: expected value of acting now = p x avoided - (1 - p) x
cost x (1 - salvage) - p x cost. Act when that is positive (p above the break-even ratio). The range
comes from the forecast scenarios (O2), not just the mean.

### O2. "Not yet, but if ..." (the value of waiting)

Waiting gives a fresher forecast but less notice (cost may rise, avoided loss falls). Act now if
EV(now) beats the expected EV of deciding at the next forecast update. The update's outcomes come
from the **ECMWF ensemble members** (we store all 11 per start date): group members by the coming
week's weather (e.g. rain below/above a threshold, heat), re-run M2 with each group's forecast
features, and get the risk *given* that weather. That gives triggers a farmer can check against
their own local forecast or rain gauge: "Not yet. If less than about half an inch of rain falls this
week, risk rises to 41% and acting is worth about $X/acre; otherwise risk drops to 9% and holding is
right." Triggers are on weekly totals (our forecast data is weekly); daily triggers ("if tomorrow is
above 84 F") would need daily forecast data and stay a later idea. A "what-if" selector on the
dashboard shows the same: pick a rain/heat scenario, see risk and value change.

### O3. Actions catalogue (first pass; defaults from literature or left for the user)

| Action | Who | Typical notice | Why notice matters |
| --- | --- | --- | --- |
| Start irrigation early / top up the profile | irrigated growers (e.g. Nebraska, Kansas) | 3-10 days | Once the root zone empties in a heat wave, pivots can't put water back fast enough |
| Cut or delay in-season nitrogen | corn, at side-dress (June) | 7-14 days | Saves input cost where drought caps yield; false alarm costs some yield |
| Buy hay/feed or secure grazing | livestock | 14-28 days | Prices rise once drought is official; stored feed keeps (high salvage) |
| Early weaning, moving or selling stock | cow-calf | 14-28 days | Cheaper before pasture fails and sale barns fill |
| Hold off soil-drying field work | all | 3-7 days | Almost free |
| Fire crews, burn bans; water-use messaging | agencies, utilities | 7-14 days | Non-farm users of the same signal |

**Left out on purpose:** grain hedging or marketing recommendations. That is trading advice, and
markets already price weather forecasts. Futures prices are used only to *value* a yield change (O4).

### O4. From soil to dollars (James)

- **Loss given onset**, by crop and stage. Two sources:
  - the FAO yield-response approach (FAO Irrigation and Drainage Paper 33/66: relative yield loss =
    Ky x relative water shortfall, with Ky by crop and growth stage; maize is most sensitive around
    flowering);
  - our own evidence: link onsets to NASS county yields 2005-2021 (we have a NASS key; L5). For
    example, "county-years with a June-July onset over most of the cropland: corn yield X% below
    trend". That gives a data-backed default and also answers the O0 caveat (do onsets the Monitor
    ignores cost yield?).
- **Price:** the user enters their expected price; optionally fill in the current December corn or
  November soybean futures quote. Dollars per acre = yield loss x expected yield x price.
- All defaults are shown with their source and can be overwritten. FlashWatch's numbers (risk,
  scenarios) are the forecast; the economics are the user's.

### O5. Dashboard ("FlashWatch for your farm")

A single page per location, open on the desktop through the season:
- choose a place (click the map, county search or lat/lon) and a farm profile (crop, irrigated or
  not, planting date, acres, expected yield and price);
- action cards: **Act now / Not yet / Watch trigger**, with expected value now in $/acre and for the
  farm, its range across forecast scenarios, the trigger sentence and the days of notice left;
- risk timeline: the past weeks' forecasts against what happened, with the Drought Monitor
  category underneath (the lead shows directly);
- the what-if selector (O2) and an editable costs table (kept in the browser);
- **replay mode** for the demo (2012 and other development years; test years only after the freeze).
  A live mode needs current ERA5-Land (5-day latency) and current ECMWF ensembles. ECMWF's free open
  data has the ensemble out to 15 days (to check: licence and fields), which covers weeks 1-2, but M2
  was trained on the 11-member reforecasts. Live mode is later, not for the hackathon.

Technically, a static page: the Python side precomputes per location and issue date the risk, the
scenario risks and the drivers (JSON); the economics are computed in the browser from the user's
config, so editing a cost updates instantly with no server. Muse needs a server (the key can't be in
the page), so the hackathon site shows precomputed agent output, and a small local server runs live
questions in the demo.

### O6. The agent's job here (Muse Spark; extends G1)

- New tools: `action_values(place, profile)` (expected value per action, scenario range, trigger) and
  `scenarios(place)` (risk given each weather group). The number checker applies as before.
- **Alerts:** when an action flips from "not yet" to "act now", or a trigger is met, the agent drafts
  a short text message (SMS length) and a longer email with the reason and the numbers. In the demo
  they go to an alert log on the page; really sending them needs a messaging account (e.g. Twilio) and
  James's go-ahead.
- **Questions:** "should I start watering the north pivot?" → the agent calls the tools with that
  farm's profile and answers with the trigger.
- Later: push alerts into farm-management software (John Deere Operations Center and Climate
  FieldView, for example, have developer APIs); export as webhook or CSV. Not for the hackathon.

### O7. Honest evaluation in decision terms

- Value curves (O0) by lead, on development years now and on the test years once, after the freeze.
- Replay track record: for 2012 and a quiet year, what a user following the rule would have spent and
  saved against never/always acting, with the (placeholder) costs stated.
- The O0 caveat reported, and the yield link (O4) if built.

### O8. Build order (proposed; see QUESTIONS Q8)

1. Decision engine (O1) in Python with tests; value curves into the report. About half a day.
2. Scenario re-scoring from ensemble members (O2). Half a day to a day.
3. Dashboard page (O5) in replay mode, replacing the plain replay page in D. About a day.
4. Agent tools and alert drafting (O6). Half a day.
5. NASS yield link (O4). Half a day; if short of time, literature defaults only.

To fit by Nov 4, something else gives. Recommended cut order: G3 (voice), L1 (M3), then a smaller
G2 (Glimmer). Never cut: the burst and freeze, the test-year scoring, the agent number checks.

## D. Deliverables (submission: Nov 4, 9 PM ET)

Required: public repo, a 3-minute demo video (judges score it), every contributor named, a working
product (judges assess "video, code and live product").

- **Live product**: a static site (GitHub Pages or similar) with the replay, skill-by-horizon charts and
  this week's agent briefings. If O is agreed, the replay becomes the decision dashboard (O5). Static means it can't break live. Q2 decides whether M2's forecast-derived
  maps can be on it.
- **3-minute video** (draft storyboard):
  - 0:00-0:10: one line that everything shown was built during the hackathon (the FAQ asks videos to
    say what, if anything, was pre-existing).
  - 0:10-0:35: the problem. Flash droughts arrive in weeks (2012 cost tens of billions); official maps
    describe them after they arrive.
  - 0:35-1:30: 2012 replay. What happened, the land-only forecast, then the forecast with weather
    forecasts catching the waves; the lead time over the Drought Monitor.
  - 1:30-2:20: the agent. This week's briefing for a county, numbers checked live, the map cross-check,
    the Glimmer narrative.
  - 2:20-2:50: honest skill. Skill by lead with uncertainty ranges; the test years were scored once, after the freeze.
  - 2:50-3:00: who benefits; open source; link.
- **Repo**: README as the front door (what it does, results, how to run, architecture diagram); the
  process docs (DESIGN, QUESTIONS) are a strength for "technical depth" and stay, tidied.
- **Most Viral bonus**: post the video on X by Nov 8 (optional).

---

## V. Report and visualisation

The demo is 25% of the score, and a striking visual story probably matters more than tables.
Plan the story first, then make sure the pipeline writes the data it needs.

- **Hero: 2023 replay.** Animated map, week by week: FlashWatch week-2 probability as a glowing
  field, onsets appearing as they happen, Drought Monitor outlines arriving later. "We saw it N weeks earlier."
- **Lead-time race.** For each episode, FlashWatch's first alert vs the Drought Monitor's first worsening.
- **Skill map.** Where the model is better than climatology (tests the west-vs-east expectation).
- **Reliability and skill by horizon**, with uncertainty ranges, against B0, B1 and raw ECMWF.
- **"Why this week" map**: top drivers by region, which also feed the agent's emails.
- **County alert cards** from the agent, with its own track record.
- Format: a static scrollytelling page; the 3-minute video cut from it. Interactive click-map: L2.
- Build it first on 2012 (dev-year predictions), so the page exists before the model is frozen.

**Prototype (Oct 28):** `python -m flashwatch.viz.replay --oof <stem> --year 2012` writes
`data/viz/replay_2012.html` (map per issue date: week-2 risk, cells already in drought, actual onsets,
Drought Monitor D1+ outline; play/scrub; season chart of mean forecast vs actual onset rate). Serve with
the `viz` entry in `.claude/launch.json`. **Finding for the story:** in 2012 the actual week-2 onset rate
spikes to ~35% of eligible cells in early May and ~20% in August, while the light M1's mean forecast stays
at 3-5%: it ranks cells sensibly but does not see the regional bursts coming. Forecast features (M2) are
the obvious test of whether they can; the replay should probably show relative risk (vs normal for the
date) rather than raw probability so the map isn't uniformly dim.
