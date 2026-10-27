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

**F9. Other sources**: Drought Monitor category and its 4-week change at the cell; ESI and EDDI (NOAA drought indices)

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
its past alerts. In the demo, emails are drafted and shown, not actually sent. Meta API credits not
granted yet (Oct 28); fallback is the open-weight Glimmer model run locally.

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
