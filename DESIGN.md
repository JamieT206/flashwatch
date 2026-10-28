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
- *P11, M2 member by member* (Oct 28, `--members`; for priority 1). The diagnosis above says the
  ensemble *mean* blurs the strong hot, dry spells. So: train M2 on each of the 11 members' own weather
  (z against all members' climatology; one copy of each training row per member), and average its
  predictions over the members. The ensemble-level features (spread, share hot-and-dry) stay as they
  are. This is the same machinery O2 needs for "risk given this kind of week".
  **Result (Oct 28, 13 start dates, `m2mem13` vs `m2fc13`, paired year-bootstrap): no gain.** Week 1
  15.3% vs 16.3% (-1.2 points, 95% CI -3.0 to -0.05); week 2 2.4% vs 2.6% (-0.3, CI -0.7 to +0.2); any
  28 days 6.4% vs 7.2%. A single member's weather is a noisier guide than the ensemble mean, and
  averaging member-level predictions doesn't recover the strong spells either. So the week-2 limit
  really is in the forecasts, not in how we summarise them. **Decision:** the forecast stays on the
  ensemble mean (the standard M2). Per-member predictions are used only for the "what are we waiting
  for" triggers (O2), where the members stand for the next forecast's possible outcomes.
  **Re-run with 16 start dates (Oct 29, `m2fc16` / `m2mem16`; Apr 1 - May 19, gaps at Apr 19-23, May 9, May 13-17;
  hindcasts of those calendar days in 2005-2024).** Standard M2 week 1 **16.8%** (90% range 11.8-20.3), week 2 **3.8%** (1.2-5.4), any 28 days
  **8.4%**; M1 on the same dates 1.7% / 0.8% / 1.4%. A little better than at 13 dates (16.3 / 2.6 /
  7.2), mostly at week 2. Member by member is again slightly worse (week 1 -1.2 points, CI -2.1 to
  -0.5; week 2 -0.4, CI -0.6 to 0.0), so the decision stands. Still spring dates only: the site keeps
  M1 for the summer, and the roll-up keeps the blurred-forecast stand-in for M2 until June-August
  start dates arrive.
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

- L1. M3: deep-learning model on the map (U-Net/ConvLSTM). **Started Oct 29** (`models/cnn.py`, model `M3` in eval.cv): the M1
  feature set laid out as one image per forecast date (a channel per feature plus a row-present channel),
  a small U-Net (two scales, a dilated middle block), masked cross-entropy on cells with a row, the latest
  training year held back for early stopping and a Platt map. Trained inside each CV fold on its training
  years only, so the leakage discipline is the fold's. First comparison: `--models B0 M1 M3 --fast`.
  **First result (Oct 29, `m3`, land features, all dates, 6 folds).** M3 alone: week 1 Brier skill 2.4%
  (M1 4.3%) but PR-AUC 0.116 (M1 0.113): it ranks as well as M1 and is worse calibrated (slope 0.85; the
  Platt map rests on one held-back year). **Blend** (average of M1's and M3's log-odds, 50/50 fixed in
  advance, `m3blend`): week 1 **5.4%** (90% range 4.2-6.4), PR-AUC **0.128**; paired against M1 +1.1 points
  (0.8 to 1.5). Week 2: 1.6% vs 1.2% (+0.3, CI -0.3 to +0.8). So the map model sees something the
  per-cell model doesn't, at week 1. Next: M3 with the forecast features against M2 (`m3fc`); a
  cross-fitted calibration for M3; if it holds, the blend goes into the Nov 2 burst (B4 ensembling).
  **With the forecast features (`m3fc`, 16 spring forecast dates):** no gain. M3 alone week 1 10.6% (M2
  16.8%); the M2+M3 blend 16.3% (paired -0.6, CI -3.8 to +2.2); week 2 3.5% vs 3.8%. Only ~16 pictures a
  year to learn from on forecast dates, against ~100 on all dates, so the map model is starved. Re-run when
  the summer forecast dates land; until then the blend is a land-only (M1) improvement.
  **Two M3s averaged (`m3seeds`: M3 and M3b, a wider net with another seed; then 50/50 with M1, fixed in
  advance).** Week 1 Brier skill **6.0%** (90% range 4.8-7.0) against M1's 4.3%: paired **+1.7 points (1.2 to
  2.3)**; PR-AUC 0.141 against 0.113 (+25%). Week 2: 1.7% against 1.2%, paired **+0.5 (0.1 to 0.8)**, now
  clear of zero. The single-M3 re-run moved from 2.4% to 2.8% (GPU training isn't bit-for-bit repeatable),
  which is why averaging several nets helps. Plan for the burst: M1 + mean of 3-5 M3s; and the same with
  M2 once summer forecast dates land.
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
- L8. **Should I install irrigation? (James, Oct 29; a separate product, parked.)** A business case for
  a rain-fed field: how often flash droughts hit it now (ERA5-Land, our labels) and under climate
  projections (e.g. NASA NEX-GDDP-CMIP6), the irrigated vs rain-fed yield gap (NASS county yields),
  installation and pumping costs (USDA Irrigation and Water Management Survey), and whether water is
  available at all (USGS groundwater levels, state and district well permits; many Nebraska districts
  restrict new wells, the Ogallala is falling). Output: a map of where irrigation newly pays, now and
  around 2040. Most promising in the humid east (IN, OH, MI, WI), where supplemental irrigation is growing
  and water is easier to get.
- L9. **Rain-fed decisions with equal billing (James, Oct 29; QUESTIONS Q14).** 130 million rain-fed
  acres against 13 million irrigated in the coverage area. Candidates with 1-2 weeks' notice: skip or cut
  side-dress nitrogen (built, rough), skip the mid-July fungicide pass, hay and feed purchases and herd
  numbers (built, rough), early silage harvest. Not crop insurance (choices close in March). Not grain
  marketing (prices already carry the forecasts; edges into trading advice).

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

**G2 data ready (Oct 28 night).**
- *Narratives.* `ingest.usdm_text` fetched all 1,044 weekly Drought Monitor narratives, 2005-2024.
- *Training pairs.* `agent.glimmer_data` pairs each week's High Plains / Midwest sections with
  that week's numbers, by region and state:
  - ERA5-Land rain as a percent of normal and temperature departure, with normals from 2005-2019;
  - soil moisture against normal and its weekly change;
  - new flash-drought onsets;
  - the Monitor's category shares and their change.
  That gives **749 training weeks (2005-2019) and 104 validation weeks (2020-2021)**; 33 weeks had
  no matching section. Test years are not written.
- *Limitation.* Older narratives also mention places outside our grid (Texas, Arkansas), which the
  inputs can't support.
- *Scoring* (`eval.glimmer_eval`):
  - share of figures not traceable to the input;
  - F1 of the states named;
  - direction of change. This is a crude word count, and unreliable: it misreads some real
    narratives, so treat it as indicative only;
  - ROUGE-L against the real text.
- *Muse Spark 1.1, zero-shot, 4 summer validation weeks:* untraced figures 0%, states F1 0.69,
  direction 0.25, ROUGE-L 0.09. About 35k tokens, mostly the model's reasoning. Spark is accurate
  but writes a recital of the numbers, not the Monitor's prose ("Drought expanded in parts of
  Wyoming ... that missed out on the heaviest rainfall"). Style and selection are what the Glimmer
  LoRA has to learn. The bar to beat: keep untraced at about 0 and raise ROUGE-L and states F1.

**G3 built (Oct 28 night, `agent/voice.py`).**
- *Endpoint.* The Meta API lists `muse-voice-transcribe-1.0`; file transcription is `POST
  {base}/asr/transcribe` (multipart: a JSON `request` part with `model`, `audioEncoding: "WAV"`,
  `sampleRateHertz`, plus the `audio` part; 16-bit mono WAV at 16 or 24 kHz). This was found by trial,
  as the OpenAI-style `/audio/transcriptions` gives 404.
- *Test.* A 17-second synthetic voice note ("Corn on the sandy pivot, N one, is curling by noon on the
  ridges ... the south beans look fine") was transcribed accurately. Spark, with a `farm_pivots`
  tool, mapped "the sandy pivot" to N1 (severity 2: stress showing) and the beans to S1 (fine), at
  about 10k tokens.
- *How reports are used.* They go to `site/data/ops/reports_<year>.json`. The operator's overview
  carries them, and the guard accepts a start backed by a recent stress report (the farmer's own
  eyes are the usual practice). The farm page quotes the note under "Your pivots".
- *Next.* A real recorded voice, and replies by voice.

**Model note.** The API also lists `muse-spark-1.2` and `muse-spark-1.3`. 1.3 ran the July 3 operator
morning well: it quoted the farmer, held N2 for the well, and used about 10k tokens against 14k on
1.1. The default stays 1.1 until the briefing and farm notes are re-checked on 1.3 (set
`META_MODEL`).

**G1 live test (Oct 28, Muse Spark `muse-spark-1.1`, 2012-05-11 pack, Champaign County IL).** Tool
calling works. First run: the checker flagged 5 figures, but they were real values written as "2.6
below normal" (sign dropped) and "28 days", so the checker now ignores sign and exempts time spans;
the tools now return percentages, and the prompt asks for whole-number percentages and plain words for
standard deviations. Second run: passed on the first draft, in plain language for farmers (28% this
week vs 4% normally, about 7 times normal; drivers: forecast rain well below normal, very dry air, 83%
of forecast scenarios hot and dry). About 7,600 tokens per briefing. Map check: Muse named Sioux
Falls, Chicago, St. Louis; the hotspot tool's top 3 were Des Moines, Chicago, Detroit (1 of 3 agree).
Looked into (Oct 28 night): the "truth" was the nearest city to the top single cells, but the model reads patches. Against cities ranked by the average risk within 150 km (`brief.city_truth`), the same answer agrees on 2 of 3 (Chicago, Sioux Falls), and its third pick (St. Louis) is 4th. Spark 1.3 gives the same answer. Both measures are now reported.

## O. From forecast to action: the decision dashboard (James, Oct 28; scope agreed, Q8)

James: an abstract predictor isn't compelling; a tool that tells a farmer *what to do this week, at
their place* is. The forecasting spec doesn't change; this is a layer on top of it. Nothing here is
built yet except the evidence in O0.

### O0. Evidence so far (16 spring forecast dates since Oct 29, 6 development years, fast loop; `eval/lead.py`)

*Watch list.* Flag the riskiest 10% of land cells each issue date and count real onsets:

| | Week 1: onsets caught, hit rate | Week 2 |
| --- | --- | --- |
| Seasonal normal (B0) | 12%, 6% | 10%, 5% |
| Land only (M1) | 20%, 11% | 15%, 8% |
| With ECMWF forecasts (M2) | **42%, 23%** | 21%, 11% |

*Flagged farms, pooled (Oct 29, James: "top 10% of all farm-days", so a quiet date can flag nothing; `evidence.flagged`,
16 spring dates).* Flash drought begins in the next 7 days on 28.8% of the farm-days M2 rates riskiest, against 5.3% overall:
**5.4x**, and those flags hold **54%** of all onsets (per-date top 10%: 4.2x, 43%). Days 8-14: 2.4x, 24%. The farmer's rule of
thumb **H1** (logistic on the last two weeks' rain and temperature against normal; `models/baselines.WeatherRule`, run
`h1fc16`): 1.65x in week 1, 0.5x in week 2, below the seasonal calendar alone (B0 pooled: 1.9x, because it picks the
high-risk time of year). Brier skill: H1 0.8% (CI -0.2 to 1.7) at week 1, -0.2% at week 2; M2 16.8% and 3.8%. H1 replaces
M1 as the public comparison (M1 stays as our internal ablation).

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

### O1a. How much is one more day of notice worth? (built: `flashwatch/decide`, Oct 28)

James: without effectiveness falling as notice shrinks, an engine would always wait until the last
moment to maximise certainty. So each action needs a value-by-notice curve.

**Irrigation, from physics rather than guesses** (`decide/irrigation.py`). It is a daily root-zone
water balance in the FAO-56 style through a rainless spell, starting from a full profile:
- stress begins once depletion passes 55% of the available water;
- yield loss = Ky x the crop's water shortfall / seasonal water use (FAO-33; Ky 1.5 around flowering);
- the pivot adds at most its capacity each day and can't overfill the profile;
- the usual practice starts the pivot 2 days after stress begins (sensor or crop symptoms);
- FlashWatch starts it k days *before* stress begins.

Typical Nebraska settings (silt loam, 3 ft roots, crop using 0.30 in/day, 220 bu at $4.50, $10 per
acre-inch), avoided loss in $/acre against the usual practice:

| Pivot capacity | 0 days | 3 days | 5 days | 7 days | 10 days | Extra pumping at 7 days |
| --- | --- | --- | --- | --- | --- | --- |
| 4 gpm/acre (0.18 in/day reaching the roots) | 21 | 51 | 70 | 87 | **109** | 17 |
| 5 gpm/acre (0.23 in/day) | 26 | 61 | 78 | **85** | 85 | 16 |
| 6.5 gpm/acre (0.29 in/day, about the crop's use) | 24 | 24 | 24 | 24 | 24 | 5 |

So, to James's "8 days vs 7": with a 5 gpm pivot each day of notice is worth about $10-12/acre up to
about 6 days, then nothing, because the profile is already full when the pivot starts. A
low-capacity pivot keeps gaining to about 10 days, while a pivot that can match the crop's use only
needs to start on time. On sandy soil (half the stored water) stress comes sooner, and notice stops
helping after about 4-5 days. Nothing here is fitted to FlashWatch data. It is textbook physics with
every setting in `config/actions.json` (sources noted), to be replaced with the farm's own.

**Hay/feed and nitrogen:** value-by-notice tables in the config, all placeholders for now. Hay's value
mostly comes from buying before the market moves. That tends to happen when the Monitor shows drought,
a median 22 days after the soil crosses the line (O0), so extra notice matters less than for
irrigation. James's point stands: buying early is partly zero-sum between neighbours, so irrigation
leads the story.

**Engine** (`decide/engine.py`, `python -m flashwatch.decide --week1 0.28 --week2 0.15`):
- daily onset chances from the weekly forecast;
- expected value of acting today against the usual practice;
- expected value of deciding again in 3 days, over forecast-revision scenarios; until O2 is wired, a
  placeholder of risk x1.6 or x0.4 at 50/50;
- break-even chance of onset.

Champaign-like risk (28% week 1, 15% week 2) with a 5 gpm pivot: **act now**, +$19/acre today vs
+$11 if deciding in 3 days. At 4% / 10%: **not yet**: -$0.6 today, but if the next forecast raises
the risk 1.6x, acting then is worth +$3.5. Break-even is about a 15% chance of onset in the next 14
days.

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

**Built (Oct 28 evening, James: "what's the info we're waiting for tomorrow?"):**
- `decide/engine.decide_members`: each of the 11 members is a possible next forecast. Its risk is
  pulled halfway toward the ensemble mean (lam = 0.5), because the next forecast resolves only part
  of the spread.
- `decide/trigger.py`: finds the coming-week rainfall or mean-temperature threshold at the farm that
  best separates the members where starting at the next forecast would pay from the rest. It needs
  90% agreement; otherwise it falls back to a risk threshold.
- The farm page says, e.g., "What we're waiting for: if the next forecast shows **less than 0.4 in of
  rain over the coming week** at your farm, start the pivot then (3 of 11 of today's scenarios look
  like that)".
- The same code is in JS (identical results on a test case).
- It needs per-member M2 predictions. `cv --members` now keeps them, with each member's week-1
  weather (`M2_week1_m<j>`, `wx_rain1_m<j>`, `wx_temp1_m<j>`), and the site build passes them through.
- Daily triggers ("day 7 above X F") would need daily fields. The decoder computes daily means before
  averaging to weeks, so they are within reach.

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
- **Built: the yield link (Oct 28 night, `eval/yields.py`; NASS county survey yields 2005-2021,
  `python -m flashwatch.ingest.nass yields`).** Each county-year (development years only) gets:
  - an exposure: the share of its land within 40 km where a 0-40 cm onset began June-August;
  - whether the Drought Monitor reached D1 over half of that land in June-September;
  - the yield against the county's own trend, fitted on its other years.

  Results (90% year-bootstrap ranges, 300 resamples). "Most" means half or more of the county
  exposed, "none" means no exposure:

  | Crop | County-years | Most exposed | None | Slope per full exposure | Same, after removing each year's average |
  | --- | --- | --- | --- | --- | --- |
  | Corn, rain-fed (non-irrigated split) | 906 | **-20%** (-33 to -4) | +13% | -39 points (-52 to -22) | -8 points (-15 to -3) |
  | Corn, irrigated (same counties) | 906 | -2% (-3 to 0) | +1% | -3.6 (-6.4 to -0.2) | -1.5 (-2.9 to 0) |
  | Corn, all | 16,273 | -2% | +2% | -5.3 (-8.0 to -1.8) | -2.7 (-5.8 to +1.2) |
  | Soybeans, rain-fed | 445 | -16% | +10% | -31 (-46 to -14) | -6 (-13 to -2) |
  | Soybeans, irrigated | 445 | -1% | +1% | -3.2 (-5.5 to -0.7) | -0.8 (-2.4 to +0.5) |

  - **Onsets cost rain-fed crops a lot, and irrigation saves most of it.** Irrigated corn still loses
    about 1.5-3.6% of yield per fully exposed year. That is the most earlier pumping could win back.
  - **The O0 caveat is answered.** In heavily exposed county-years where the Monitor never showed
    D1, corn yield was +2.9% (0.8 to 5.2), so there was no loss. Where it did show D1, yield was -6.0%.
    So the onsets the Monitor ignores are ones that broke in time; the damaging ones are the ones it
    eventually confirms. This also means FlashWatch's labels include harmless onsets. The value per
    *damaging* onset is higher than the average suggests, but many warnings are about spells that
    won't matter.
  - **The irrigation model was too harsh.** With a 35-day rainless spell (the default), a careful
    irrigator at 5 gpm/acre loses 8.6% of revenue per onset: about 3 times what irrigated counties
    show. A 21-day spell gives 2.7%, in the middle of the observed range (18 days: 1.6%; 25: 4.2%).
    Re-run of the roll-up with `--set spell_days=21` (tags `*_cal`): **US irrigated corn and soy
    about $23M a year** (ECMWF-quality estimate; ceiling with perfect weather $33M; land-only
    about zero), against $99M / $131M before. The region page now leads with the calibrated number,
    with a selector for the textbook one (Q11). The farm page still defaults to 35 days until James
    decides.
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

**Built (Oct 28 evening):** `site/index.html` (the farm page) and `site/region.html` ("the big
picture" for judges). Data comes from `python -m flashwatch.site.build`, `flashwatch.site.evidence`
and `flashwatch.decide.rollup` into `site/data/` (git-ignored until Q2). The engine is ported to
`site/assets/engine.js`; `python -m flashwatch.site.parity` checks it gives identical numbers to
the Python. The farm page:
- the verdict (start the pivot today / not yet / no need), with expected value of today vs in 2 days;
- what would change it: the forecast risk at which starting at the next forecast pays (weather
  triggers come with O2);
- risk tiles, a season chart with FlashWatch's starts and the real onsets, soil moisture and the
  Drought Monitor;
- the season tally against both usual practices;
- the value-of-notice curve;
- editable farm settings, kept in the browser.

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

**Built (Oct 28 evening): `agent/farm.py`.** Muse Spark writes the farm's morning note and a text
message (at most 160 characters). Its one tool, `farm_status`, reads the site data and runs the
engine. Numbers are checked as in G1, and the farm page shows the result. First run, Edmunds County
SD, 2012:
- July 1: "no need to start early" (3%, normal);
- July 3: "Start pivot today. 14% flash drought risk next 7 days, 4.8x normal ..."; the flash drought
  began July 5;
- August 21: "not yet".

The checker caught an untraced "14" (from "14 days") twice, and the redraft fixed it. Three notes
used about 25k tokens.

### O7. Honest evaluation in decision terms

- Value curves (O0) by lead, on development years now and on the test years once, after the freeze.
- Replay track record: for 2012 and a quiet year, what a user following the rule would have spent and
  saved against never/always acting, with the (placeholder) costs stated.
- The O0 caveat reported, and the yield link (O4) if built.

### O9. Backtest and the headline number (built Oct 28 evening)

`decide/backtest.py` replays the farm page's rule day by day (every forecast date, June 1 - August
31, six development years, every other grid cell):
- **start the pivot** when the engine says act now;
- **a hit:** an onset within 14 days earns the avoided loss for that much notice, minus extra pumping;
- **a false alarm:** no onset comes; the farmer pays a week of mostly wasted pumping;
- **missed onsets** are handled the usual way, worth 0.

Two usual practices: careful (starts 2 days after stress) and calendar (starts July 1; before that,
7 days after stress). `decide/rollup.py` maps counties (NASS Census 2022 irrigated corn and soybean
acres, Census Gazetteer centroids) to the nearest cell. Scope 1: the study region. Scope 2: all US
irrigated corn and soy at the region's $/acre (an extrapolation). Scope 3: rain-fed corn, nitrogen
cut (placeholder costs).

Results, per year, against a careful irrigator. The full-season ECMWF forecasts are not downloaded
yet, so the middle row is an *estimate*: weather blurred to correlation 0.7 with the truth, between
ECMWF's measured week-1 (0.8-0.97) and week-2 (0.4-0.65) accuracy.

| Forecasts | Region, irrigated | All US irrigated corn + soy | $/acre (corn / soy) | People fed (US) | Weight vs US retail food loss |
| --- | --- | --- | --- | --- | --- |
| Land only (measured) | $9M | $12M | 1.05 / 0.00 | 1.9M | 2.0% |
| ECMWF quality (estimate) | $76M | **$99M** | 8.33 / 0.86 | **6.6M** | **6.7%** |
| Perfect weather (ceiling) | $99M | $131M | 10.79 / 1.38 | 6.8M | 6.9% |

Against a calendar irrigator, the estimate is $30M (US) and land-only is negative (false alarms cost
more than the early starts save). Rain-fed nitrogen (placeholder costs): $50M under the estimate.
The forecast-revision spread in the engine is 0.5, measured: M2's week-1 risk moves by 51% on
average between forecasts 2 days apart (M1: 30%).

**Honest reading:**
- the number is **about $100 million a year, not billions**;
- that is because irrigated corn is ~11M acres, flash-drought onsets hit a given field in roughly
  1 summer in 3, and a week's notice is worth about $80/acre per onset;
- "people fed" counts the grain's calories (most corn goes to feed and ethanol);
- the retail-loss comparison is by weight, dry grain vs mixed food;
- per-farm numbers are larger where pivots are under-sized (4 gpm/acre roughly doubles the value of
  notice), and smaller for soybeans (less sensitive, less value per acre);
- the year-to-year swing is wide: drought years carry most of it.

### O10. From advice to the pivot: equipment and farm software (James, Oct 28 night; built)

James: "agentic integration into farm-management software, pivot control systems, etc. Makes it
more useful and cooler, and also does good agent work."

**What exists (web research, Oct 28):**
- **Pivot control.** Lindsay FieldNET and Valley 365 (with AgSense) start and stop pivots remotely
  from their own apps. Neither publishes an open API to *start* a pivot. Data sharing runs through
  partner deals (Valley lists API monitoring links to about ten ag-software partners; Lindsay has
  two-way data links with Climate FieldView and Nutrien).
- **Reading pivot data.** Leaf (withleaf.io), a farm-data API, already reads FieldNET and Valley:
  pivots, and the water each one actually applied, by day.
- **Farm-management software.** The John Deere Operations Center has a developer program
  (including a Work Plans API). Climate FieldView has a free partner program with OAuth and upload
  endpoints. Both need partner approval and the farmer's consent per account.

**Built** (`flashwatch/connect/`, `flashwatch/agent/operator.py`, `config/farm_demo.json`, tests in
`tests/test_connect.py`):
- **Connectors.** All share one interface: list pivots, status, water applied, send a command.
  - `simulated`: pivots that run the water balance; used for the demo.
  - `webhook`: a signed JSON command (HMAC-SHA256) to any URL the farmer chooses, such as their own
    automation, a dealer system or a relay.
  - `leaf`: read-only pivots and applied water from FieldNET and Valley. Written from Leaf's docs;
    untested, as we have no account.
  - `manual`: the honest default today. The message tells the farmer what to press in FieldNET or
    Valley 365.
- **Guard** (`connect/guard.py`). This is code, not the model. It refuses a command, with a reason
  the agent sees, when:
  - the pivot is unknown, offline or already in that state;
  - today's advice isn't "start";
  - it's out of season;
  - the depth per pass is out of range;
  - the water allocation is short (for example, a Nebraska NRD allocation);
  - or a shared well would be over-drawn.
  Commands are taken in order, so the agent lists the most valuable pivot first. Nothing is sent
  without the farmer's YES. Connectors refuse unapproved commands as a second lock.
- **Operator agent** (Muse Spark, two tools). `farm_overview` gives each pivot's advice from the
  decision engine, with that pivot's own crop, soil and capacity, plus running state, water applied,
  wells, allocation and a suggested order. `propose_commands` runs the guard. The agent proposes,
  reads the refusals, revises, then writes an approval text and a note. Numbers are checked as in G1.
  Applied water goes back into the next morning's overview: allocation used, pivots now running.
- **Farm page:** a "Your pivots" panel (pivot table, the approval text, what was sent), shown for
  the demo farm's county.

**Live test (Muse Spark, 2012 replay, demo farm in Edmunds County SD: four pivots, N1 and N2 share
a 1,100 gpm well, S2 has 0.5 in of allocation left).**
- *July 3.* Muse proposed N1, N2 and S2. The guard refused N2: the well can't run both northern
  pivots (1,210 gpm needed). Muse kept N1 (worth $533 against N2's $288), cut S2 to a 0.5 in pass
  for its allocation, and left out the soybeans (S1: the advice says no need).
- *The checker.* It caught a total Muse had worked out itself ($779) and had it rewritten.
- *Cost.* About 14k tokens.
- *Rules on later days.* The rule-based plan stopped S2 on July 5, once its allocation was used.

**Not done / needs James:** real accounts (Leaf, Deere, FieldView partner programs); whether the
agent may ever act without a YES (Q10; default never); a reply-by-text loop (needs Twilio or
similar and James's go-ahead).

### O8. Build order (proposed; see QUESTIONS Q8)

1. Decision engine (O1) in Python with tests; value curves into the report. About half a day.
2. Scenario re-scoring from ensemble members (O2). Half a day to a day.
3. Dashboard page (O5) in replay mode, replacing the plain replay page in D. About a day.
4. Agent tools and alert drafting (O6). Half a day.
5. NASS yield link (O4). Half a day; if short of time, literature defaults only.

**Nothing is cut** (James, Q8): what doesn't fit by Nov 4 continues after. Order of work: (1) whatever
meaningfully improves prediction, (2) the farmer tool (this section, irrigation first, then hay and
feed), (3) modelling nice-to-haves (M3, L-list). Never skipped: the burst and freeze, the test-year
scoring, the agent number checks.

## D. Deliverables (submission: Nov 4, 9 PM ET)

Required: public repo, a 3-minute demo video (judges score it), every contributor named, a working
product (judges assess "video, code and live product").

- **Live product**: a static site (GitHub Pages or similar) with the replay, skill-by-horizon charts and
  this week's agent briefings. If O is agreed, the replay becomes the decision dashboard (O5). Static means it can't break live. Q2 decides whether M2's forecast-derived
  maps can be on it.
- **3-minute video** (storyboard v2, Oct 28: farmer first, per James):
  - 0:00-0:10: everything shown was built during the hackathon (the FAQ asks videos to say what, if
    anything, was pre-existing).
  - 0:10-0:30: a farmer's problem. Early July, corn about to tassel, the pivot is off. Flash droughts
    empty the soil in two or three weeks, and the Drought Monitor reports them weeks later.
  - 0:30-1:20: the farm page, Edmunds County SD, 2012 replay:
    - July 1: "no need" at 3%;
    - July 3: "start the pivot today", 14%, about 5x normal, with the expected dollars per acre;
    - the flash drought begins July 5;
    - the text message from the Muse adviser;
    - scrub the date slider through the season;
    - change the pivot capacity and watch the advice and the value of notice change.
  - 1:20-1:50: "not yet, but if ...": a not-yet day and what the next forecast would have to show.
    Ensemble members become weather triggers once the member predictions land.
  - (added Oct 28 night) 1:35-2:05: the agent acts. The farmer leaves a voice note ("the sandy pivot
    is curling by noon"); Muse Voice Transcribe plus Spark turn it into a report on N1. The Muse
    operator plans the morning across four pivots. The farm's rules refuse N2 (the shared well can't
    run both), so the agent starts the more valuable one and trims S2 to its water allocation. The
    text asks for YES; on YES the pivots start (simulated). Two days later S2 stops when its
    allocation runs out. Pivot makers' apps (FieldNET, Valley 365) are the path to real pivots.
  - 1:50-2:30: the big picture: the regional risk map catching the 2012 wave; skill (week 1 is
    where it pays); 22 days ahead of the Drought Monitor. County yields: flash droughts cost rain-fed
    corn about 20% and irrigated corn about 2%. The honest roll-up is about $23M a year,
    calibrated to those yields (Q11), and we say why it isn't billions.
  - 2:30-2:50: how it's built: ERA5-Land + ECMWF ensembles -> calibrated model -> water balance ->
    decision -> Muse Spark adviser with number checking; test years scored once, after the freeze.
  - 2:50-3:00: who it's for (irrigators first; hay and feed next); open source; link.
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
