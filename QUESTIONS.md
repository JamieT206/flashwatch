# Open questions and decisions

Questions for James are at the top; answer inline (edit this file, or reply in chat) and
I'll pick them up. Below them are defaults I've taken; say if you disagree.
Public repo: nothing private goes here.

## Questions

### Q1. Meta tooling is 25% of the score: promote the Glimmer work up the plan? (needs answer by Oct 30)
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

### Q3. Hyperparameter tuning: fully nested, or tuned once on the dev years? (needs answer by Oct 29)
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
Until you answer, I'm running with (c).

### Q4. Compute budget for M1/M2, and your PC's RAM and core count (needs answer by Oct 28)
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
- Isotonic calibration per lead, fitted on inner out-of-fold predictions (4 year-groups
  inside each outer fold's training years).
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
