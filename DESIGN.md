# FlashWatch: design decisions and work plan

Working rule (James, Oct 28): get the whole pipeline running with **light training and short
cycles**, iterate the design while training stays light, and train heavily only once the design
is in place. Decisions fall in two groups:

- **A. Decide now, under light training.** Things we can be confident about without much compute:
  the target, what is in and out of the training data, feature families, the evaluation protocol.
  Decided on label/data evidence wherever possible, not on model skill (no target or feature
  shopping on CV scores; each decision records what it was based on).
- **B. Open until the late burst.** Things that may look different with more compute behind them
  and **don't change what we build**: model library, hyperparameters, ensembling, seeds. Settled in
  one intensive burst at ~70% of the way through (**Nov 2, Sydney**), with the decision rule written
  down before the burst; then freeze, then score the test years once.
  Exception: if an open question could send us down a dead end (it changes the data pipeline or
  what we build), do a heavier run now instead of leaving it open.

Iteration loop: `python -m flashwatch.eval.cv --models B0 B1 M1 --leads week1 week2 any28 --fast --tag <variant>`
(6 held-out years incl. 2012, every other cell, LightGBM lr 0.1 x 150 rounds; 6.8 min per cycle on Oct 28).

---

## A. Decided, or to decide now

| # | Topic | Status | Decision / current default | Based on |
| --- | --- | --- | --- | --- |
| A1 | Study period and splits | Decided | 2005-2024 on one ECMWF cycle; CV 2005-2021 LOYO; test 2022-2024 opened once | ECDS constraints (D1) |
| A2 | Target: onset rule | **To align** | 40->20 percentile drop, P>=40 within d-20..d-5, stays <=20 for 10 days; root zone 0-100 cm (layers 1-3, 7/21/72) | Spec. 2012 check shows 0-100 cm onsets come in May, before the Jun-Jul window: see T1 |
| A3 | Target: horizons | Decided | Week 1-4 and any onset in days 1-28 | Spec |
| A4 | Training rows | **To align** | Eligible if P(issue - 5 d) > 20; 101 issue dates Apr 1 - Oct 15; land fraction >= 0.2; snow/frozen masked | Spec; see T2-T4 |
| A5 | Leakage rules | Decided | Features only via `features.asof`; fold climatologies from training years; perturbation test per feature family | Spec |
| A6 | Reference forecast | Decided | B0 tuned by year-grouped CV (heavy shrinkage to seasonal regional rate) | Q6a, Oct 28 |
| A7 | Calibration structure | **To decide now** | Isotonic on inner OOF (2 year-groups). M1 is under-confident (slope 1.5-2.0) | Fast CV Oct 28. Structural, so not left for the burst |
| A8 | Feature families | **In progress** | ~30 observed + forecast features (HANDOVER 6.2). Deep feature work: see F-list | |
| A9 | M3 (U-Net/ConvLSTM) | **To decide now** | Recommend cut: a different pipeline and a dead-end risk at this timescale; spatial context goes in as GBM features (F5) instead | |
| A10 | Per-lead models vs one model with lead as a feature | To decide now | Per lead | Changes the training table, so not a burst item |
| A11 | Glimmer narrative LoRA | Decided | In plan (Q1); SMAP cut first | Q1 |

### Target questions to align on (A2, A4)
- **T1. Soil depth.** 0-100 cm is dominated by 28-100 cm (72% weight) and reacts slowly; flash
  drought definitions in the literature mostly use shallower layers (e.g. 0-40 cm). Option: compare
  0-28 cm, 0-100 cm and a blend on **label evidence only** (agreement with SMVI and USDM worsening,
  known-event timing, episode coherence), then pick before any more model runs. ~5 min per variant.
- **T2. Eligibility.** P > 20 at issue drops cells already in drought (92% of the 2012 box by Jun 1).
  Right for an onset product; worth saying so explicitly in the write-up.
- **T3. Season.** Apr 1 - Oct 15 issue dates. Onsets in Sep-Nov are frequent (cross-check counts);
  keep, since late-season flash drought matters for winter wheat planting.
- **T4. Masks.** Land fraction >= 0.2 interim; MODIS vegetated fraction planned. Irrigated cells
  (Nebraska, Kansas) are not irrigated in ERA5-Land, so labels and features agree; flag for impact framing.
- **T5. Recovery/false-start exclusion.** The 10-day persistence condition already drops brief dips.

---

## B. Open until the burst (Nov 2, Sydney)

| # | Question | Candidates | Tooling ready |
| --- | --- | --- | --- |
| B1 | GBM library | LightGBM (current), XGBoost (GPU), CatBoost | LightGBM only |
| B2 | Hyperparameters and rounds | Random search per lead on dev years (Q3a) | `models.tune` |
| B3 | Inner calibration groups | 2 vs 4 (if the A7 fix keeps inner models) | `GBM(inner_groups=...)` |
| B4 | Ensembling | Seed bagging; M1+M2 blend; stacking with B2 | No |
| B5 | Model ladder for the headline | M1, M2, B2 (raw ECMWF, recalibrated) | M1/M2 yes, B2 no |
| B6 | Training rows used | Full grid vs thinned (does thinning cost skill?) | `--thin` |

Burst decision rule (to write before Nov 2): pick by mean Brier over the full 17-fold CV per lead,
prefer the simpler option unless the paired year-bootstrap CI excludes zero.

---

## F. Feature generation backlog (A8)

Status: [x] built, [ ] idea. Every new family gets a perturbation test.

**F1. Soil moisture dynamics**
- [x] root-zone and surface percentiles; 5/10/20-day changes; days since P >= 40; 5-day root-zone z
- [ ] rate of decline and acceleration (second difference) over several windows
- [ ] days-to-threshold: (P - 20) / current decline rate, capped
- [ ] surface minus root-zone percentile (drying from the top down)
- [ ] recent drydown e-folding time (how fast this cell loses water after rain)

**F2. Atmospheric demand**
- [x] 14/30-day z of temperature, VPD, PET, solar; ET/PET (raw and z)
- [ ] Rapid Change Index of ET/PET (Otkin et al.): standardised multi-week declines
- [ ] hot-day counts and heat-wave persistence; VPD x soil moisture interaction ("drydown pressure")
- [ ] EDDI-style demand percentile on our grid (from ERA5-Land PET)

**F3. Precipitation**
- [x] 14/30/60-day z
- [ ] dry-spell length; days since > 5 mm; fraction of normal; SPI-like multi-scale

**F4. Vegetation and land surface**
- [ ] MODIS NDVI/EVI anomaly and its 16-day change (Earth Engine, latency ~ 2-3 weeks)
- [ ] MODIS LST anomaly; LST - air temperature (ESI-like stress)
- [ ] growing degree days since Apr 1 (crop stage)

**F5. Spatial context** (replaces M3)
- [ ] neighbourhood means of percentiles and changes (3x3, 7x7); upwind/west neighbours
- [ ] fraction of neighbours with an onset in the last 14 days (drought spreads)
- [ ] distance to the nearest USDM D1+ edge, as of issue

**F6. Large-scale drivers**
- [x] Nino 3.4
- [ ] MJO (RMM phase and amplitude, daily), PNA, PDO, AO (monthly indices, as-of)
- [ ] regional-mean soil moisture anomaly (Plains-wide dryness)

**F7. Static**
- [x] aridity, growing-season rain (fold climatology)
- [ ] soil water holding capacity (SoilGrids), cropland fraction (USDA CDL), irrigated fraction

**F8. Forecast (M2)**
- [x] per-week ensemble z, spread ratio, P(lower tercile) for precip, temperature, VPD, soil moisture
- [ ] forecast soil moisture translated to percentile vs reforecast climatology (closest to the target)
- [ ] accumulated forecast precipitation deficit over weeks 1-k

**F9. Other observed sources**
- [ ] USDM category and 4-week change at the cell, as of issue (strong, public, 2-day latency)
- [ ] ESI (NOAA/USDA), EDDI (NOAA PSL) as direct features where coverage allows

---

## V. Report and visualisation (start now)

The demo is 25% of the score and a striking visual story probably matters more than tables.
Plan the story first, then make sure the pipeline writes the data it needs.

- **Hero: 2023 replay.** Animated map, week by week: FlashWatch week-2 probability as a field,
  onsets appearing as they happen, USDM D1+ outlines arriving later. Message: "we saw it N weeks earlier".
- **Lead-time race.** Per episode, FlashWatch first alert vs USDM's first worsening (distribution of lead days).
- **Skill map.** BSS by region (tests pre-registered expectation 4, west vs east).
- **Reliability and skill-by-lead** with CIs, against B0, B1, raw ECMWF.
- **"Why this week" driver map**: SHAP top drivers by region, feeding the agent's advisories.
- **County advisory cards** from the Meta model with its own track-record scorecard.
- Format: a static scrollytelling page (MapLibre + pre-rendered tiles/frames), the 3-min video cut from it.
- Pipeline needs: per-cell, per-issue OOF probabilities for dev years (have), test-year predictions
  after the freeze, episode tables, USDM rasters (have), SHAP values per cell-issue.
- Dev-year version first (2012 from OOF predictions) so the page is built before the freeze.
