# Model card: FlashWatch flash-drought onset forecasts

Development results (2005-2021). The test years 2022-2024 are scored once, after the model is frozen
(TODO: test-year results).

## What it predicts

The chance that a flash drought **begins** at a place in the next 7 days (week 1) and in days 8-14 (week 2),
issued every other day from April to mid-October, on a 0.25 degree grid (about 25 km) over the Corn Belt and northern
Plains (36-49 N, 104-80 W).

A flash-drought onset is when the soil moisture in the top 40 cm (ERA5-Land) falls from the 40th percentile or
above to the 20th or below for the date, and stays at or below the 20th for 10 days. Only places not already
dry are forecast. Places under snow or frozen ground are left out.

## The models

| Name | What it is | Used for |
| --- | --- | --- |
| **M2** | LightGBM on observed land features plus ECMWF extended-range forecast features (11 ensemble members), calibrated | The headline results |
| **E1** | M1 (LightGBM on observed land features only) averaged with four small U-Nets that see the whole map | The farm page and maps, until the forecasts for late July and August finish downloading |
| B0, B1, H1, B2 | Seasonal normal; logistic regression on soil moisture; a farmer's rule of thumb ("hot and dry lately"); the raw ECMWF forecast, recalibrated | Comparisons |

- **Inputs** use only what was public on the forecast date, allowing for ERA5-Land's 5-day delay:
  - soil moisture percentiles and their recent change;
  - rain, temperature, vapour-pressure deficit and evaporation against normal over 14-60 days;
  - the share of nearby places already drying;
  - for M2, the forecast's rain, temperature, dryness and soil moisture for weeks 1-4.

  There are no location features (no latitude, longitude or cell id). An automatic test changes the data after the
  forecast date and checks that no input moves.
- **Training:** one model per lead time, with plain log loss, plus monotone constraints where the physics is
  one-way (for example, more rain can't raise the risk). LightGBM was kept after XGBoost and CatBoost scored no
  better.
- **Calibration:** four models are each trained without a quarter of the training years. A logistic map, fitted
  on their predictions for the years they didn't see, turns scores into probabilities. The prediction is the
  average of the four.

## How it was tested

- Cross-validation by year, 2005-2021: each year is predicted by models that never saw it. Normals and
  percentile thresholds come from the training years only.
- Ranges are 90% ranges from resampling whole years.
- M2 is scored on the 41 ECMWF forecast start dates downloaded so far (April 1 to July 13); the rest of the
  season is still downloading.
- Four expectations were written down before any model ran (results below).

## Results (development years)

| | Week 1 | Week 2 |
| --- | --- | --- |
| M2, Brier skill against the seasonal normal | 16.5% (13.4 to 18.5) | 3.7% (1.7 to 5.2) |
| M2, riskiest tenth of farm-days: onsets against normal | 6.2 times (25% against 4.0%) | 3.1 times |
| Rule of thumb (H1), riskiest tenth | 1.7 times | 0.7 times |
| Raw ECMWF, recalibrated (B2), Brier skill | 2.2% | 0.3% |
| Land only (E1), all dates, Brier skill | 6.1% | 1.7% |

- Of all onsets, 62% fall in M2's riskiest tenth at week 1.
- The soil crosses the drought line a median 22 days before the Drought Monitor first marks moderate
  drought (D1) at that place.
- **Pre-registered expectations:**
  1. The land model beats a simple soil-moisture baseline at weeks 1-2: holds.
  2. Forecasts help only at weeks 3-4: wrong; they help most at week 1.
  3. The raw ECMWF forecast is badly calibrated: holds; Brier skill -166% before recalibration.
  4. More skill in the western Plains than the humid east: holds; week 1 19% against 13%.

## Limits and known problems

- **The test years are not scored yet.** Development numbers on 41 spring and early-summer dates may not hold
  for late summer.
- **Many onsets don't cost anything.** About half the onsets warned about break before they hurt yields. A
  "damaging drought" label is the first change planned after the hackathon.
- **Calibration:** M2 is slightly too cautious at the extremes. Straightening it scored no better, so it was
  left as is.
- **Data:** the soil moisture is ERA5-Land reanalysis, a model of the soil checked against observations, not a
  probe in the field. At 25 km, one cell holds many farms with different soils.
- **Replay only.** A live service needs current ERA5-Land and ECMWF's real-time forecasts, which need a licence.
- **The dollar values** come from a textbook water balance and extension-service costs, calibrated to USDA
  county yields, not from field trials. Rain-fed values are rougher.

## Intended use

For research and as decision support for irrigators and agronomists, alongside their own field checks. It is
not a substitute for local observation, and not for insurance, lending or policy decisions.

## The agents

The Meta models (Muse Spark, Muse Glimmer, SAM 3.1, Voice Transcribe) explain the forecasts and carry out
tasks; they don't make the forecasts. Every figure an agent writes is checked against a tool result, and pivot
commands need the farmer's YES. See the Agents page and `docs/writeup.md`.

## Data and licences

ERA5-Land (Copernicus Climate Change Service); ECMWF S2S reforecasts (CC BY-NC 4.0; no ECMWF data is
redistributed here); US Drought Monitor; USDA NASS. Full list and attributions in the README.
