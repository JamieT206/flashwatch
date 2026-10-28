# FlashWatch: a week's warning of flash drought, and what to do about it

*Draft for the Meta Global AI Developer Hackathon submission (Push the Frontier). Numbers marked
TODO are filled in after the model freeze, when the test years 2022-2024 are scored once.*

## The problem

A flash drought dries a field out in two or three weeks, usually in a summer hot spell. By the time
the US Drought Monitor shows it, the damage is often done. In USDA county yields from 2005 to 2021,
rain-fed corn yielded about **20% below trend** where most of the county had a summer flash drought.
Irrigated corn in the same counties lost only about **2%**: irrigation saves most of the crop, if the
pivot is running in time. A centre pivot can't catch up once the root zone is empty in a heat wave.
So the question an irrigator faces every summer morning is simple: *start the pivot today, or wait?*

## What FlashWatch does

- **Your farm.** Enter a latitude and longitude. FlashWatch shows:
  - the chance that the soil at that place drops into drought in the next 7 days and in days 8-14,
    against normal for the date;
  - a verdict, *start the pivot today / not yet / no need*, with the expected dollars per acre of
    starting today against deciding at the next forecast two days later;
  - what the next forecast would have to show to change the answer;
  - what a day of warning is worth on that field, from a water-balance model with the farm's own
    pivot capacity, soil, crop, yield and prices (all editable);
  - hay/feed and nitrogen decisions as secondary actions.
- **An adviser that writes.** Each morning, Muse Spark writes a text message and a short note from
  FlashWatch's numbers, through tool calls. A checker traces every figure back to a tool result and
  sends the draft back to be fixed if it can't.
- **An operator that acts, with the farmer's YES.** For a farm with several pivots, a Muse Spark
  agent plans the morning:
  - it reads each pivot's advice, which pivots share a well, how much water allocation is left, and
    any voice note from the farmer;
  - it proposes starts and stops, and fixed farm rules check every command (well capacity,
    allocation, season, the advice);
  - it revises when a command is refused, and asks for approval by text.
  
  Approved commands go to the pivots: simulated in the demo, through a signed webhook, or as a to-do
  in the pivot maker's app.
- **Voice field reports.** "Corn on the sandy pivot is curling by noon." Muse Voice Transcribe turns
  the note into text, and Spark maps "the sandy pivot" to N1 with a severity. The operator treats a
  farmer-reported stressed crop as reason enough to start.
- **The big picture.** A regional risk map, forecast skill, how far ahead of the Drought Monitor
  FlashWatch is, the county-yield evidence, and what it would be worth if every irrigated corn and
  soybean farm used it.

## How it works

```
ERA5-Land soil, rain, heat (as of each forecast date)     ECMWF extended-range ensemble (11 members)
                       \                                   /
        LightGBM onset model (calibrated): chance of a flash drought in days 1-7 and 8-14
                                         |
        decision engine: FAO-56 water balance of the field x days of warning x
        start today vs decide in 2 days (ensemble members as the next forecast's outcomes)
                                         |
   farm page (static)  +  Muse Spark adviser and operator (tool calls, number checks, farm-rule guard)
                       +  Muse Voice Transcribe field reports  ->  pivots (with the farmer's YES)
```

- **Labels.** A flash-drought onset is when the 0-40 cm soil-moisture percentile falls from 40 or
  above to 20 or below and stays below for 10 days (0.25 degree grid, 36-49 N, 104-80 W, 2005-2024).
- **Honest testing.**
  - Every prediction shown comes from a model that never saw that year.
  - Features read only what was public on the forecast date, and an automatic leakage test perturbs
    the future to prove it.
  - The test years 2022-2024 are scored once, after the model is frozen.
- **Meta models:**
  - Muse Spark (adviser, operator, map cross-check, voice-note structuring);
  - Muse Voice Transcribe (field reports);
  - Muse Glimmer, open weights (TODO: LoRA-tuned on 15 years of real Drought Monitor narratives to
    write the regional summary from FlashWatch's numbers, scored against held-out weeks).

## Results (development years 2005-2021; test years TODO)

- **A week ahead is where the forecasts pay.** With ECMWF weather forecasts:
  - on the tenth of farm-days FlashWatch rates riskiest, a flash drought begins in the next 7 days
    **5.4 times** as often as normal (29% against 5.3%), and those flags hold 54% of all onsets;
  - a farmer's rule of thumb ("hot and dry lately", fitted the same way) manages 1.7 times;
  - week-1 skill is about 17% better than the seasonal normal (Brier skill score);
  - beyond two weeks, the forecasts can't see the dry spells (week 2: about 4%).
- **Ahead of the Drought Monitor.** The soil crosses the drought line a median 22 days before the
  Monitor first shows moderate drought (D1) at that place. FlashWatch is first in 86% of cases.
- **Flash droughts cost crops** (county yields, above). Onsets the Monitor never flags cost no yield
  on average: those spells broke in time.
- **What a warning is worth.** For a typical Nebraska pivot (5 gpm/acre), each day of warning is
  worth about $10 an acre up to about a week, then nothing more. After that, the soil is already full
  when the pivot starts.
- **If every irrigated corn and soybean farm used it:** about **$23 million a year**, with yield
  losses calibrated to the county data (ceiling with perfect weather forecasts: $33M). It isn't
  billions: irrigation already saves most of the crop, and FlashWatch wins back the last 2-3%. The
  larger prize is rain-fed farms, whose decisions (nitrogen, marketing, insurance) are next.
- **Pre-registered expectations**, written before any model run:
  1. M1 beats B1 at weeks 1-2: TODO.
  2. "M2 beats M1 at weeks 3-4 but not at week 1": **wrong, in the opposite direction.** The gain
     is largest at week 1.
  3. Raw ECMWF badly calibrated: not tested (B2 not built).
  4. More skill in the western Plains than the humid east: TODO.
- **Null results we kept:**
  - Drought Monitor features: no gain.
  - Directional neighbourhood features: no gain.
  - Year-level climate indices: worse, so removed.
  - Training on each ensemble member separately: slightly worse.

## Limitations

- Replay only: a live version needs current ERA5-Land (5-day delay) and live ECMWF ensembles.
- The irrigation economics are textbook physics calibrated to county yields, not field trials.
- About half of the onsets FlashWatch warns about break before they cost anything. A stricter
  "damaging drought" label is the first experiment after the hackathon.
- Real pivot control needs the makers' partner programs. Neither Lindsay nor Valley publishes an
  open start/stop API; Leaf can read their data today.

## Built with

Python (LightGBM, xarray, pandas), d3, Meta Model API (Muse Spark, Muse Voice Transcribe), Muse
Glimmer (TODO). Data: ERA5-Land (Copernicus Climate Change Service), ECMWF S2S reforecasts, US
Drought Monitor (NDMC, USDA, NOAA), USDA NASS, US Census Gazetteer. Everything in this repository was
built during the hackathon.
