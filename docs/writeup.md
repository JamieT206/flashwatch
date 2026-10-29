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

- **Your farm, irrigated.**
  - Pick the farm on a map or search an address.
  - "Extra irrigation today?" shows the chance of a flash drought in the next 7 days and in days 8-14
    against normal for the date, and a verdict: *start extra irrigation today*, *not yet: keep checking
    forecasts*, or *irrigate as usual*. It gives the dollars per acre of starting now against deciding
    at the next forecast, and what that forecast would have to show to change the answer.
  - A water account for the field: rain and irrigation in, crop water use out (FAO-56, with crop
    coefficients by growth stage). The farmer logs what they've put on, or it's read from the pivot's
    records. The advice then starts from the actual deficit and says how many days of pumping fill
    the root zone.
- **Your farm, rain-fed.** 130 million of the region's 143 million corn and soybean acres aren't
  irrigated. A card per decision:
  - skip the mid-July fungicide pass (a dry spell means little disease);
  - trim side-dress nitrogen;
  - get ready to chop drought-hit corn for silage;
  - buy hay before prices rise.
  Each shows today's chance against the chance at which acting pays, the dollars either way, adjustable
  assumptions, and the season replayed.
- **Today and Replay.** Today follows the latest forecast and moves on by itself each day; Replay steps
  through a past season. Starting values are local (USDA state prices, county yields) and labelled.
- **Agents on Meta's models**, each working through tools and checked before anything reaches the
  farmer or a pivot:
  - *Pivot finder (SAM 3.1).* From the farm's location, a USDA aerial photo; SAM 3.1 finds every
    centre-pivot circle; each becomes a pivot with its acres. A second question to SAM ("crop field shaped like a
    circle or part of a circle") picks up the circles with a slice left out and the half circles. In a
    Holt County, Nebraska test: the 11 full circles in a 4.8 km square, plus 4 part and half circles.
  - *Operator (Muse Spark).* For a farm with several pivots it reads each pivot's advice, water account,
    well and allocation and any voice note, and proposes starts and stops. Fixed farm rules check every
    command (well capacity, allocation, season, the advice); refusals go back to the agent with the
    reason. It asks on WhatsApp, with YES and NO buttons, and acts only on YES, through a signed webhook,
    Lindsay FieldNET data via Leaf, or a to-do in the maker's app. On the same WhatsApp thread the farmer
    can ask a question or send a voice note, and the watcher's texts arrive there too.
  - *Ask FlashWatch (Muse Spark).* The farmer asks in plain words ("I put an inch on yesterday, do I
    still need to start?"); the agent answers through the advice, decision and water-account tools, and
    logs irrigation it's told about.
  - *Voice field reports (Muse Voice Transcribe + Spark).* "Corn on the sandy pivot is curling by
    noon" becomes a stress report on pivot N1.
  - *Watcher (Muse Spark).* Follows each subscribed farm through the season and texts only when
    something changes: the advice turns to start, a rain-fed decision's moment comes, the pivot can't
    keep up. It keeps score of itself: every forecast text is checked against what happened in the
    next two weeks. Over six past seasons on 148 farms, a flash drought followed its "start the pivot"
    texts 26% of the time, 4.8 times the normal 5.4%; rain-fed "act now" texts 3.7 times; "risk eased"
    texts still 1.9 times (eased means lower, not gone). A third of onsets had a warning text in the
    two weeks before.
  - *Weekly briefing (Muse Spark)* for a county or extension office.
  - *Drought Monitor-style narrative (Muse Glimmer 30B, open weights)*: writes the Monitor's weekly
    High Plains and Midwest summary from that week's numbers. It runs locally (4-bit, one 12 GB graphics
    card, llama.cpp). On 7 held-out weeks from 2020-21, on its own it writes like Muse Spark does: correct
    states, no made-up figures, but a list of numbers (9% of wording shared with the real text). Shown two
    real past weeks first, it picks up the Monitor's style (15%) but borrows details it wasn't given, which
    the number checker flags (7% of figures). Next: a LoRA fine-tune on 749 weeks of the real narratives
    (TODO: results).
    The same model also runs FlashWatch's agents offline, with the same tools, rules and checker. For
    the demo farm's 3 July 2012 morning it made the same plan as Muse Spark, every figure traced on the
    first draft, in about 8 minutes. Asked three farmer questions, it answered all correctly but took 1-4
    minutes each (Spark: 13-32 seconds) and was less thorough (it skipped the water account once and
    week 2 once). Spark stays the default; Glimmer is for no connection, or data that shouldn't leave the farm.
  Every figure an agent writes is traced to a tool result; untraced ones go back for a redraft and
  are cut if they can't be fixed.
- **The big picture.** Value to a typical irrigated and rain-fed farm, how good the forecasts are,
  value by corn price, a regional risk map, and the lead over the Drought Monitor.

## How it works

```
ERA5-Land soil, rain, heat (as of each forecast date)     ECMWF extended-range ensemble (11 members)
                       \                                   /
   LightGBM onset model per grid cell  +  U-Net on the whole map (PyTorch), averaged:
                   chance of a flash drought in days 1-7 and 8-14
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
- **Meta models:** SAM 3.1 (pivot finder); Muse Spark (operator, Ask FlashWatch, morning text,
  briefing, voice-note structuring, map cross-check); Muse Voice Transcribe (field reports); Muse
  Glimmer 30B, open weights, run locally (narrative writer, and an offline fallback for the operator
  and Ask FlashWatch); WhatsApp Cloud API (texts, YES/NO approval, questions).

## Results (development years 2005-2021; test years TODO)

- **A week ahead is where the forecasts pay.** With ECMWF weather forecasts:
  - on the tenth of farm-days FlashWatch rates riskiest, a flash drought begins in the next 7 days
    **5.9 times** as often as normal (27% against 4.6%), and those flags hold 59% of all onsets;
  - a farmer's rule of thumb ("hot and dry lately", fitted the same way) manages 1.7 times;
  - week-1 skill is about 17% better than the seasonal normal (Brier skill score);
  - beyond the first week the edge fades (week 2: about 3%, 2.7 times normal on the top tenth);
  - scored on the 31 ECMWF forecast dates downloaded so far, April to early July.
- **Seeing the map helps.** Without the weather forecasts, a small convolutional network that sees
  the whole map at once (where a drying patch is, and what surrounds it) adds to the per-cell model:
  week-1 skill 6.1% against 4.3% when the two are averaged (paired gain 1.8 points, 90% range 1.3-2.4).
  The farm page runs on this pair until the summer ECMWF forecasts finish downloading.
- **Ahead of the Drought Monitor.** The soil crosses the drought line a median 22 days before the
  Monitor first shows moderate drought (D1) at that place. FlashWatch is first in 86% of cases.
- **Flash droughts cost crops** (county yields, above). Onsets the Monitor never flags cost no yield
  on average: those spells broke in time.
- **What a warning is worth.**
  - A flash drought with a week's warning is worth about $21 an acre on irrigated corn (yield saved,
    less extra pumping; a typical Nebraska pivot) and about $28 an acre on rain-fed corn (the fungicide
    pass a dry spell makes pointless).
  - Averaged over all years, including the many without a flash drought at a given farm, that's about
    $19 million a year across the region's irrigated acres and $171 million across its rain-fed acres.
    Rain-fed values rest on extension averages and are rougher.
  - Dry spells are set to 21 days, which matches USDA county yield losses (irrigated counties lost 1.5
    to 3.6% of yield per flash drought). Higher crop prices raise the value: $1.29 an acre a year on
    irrigated corn at $3.50, $4.80 at $7.50.
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
  open start/stop API; Leaf can read FieldNET's irrigation data today.
- The water account uses daily means (ERA5-Land has no daily max and min here) and the FAO default wind;
  a farm's own rain gauge and weather station would sharpen it.
- The pivot finder estimates a part circle's acres from its box (masks not decoded yet).
- WhatsApp is built and tested without an account (messages go to a local outbox); it has not yet
  sent to a real phone.

## Built with

Python (LightGBM, PyTorch, xarray, pandas), d3, Leaflet, Meta Model API (SAM 3.1, Muse Spark, Muse Voice
Transcribe), WhatsApp Cloud API, Muse Glimmer 30B (llama.cpp). Data: ERA5-Land (Copernicus Climate Change Service), ECMWF S2S
reforecasts, US Drought Monitor (NDMC, USDA, NOAA), USDA NASS, USDA NAIP imagery (USGS), US Census
Gazetteer, OpenStreetMap. Everything in this repository was
built during the hackathon.
