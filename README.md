# FlashWatch

A flash drought can turn a green field brown in two or three weeks. FlashWatch forecasts the chance
that the soil at a farm drops into drought in the next one and two weeks, and works out what to do
about it: start extra irrigation today or wait, skip the fungicide pass, trim side-dress nitrogen, buy
hay before prices rise. AI agents built on Meta's models deliver the advice by text, find a farm's
pivots from aerial photos, answer the farmer's questions, and with the farmer's YES pass commands to
the pivots.

It covers the Corn Belt and northern Plains (36-49 N, 104-80 W): 143 million acres of corn and soybeans,
13 million of them irrigated.

## The site

`python -m http.server 8812 --directory site`, then open http://localhost:8812.

| Page | What's on it |
| --- | --- |
| **Your farm** | Pick the farm on a map (or search an address). **Irrigated**: "Extra irrigation today?" with the dollars of starting now against waiting for the next forecast, what would change the answer, the field's water account (root zone % full, days to stress, irrigation logged or read from the pivot), the season replayed. **Rain-fed**: a card per decision (fungicide, side-dress nitrogen, silage, hay) with a gauge of today's chance against the chance at which acting pays, adjustable assumptions and the season replayed. **Today** follows the latest forecast; **Replay** steps through a past season. Starting values are local (USDA prices, county yields) and labelled. |
| **The big picture** | What FlashWatch is worth to a typical irrigated and rain-fed farm, how good the forecasts are, value by corn price, a regional risk map, and the lead over the US Drought Monitor. |
| **Agents** | Real runs of each agent: SAM 3.1 finding pivots (full, part and half circles), the Muse Spark operator planning a morning (with the farm-rule checks and the farmer's approval), the watcher's texts through a season with how often they were right, a voice note, a weekly briefing, and the Glimmer narrative writer. |
| **Help, About** | What every number means; contact. |

## Results so far

Development years 2005-2021, each scored by a model that never saw that year. The test years 2022-2024
are scored once, after the model is frozen.

- **Forecasts.** On the tenth of farm-days FlashWatch rates riskiest, a flash drought begins in the
  next 7 days 5.9 times as often as normal (27% against 4.6%); those flags hold 59% of all flash
  droughts. A farmer's rule of thumb ("hot and dry lately", fitted the same way) manages 1.7 times.
  Days 8-14: 2.7 times. Scored on the 31 forecast dates downloaded so far (April to early July).
- **Ahead of the Drought Monitor.** The soil crosses the drought line a median 22 days before the
  Monitor first marks moderate drought (D1) at that place, before the forecast adds its one to two weeks.
- **Value.** A flash drought with a week's warning is worth about $21 an acre on irrigated corn (yield
  saved less extra pumping) and about $28 an acre on rain-fed corn (the fungicide pass a dry spell makes
  pointless). Averaged over all years, including the many without a flash drought at a given farm:
  about $19M a year across the region's irrigated acres and about $171M across its rain-fed acres
  (rain-fed values are rougher). Dry spells are set to 21 days, matching USDA county yield losses.
- **Null results we kept:** Drought Monitor features, directional neighbourhood features, training on
  each forecast run separately (no gain); year-level climate indices (worse, removed).

## How it works

```
ERA5-Land soil, rain, heat (as known on the forecast date)      ECMWF extended-range forecasts (11 runs)
                        \                                         /
          LightGBM model (calibrated): chance of a flash drought in days 1-7 and 8-14
                                           |
   decision engine: FAO-56 water balance of the field (pivot, soil, crop, the root zone's current water)
   and value tables for the rain-fed decisions; act today vs decide at the next forecast
                                           |
   site (static)  +  agents on Meta models, each working through tools and checked:
     SAM 3.1          finds the farm's pivots in USDA aerial photos
     Muse Spark       morning text, pivot operator, weekly briefing, Ask FlashWatch chat
     Voice Transcribe field voice notes
     Muse Glimmer     Drought Monitor-style narrative (open weights, run locally; LoRA next)
                                           |
   farm rules (well, allocation, season, advice) -> farmer's YES (on the page or WhatsApp) -> pivot connector
   (signed webhook, Lindsay FieldNET data through Leaf, or a to-do in the maker's app)
```

Every number an agent writes is traced back to a tool result before it's sent; untraced figures go
back for a redraft and are cut if they can't be fixed. Farm rules are code, not prompts.

## Run it

Windows PowerShell from the repo folder; Python venv in `.venv`; keys in `flashwatch.env` (never committed).

```powershell
py -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
python -m flashwatch.ingest all                  # downloads (resumable; ECMWF S2S takes days)
python -m flashwatch.obs                         # data/obs.zarr from ERA5-Land
python -m flashwatch.labels --variants           # flash-drought onsets
python -m flashwatch.forecast build              # data/fcst.zarr from the S2S files
python -m flashwatch.eval.cv --models B0 H1 M1 M2 --leads week1 week2 any28 --fast --fc-dates
python -m flashwatch.eval.cv --models B0 M1 M3 M3b --leads week1 week2 --fast --tag m3seeds   # M3 map model (GPU; requirements-dl.txt)
python -m flashwatch.eval.blend <m3seeds oof> <m3seeds2 oof> --blend M1 M3+M3b+M3c+M3d --write E1   # M1 + mean of four M3s
python -m flashwatch.eval.final freeze --models ... --blend E1=M1,M3+M3b+M3c+M3d   # after the burst (clean tree)
python -m flashwatch.eval.final score            # once: train 2005-2021, score 2022-2024
python -m flashwatch.ingest nass                 # county crop acres (NASS_API_KEY)
python -m flashwatch.ingest.nass local           # state prices and farm counts
python -m flashwatch.site.build --oof data\oof\<file>.parquet --model M2   # or the E1 file, --model E1 (current site)
python -m flashwatch.site.defaults               # local starting values for the farm page
python -m flashwatch.site.evidence --m1 <oof> --m2 <oof> --h1 <oof>
python -m flashwatch.decide.rollup --oof <oof> --model M2 --tag <name> --set spell_days=21
python -m flashwatch.site.agents                 # agent runs for the Agents page
python -m pytest
```

Agents (need `META_API_KEY`):

```powershell
python -m flashwatch.agent.pivots --lat 42.45 --lon -98.60          # SAM 3.1 pivot finder
python -m flashwatch.agent.operator --year 2012 --dates 07-01 07-03 07-05 --llm-dates 07-03 --approve "demo"
python -m flashwatch.agent.chat --lat 40.87 --lon -97.99 --year 2012 --date 07-03   # Ask FlashWatch
python -m flashwatch.agent.serve                 # Ask FlashWatch on the farm page (127.0.0.1:8813)
python -m flashwatch.agent.voice note.wav --year 2012 --date 07-03   # voice field report
python -m flashwatch.agent.farm --lat 45.38 --lon -99.38 --year 2012 --dates 07-03  # morning text
python -m flashwatch.agent.watch --year 2012 --llm 3                # the watcher's texts for a season
python -m flashwatch.agent.scorecard             # how often its texts were right, six seasons (no API)
python -m flashwatch.agent.whatsapp operator --to <phone> --year 2012 --date 07-03   # plan with YES/NO buttons
python -m flashwatch.agent.whatsapp inbox --from <phone> --text YES                # the farmer's reply
modal run flashwatch/agent/glimmer_train.py      # Glimmer LoRA (Modal account)
# Glimmer locally: llama.cpp (b10353+) llama-server with meta-models/Muse-Glimmer-30B-GGUF Q4_K_M, --fit on --jinja
# -a muse-glimmer-30B --port 8080, then:
META_API_BASE=http://127.0.0.1:8080/v1 META_MODEL=muse-glimmer-30B python -m flashwatch.eval.glimmer_eval --n 12 --shots 2 --timeout 1500
```

## Where things are

| Folder | Contents |
| --- | --- |
| `flashwatch/ingest` | Downloads: ERA5-Land, ECMWF S2S reforecasts, Drought Monitor maps and narratives, USDA NASS |
| `flashwatch/labels`, `features`, `models`, `eval` | Flash-drought labels, as-of features (leakage-tested), models, cross-validation |
| `flashwatch/decide` | Water balance, field water account, decision engine, backtest, roll-up |
| `flashwatch/agent` | The agents, the number checker, the LLM client |
| `flashwatch/connect` | Pivot connectors (simulated, webhook, Leaf, manual) and the farm-rule guard |
| `site` | The static site |
| `docs` | Write-up, agents, video plan |

Working records: `DESIGN.md` (decisions, tests, results), `QUESTIONS.md`, `HANDOVER.md`.

## Data and licences

ERA5-Land (Copernicus Climate Change Service), ECMWF S2S reforecasts, US Drought Monitor (NDMC, USDA,
NOAA), USDA NASS (yields, Census of Agriculture, prices), USDA NAIP aerial imagery (USGS National Map),
US Census Gazetteer, OpenStreetMap. ECMWF-derived data files are not redistributed here. See
HANDOVER.md section 9.
