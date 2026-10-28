# FlashWatch

**Should you start irrigating today?** Flash droughts turn a green field brown in two or three weeks.
FlashWatch forecasts, every other day, the chance that the soil at your farm crashes into drought in
the next one and two weeks. It then works out whether starting your centre pivot early pays,
against waiting two days for a fresher forecast.

- **Your farm** (`site/index.html`): enter a latitude and longitude, and get the following:
  - today's verdict (*start the pivot today / not yet / no need*), with the expected dollars per
    acre of each choice;
  - what the next forecast would have to show to change it;
  - the season so far against what really happened;
  - what a day of warning is worth on *your* field (pivot capacity, soil, crop, prices: all editable);
  - a morning text message and note from an adviser agent built on Meta's Muse Spark, with every
    number checked against the forecast tools;
  - **your pivots**: for a farm with several pivots, the Muse operator agent plans which ones to start
    today. Fixed farm rules (well capacity, water allocation, the advice) check every command, and
    nothing moves until the farmer replies YES. The commands then go to the pivots (simulated in the
    demo; a signed webhook, or a to-do in FieldNET or Valley 365). See DESIGN.md O10.
- **The big picture** (`site/region.html`):
  - what FlashWatch would be worth if every irrigated corn and soybean farm used it;
  - a regional risk map;
  - forecast skill, and when a warning is worth acting on;
  - how far ahead of the US Drought Monitor it is.

## What we found (development years 2005-2021; test years 2022-2024 are scored once, after the model freeze)

- **A week's warning is worth acting on.**
  - With ECMWF weather forecasts, the riskiest tenth of the region catches 42% of the flash droughts
    that begin in the next 7 days.
  - That captures 40-60% of a perfect forecast's value for cheap actions like starting a pivot early.
  - Beyond two weeks, no weather forecast can see the dry spells.
- **Notice is worth about $10/acre a day for a typical Nebraska pivot,** up to about a week. After that
  the soil is already full when the pivot starts. This comes from a textbook FAO-56 water balance; see
  `flashwatch/decide/irrigation.py`.
- **Ahead of the Drought Monitor:** the soil crosses the drought line a median 22 days before the
  Monitor first shows moderate drought (D1) at that place.
- **Roll-up:** about **$100 million a year** for US irrigated corn and soybeans, or about 6.6 million
  people's yearly calories in grain. This is an estimate until the full-season forecasts finish
  downloading. It is honest arithmetic, not a billion-dollar claim; see DESIGN.md O9.

The full working record is in `DESIGN.md` (decisions, tests, results), `QUESTIONS.md` and
`HANDOVER.md`.

## How it works

```
ERA5-Land soil, rain, heat (as of the forecast date)    ECMWF S2S ensemble forecasts (11 members)
                     \                                    /
            LightGBM onset model: chance of a flash drought in days 1-7, 8-14 (calibrated)
                                        |
          decision engine: FAO-56 water balance of your field x days of warning x
          act today vs decide in 2 days (ensemble members as the next forecast's outcomes)
                                        |
      farm page (static site)   +   Muse Spark adviser (tool calls, number-checked, SMS + note)
                                        |
      Muse operator: plans pivot starts -> farm-rule guard -> farmer's YES -> pivot connector
```

## Run it

Windows PowerShell, from the repo folder (Python venv in `.venv`; keys in `flashwatch.env`, never
committed):

```powershell
py -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
python -m flashwatch.ingest all                  # downloads (resumable; ECMWF S2S takes days)
python -m flashwatch.obs                         # data/obs.zarr from ERA5-Land
python -m flashwatch.labels --variants           # flash-drought onsets
python -m flashwatch.forecast build              # data/fcst.zarr from the S2S files
python -m flashwatch.eval.cv --models B0 M1 M2 --leads week1 week2 any28 --fast --fc-dates --members
python -m flashwatch.ingest nass                 # county crop acres (needs NASS_API_KEY)
python -m flashwatch.site.build --oof data\oof\<file>.parquet --model M2
python -m flashwatch.site.evidence --m1 <oof> --m2 <oof>
python -m flashwatch.decide.rollup --oof <oof> --model M2 --tag <name>
python -m flashwatch.agent.farm --lat 45.38 --lon -99.38 --year 2012 --dates 07-03   # needs META_API_KEY
python -m flashwatch.agent.operator --year 2012 --dates 07-01 07-03 07-05 --llm-dates 07-03 --approve "demo"
python -m http.server 8812 --directory site      # then open http://localhost:8812
python -m pytest
```

Try the decision engine on its own:

```powershell
python -m flashwatch.decide --week1 0.28 --week2 0.15 --set gpm_per_acre=4
```

Data: ERA5-Land (Copernicus Climate Change Service), ECMWF S2S reforecasts, US Drought Monitor
(NDMC, USDA, NOAA), USDA NASS Census of Agriculture, US Census Gazetteer. See HANDOVER.md section 9
for licences and attribution.
