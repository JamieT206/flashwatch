# FlashWatch: handover

Written Tue Oct 27, 2026, 04:20 Sydney time (Mon Oct 26, 17:20 UTC), about an hour after the hackathon kickoff.
It is for whoever picks this up next, human or AI coding assistant. Read sections 1–4 before touching anything.

---

## 0. If you are an AI assistant reading this

- You are helping **James Twiss** (Sydney, Australia, UTC+11), an expert in predictive analytics and ML. He is technical, so skip basic explanations, but he wants clear status and decisions surfaced.
- The project is a 10-day hackathon entry. **Hard deadline: Nov 4, 9:00 PM US Eastern = Nov 5, 02:00 UTC = Nov 5, 1:00 PM Sydney.** Plan backwards from it with at least 3 hours of buffer.
- The design is settled (section 6 and `FlashWatch Design Spec.pdf`). Don't redesign it. Execute it, and raise any deviation with James first.
- **One decision is open** and needs James's answer before ingesting forecasts (section 5, D1). Ask him first.
- Credentials live in a private `.env` file (section 3). Never print secrets into logs, chat, commits or docs.
- Pre-registered expectations exist (section 6.6). Report results against them honestly, including nulls.

---

## 1. The project in one paragraph

**FlashWatch** forecasts the *onset* of flash drought 1–4 weeks ahead on a 0.25° grid over the US Corn Belt and Plains, then turns the forecasts into county advisories using a Meta-model agent that checks its own track record. The forecasting core is classical ML (LightGBM on reanalysis and satellite land state plus ECMWF sub-seasonal reforecasts). The agent explains and verifies; it never forecasts. Headline claim to test: calibrated onset probabilities that beat climatology, persistence and raw ECMWF output on held-out years, at least at 1–2 week leads. The demo replays the 2023 Midwest flash drought week by week against the US Drought Monitor.

- Live spec (Claude Doc, editable): https://claude.ai/code/artifact/8f6f3a21-ca1c-446c-a32f-253272029e7c
- PDF snapshot of the spec, Oct 17: `FlashWatch Design Spec.pdf` in this bundle (20 pages, includes the agent-loop and schedule diagrams)

---

## 2. Hackathon facts

| Item | Detail |
| --- | --- |
| Event | Meta Global AI Developer Hackathon (run with Cerebral Valley), virtual |
| Dates | Kickoff Oct 26, 12:00 PM EDT (Oct 27, 3:00 AM Sydney). Workshops and Discord office hours Oct 27 – Nov 4. Submissions due Nov 4, 9:00 PM EST. Judging Nov 4–8. Finals Nov 9 |
| Status | **James has been accepted.** He has no government affiliation (eligibility checked) |
| Track | Indie developer: "judged on creativity, technical ambition, and whether it works" |
| Themes | "Build for Your World" and "Push the Frontier". FlashWatch is pitched under Push the Frontier (understanding and anticipating a physical system) |
| Prizes | $1M pool across two tracks; no per-track split published |
| Credits | $150 Meta Model API (selected participants). Partner credits for select participants: Modal $100, Baseten $100 + 2M TPM, Fireworks $25. **Not yet claimed**: look in the acceptance email or Discord |
| Official rules | Not published as of Oct 22; promised before kickoff. **Read them now**: pre-work rules, required Meta-model use, submission format, video length, open-source/licence terms |
| Meta models | Via Meta Model API: Muse Spark series (1.1 is $1.25/$4.25 per M tokens, 1M context, OpenAI-compatible, image/video/PDF input, tool calls, structured outputs), **Muse Wonder** (launched Oct 12; $4/$20 per M; Muse Wonder Full is Muse Code subscribers only), Muse Image ($0.01/image), SAM 3.1. Open weights: **Muse Glimmer 30B** (Apache 2.0) on Hugging Face as `meta-models/Muse-Glimmer-30B`, `meta-models/Muse-Glimmer-30B-GGUF` and `unsloth/Muse-Glimmer-30B-GGUF`. Also Muse Code, Meta's agentic coding tool |
| API | Base URL `https://api.meta.ai/v1`, model IDs like `muse-spark-1.1`. It also exposes an Anthropic-style `/v1/messages`. The preview launched US-only; whether James's key works from Australia is **untested** |

The application text James submitted (project idea and the conflict-model project) was drafted in the earlier chat. Its substance is the spec and section 8.

---

## 3. Accounts and credentials

All accounts below are created and working (tested Oct 17 from James's old Windows PC).
Credentials are in **`flashwatch.env`**, a separate private file James has. Rename it to `.env` in the project root. It is gitignored. `check_access.py` loads it automatically.

| Service | Variables in `.env` | Status | Notes |
| --- | --- | --- | --- |
| NASA Earthdata (SMAP, ECOSTRESS, AppEEARS) | `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | Working | Username is `jtwiss`. `earthaccess.login(strategy="environment")` works |
| Google Earth Engine | `EE_PROJECT=everything-machine-450219` | Working | Registered for noncommercial use until Apr 2028, Community tier. **On a new machine, run `earthengine authenticate` once** (opens a browser); the credentials don't travel with `.env` |
| ECMWF account | `ECMWF_API_URL/KEY/EMAIL` | Account fine, **API route retired** | The old Web API (`api.ecmwf.int`, `ecmwfapi`) lost S2S access on 27 May 2026. Keep the key; use the ECMWF Data Store instead (next row) |
| ECMWF Data Store (ECDS): S2S reforecasts | uses `CDSAPI_KEY`; `ECDS_API_URL` defaults to `https://ecds.ecmwf.int/api` | Working | S2S licence accepted on ECDS. Same login as ECMWF/CDS; the CDS key works |
| Copernicus CDS (ERA5-Land backup) | `CDSAPI_URL`, `CDSAPI_KEY` | Working | ERA5-Land licence accepted |
| USDA NASS Quick Stats | `NASS_API_KEY` | Working | |
| Meta Model API | `META_API_KEY` (**not yet in `.env`**) | Untested | James has the key from his acceptance; add it, then run `check_access.py --only meta` |
| Hugging Face | none needed for listing | Reachable | Downloading Glimmer may need login or terms acceptance |
| GitHub | n/a | Not set up | Create a private repo now; make it public at submission |

Security note: the Earthdata password was shared in chat, so James may want to rotate it after the hackathon.

---

## 4. Where things stand

### Done
- Design spec complete (section 6, PDF, live doc).
- All accounts created; every data source reached with working credentials (results below).
- Diagnostic `check_access.py` written and run. It's a throwaway access tester, not project code.

### Access test results (Oct 17, James's old PC)

| Source | Result | Implication |
| --- | --- | --- |
| Earth Engine | ERA5-Land daily (`ECMWF/ERA5_LAND/DAILY_AGGR`): all 150 bands present, none of the needed ones missing; region mean of soil water layer 2 on 2023-06-15 = 0.227. SMAP L4 v008 (`NASA/SMAP/SPL4SMGP/008`): 8 images that day, region root-zone mean 0.253 | Primary route for ERA5-Land and SMAP |
| ECDS S2S reforecast | 1 member, 1 variable (`tp`), 46 days, region subset: 23 KB in 191 s, mostly queue | Works. Bundle members, years and variables per request to amortise the queue |
| CDS ERA5-Land | 1 variable, 1 day: 133 KB in 86 s, mostly queue | Too slow for 22 years of requests; backup only |
| SMAP via NASA CMR/earthaccess | 144–153 MB per granule, 8 granules a day; one granule downloaded in 33 s | Full download about 4.6 TB, so never bulk-download; use Earth Engine (or AppEEARS/Harmony subsetting) |
| Earthdata login, AppEEARS token | Pass | ECOSTRESS and SMAP backup routes |
| Drought Monitor weekly shapefiles | 1.7 MB per week | About 2 GB for 2004–2025 |
| Drought Monitor statistics API | Returned `[]` for `aoi=IL` | Not needed |
| SMVI inventory (HydroShare) | 21 files, 5.8 GB, `SMVI_Composites_YYYY_YYYY.tar.gz` (5-year archives) | Fetch only the 2004–2021 archives |
| NASS Quick Stats | 115 rows for Illinois corn condition, 2023 | Works |
| NOAA GEFSv12 reforecast (`noaa-gefs-retrospective` S3, anonymous) | Bucket listable | Fallback forecast source |
| Hugging Face Muse Glimmer | Repos found (above) | Local agent development possible |

### Not done yet (do these first, in order)
1. **Read the official rules** (cerebralvalley.ai/e/meta-ai-developer-hackathon or dev.meta.ai/events/global-hackathon).
2. **Ask James decision D1** (section 5).
3. Set up the new machine (section 7.1), then run `python check_access.py` and `python check_access.py --throughput`. The throughput tests are new and have **never been run**: they time an Earth Engine one-year pull and a bundled 10-member × 20-year ECMWF request, and check that GRIB decoding (cfgrib/eccodes) works on this machine.
4. Add `META_API_KEY` and run `python check_access.py --only meta`. This confirms the key works from Australia and lists the available models.
5. Claim the Modal, Baseten and Fireworks credits; join the Discord.
6. Check the GPU (`nvidia-smi`). Glimmer 30B at about 4-bit needs roughly 20 GB VRAM. Without that, develop the agent against the Meta API directly.
7. Optional speed-up: Claude's sandboxes block these data hosts by default. If James adds them under claude.ai Settings → Capabilities, an AI assistant can run downloads itself instead of asking James to paste commands. Hosts: `earthengine.googleapis.com`, `oauth2.googleapis.com`, `ecds.ecmwf.int`, `cds.climate.copernicus.eu`, `urs.earthdata.nasa.gov`, `cmr.earthdata.nasa.gov`, `appeears.earthdatacloud.nasa.gov`, `droughtmonitor.unl.edu`, `www.hydroshare.org`, `quickstats.nass.usda.gov`, `api.meta.ai`, `huggingface.co`, `noaa-gefs-retrospective.s3.amazonaws.com`. Some users report the setting doesn't always apply, so test it. This isn't needed if you're a coding assistant running directly on James's machine with normal internet.

---

## 5. Decisions

### Open: needs James
**D1. Study period and forecast model cycle.** ECMWF reforecasts come in blocks tied to a model cycle, each covering about 20 years before its issue dates. Reforecasts issued with 2024 forecasts (Cy48r1, Mon/Thu starts) cover roughly 2004–2023. So the spec's planned 2024–2025 test years would need real-time forecasts (101 members, newer model) or a later cycle's reforecasts, which mixes forecast systems.
- **Recommended:** shift the study period to **2005–2024** and use one cycle throughout: the reforecasts issued with 2025 forecasts (Cy49r1, which runs reforecasts every other day from 12 Nov 2024). Development cross-validation would be 2005–2021, with **2022–2024 as the test years**. The 2012 and 2023 case studies are kept; 2025 is dropped.
- Alternative: keep 2004–2025 and use real-time forecasts subsampled to 11 members for 2024–2025.
- Whichever is chosen, **verify the exact hindcast years** with one small ECDS request for the chosen issue dates before bulk download.

### Locked (from the spec; change only with James)
| Topic | Decision |
| --- | --- |
| Region | 36–49°N, 104–80°W; 0.25° grid, 52 × 96 = 4,992 cells |
| Season | Issue dates Apr 1 – Oct 15, on the reforecast start days |
| Leads | Week 1, 2, 3, 4 separately, plus "any onset in days 1–28" (headline) |
| Eligibility | Only cells with root-zone percentile > 20 at issue (onset, not continuation) |
| Main model | LightGBM per lead, plain log loss, no downsampling or class weights, monotone constraints, isotonic calibration per lead |
| Evaluation | Leave-one-year-out CV, a test set opened once after the model freeze, Brier skill vs climatology, year-block bootstrap CIs |
| Land-state backbone | ERA5-Land via Earth Engine; SMAP as a JPL experiment on 2015+ |
| Forecast inputs | ECMWF S2S reforecasts via ECDS; NOAA GEFSv12 as fallback |
| Agent | Meta model explains and verifies; deterministic code does everything else; a number validator; anonymised replay |

---

## 6. Design summary (full detail in the PDF)

### 6.1 Target and labels
- P(d) = percentile of the 5-day running mean of root-zone soil moisture (0–100 cm; ERA5-Land layers 1–3, depth-weighted) on day d. It is computed against the same calendar window (±15 days) in other years.
- **Onset on day d** when all three hold: (1) P(d) ≤ 20 and P(d−1) > 20; (2) P ≥ 40 on at least one day in [d−20, d−5]; (3) P ≤ 20 for all of [d+1, d+10]. Labels may use future data; features never may.
- Masks: snow or frozen soil (ERA5-Land snow depth, soil temperature); cells under 20% vegetated land.
- **Climatologies are rebuilt per CV fold from training years only**, for both labels and features.
- Label for issue date τ and lead week w: 1 if an onset day falls in (τ + 7(w−1), τ + 7w]. The 28-day label is the OR of weeks 1–4.
- Episodes: space-time clusters (8-connected cells, onset days within ±7). Use them for event-level scoring and to state how many independent events exist.
- Day-2 cross-checks: known events (2012 central US, 2017 Northern Plains, 2022, 2023 Midwest); overlap with the SMVI inventory (2004–2021); Drought Monitor worsening of 2+ categories within 4 weeks; SMAP-based onset within ±7 days; threshold sensitivity (30/20, 40/10).
- The base rate per eligible cell-week is unknown; the guess is 1–3%.

### 6.2 Features (about 60), every one as of the issue date minus source latency
| Group | Features | Latency |
| --- | --- | --- |
| Soil moisture state | Root-zone and surface percentiles; change over 5, 10 and 20 days; days since last above the 40th percentile | ERA5-Land 5 days (SMAP 3) |
| Atmospheric demand | 14- and 30-day anomalies of temperature, vapour pressure deficit, potential evaporation, radiation | 5 days |
| Water balance | 14-, 30- and 60-day precipitation deficits (SPI-style); evaporation ÷ potential evaporation (ESI proxy) | 5 days |
| Forecast, weeks 1–4 | Ensemble mean and spread of precipitation, temperature, vapour pressure deficit and soil-moisture anomalies; P(precipitation in lower tercile) | 0 |
| Season and climate mode | Day of year (sine and cosine); Niño 3.4 (latest published monthly value) | |
| Static | Soil water capacity (gNATSGO/SoilGrids), cropland fraction (CDL, year before), aridity, 30-year mean growing-season precipitation | |

Rules: no lat/lon or cell ID features. Forecast anomalies are taken against the reforecast's own climatology per start date and week. NaNs stay NaN. A single `as_of(source, issue_date)` function, with a unit test that fails on any read past it.

### 6.3 Model ladder
B0 climatology (smoothed onset rate per cell and week of year) → B1 logistic on current percentile and 10/20-day trend → B2 raw ECMWF (onset rule applied to the forecast soil moisture, then a logistic recalibration) → **M1** LightGBM, land only → **M2** LightGBM, land plus forecast (main) → M3 U-Net/ConvLSTM (stretch, only if errors show spatial structure). The key ablation is M2 − M1 per lead, with paired year-bootstrap CIs. SHAP top-3 drivers per county feed the agent.

### 6.4 Evaluation
Leave-one-year-out over the development years with nested tuning (about 30 trials per lead) and calibration. Test years are scored once with the frozen model. Metrics per lead: Brier and BSS vs B0, reliability (slope and intercept), PR-AUC alongside the base rate, hit rate and false alarm ratio at a fixed threshold, episode lead time, and lead over the Drought Monitor's first one-category worsening. 1,000 year-bootstrap resamples. Always report the episode count.

### 6.5 Agent (Meta model)
Weekly cycle: fetch (as_of) → run frozen M2 + SHAP → render maps and county table → **[LLM] read and cross-check** vs Drought Monitor and crop condition → **[LLM] draft advisory JSON** → validate numbers (every figure must trace to a tool output; at most 2 redrafts, then drop flagged sentences) → publish → score forecasts from 28+ days earlier.

Tools, all taking `as_of`: `get_forecast`, `get_drivers`, `get_map` (PNG), `get_drought_monitor`, `get_crop_condition`, `get_scorecard`, `publish_advisory`.

Advisory schema per county: FIPS and name; risk level (Low, Elevated, High, Severe, from fixed probability bands); probabilities for weeks 1–4; top 3 drivers; a confidence sentence; what to watch; actions for producers/extension and for water managers; the tool calls relied on.

`agent/llm.py` wraps any OpenAI-compatible endpoint, so Glimmer runs locally and the Meta API is a config switch.

Leakage guard for the replay: relative dates ("week 22"), anonymised county IDs, prompts forbidding outside knowledge, and the number validator.

Stretch: a LoRA on Glimmer (Modal) that writes Drought Monitor-style narratives from structured weekly state.

### 6.6 Pre-registered expectations (written before any run)
1. M1 beats B1 at weeks 1–2.
2. M2 beats M1 at weeks 3–4 but not at week 1.
3. B2 is poorly calibrated before recalibration.
4. Skill is higher in the western Plains than in the humid east.

### 6.7 Demo and submission
- Replay May 1 – Aug 31, 2023 week by week. Left panel: FlashWatch week-2 probability with the alert outline. Right panel: actual onsets plus Drought Monitor changes. Below: the agent's advisories and scorecard. A timeline strip for focus counties. An ECOSTRESS 70 m close-up if coverage allows. Second clip: 2012 (out of fold).
- Package: public GitHub repo (README, results table, one-command report reproduction from the cached cube), about a 3-minute video (problem 20 s, replay 90 s, skill and reliability 40 s, agent 30 s; adjust to the official rules), a static replay page, a two-page write-up and a model card.

### 6.8 Repository layout (planned)
```
flashwatch/
  config/     region.yaml, sources.yaml, labels.yaml, model.yaml
  ingest/     era5land.py, smap.py, s2s.py, usdm.py, smvi.py, nass.py, static.py
  grid/       regrid.py
  labels/     onset.py, episodes.py
  features/   asof.py, build.py
  models/     baselines.py, gbm.py, calibrate.py, explain.py
  eval/       cv.py, metrics.py, bootstrap.py, report.py
  agent/      llm.py, tools.py, loop.py, schema.py, validate.py, prompts/
  demo/       replay.py, site/
  scripts/    check_access.py
  tests/      test_asof.py, test_labels.py, test_metrics.py
data/         obs.zarr, smap.zarr, fcst.zarr, train_lead{1..4}.parquet  (gitignored)
```
CLI: `fw ingest|labels|features|train|report|agent|replay`. Stores: `obs.zarr` ~1.6 GB, `smap.zarr` ~0.2 GB, `fcst.zarr` ~0.5 GB, training tables ~1 GB per lead (estimates).

---

## 7. How to start on the new machine

### 7.1 Setup
```bash
# Python 3.11+ recommended. James's old PC used Windows + Python 3.12.4 + the `py` launcher.
mkdir flashwatch && cd flashwatch
# copy in: check_access.py, requirements-check.txt, README-access-check.md, and flashwatch.env renamed to .env
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-check.txt
earthengine authenticate                               # once per machine
python check_access.py                                 # quick checks
python check_access.py --throughput                    # bulk timing + GRIB decode (never run yet)
```
Project stack to add once coding starts: `xarray zarr netCDF4 cfgrib eccodes numpy pandas pyarrow geopandas rioxarray shapely lightgbm scikit-learn shap matplotlib earthengine-api earthaccess cdsapi requests pyyaml typer openai`. On Windows, test `cfgrib`/`eccodes` early; GRIB decoding is the usual pain point.

### 7.2 Day 1 (US Oct 26 / Sydney Oct 27) priorities, after D1 is answered
1. **Queue the ECMWF reforecast downloads first.** They're the longest pole and run in the background. One request per issue date bundles all hindcast years, all members (control `type: cf` plus perturbed `type: pf`, `number: 1/to/10`) and all variables for one parameter group. Run a few requests concurrently. About 60 issue dates per season.
2. **Pull ERA5-Land from Earth Engine** with `ee.data.computePixels` on the fixed 0.25° grid (7.3), one call per variable-year. A year of daily data is 365 bands, well under per-request limits. Write to `obs.zarr`.
3. Download Drought Monitor weekly shapefiles (Tuesday valid dates) and the SMVI archives for 2004–2021.
4. Build the onset labels as soon as soil moisture lands (Day 2 work can start early).

### 7.3 Technical notes learned the hard way

**Earth Engine**
- ERA5-Land daily band names include `volumetric_soil_water_layer_1..4`, `temperature_2m`, `dewpoint_temperature_2m`, `total_precipitation_sum`, `total_evaporation_sum`, `potential_evaporation_sum`, `surface_solar_radiation_downwards_sum`, `soil_temperature_level_1`, `snow_depth`. All are present.
- SMAP L4 v008 bands include `sm_rootzone` and `sm_surface`; 3-hourly, 8 images a day. Make daily means server-side.
- The fixed grid used in tests:
  ```python
  GRID_025 = {"dimensions": {"width": 96, "height": 52},
              "affineTransform": {"scaleX": 0.25, "shearX": 0, "translateX": -104,
                                  "shearY": 0, "scaleY": -0.25, "translateY": 49},
              "crsCode": "EPSG:4326"}
  arr = ee.data.computePixels({"expression": img.toBands(), "fileFormat": "NUMPY_NDARRAY", "grid": GRID_025})
  ```
  The default resampling is nearest-neighbour. For proper area means from 0.1° (ERA5-Land) and 9 km (SMAP), use `reduceResolution(ee.Reducer.mean(), maxPixels=...)` at the native projection before reprojecting to the grid.

**ECMWF S2S via the ECMWF Data Store**
- The old Web API was decommissioned for S2S and TIGGE on **27 May 2026**. Use the ECDS (opened 21 Apr 2026): `https://ecds.ecmwf.int`, API `https://ecds.ecmwf.int/api`, datasets `s2s-reforecasts` and `s2s-forecasts`, through the `cdsapi` client with the CDS key. It accepts **MARS-like syntax**.
- The request that worked:
  ```python
  import cdsapi
  c = cdsapi.Client(url="https://ecds.ecmwf.int/api", key=CDSAPI_KEY)
  c.retrieve("s2s-reforecasts", {
      "class": "s2", "expver": "prod", "model": "glob", "origin": "ecmf", "stream": "enfh",
      "type": "cf", "levtype": "sfc", "date": "2024-06-13", "hdate": "2004-06-13",
      "time": "00:00:00", "param": "228228",                # total precipitation
      "step": "/".join(str(h) for h in range(0, 1105, 24)),  # daily to day 46
      "area": "49/-104/36/-80"}, "out.grib")
  ```
- **Untested** (the `--throughput` checks try them): bundling `type: pf, number: 1/to/10` with 20 hdates in one request, and daily-averaged parameters such as 2 m temperature (`param: 167`) with step ranges `"0-24/24-48/…/1080-1104"`.
- Parameter codes other than `228228` need verifying (guesses: `167` 2 m temperature, `168` 2 m dewpoint, `228086`/`228087` soil moisture top 20 cm / 100 cm). Use ECMWF's MARS-to-CDS dictionary `ecds-s2s.yaml` (https://confluence.ecmwf.int/spaces/DAC/pages/528424942/ecds-s2s.yaml) or generate a request from the ECDS web form ("Show API request code").
- ECMWF cycles: Cy48r1 from Jun 2023 (reforecasts Mon/Thu, 11 members, 20 years); **Cy49r1 from 12 Nov 2024** (reforecasts every other day; the start-date/hdate pattern changes); Cy50r1 from 13 May 2026.
- ECDS date ranges use `2024-02-01/2024-02-03`, not `/to/`.
- Real-time S2S ECMWF data has a 48-hour delay.

**NASA**
- SMAP granules are about 144 MB, 8 a day: never bulk-download them. Use Earth Engine, or AppEEARS or Harmony subsetting.
- `earthaccess` with `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` env vars works. The AppEEARS login endpoint returns a token.
- ECOSTRESS: use V2 products only (V1 was decommissioned Jan 30, 2026). Overpasses are patchy, so use it for demo visuals only.

**Other sources**
- Drought Monitor shapefile URL pattern: `https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_YYYYMMDD_M.zip`. Dates are the Tuesday valid date, e.g. `USDM_20230613_M.zip`.
- SMVI flash drought inventory: HydroShare resource `b6d1d853b82247e1aeaf907ac254c99e`, file list at `https://www.hydroshare.org/hsapi/resource/<id>/files/`.
- NASS Quick Stats: `https://quickstats.nass.usda.gov/api/api_GET/?key=…&commodity_desc=CORN&statisticcat_desc=CONDITION&year=2023&state_alpha=IL&agg_level_desc=STATE&format=JSON`. State level only.
- CDS ERA5-Land requests queue for about 90 s each, so use it as a backup. ECMWF also has a beta "ARCO data lake" of geo-chunked ERA5-Land Zarr, reached through the CDS dataset page's "Analysis ready data" tab.

**Sandbox networking:** Claude cloud and desktop sandboxes reach only package registries and GitHub by default. Every data host above was refused there, so James ran `check_access.py` in his own terminal. See section 4, step 7.

---

## 8. Context worth knowing

- **James's relevant past work** (useful for the write-up and for the design rationale):
  - A monthly **interstate conflict forecasting model**: 100M+ GDELT records, PySpark, about 4.6M dyad-months, 201 features, a PyTorch MLP, a strict time split. Two lessons shaped FlashWatch. Persistence inflated skill, hence the onset-only target. Downsampling broke calibration (predictions ran at 0.37 of the observed rate), hence plain log loss plus isotonic calibration.
  - **Project Justice**: litigation-outcome prediction from court complaints, using LLM extraction through fixed tool schemas with leakage guards and a pre-registered null result. Those leakage-guard habits are reused in the agent's replay mode.
- **Research positioning:** ML subseasonal skill comes mostly from the current land state, and improvements mainly predict moistening rather than drying ([Lorenz & Otkin](https://repository.library.noaa.gov/view/noaa/69335)). Hybrid deep learning plus dynamical models improve subseasonal soil-moisture drought forecasts beyond 2 weeks ([Lesinger & Tian 2025, Nat. Comms](https://ideas.repec.org/a/nat/natcom/v16y2025i1d10.1038_s41467-025-62761-3.html)). Onset remains an open problem for all S2S systems ([SMCast, arXiv 2610.07060](https://arxiv.org/pdf/2610.07060)). Global onset prediction to 10 days was shown recently ([Gou et al. 2026, GRL](https://www.citedrive.com/en/discovery/global-flash-drought-prediction-through-evaporative-stress-forecasting/)).
- **NASA budget context:** the FY2027 request cuts NASA science sharply, but the House bill rejected the worst cuts. SMAP and ECOSTRESS are operating; ECOSTRESS is approved through FY2026, with possible extension to FY2029.

---

## 9. Licences and required attribution

- **ECMWF S2S data: CC BY-NC 4.0, non-commercial research only.** Results must be openly available. Acceptable for an open-source indie entry; a commercial product would need ECMWF's freely licensed open data or NOAA GEFS instead. Required wording:
  - "This work is based on S2S data. S2S is a joint initiative of the World Weather Research Programme (WWRP) and the World Climate Research Programme (WCRP). The original S2S database is hosted at ECMWF as an extension of the TIGGE database."
  - Cite Vitart et al., *The Sub-seasonal to Seasonal (S2S) Prediction Project Database*, BAMS 98(1), 163–176, doi:10.1175/BAMS-D-16-0017.1.
  - Also include: Copyright "© [year] ECMWF", the source (ECMWF Data Store), the CC BY-NC 4.0 licence link, ECMWF's no-liability disclaimer, and a note of modifications made.
- ERA5-Land (Copernicus): credit as "Contains modified Copernicus Climate Change Service information [year]".
- NASA SMAP and ECOSTRESS: cite the dataset DOIs from the NSIDC and LP DAAC pages.
- Earth Engine: noncommercial registration; credit Google Earth Engine.
- US Drought Monitor: credit the National Drought Mitigation Center, USDA and NOAA.
- Muse Glimmer: Apache 2.0.

---

## 10. Schedule (US Eastern dates; Sydney is the next calendar day)

| Day | Date | Exit criterion |
| --- | --- | --- |
| 1 | Oct 26 | `obs.zarr` built for the region and all years; reforecast requests queued; Drought Monitor and SMVI downloaded |
| 2 | Oct 27 | Onset labels and episodes built; cross-check table filled; base rate known |
| 3 | Oct 28 | Training tables for all four leads; `as_of` leakage test passing; B0–B1 scored in the CV harness |
| 4 | Oct 29 | M1 calibrated and scored. **Gate:** if M1 doesn't beat B1 at week 2, narrow the headline to week 1 and the 28-day window |
| 5 | Oct 30 | M2 and B2 scored; ablation with CIs; **model frozen**; test years scored once |
| 6 | Oct 31 | SMAP variant (stretch); SHAP drivers per county; map rendering |
| 7 | Nov 1 | Agent tools, schema, validator; one full cycle on one issue date |
| 8 | Nov 2 | Cross-checks and self-scoring; replay engine running the 2023 season |
| 9 | Nov 3 | Demo site and both case studies; draft write-up; Glimmer fine-tune if time allows |
| 10 | Nov 4 | Video, README, model card; **submit by 9 PM ET** (Nov 5, 1 PM Sydney) |

Cut order if behind: Glimmer fine-tune, then the SMAP variant, then M3. **Never cut:** calibrated out-of-fold evaluation, the as-of leakage test, the 2023 replay.

---

## 11. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| ECMWF downloads slow (queue about 3 min per request) | High | Bundle per issue date, run several concurrently, start Day 1; GEFSv12 fallback; M1 doesn't need forecasts |
| Little or no skill at weeks 3–4 | High | Pre-registered; report with CIs; the demo works on week 1–2 skill |
| Too few independent episodes | Medium | Report episode counts; year bootstrap |
| Labels sensitive to thresholds | Medium | Two threshold variants; triangulate with SMVI, Drought Monitor, SMAP |
| Reforecasts (11 members) vs real-time (101 members, newer cycle) | Certain | Decision D1; use mean, spread and terciles; note in the model card |
| LLM already knows 2023 | Medium | Anonymised replay, validator, prompts |
| Meta API key not working from Australia | Medium | Test now; develop on Glimmer through the same client |
| Rules restrict something in this plan | Unknown | Read the rules first |
| GRIB decoding fails on Windows | Medium | Test with `--throughput`; fall back to WSL or Linux for the decode step |

---

## 12. Files in this bundle

| File | What it is |
| --- | --- |
| `HANDOVER.md` | This document |
| `FlashWatch Design Spec.pdf` | Full spec snapshot, Oct 17 (the live doc is the master copy) |
| `check_access.py` | Access and throughput diagnostic; loads `.env`; flags `--slow`, `--throughput`, `--only <words>`; writes `access_report.json` |
| `requirements-check.txt` | Dependencies for the diagnostic |
| `README-access-check.md` | How to run the diagnostic and where each key goes |
| `.gitignore` | Keeps `.env`, keys, reports and `data/` out of git |

Delivered separately and privately: **`flashwatch.env`**, the credentials. Rename it to `.env`.
