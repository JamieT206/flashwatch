# flashwatch
Predicting flash droughts 1-4 weeks in advance

## Day 1: start the downloads (Windows PowerShell, from the repo folder)

```powershell
git pull
copy flashwatch.env .env            # or leave it as flashwatch.env; both are read, neither is committed
py -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
earthengine authenticate            # once per machine (opens a browser)
python -m flashwatch.ingest all     # leave this window open; set the PC not to sleep
python -m flashwatch.ingest status  # in another window, any time
```

Everything is resumable: re-running skips finished files and retries failures. Output goes to
`data/raw/{s2s,era5land,usdm,smvi}`, logs to `data/logs/`.

## Day 2+: build the stores, labels and baselines (after downloads finish)

Needs about 10 GB of free RAM at full scale. Each step prints what it did.

```powershell
git pull; pip install -r requirements.txt
python -m flashwatch.ingest nino                 # small; Nino 3.4 monthly index
python -m flashwatch.obs                         # data/obs.zarr from the ERA5-Land downloads (minutes)
python -m flashwatch.labels --variants           # onsets, episodes, base rates; 3 threshold sets (~5 min)
python -m flashwatch.labels.crosscheck           # known events, SMVI, Drought Monitor (first run rasterises USDM, ~10 min)
python -m flashwatch.forecast inspect data\raw\s2s\2025-04-01_daily.grib   # paste the output to Claude
python -m flashwatch.forecast build              # data/fcst.zarr (once the S2S files are in)
python -m flashwatch.eval.cv --models B0 B1      # leave-one-year-out CV of the baselines (~35 min)
python -m flashwatch.eval.cv --models B0 B1 M1   # adds LightGBM, land only (hours)
python -m pytest                                 # unit tests, incl. the as-of leakage tests
```
