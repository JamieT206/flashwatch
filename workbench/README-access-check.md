# FlashWatch access check

A pre-kickoff diagnostic: one small request per data source, results in `access_report.json`.
It is not project code. Run it in your own terminal, because Claude's sandbox can't reach these sites.

## 1. Run it now (no accounts needed)

In PowerShell, from this folder:

```powershell
py -m pip install -r requirements-check.txt
py check_access.py
```

The anonymous checks run immediately: NASA CMR, the Drought Monitor, HydroShare, NOAA on AWS
and Hugging Face. The rest are skipped with a note saying what each one needs.

## 2. Add credentials as you get them, then re-run

| Source | What to set up |
| --- | --- |
| All keys | Easiest: put them in a `.env` file in this folder (see `flashwatch.env`); the script loads it |
| NASA Earthdata | `$env:EARTHDATA_USERNAME="you"; $env:EARTHDATA_PASSWORD="..."` |
| Earth Engine | `earthengine authenticate` once (opens a browser), then `$env:EE_PROJECT="your-cloud-project"` |
| Copernicus CDS | Save `%USERPROFILE%\.cdsapirc` with two lines: `url: https://cds.climate.copernicus.eu/api` and `key: <your key>`. Accept the ERA5-Land licence on its dataset page. |
| ECMWF Data Store (S2S) | No extra setup: uses the CDS key. Log in once at https://ecds.ecmwf.int and accept the S2S licence |
| USDA Quick Stats | `$env:NASS_API_KEY="..."` |

## 3. Timed downloads

```powershell
py check_access.py --slow
```

This adds a real CDS request, an ECMWF reforecast request and one full SMAP granule (large),
timing each. `py check_access.py --throughput` times bulk pulls (Earth Engine, bundled ECMWF request) and checks GRIB decoding. `--only ecmwf` (or any word from a check's name) runs a subset.

When it finishes, tell Claude. It can read `access_report.json` from this folder directly.
