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
