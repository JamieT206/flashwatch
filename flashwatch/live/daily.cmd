@echo off
rem Daily live-feed run for Windows Task Scheduler (README: the live feed): update, forecast, publish.
rem Logs to data\logs\live_daily.log. Registered as "FlashWatch live feed", daily 21:00, catching up when the PC is next on.
cd /d "%~dp0..\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ==== %date% %time% >> data\logs\live_daily.log
.venv\Scripts\python.exe -m flashwatch.live.run today --publish >> data\logs\live_daily.log 2>&1
echo exit %errorlevel% >> data\logs\live_daily.log
