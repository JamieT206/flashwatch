#!/bin/bash
# Post-download rebuild (everything before the freeze), started on its own when the ECMWF downloads finish.
#
#   powershell: Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList "workbench/burst.sh" -WindowStyle Hidden
#   DRY=1 bash workbench/burst.sh     # print the steps without waiting or running them
#
# Every hour it counts the S2S issue dates that have both files (daily and accum). It starts when all are in, or at
# CUTOFF with whatever is in. If the downloader stops before then, it restarts it (at most 3 times).
# Before the heavy steps it waits for 12 GB of free memory (full CV peaks around 11 GB), up to 6 hours.
# Steps: tests, forecast build, M2 CV, evidence and site data, false-start check (not written), roll-ups, agent re-runs.
# Each step logs to data/logs/burst_<step>.log; progress and the last line (DONE, or FAILED at a step) go to
# data/logs/burst.log. The freeze and the test-year scoring are left for review.

set -u
cd /c/Users/Admin/code/flashwatch
source .venv/Scripts/activate
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=.
LOG=data/logs/burst.log
TAG=${TAG:-m2full}
CUTOFF=${CUTOFF:-2026-11-03T12:00+11:00}
DRY=${DRY:-0}
E1=data/oof/B0_M1_M3_M3b_week1_week2_fast_d0-40_m3seeds_E1.parquet
M1=data/oof/B0_B1_M1_week1_week2_any28_fast_d0-40_lakefix.parquet
PERFECT=data/oof/B0_M1_M2_week1_week2_any28_fast_d0-40_oracle_native.parquet
N=data/oof/B0_H1_M1_M2_week1_week2_any28_fast_d0-40_$TAG.parquet

say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }

complete_dates() {   # "<complete> <total>"
  python -c "
import json; from pathlib import Path
ds = json.loads(Path('flashwatch/config/s2s_dates.json').read_text())['model_version_dates']
ok = [d for d in ds if all(Path(f'data/raw/s2s/{d}_{g}.grib').exists() for g in ('daily', 'accum'))]
print(len(ok), len(ds))"
}

downloader_running() {
  powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { \$_.CommandLine -match 'flashwatch.ingest s2s' }).Count"
}

keep_awake() {   # holds off idle sleep (not a settings change) until this script exits
  powershell -NoProfile -c "Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);';
    [W.P]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null; while (Get-Process -Id $1 -ErrorAction SilentlyContinue) { Start-Sleep 60 }" &
}

restart_downloader() {
  powershell -NoProfile -c "\$r = (Get-Location).Path; Remove-Item data\raw\s2s\*.part -ErrorAction SilentlyContinue;
    \$env:FW_S2S_PACK = '16'; \$env:FW_S2S_WORKERS = '3';
    Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList '-m','flashwatch.ingest','s2s' -WorkingDirectory \$r -WindowStyle Hidden -RedirectStandardOutput data\logs\s2s.stdout.txt -RedirectStandardError data\logs\s2s.stderr.txt"
}

free_gb() {
  powershell -NoProfile -c "[math]::Floor((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue/1024)"
}

past_cutoff() { [ "$(date +%s)" -ge "$(date -d "$CUTOFF" +%s)" ]; }

step() {   # step <name> <command...>
  local name=$1; shift
  if [ "$DRY" = 1 ]; then echo "[$name] $*"; return; fi
  say "start $name"
  if ! "$@" > "data/logs/burst_$name.log" 2>&1; then
    say "FAILED at $name (see data/logs/burst_$name.log)"
    tail -5 "data/logs/burst_$name.log" >> "$LOG"
    exit 1
  fi
  say "done $name"
}

# 1. Wait for the downloads.
[ "$DRY" = 1 ] || keep_awake "$(cat /proc/$$/winpid)"
restarts=0
while [ "$DRY" != 1 ]; do
  read -r n total <<< "$(complete_dates)"
  running=$(downloader_running | tr -d '\r')
  say "S2S: $n of $total dates complete; downloader processes: $running"
  [ "$n" -ge "$total" ] && break
  if past_cutoff; then say "cutoff $CUTOFF reached: going with $n of $total dates"; break; fi
  if [ "$running" = 0 ]; then
    if [ "$restarts" -ge 3 ]; then say "downloader stopped 3 times: going with $n of $total dates"; break; fi
    restarts=$((restarts + 1)); say "downloader not running: restart $restarts"; restart_downloader
  fi
  sleep 3600
done

# 2. Wait for memory.
for i in $(seq 1 36); do
  [ "$DRY" = 1 ] && break
  gb=$(free_gb | tr -d '\r')
  [ "$gb" -ge 12 ] && { say "memory: $gb GB free"; break; }
  say "LOW MEMORY: $gb GB free, waiting for 12 (close other apps)"
  sleep 600
done

# 3. Rebuild.
mkdir -p data/site_backup
[ "$DRY" = 1 ] || cp -r site/data "data/site_backup/before_$TAG"
step tests python -m pytest -q
step forecast python -m flashwatch.forecast build
step cv python -u -m flashwatch.eval.cv --models B0 H1 M1 M2 --leads week1 week2 any28 --fast --fc-dates --tag "$TAG"
step evidence python -m flashwatch.site.evidence --m1 $M1 --m2 "$N" --perfect $PERFECT --h1 "$N"
step site python -m flashwatch.site.build --oof "$N" --model M2 --fill $E1 --fill-model E1 \
  --note "ECMWF forecast model (M2) where a forecast exists, land-only model (E1) on other dates"
step falsestart python -m flashwatch.decide.falsestart --oof "$N"
step rollup_water python -m flashwatch.decide.rollup --oof "$N" --model M2 --tag "${TAG}_water" --set spell_days=21
step rollup_month python -m flashwatch.decide.rollup --oof "$N" --model M2 --tag "${TAG}_month" --no-water --set spell_days=21
for p in 3.5 4.5 5.5 6.5 7.5; do
  t=price${p/./}
  step "rollup_$t" python -m flashwatch.decide.rollup --oof "$N" --model M2 --tag "$t" --no-water --set spell_days=21 price=$p
done
step operator python -u -m flashwatch.agent.operator --year 2012 --dates 07-01 07-03 07-05 07-07 07-09 --llm-dates 07-03 \
  --approve "demo farmer"
step watch python -u -m flashwatch.agent.watch --year 2012 --llm 3
step ask python -u data/logs/ask_compare.py
step scorecard python -u -m flashwatch.agent.scorecard
step agents python -m flashwatch.site.agents
[ "$DRY" = 1 ] || say "DONE: rebuild on tag $TAG; next: review, freeze, score the test years"
