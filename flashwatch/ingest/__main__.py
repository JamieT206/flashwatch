"""Day-1 downloads. Resumable: re-running skips finished files and retries failures.

    python -m flashwatch.ingest all         # every source in parallel, logs in data/logs/
    python -m flashwatch.ingest s2s         # one source: s2s | era5land | usdm | smvi | nino | indices
    python -m flashwatch.ingest status      # what has landed so far
"""

from __future__ import annotations

import subprocess
import sys
import time

from flashwatch.config import LOGS, RAW, load_env

SOURCES = ["s2s", "era5land", "usdm", "smvi", "nino", "indices"]


def status():
    for s in SOURCES:
        d = RAW / s
        files = [f for f in d.glob("*") if f.is_file() and not f.name.startswith("_")
                 and ".part" not in f.name] if d.exists() else []
        mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"{s:9s} {len(files):5d} files {mb:9.1f} MB")
    for s in SOURCES:
        log = LOGS / f"{s}.log"
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            errs = sum(" ERROR " in ln for ln in lines)
            print(f"\n--- {s}.log ({errs} errors), last line:\n{lines[-1] if lines else ''}")


def run_all():
    LOGS.mkdir(parents=True, exist_ok=True)
    procs = {}
    for s in SOURCES:
        out = open(LOGS / f"{s}.console.txt", "a", encoding="utf-8")
        procs[s] = (subprocess.Popen([sys.executable, "-m", "flashwatch.ingest", s],
                                     stdout=out, stderr=subprocess.STDOUT), out)
        print(f"started {s} (pid {procs[s][0].pid})")
    print(f"logs: {LOGS}\nleave this window open; `python -m flashwatch.ingest status` in another shows progress")
    while procs:
        time.sleep(30)
        for s, (p, out) in list(procs.items()):
            if p.poll() is not None:
                out.close()
                print(f"{time.strftime('%H:%M')} {s} exited with code {p.returncode}")
                del procs[s]
    status()


def main():
    load_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "all":
        run_all()
    elif cmd == "status":
        status()
    elif cmd in SOURCES:
        import importlib
        importlib.import_module(f"flashwatch.ingest.{cmd}").main()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
