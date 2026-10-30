"""Publish site/ (pages, assets and the built data) to the gh-pages branch, served by GitHub Pages.

    python -m flashwatch.site.publish            # commit site/ as the new gh-pages tip and push it
    python -m flashwatch.site.publish --dry-run  # list what would go up

The branch holds only the built site (its own history, one commit per publish); the working tree and the
current branch are not touched. site/data is model output on ERA5-Land (land-only model E1) plus summary
results; no ECMWF forecast data goes up. Test years are refused by site.build before they get here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from flashwatch.site.build import SITE

ROOT = SITE.parent
SKIP = ("data/live.json",)   # a local Today feed, if one exists


def git(*args, env=None, inp=None) -> str:
    # bytes, not text mode: on Windows text mode turns \n into \r\n and git drops every --index-info line
    return subprocess.run(["git", *args], cwd=ROOT, env=env, input=None if inp is None else inp.encode(),
                          capture_output=True, check=True).stdout.decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remote", default="origin")
    args = ap.parse_args()
    files = sorted(p for p in SITE.rglob("*") if p.is_file() and p.relative_to(SITE).as_posix() not in SKIP)
    assert (SITE / "data" / "meta.json").exists(), "build the site data first (README: site)"
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    if args.dry_run:
        return
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        blobs = git("hash-object", "-w", "--stdin-paths", inp="\n".join(str(p) for p in files)).split()
        empty = git("hash-object", "-w", "--stdin", inp="")   # .nojekyll: Pages serves the files as they are
        index = [f"100644 blob {b}\t{p.relative_to(SITE).as_posix()}" for b, p in zip(blobs, files)]
        git("update-index", "--index-info", env=env, inp="\n".join(index + [f"100644 blob {empty}\t.nojekyll"]) + "\n")
        tree = git("write-tree", env=env)
    n = len(git("ls-tree", "-r", "--name-only", tree).splitlines())
    assert n == len(files) + 1, f"tree has {n} files, expected {len(files) + 1}"
    try:
        parent = ["-p", git("rev-parse", f"{args.remote}/gh-pages")]
    except subprocess.CalledProcessError:
        parent = []
    src = git("rev-parse", "--short", "HEAD")
    commit = git("commit-tree", tree, *parent, "-m", f"Site from {src}")
    git("push", args.remote, f"{commit}:refs/heads/gh-pages")
    print(f"pushed gh-pages {commit[:7]} (site from {src})")


if __name__ == "__main__":
    main()
