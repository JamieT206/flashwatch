# FlashWatch: standing instructions for Claude Code

Read `HANDOVER.md` (sections 0-5) at the start of every session, then `QUESTIONS.md`. They hold
the project state, decisions, open questions and the next steps in order.

- **Public repo.** Never commit secrets, account names, `flashwatch.env`/`.env`, anything under
  `data/`, or ECMWF-derived data files. Never print credentials.
- **Branch**: work on `claude/practical-planck-eyczp0`; commit and push regularly with clear messages.
  James reviews from GitHub during the day, so push before long runs and when something is done.
- **Questions for James** go in `QUESTIONS.md` (with a recommendation and a default you'll use until
  he answers), as well as in chat.
- **Don't redesign.** The spec is settled; raise any deviation with James first.
- **Leakage discipline**: features read observed data only via `flashwatch.features.asof`; every new
  feature family gets a perturbation test in `tests/test_features.py`. Climatologies come from the
  fold's training years only. The test years (2022-2024) are scored once, after the model freeze.
- **Report honestly** against the pre-registered expectations (HANDOVER 6.6), including nulls.
- Run `python -m pytest` before pushing. Long jobs: run in the background, log to `data/logs/`,
  watch memory (full-scale CV peaks around 11 GB).
- Platform: Windows, PowerShell (Git Bash also installed), Python venv in `.venv`. Commands are `python -m flashwatch...`
  (see `README.md` for the run order).
- Deadline: **Nov 4, 9 PM US Eastern (Nov 5, 1 PM Sydney)**. James is in Sydney (UTC+11).
