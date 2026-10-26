# Open questions and decisions

Questions for James are at the top; answer inline (edit this file, or reply in chat) and
I'll pick them up. Below them are defaults I've taken; say if you disagree.
Public repo: nothing private goes here.

## Questions

### Q1. Meta tooling is 25% of the score: promote the Glimmer work up the plan? (needs answer by Oct 30)
The official rules (docs/hackathon-rules.pdf) score Impact 30%, **Use of Meta AI tooling 25%**
("how creatively... beyond a basic integration... surfaced unexpected capabilities"),
Technical depth 20%, Demo 25%. The spec makes the Meta model an explainer/verifier and puts the
Glimmer LoRA **first on the cut list**. That's sound science but under-sells a quarter of the marks.
Options, none of which touches the forecasting core:
- (a) Keep the plan; lean on the agent's multimodal map reading and self-scoring in the pitch.
- (b) **Recommended:** keep the Glimmer narrative LoRA (Modal credits) and move it off the cut list;
  cut the SMAP variant first instead. Pitch: a 30B open model fine-tuned to write Drought
  Monitor-style narratives from structured state, scored against the real USDM narratives.
- (c) Also use SAM 3.1 or Muse Image in the demo. I don't see a natural fit; I'd skip it.

### Q2. ECMWF licence vs the submission licence
S2S data is CC BY-NC (non-commercial). The rules make each entrant grant the organisers a
perpetual licence to "exploit the Submission for any purpose without restriction, including
marketing". My default: the submission is code, results tables and figures with the required
ECMWF attribution; **no ECMWF-derived data files** go in the repo or the demo bundle. If you
want more caution, the fallback is to drop B2/M2 from the public demo, or use GEFSv12 (public domain).
Your call on how much risk to carry.

## Defaults taken (Day 2 code)
- D1 resolved: 2005-2024, CV 2005-2021, test 2022-2024 (the PDF spec still says 2023-2025).
  101 issue dates/season, verified against ECDS's constraints table.
- 5-day running mean of root-zone moisture is **trailing** (days d-4..d), not centred.
- Percentiles: each value's pool is the +/-15-day calendar window in the fold's training years
  **excluding its own year** ("other years"); held-out years use all training years. Mid-rank ties.
  Feb 29 shares Feb 28's calendar day.
- Root zone 0-100 cm = layers 1-3 weighted 7/21/72.
- Onset mask: snow depth > 1 cm or top soil layer <= 0 C on the onset day.
- "Under 20% vegetated land" mask: interim rule is ERA5-Land land fraction >= 0.2. I plan to
  replace it with MODIS land cover (MCD12Q1 via Earth Engine) vegetated fraction on Day 3.
- Drought Monitor as-of latency 2 days (maps are valid Tuesday, released Thursday).
