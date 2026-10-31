# Submission form: draft answers (Oct 31)

Numbers are from the development years (2005-2021). After the test years are scored (Nov 2), swap in the test-year
figures where marked [test].

---

## Project Description: Describe your project and what problem it solves

A flash drought dries a field out in two or three weeks, usually in a summer heat wave. By the time the US Drought
Monitor shows it, the damage is often done: in USDA county yields from 2005 to 2021, rain-fed corn yielded about 20%
below trend where a summer flash drought hit, while irrigated corn in the same counties lost only about 2%. Irrigation
saves the crop, but only if the pivot is already running. A centre pivot can't catch up once the root zone is empty in
a heat wave. So every summer morning an irrigator has to decide: start the pivot today, or wait?

FlashWatch answers that question for a specific field. Every forecast date it gives the chance that the soil at the
farm drops into flash drought in the next 7 days and in days 8-14, against normal for the date. It combines today's
observed soil moisture, rain and heat (ERA5-Land) with ECMWF's ensemble weather forecasts, using a gradient-boosted
model per grid cell and a convolutional network that sees the whole map. A water balance of the field then turns the
chance into dollars: what starting extra irrigation today is worth per acre, against waiting for the next forecast.
The advice is one of three plain answers: start extra irrigation today, not yet, or irrigate as usual.

On the tenth of farm-days FlashWatch rates riskiest, a flash drought begins in the next 7 days 6.3 times as often as
normal [test], and the soil crosses the drought line a median 22 days before the Drought Monitor first shows drought
at that place. Across the High Plains and Midwest that's worth about $120 million a year: about $23 million on irrigated
corn and $94 million on rain-fed farms, where the decisions are different (skip a fungicide pass, trim side-dress
nitrogen, buy hay before prices rise).

Agents built on Meta's models carry the advice to the farm. SAM 3.1 finds a farm's pivots in an aerial photo. Muse
Spark plans which pivots to run each morning, checks the plan against the farm's rules (well capacity, water
allocation, season), asks the farmer on WhatsApp, and starts the pivots only on a YES. Farmers can ask questions or
send a voice note from the field, and a watcher texts only when something changes. Every figure an agent writes is
traced back to FlashWatch's own numbers before it reaches the farmer.

The testing is strict: every prediction shown comes from a model that never saw that year; features only read what was
public on the forecast date (an automatic test perturbs the future to prove it); and the test years 2022-2024 are
scored once, after the model is frozen. Null results are reported alongside the wins.

Live demo: https://jamiet206.github.io/flashwatch/ (replays past seasons; the forecast archive it trains on is
released three weeks late). Code: https://github.com/JamieT206/flashwatch

---

## Describe your use of Meta products to build the submitted project

- **Muse Spark 1.1 (Meta Model API).** The main agent model, used through tool calls on FlashWatch's own data:
  - the irrigation operator, which plans pivot starts and stops for a farm with several pivots, revises when the farm's
    rules refuse a command, and writes the approval message;
  - Ask FlashWatch, a chat on the farm page and on WhatsApp ("I put an inch on yesterday. Do I still need to start?"):
    it answers through the advice and water-account tools and logs the irrigation it's told about;
  - the watcher, which follows a farm through the season and texts only when the advice or the soil changes;
  - the morning text, a weekly briefing for a county extension office, and turning voice notes into structured field
    reports.

  Every number Spark writes is checked against the tool results; an untraced figure sends the draft back, and is cut if
  it can't be fixed.
- **SAM 3.1.** The pivot finder: given the farm's location, FlashWatch fetches a USDA aerial photo and SAM 3.1 outlines
  every centre-pivot circle. A second prompt ("crop field shaped like a circle or part of a circle") finds the part and
  half circles. Each circle becomes a pivot with its acres in the farm file the operator uses, so setting up a farm
  takes one click instead of typing coordinates.
- **Muse Voice Transcribe.** Field reports: a farmer says "corn on the sandy pivot is curling by noon" and it becomes a
  stress report on the right pivot, which the operator treats as a reason to water.
- **Muse Glimmer 30B (open weights).**
  - Run locally (4-bit, llama.cpp, one 12 GB graphics card) as an offline fallback for the agents, with the same tools,
    rules and checker. On the demo morning it made the same plan as Spark.
  - Fine-tuned with a LoRA adapter on Modal on 15 years of the Drought Monitor's weekly narratives, so it writes the
    Monitor's kind of regional summary from FlashWatch's numbers: 18% of its wording shared with the real text,
    against 9% for the base model.
  - We also tried a version that writes before the Monitor's map comes out. It kept the style but drifted to places
    outside its input, and we report that as a null result.
- **WhatsApp Cloud API.** The farmer's side of the operator: the plan arrives with YES and NO buttons, YES starts the
  pivots, and the same thread takes questions and voice notes. Tested end to end on a real phone through Meta's test
  number.

---

## Why are Meta's models important to your solution?

The forecast alone doesn't save a crop; a farmer acting on it in time does. Meta's models are how FlashWatch reaches
farmers where they already are and acts on their behalf:

- **WhatsApp and voice fit farm work.** A farmer in a field won't open a dashboard, but will read a text, tap YES, or
  leave a voice note with dirty hands. Spark, Voice Transcribe and the WhatsApp API together make that loop work.
- **Tool use makes the agents trustworthy.** Spark works through FlashWatch's tools rather than from memory, so every
  dollar figure and probability it writes can be traced and checked, and the farm's rules can refuse a bad command
  before anything moves.
- **SAM removes the setup barrier.** Finding and measuring pivots from a photo means a farm is set up from its address.
- **Open weights cover what an API can't.** Glimmer runs on a farm office PC with no connection, keeps farm data local,
  and can be fine-tuned on public data such as the Drought Monitor's narratives. One model family covers both the API
  and the open weights, through the same client code.

---

## Provide feedback from your experience using partner technologies

**Modal** (fine-tuning Muse Glimmer 30B):

- **Good:**
  - Setup was quick: `modal setup`, a Hugging Face token stored as a secret, and a Python file that defines the image,
    GPU and volume.
  - A QLoRA run on one A100 80 GB took about an hour and cost about $4.
  - Caching the model weights (about 60 GB) on a volume meant the second training run and every scoring run skipped the
    download.
  - Logs streamed back to the terminal, which made it easy to watch the loss.
- **Problems:**
  - We tried serving the model with a vLLM web endpoint. The engine needed a CUDA compiler the slim image lacked, so the
    server crashed, and every incoming request started another GPU container. About 10 GPUs and $6 went before we
    noticed.
  - We fixed it with one container at most and a hard timeout on every function. A default limit on GPU containers for
    new apps, or a warning when crashed containers keep restarting, would have prevented it.
  - The storage page showed the volume's size as "—" (it refreshes once a day), so it was hard to tell how much we were
    storing.

---

## Please specify below what matters most when choosing an AI model or coding tool

*(Your view; a starting point.)*

- **For a model:**
  - reliable tool calling and following an output format;
  - not inventing numbers. We still check every figure, but fewer redrafts means faster answers;
  - speed: Spark answered farmer questions in 13-32 seconds, against 1-4 minutes for the 30B model run locally;
  - price;
  - open weights and a permissive licence for offline use and fine-tuning.
- **For a coding tool:**
  - it can work across a whole repository for long stretches;
  - it runs the tests and long jobs itself;
  - it sticks to an agreed spec;
  - it reports results honestly, including the ones that didn't work.
