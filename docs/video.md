# Demo video: script and shot list (draft, Oct 29)

Three minutes, farmer first. About 420 words of narration at a calm pace. Every shot below is a page
state you can reach from a URL or a click, so it can be recorded in one sitting.

**Setup.**
- Run `python -m http.server 8812 --directory site` and `python -m flashwatch.agent.serve` (for the chat).
- Chrome at 1920x1080, page zoom 110%, light mode. Clear the farm settings first: in the browser console,
  `localStorage.clear()`.
- Record with OBS (or the Windows Game Bar). Record the narration separately and lay it over.

## Script

| Time | Shot | Narration |
| --- | --- | --- |
| 0:00-0:08 | Title card: "FlashWatch. A week's warning of flash drought, and what to do about it." | Everything in this video was built during the hackathon, on public data and Meta's models. |
| 0:08-0:25 | Aerial photo of pivots (docs/img/sam_pivots_holt_ne.png), then the July 2012 drought map on the big picture page | A flash drought can dry a field out in two or three weeks. In July 2012 one ran across the Corn Belt. By the time the Drought Monitor showed it, the damage was done. For a farmer the question is simple: water now, or wait? |
| 0:25-0:45 | Farm page, Replay, 2012, Edmunds County SD (S1). Click the date bar from Jul 1 to Jul 3 | FlashWatch forecasts the chance of a flash drought at your farm one and two weeks ahead. July 1: irrigate as usual. July 3: the chance is five times normal, and starting extra irrigation today is worth this much an acre more than waiting. Two days later, the flash drought begins. |
| 0:45-1:00 | Scroll to "Your field's water"; click "Add irrigation data", then "Fill in demo numbers" | It keeps a water account for the field: rain and irrigation in, crop water use out. Read from the pivot's own records, it knows how full the root zone is, and the advice starts from there. |
| 1:00-1:15 | Ask FlashWatch box: type "I put an inch on yesterday. Do I still need to start?" | Farmers can just ask. Muse Spark answers through FlashWatch's tools, logs the inch, and every number it says is checked against the tools before you see it. |
| 1:15-1:35 | Agents page: SAM pivot image, then the operator trace | Setting up a farm takes one click. SAM 3.1 finds the pivots in a USDA aerial photo and measures each one. Each morning the operator agent plans which pivots to run. Fixed farm rules check every command: here the shared well can't run two pivots at once, so it starts the driest. Nothing moves until the farmer taps YES on WhatsApp. |
| (in 1:15-1:35) | Phone screen recording: the plan arrives on WhatsApp, tap YES, "Done" comes back | (no extra narration; the YES line above covers it) |
| 1:35-1:45 | Voice note step on the Agents page (play the WAV under it) | And a farmer can just say what they see. Muse Voice Transcribe and Spark turn it into a report on the right pivot. |
| 1:45-2:10 | Farm page, toggle Rain-fed; fungicide card; drag a slider | Most corn and soybeans aren't irrigated: 130 million acres in our region. For those farms the decisions are different: skip the fungicide pass when a dry spell means no disease, trim side-dress nitrogen, buy hay before prices rise. Each shows the chance today against the chance at which acting pays. |
| 2:10-2:35 | Big picture: the 6x chart, the farm table, the Drought Monitor numbers | How good is it? On the farms FlashWatch flags, a flash drought follows six times as often as normal; a hot-and-dry rule of thumb manages under two. It sees drought three weeks before the Drought Monitor. Across the region, that's worth about $120 million a year, most of it on rain-fed farms. |
| 2:35-2:52 | README architecture diagram, then the Glimmer section | Under the hood: ERA5-Land soil data and ECMWF weather forecasts feed a machine-learning model; a water balance turns it into dollars; Meta's models carry it to the farm. The test years are scored once, after the model is frozen. |
| 2:52-3:00 | End card: "FlashWatch, by Jamie Twiss", repo link, site link | FlashWatch. The code is on GitHub. |

## Shot list (URLs)

| # | What | How to get there |
| --- | --- | --- |
| 1 | July 3 advice, demo farm | `index.html#lat=45.38&lon=-99.38&mode=replay&year=2012&date=07-03` |
| 2 | July 1 for contrast | same, `date=07-01` |
| 3 | 2012 risk map | `region.html`, season 2012, date Jul 3 |
| 4 | Water account | shot 1, scroll to "Your field's water", click "Add irrigation data", then "Fill in demo numbers" |
| 5 | Ask FlashWatch | shot 1 with the local server running |
| 6 | SAM pivots, operator, voice | `agents.html` |
| 6a | WhatsApp on the phone | server + tunnel running, then `python -m flashwatch.agent.whatsapp operator --to <phone> --year 2012 --date 07-03 --llm`; record the phone screen (the test number takes about a minute) |
| 7 | Rain-fed fungicide | shot 1, toggle Rain-fed, date Jul 19 |
| 8 | Forecast quality, farm table | `region.html` top half |

## To check before recording

- Forecasts to mid-August are in ("six times" is current; 6.3x). The farm page now uses M2 on forecast dates: re-check
  July 3's numbers before recording (Oct 31: July 3 says "start extra irrigation today" again, 4.1x normal, now that a false
  start is priced on real weather). "About $120 million" = $94M rain-fed + $23M irrigated on real forecasts.
- Test-year results (after the freeze) may change the "how good is it" line: use whatever they say.
- End card names the only contributor, Jamie Twiss (rules).
