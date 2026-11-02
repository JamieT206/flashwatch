# Demo video: script and shot list (draft, Oct 29)

Three minutes, farmer first. About 420 words of narration at a calm pace. Every shot below is a page
state you can reach from a URL or a click, so it can be recorded in one sitting.

**Setup.**
- Run `python -m http.server 8812 --directory site` and `python -m flashwatch.agent.serve` (for the chat).
- Chrome at 1920x1080, page zoom 110%, light mode. Clear the farm settings first: in the browser console,
  `localStorage.clear()`.
- Record with OBS (or the Windows Game Bar). Record the narration separately and lay it over.

## Narration as recorded (Nov 2): two cuts

Two takes: `video/in/hack.m4a` (every line) and `video/in/general.m4a` (the general-purpose lines and a new
"farmers" line, used in both cuts). The hackathon cut runs 2:44, the general one 2:48 (`video/edit.py`, CUT=hack or
general). Where a line was read twice, the second take is used unless noted.

**Hackathon cut**

1. Welcome to FlashWatch. Everything in this video is built during the hackathon, using public data and Meta's models
   and agents.
2. Farmers are the backbone of America, and we should be building more tools to support them. I built FlashWatch to do
   exactly that. *(general take, both cuts)*
3. A flash drought can dry out a field in two or three weeks.
4. The government's Drought Monitor tracks these, but by the time the drought is actually happening, it's often too
   late.
5. Watering early reduces the damage, but it costs money.
6. FlashWatch uses soil moisture data and weather forecasts to predict flash droughts and to advise farmers on whether
   and when to start extra irrigation. *("satellite" cut from the take: ERA5-Land soil moisture is modelled, not
   satellite-measured)*
7. July 1st, irrigate as usual.
8. July 3rd, we're now at five times the normal risk. FlashWatch recommends starting extra irrigation.
9. Two weeks later, the flash drought hits.
10. FlashWatch calculates the water balance for each field and adjusts its advice accordingly. *("reading irrigation
    data straight from the pivot" cut: the pivot connector is written but untested)*
11. Farmers can also query FlashWatch directly, powered by Muse Spark.
12. To set up a farm, SAM 3.1 finds the pivots in an aerial photo and measures each one.
13. Each morning, the operator agent plans which irrigation pivots to run and checks them against the farm's configured
    rules.
14. Nothing happens until the farmer taps YES on WhatsApp.
15. Farmers can send a voice note transcribed by Muse and incorporated into the model.
16. For rain-fed farms, FlashWatch recommends other drought mitigation actions. Skip a fungicide pass, trim side-dress
    nitrogen, or buy hay early.
17. In our test sample, farms flagged as higher risk saw more than five times as many droughts as the baseline. This
    is more than twice as good as weather-based rules of thumb. The total savings potential is about $140 million a
    year.
18. Under the hood, ERA5-Land soil data and ECMWF forecasts feed a machine-learning model, and Meta's models and agents
    carry it to the farmer. *(first take; the second says the agents interact with the farm equipment directly)*
19. I'm Jamie Twiss, and I built FlashWatch to help American farmers. Thank you.

**General-purpose cut**: the same, with these swapped in.

- 1. Welcome to FlashWatch, software that provides one to two weeks' warning of flash drought for farmers and advises
  them on what to do about it.
- 11. Farmers can also query FlashWatch directly.
- 12. To set up a farm, FlashWatch finds the pivots in an aerial photo and measures each field.
- 15. Farmers can also send a voice note, and FlashWatch turns it into a report on the correct pivot. *(first take)*
- 18. Under the hood, ERA5-Land soil data and ECMWF weather forecasts feed a proprietary machine-learning model and
  calculate a water balance that in turn is used to power the recommendations.
- 19. I'm Jamie Twiss, and I'm building FlashWatch to help American farmers. See how it works at the link below.

Pictures added on Nov 2, in both cuts: under line 3, a caption over the 2012 risk map ("Flash drought: the soil goes
from normal to drought in a few weeks, not months, usually under heat and little rain."); under line 4, the U.S.
Drought Monitor's national maps for July 3, 10, 17 and 24, 2012 (droughtmonitor.unl.edu, credited on screen), with
"In 2006-2021 the soil reached drought a median 22 days before this map showed it. FlashWatch warns one to two
weeks before that." (README; evidence.json, monitor.crossing_vs_d1). Not used: the Monitor at the demo farm itself,
which showed moderate drought (D1) from July 3 to mid-August 2012 and never worse.

Numbers: line 8, farm page 2012-07-03 (5.3x normal, start today); line 17, test years 2022-2024 scored once on Nov 2
(5.6x normal against 1.9x for the rule of thumb); about $140M a year = $28M irrigated + $109M rain-fed (2006-2021,
real ECMWF forecasts).

The archived ECMWF forecasts (and the free Earth Engine tier for the soil data) are licensed for non-commercial use
only: showing the general cut is fine; selling the product would need data feeds licensed for commercial use.

## Full-length narration (3:51, Nov 2)

The longer version, kept for reference.

1. Hello and welcome to FlashWatch. Everything in this video was built during the hackathon, using public data and
   Meta's models and agents.
2. Farmers are the backbone of America. They're smart and technology-forward, but not nearly enough startups are
   looking to support farmers and the agricultural sector. FlashWatch is an effort to redress that balance.
3. A flash drought can dry out a field in two or three weeks, with a significant impact on productivity.
4. In July 2012, one such drought ran across the Corn Belt. By the time the government's official Drought Monitor
   showed it, the damage was already done.
5. Farmers can mitigate the worst effects of a flash drought by watering in advance, but this costs money. So the
   question is simple: should I water my field now, if a drought seems likely, or should I wait?
6. FlashWatch predicts the probability of a flash drought at your farm, one and two weeks ahead.
7. July 1st: irrigate as usual.
8. On July 3rd, the chance of a flash drought is five times the normal level, and FlashWatch says to
   start extra irrigation today.
9. Two weeks later, the flash drought hits.
10. FlashWatch keeps a water account for each field: rain and irrigation in, crop water use out. It's
    built to read the irrigation straight from the pivot's control system, so it knows how much water is left in the
    soil, and adjusts its advice.
11. Farmers can just ask FlashWatch. Muse Spark answers using FlashWatch's own data, and logs the extra
    irrigation. If it quotes a number that isn't in that data, the answer is sent back and rewritten.
12. Set up your farm. SAM 3.1 finds the pivots in an aerial photo, and measures each one from the image.
13. Every morning, the operator agent plans which pivots to run. These are checked against any configuration or
    rules for the farm.
14. For example, one well can't run two pivots at the same time, so the agent pauses one and starts the field where
    the forecast says watering is most valuable.
15. Nothing happens until the farmer taps YES on WhatsApp.
16. Farmers can also send a voice note. Muse Voice Transcribe and Spark turn it into a report on the right pivot,
    and the operator agent takes it into account.
17. Most corn and soybeans aren't irrigated: 130 million acres in FlashWatch's coverage area. Those farms face
    different decisions when a drought is coming. Skip the fungicide pass when a dry spell means disease is less
    likely, trim side-dress nitrogen, or buy hay before prices rise. FlashWatch calculates the value of each step
    from the chance of a drought, and tells the farmer whether it makes sense to act now.
18. How good is FlashWatch? We tested it on 2022 to 2024, years the model never saw. On the farms it
    flags, a flash drought followed more than five times as often as normal. A hot-and-dry rule of thumb, like many
    farmers use, manages less than twice. FlashWatch sees droughts three weeks before the Drought Monitor does, and in
    past seasons that was worth about 140 million dollars a year across the region.
19. FlashWatch is powered by ERA5-Land soil data and ECMWF weather forecasts, which feed a machine-learning model; a
    water balance then turns that into actions and dollars. Meta's models and agents carry it to the farmer, and help
    the farmer interact with it.
20. My name is Jamie Twiss, and I built FlashWatch to help American farmers. Thank you.

## Script

| Time | Shot | Narration |
| --- | --- | --- |
| 0:00-0:08 | Title card: "FlashWatch. A week's warning of flash drought, and what to do about it." | Everything in this video was built during the hackathon, on public data and Meta's models. |
| 0:08-0:25 | Aerial photo of pivots (docs/img/sam_pivots_holt_ne.png), then the July 2012 drought map on the big picture page | A flash drought can dry a field out in two or three weeks. In July 2012 one ran across the Corn Belt. By the time the Drought Monitor showed it, the damage was done. For a farmer the question is simple: water now, or wait? |
| 0:25-0:45 | Farm page, Replay, 2012, Edmunds County SD (S1). Click the date bar from Jul 1 to Jul 3 | FlashWatch forecasts the chance of a flash drought at your farm one and two weeks ahead. July 1: irrigate as usual. July 3: the chance is five times normal, and starting extra irrigation today is worth this much an acre more than waiting. Two weeks later, the flash drought hits (the cursor runs along the date bar to July 17, the first day in drought). |
| 0:45-1:00 | Scroll to "Your field's water"; click "Add irrigation data", then "Fill in demo numbers" | FlashWatch keeps a water account for each field: rain and irrigation in, crop water use out. It's built to read the irrigation straight from the pivot's control system, so it knows how much water is left in the soil, and adjusts its advice. (The Leaf connector is written but untested: "built to", not "does".) |
| 1:00-1:15 | Ask FlashWatch box: type "I put an inch on yesterday. Do I still need to start?" | Farmers can just ask FlashWatch. Muse Spark answers using FlashWatch's own data, and logs the extra irrigation. If it quotes a number that isn't in that data, the answer is sent back and rewritten. |
| 1:15-1:35 | Agents page: SAM pivot image, then the operator trace | Setting up a farm takes one click. SAM 3.1 finds the pivots in a USDA aerial photo and measures each one. Each morning the operator agent plans which pivots to run. Fixed farm rules check every command: here one well can't run two pivots at once, so it pauses one and starts the field where the forecast says water is worth most. Nothing moves until the farmer taps YES on WhatsApp. |
| (in 1:15-1:35) | Phone screen recording: the plan arrives on WhatsApp, tap YES, "Done" comes back | (no extra narration; the YES line above covers it) |
| 1:35-1:45 | Voice note step on the Agents page (play the WAV under it) | And a farmer can just say what they see. Muse Voice Transcribe and Spark turn it into a report on the right pivot. |
| 1:45-2:10 | Farm page, toggle Rain-fed; fungicide card; drag a slider | Most corn and soybeans aren't irrigated: 130 million acres in our region. For those farms the decisions are different: skip the fungicide pass when a dry spell means no disease, trim side-dress nitrogen, buy hay before prices rise. Each shows the chance today against the chance at which acting pays. |
| 2:10-2:35 | Big picture: the 6x chart, the farm table, the Drought Monitor numbers | How good is it? Tested on 2022-2024, years the model never saw: on the farms FlashWatch flags, a flash drought followed more than five times as often as normal; a hot-and-dry rule of thumb manages under two. It sees drought three weeks before the Drought Monitor. Across the region, that's worth about $140 million a year, most of it on rain-fed farms. |
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
| 7 | Rain-fed fungicide | shot 1, toggle Rain-fed, date Jul 13 ("Skip the fungicide pass", risk high; the soil is in drought from Jul 17) |
| 8 | Forecast quality, farm table | `region.html` top half |

## To check before recording

- All 101 forecast dates are in ("six times" is current; 6.3x). The farm page now uses M2 on forecast dates: re-check
  July 3's numbers before recording (Oct 31: July 3 says "start extra irrigation today" again, 4.1x normal, now that a false
  start is priced on real weather). "About $140 million" = $109M rain-fed + $28M irrigated on real forecasts (Nov 2).
- July 3 margin is small: start today +$1.1 an acre against +$0.8 for waiting, 5.3x normal (Nov 2). Narrate "FlashWatch says start
  extra irrigation today" rather than "worth this much more than waiting"; July 5 (9.1x, +$2.1 against +$1.4) is stronger.
- Browser shots are scripted: `python video/record.py <shot>` (Playwright, drawn cursor), output in video/out/.
- Test years scored Nov 2: 5.6x normal (rule of thumb 1.9x); the "how good is it" line uses them.
- End card names the only contributor, Jamie Twiss (rules).
