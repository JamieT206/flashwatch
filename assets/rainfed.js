// Rain-fed (and livestock) drought decisions: what each is worth if a flash drought comes and if it
// doesn't, by days of notice. Mirrors flashwatch/config/actions.json and flashwatch/decide.outcomes.
"use strict";

const RF = (() => {
  // params: [key, label, step, default, source]
  const ACTIONS = [
    { key: "fungicide_corn", crop: "corn", title: "Skip the fungicide pass", window: ["07-10", "08-05"], when: "Decided at tasseling, mid-July to early August",
      act: "Skip the fungicide pass", calm: "Spray as planned", per: "an acre", notice: [0, 7, 14], share: [1, 1, 1],
      params: [["fc_cost", "Product and application ($/acre)", 1, 32, "Between Wisconsin ($22) and Ohio State ($40) estimates"],
               ["fc_wet", "Gain if it stays wet (bu/acre)", 0.5, 9, "Fields with disease present; Iowa State's average over all fields is 3.3"],
               ["fc_dry", "Gain in a dry spell (bu/acre)", 0.5, 1, "Little disease when dry (NC State 2024 drought trial)"]],
      outcome: f => [f.fc_cost - f.fc_dry * f.price, f.fc_wet * f.price - f.fc_cost],
      explain: f => `A dry spell means little leaf disease, so the pass you'd normally make at tasseling earns little. If a flash drought comes, skipping keeps
        $${f.fc_cost} an acre of product and application. If it stays wet, you give up about ${f.fc_wet} bu of disease protection.` },
    { key: "fungicide_soy", crop: "soybeans", title: "Skip the fungicide pass", window: ["07-20", "08-20"], when: "Decided at pod set (R3), late July to mid-August",
      act: "Skip the fungicide pass", calm: "Spray as planned", per: "an acre", notice: [0, 7, 14], share: [1, 1, 1],
      params: [["fs_cost", "Product and application ($/acre)", 1, 25, "Estimate"],
               ["fs_wet", "Gain if it stays wet (bu/acre)", 0.5, 3, "Fields with disease present, estimate"],
               ["fs_dry", "Gain in a dry spell (bu/acre)", 0.1, 0.3, "Little disease when dry, estimate"]],
      outcome: f => [f.fs_cost - f.fs_dry * f.price, f.fs_wet * f.price - f.fs_cost],
      explain: f => `Dry weather means little disease, so the R3 pass earns little. If a flash drought comes, skipping keeps $${f.fs_cost} an acre;
        if it stays wet, you give up about ${f.fs_wet} bu.` },
    { key: "nitrogen", crop: "corn", title: "Trim side-dress nitrogen", window: ["06-01", "06-30"], when: "Decided at side-dress time, June",
      act: "Trim side-dress nitrogen", calm: "Keep your planned rate", per: "an acre", notice: [0, 7, 14], share: [0.5, 1, 1],
      params: [["n_cut_lb", "Trim (lb N/acre)", 5, 30, "A trim of a typical 40-60 lb side-dress; large cuts aren't advised"],
               ["n_price", "Nitrogen price ($/lb N)", 0.01, 0.66, "USDA AMS Iowa production cost report, June 2026"],
               ["n_wrong_pct", "Yield lost if rain comes (%)", 0.5, 3, "Estimate"]],
      outcome: f => { const saved = f.n_cut_lb * f.n_price; return [saved, f.n_wrong_pct / 100 * f.yield_bu * f.price - saved]; },
      explain: f => `Extension advice is to keep the planned rate unless drought cuts yield potential. A flash-drought warning is the early sign that it might:
        a crop short of water can't use all the nitrogen. Trimming ${f.n_cut_lb} lb saves $${(f.n_cut_lb * f.n_price).toFixed(0)} an acre; if rain comes,
        you lose about ${f.n_wrong_pct}% of yield.` },
    { key: "silage", crop: "corn", title: "Get ready to chop for silage", window: ["07-20", "08-31"], when: "Livestock farms, late July and August",
      act: "Line up chopping now", calm: "No change", per: "an acre", notice: [0, 7, 14], share: [0.4, 1, 1], livestock: true,
      params: [["sl_t", "Silage from drought-hit corn (tons/acre)", 0.5, 9, "Estimate"],
               ["sl_value", "Silage value ($/ton)", 1, 50, "Estimate"],
               ["sl_kept", "Value kept by chopping on time (%)", 1, 10, "Estimate"],
               ["sl_prep", "Booking and tests if no drought ($/acre)", 0.5, 4, "Estimate"]],
      outcome: f => [f.sl_t * f.sl_value * f.sl_kept / 100, f.sl_prep],
      explain: f => `Drought-hit corn is still good feed if chopped at the right moisture (60-70%). It often looks drier than it is, so test before chopping.
        A week's notice gives time to book a chopper, storage and moisture tests, instead of chopping late when everyone else is.` },
    { key: "hay", crop: "any", title: "Buy hay or feed early", window: ["04-01", "10-31"], when: "Any time in the season",
      act: "Buy hay or feed now", calm: "No need to buy early", per: "for the farm", notice: [0, 7, 14, 21, 28], share: [0.6, 0.8, 0.95, 1, 1], livestock: true,
      params: [["hay_tons", "Hay or feed to buy (tons)", 5, 50, "Example herd"],
               ["hay_price", "Hay price ($/ton)", 5, 180, ""],
               ["hay_rise", "Price rise in a drought (%)", 1, 30, ""],
               ["hay_carry", "Cost of buying early if no drought (%)", 1, 8, "Storage loss and interest, estimate"]],
      outcome: f => [f.hay_tons * f.hay_price * f.hay_rise / 100, f.hay_tons * f.hay_price * f.hay_carry / 100],
      explain: f => `Hay prices rise when drought spreads; in 2012 they rose ${f.hay_rise}% where you farm. Buying before the market moves locks in today's price.
        Partly a race with your neighbours: the market usually reacts when the Drought Monitor shows drought, about three weeks after the soil does.` },
  ];

  const DEFAULTS = Object.fromEntries(ACTIONS.flatMap(a => a.params.map(p => [p[0], p[3]])));
  const SOURCES = Object.fromEntries(ACTIONS.flatMap(a => a.params.map(p => [p[0], p[4]])));

  function curve(a, f) {
    const [gain, cost] = a.outcome(f);
    return { rows: a.notice.map((k, i) => ({ notice: k, avoided: gain * a.share[i], extraCost: 0 })), costNoEvent: cost, gain, cost };
  }
  const forFarm = (f, livestock) => ACTIONS.filter(a => (a.crop === "any" || a.crop === f.crop) && (livestock || !a.livestock));
  const inWindow = (a, md) => md >= a.window[0] && md <= a.window[1];
  const chance14 = (p1, p2) => 1 - (1 - p1) * (1 - p2);

  return { ACTIONS, DEFAULTS, SOURCES, curve, forFarm, inWindow, chance14 };
})();
