# Handover

Written 2026-10-29 for whoever (human or agent) picks this up next. The user's honest
reaction after round 5: *"performance still does not seem that good."* They're right
to push on this — read the "Honest assessment" section before doing anything else.

## Update after rounds 6–12 (2026-10-30)

Tier 1 and most of tier 2 below are done; details in `experiments/NOTES.md`.
- **Point model: nothing beat the defaults.** Spike weighting, asinh target, 90th
  percentile, Europe spread, 7-day renewables, daily network summaries, Europe gas
  share, seed bagging, intraday stretch, smaller and larger trees were all within the
  noise or worse on the year and forward splits.
- **Price bands are new and work** (`Banded`, round 10): 10–90 % bands from held-out
  year errors relative to the predicted price, per zone. They beat the point price on
  pinball loss in every held-out year (mean skill 0.151 year, 0.169 forward). In the
  report and on the dashboard.
- **Ceiling:** perfect knowledge of each zone's monthly price level would only reach
  skill 0.36, daily level 0.54. The user's goal is a 50 % MAE cut, which needs the
  backcast itself to improve.
- **Upstream is the lever:** the issue #26 fixes for 2022 (pypsa-eur-backcast
  `feat/backcast-2022`) cut raw 2022 MAE from 92 to 59, but mixing one fixed year with
  unfixed ones hurts the correction (round 9). Retrain once every year has the same
  fixes; the round 9 experiments (`tables=` in `experiments/run.py`) show how.

## 30-second orientation

- This repo corrects PyPSA model prices from `pypsa-eur-backcast` towards observed
  day-ahead prices, using LightGBM on features computed from the solved network only
  (never observed data — the model must also run on future/scenario networks).
- `experiments/NOTES.md` is the full experiment log, rounds 1–5, in order. Read it
  before proposing anything — it's long but it tells you what's already been tried
  and why it was rejected, so you don't repeat work.
- `experiments/ROADMAP.md` is the older idea backlog (written after round 2). Rounds
  3–5 already executed items 1, 2 and 5 of it. Treat it as partially stale; this file
  supersedes it for prioritization.
- `report/` is committed and drives the dashboard (`dashboard/`, Streamlit). Regenerate
  it with `pixi run report` any time `DEFAULT_FEATURES` or the `lgbm` params change —
  a test (`test_committed_report_uses_the_defaults`) enforces this.

## The evaluation contract — do not violate this

- Judge every idea on the `year` split (leave one year out) and `forward` split (train
  on earlier years only), using **mean and worst skill per held-out year**, where
  skill = `1 - MAE / raw_PyPSA_MAE`.
- **Never** use the `month` or `week` split to decide whether something helps. Mixing
  weeks across years scored skill ~0.29 vs. the year split's ~0.13–0.16 for the exact
  same model — it was mostly reading off each year's price level from neighbouring
  weeks, not generalizing. This was tested, not assumed.
- 2022 (gas crisis) stays **in training** — dropping it lowers mean skill on the other
  years (0.139 → 0.124, mostly hurting 2021). But its own error stays large, so report
  mean skill both with and without 2022 rather than let one crisis year dominate the
  headline number.
- The user wants this **simple and explainable** over squeezing out performance. A
  feature or model choice needs a plain-language reason ("which plant sets the
  price", "how full the interconnector is") — not just a number that moved.

## Honest assessment (why the user is unsatisfied)

Current defaults (commit `67177ce`), pooled over 6 years (2019–2024), median over
43 zones, out of sample:

| | raw PyPSA | corrected (`lgbm`) | change |
|---|---|---|---|
| MAE | 20.94 | 18.93 | −10% |
| r | 0.73 | 0.78 | +0.05 |
| MAE, top 5% hours (spikes) | 187.2 | 174.1 | **−7%** |
| MAE, bottom 5% hours | 39.5 | 26.9 | −32% |
| share of hours ≤ 0 | 0.0% | ~0.5–1% | market is 2–3% |

Per-year MAE (2019…2024): raw `7.9, 9.4, 25.7, 92.4, 21.0, 22.9` → corrected
`6.6, 7.2, 20.8, 80.2, 19.3, 19.6`. Mean skill 0.157, worst year (2023) +0.084.

**What this means in plain terms:**
1. **The easy part is done.** The model picks up some of the general price level and
   daily shape — that's where most of the 10% MAE gain and the bottom-5% improvement
   come from.
2. **The hard part — spikes — is barely touched (−7%).** This is the economically
   important failure mode (a trader or planner cares far more about getting a 300
   EUR/MWh hour roughly right than shaving a few EUR off a normal hour), and it's
   essentially unsolved.
3. **Negative prices are still almost never predicted**, even though they're ~2–3% of
   real hours. The backcast's own README flags this as a structural gap (nuclear/
   biomass/hydro throttle bands, no genuine curtailment pricing) — it may not be
   fixable from this repo's side at all.
4. **2022 stays badly wrong** (80 EUR/MWh MAE after correction, on a year where the
   true mean price was ~220). The correction recovers only a small fraction of PyPSA's
   massive underpricing in the crisis. This is very likely a **merit-order problem in
   the backcast itself** (gas sets the real market price in most 2022 hours; the
   model's lignite/coal plants set its price instead — see the "what sets the model
   price" dashboard chart), not something hourly ML features can patch over.
5. **The linear baseline is now *worse* than raw PyPSA** (22.05 vs 20.94 MAE) — only
   the tree model adds value, and even it is clearly fitting shape, not level, in hard
   years.

**Conclusion to act on:** the feature-engineering-on-top-of-PyPSA approach has likely
hit a soft ceiling for the problems that matter most (spikes, negative prices, crisis
years). Squeezing another default feature in is unlikely to move those numbers much —
rounds 4–5 already show diminishing and sometimes negative returns from more features.
The next agent should spend real effort on tier 1 below (targeted models for the tail,
which is still genuinely untried) and tier 3 (whether the ceiling is actually upstream,
in `pypsa-eur-backcast`'s cost model), not just more rounds of "add feature X to the
same LightGBM."

## What's been tried — do not re-run these, see NOTES.md for full detail

Rounds 1–5, summarized (MAE / mean skill, year split unless noted):

- Round 1 (2023 only, month split — later shown to be optimistic): established the
  basic feature set (margins, dispatch shares, hour/day).
- Round 2 (6 years, year split introduced): within-year splits overstate the gain by
  ~3x; absolute price levels (model price, fuel costs) hurt generalization; scale-free
  features carry over better; bus identity helps across years here but not in round 4.
- Round 3: local time/holidays — no gain. Gas-cost-scaled target — robust on `year`,
  **fails badly going forward** (overshoots 2021 by predicting from cheap-gas years).
  **Applying only 70% of the correction (`SHRINK=0.7`) was the single biggest, simplest
  gain found** — beats raw PyPSA in every held-out year. Mixed week split confirmed
  invalid (see above).
- Round 4: a leaner 8-feature set is worse; adding zone identity is roughly neutral.
  Defaults kept as-is.
- Round 5: `price_setter` and link congestion features both help on both splits and
  are explainable — now default (13 features total). Excluding 2022 from training
  hurts the other years; kept in. **`SHRINK` was NOT re-swept after these 3 features
  were added** — it's still the value chosen for the 10-feature set.

## Prioritized next steps

### Tier 1 — cheap, likely to move the numbers the user cares about

1. **Re-sweep `SHRINK` for the current 13-feature model.** One afternoon's work,
   directly addresses a known gap (see above). Do a nested sweep like round 3's (pick
   the share on the other 5 years, report skill on the held-out one) across 0.3–1.0.
2. **A genuinely separate model for the tails**, flagged since round 2's roadmap but
   never built: e.g. a classifier for "this hour is a spike/scarcity hour" (using
   `price_setter == scarcity`/load-shedding, margins, price-setter switches) feeding a
   second regressor or a quantile correction just for those hours. This is the one
   roadmap item that could plausibly move the top-5% MAE number, which nothing so far
   has touched.
3. **Seed bagging** (average `lgbm` over 3–5 seeds) — cheap, reduces the variance
   that's been visible throughout (e.g. ±2 MAE swings on small-sample experiments).
   Not yet done despite being on the roadmap since round 2.
4. **Sample weights by zone-year inverse raw-MAE**, so 2022's huge errors don't starve
   the gradient signal for normal years (roadmap item D2, still untried).

### Tier 2 — diagnostic work before building more

5. **Figure out whether the spike/2022 ceiling is a feature problem or a backcast
   problem.** Concretely: for the worst-predicted hours, check whether *any* feature
   in the table (including ones not in `DEFAULT_FEATURES`) separates them from normal
   hours, versus whether the solved network's own price for that hour is just
   structurally wrong (e.g., check `price_setter` mismatches between model and the
   known-correct `pypsa-eur-backcast` merit order on specific bad hours). If the
   network simply doesn't contain the information, no amount of ML feature work will
   fix it, and effort should redirect to tier 3.

### Tier 3 — upstream, different repo, bigger conversation with the user first

6. **Talk to the user about whether `pypsa-eur-backcast` itself needs a fix** for 2022
   and spikes specifically — e.g. a genuine scarcity/bid-markup mechanism, or the
   price-setter mismatch shown on the dashboard. Two things already found on the
   backcast side are directly relevant, and both point the same way (capacity isn't
   the binding constraint in the model, so cheap fossil plants stay marginal when the
   market prices scarcity):
   - `feat/entsoe-unavailability` (issue #13): adding ENTSO-E outage-based thermal
     availability **changed nothing for fossil carriers in 2019 or 2023** — "there is
     too much spare fossil capacity for it to bind."
   - `feat/daily-gas-prices` (issue #11): daily gas prices helped 2023 and 2019
     modestly, but their own 2022 test explicitly says **"2022 baseline is broken
     (gas underdispatch), so retest after that fix"** — i.e. someone already
     diagnosed 2022 as a backcast dispatch bug, not just a price-level gap.
   Both are recorded in `/home/cesare/.claude/projects/-home-cesare-pypsa-eur-backcast/
   memory/` (`entsoe-unavailability.md`, `daily-gas-prices.md`). Read them before
   assuming this repo's ML features can fix 2022 — the evidence so far says the fix is
   upstream in the backcast's dispatch/merit order, not here.
7. **More years, especially more crisis-like years**, if/when available — 6 years
   gives very few examples of extreme regimes, which is part of why 2022 and the gas
   crisis are hard to generalize into.

### Explicitly not recommended right now

- Neural networks / non-tree architectures — still not justified by data volume
  (effectively 6 independent year-regimes), and conflicts with the user's preference
  for simple and explainable. Revisit only if tier 1–2 are exhausted and a specific
  architecture addresses a specific, diagnosed gap.
- More incremental features on the same LightGBM without a clear story — rounds 4–5
  show this has diminishing or negative returns now.

## Practical gotchas (learned the hard way this round)

- **Launch long jobs detached**, not just via the backgrounded-shell tool alone:
  `setsid nohup your_script.sh > log.txt 2>&1 < /dev/null & disown`. A session ending
  can otherwise kill jobs started through the plain background-shell mechanism — this
  happened to a report run mid-session. Watch for completion by tailing the log file
  for a sentinel line like `=== DONE`.
- `pixi run report` takes ~90 minutes with the full feature ablation on 6 years (~15
  min with `--no-ablation`). Plan around this; don't block other work on it
  synchronously.
- The Streamlit dev server, when started via the plain background-shell tool, can die
  with a segfault/ASGI error as soon as the launching call returns. Same detached
  `setsid nohup ... & disown` pattern fixes it. Use Playwright + the system Chrome
  (`pixi exec -s python=3.12 -s playwright-python python -I script.py`,
  `p.chromium.launch(channel="chrome")`) to screenshot pages for a visual check before
  committing dashboard changes.
- The dashboard's "Zone" selector is a react-aria combobox, not a plain `<select>` —
  in Playwright, `.fill("text")` + `Enter` works; clicking an option by text times out.
- After changing `DEFAULT_FEATURES` or any `lgbm` param, rerun `pixi run report` and
  commit `report/` — `tests/test_report.py::test_committed_report_uses_the_defaults`
  will fail otherwise, by design.
- Run `pixi run -e dashboard pytest -q` (not plain `pixi run test`) when dashboard
  files changed, so the dashboard smoke test runs too.

## Commit/push conventions

- Commit as you go, with a short rationale in the message (see recent history for
  style) and the `Co-Authored-By:` line your own session's attribution reminder gives
  you — don't hardcode a name from this file, it changes between sessions/models.
- Push to `origin master` directly; there's no PR workflow on this repo (unlike the
  sibling `pypsa-eur-backcast` repo, which does use PRs).
