# Experiment notes

Log of what works and what doesn't. All scores are in EUR/MWh: median over the 43 scored buses, out of sample.
The full log is in `results.csv`, written by `run.py`. `analyse_best.py` holds the deeper analysis of the best configuration.

## Round 1: backcast 2023 only (2026-10-28)

Input: `pypsa-eur-backcast` `results/backcast-2023/networks/solved_2023.nc`, observed prices from `resources/backcast-2023/day_ahead_prices.csv`.
Check: the raw PyPSA scores reproduce the backcast's `price_metrics.csv` exactly (MAE 21.03, r 0.752).

### Splits

- `week` (every 4th week as a fold) is optimistic. With the `month` feature it reaches MAE 15.5, but 17.5 without it: `month` learns the price level of the same month from neighbouring weeks.
- `month` (leave one calendar month out) without the `month` feature is the reference. It is the most honest test within one year.
- Forward test (train Jan–Aug, test Sep–Dec) as a stand-in for an unseen period.

### Results (month split, LightGBM unless stated)

| id | change | MAE | r | top-5 % MAE | bottom-5 % MAE |
|---|---|---|---|---|---|
| — | raw PyPSA | 21.03 | 0.752 | 39.7 | 46.7 |
| — | `bus_bias` baseline | 20.77 | 0.747 | 42.8 | 41.8 |
| E04 | `linear` (Huber), default features without month | 20.74 | 0.758 | 43.0 | 40.0 |
| E18 | model price + capacity margins only | 20.03 | 0.765 | 41.2 | 34.7 |
| E04 | + hour, day of week | 18.28 | 0.796 | 35.1 | 32.7 |
| E05 | + system state (renewable share, curtailment, thermal utilisation, net imports, storage) | 16.96 | 0.825 | 34.1 | 30.1 |
| **E14** | **E05 with larger trees (63 leaves, 800 trees, lr 0.03)** | **16.66** | **0.825** | 34.6 | **27.8** |

Seed noise is about ±0.06 MAE (E05, E10, E11), so differences below ~0.15 are not meaningful.

### What works

- **LightGBM on the error (observed − model price).** It improves all 43 buses and every held-out month.
  The gains are largest in the Baltics and DE/LU (−7 to −8 MAE), smallest in Italy (−1 to −2).
- **Features from the solved dispatch**, not only from capacities. In gain, the most important are model price 21 %, thermal utilisation 15 %, hour 14 %, storage dispatch 10 % and curtailment 8 %.
- **Forward test:** MAE 24.4 → 21.3, r 0.751 → 0.819. The correction carries to later months with lower gas prices.
- **Spread and negative prices, partly:** std ratio 0.81 → 0.86, hours ≤ 0 from 0 % to 1.0 % (observed 2.2 %).
- **Larger trees** (E14) beat smaller ones (E13: 17.68).

### What doesn't (or is not yet testable)

- **Capacity margins alone** give only ~1 MAE (E18). On top of the dispatch features they add ~0.14 (E17: 16.80 without them), which is at the noise level.
- **The `month` feature** leaks the level of the year. It is dropped from the defaults.
- **Fuel cost features** (`cost_gas`, `cost_coal`, `price_to_gas`) make MAE worse (17.20 vs 16.96).
  Within one year they stand in for time, so the held-out months are extrapolated. Retest with several years, where they should carry the level.
- **Daily mean and intraday deviation of the model price** make MAE worse (17.44).
- **Bus identity** helps without system features (18.28 → 17.50), and on top of them gives a better r (0.834) and bias (1.2) but no better MAE (17.12). Revisit with more years.
- **Loss:** the default absolute-error loss gives the best MAE. Squared error (E16) gives the best spike MAE (32.4) and bias (1.1) but MAE 17.42. Huber (E15) sits in between.
- **`bus_bias` and `linear` baselines** barely beat raw PyPSA. The bias correction even worsens spike hours.

### Feature ablation (report of 2026-10-28)

`ppf report` refits the default model without each default feature and with each other feature (month split).
Three seeds give a noise band of ±0.09 MAE (2σ).
Positive numbers mean MAE rises.

| removed | Δ MAE | | added | Δ MAE |
|---|---|---|---|---|
| hour | +1.19 | | month | −0.41 (leak) |
| dayofweek | +0.65 | | bus | −0.28 |
| storage_share | +0.53 | | price_deviation | −0.22 |
| model_price | +0.17 | | margin_with_imports_mw | −0.18 |
| curtailment_share | +0.11 | | margin_domestic_mw | −0.05 |
| margin_with_imports | +0.11 | | price_daily_mean | +0.04 |
| vres_share | +0.06 | | price_to_gas | +0.49 |
| net_import_share | +0.05 | | cost_gas | +1.35 |
| firm_utilisation | 0.00 | | cost_coal | +1.51 |
| margin_domestic | −0.01 | | | |

- Calendar and storage carry the most. Storage dispatch is the model's own signal of when prices peak within the day.
- Thermal utilisation has the second highest split gain (15 %) but costs nothing to remove: the other features carry the same information.
  Split gain is therefore a poor guide to which features to keep.
- The domestic margin and the dispatch shares of renewables and net imports are within noise individually.
- Margins in MW, intraday deviation and bus identity help a little, within one year. Retest with the year split before adding them.

### Defaults changed

`DEFAULT_FEATURES` is now the E05 set, and the `lgbm` parameters are those of E14.

### Next

- More years for the year split. Then retest fuel costs, bus identity and month.
- Temperature from the atlite cutout.
- Local time hour and holidays. The hour is currently UTC, and it is the third most important feature.
- Neighbour price spreads and congestion of the NTC links.
- Spike hours remain the largest error (top-5 % MAE ~34). Try a squared-error model or a quantile ensemble for the tails.

## Round 2: six years, 2019–2024 (2026-10-29)

Input: the solved networks of `pypsa-eur-backcast` for 2019 to 2024, extracted with each year's `resources/backcast-<year>/day_ahead_prices.csv`.
`experiments/sanity.py <year>` reproduces the backcast's `price_metrics.csv` exactly for every year.

Raw PyPSA, median over zones:

| year | MAE | r | \|bias\| | zones with prices |
|---|---|---|---|---|
| 2019 | 7.89 | 0.707 | 4.9 | 40 |
| 2020 | 9.36 | 0.753 | 6.3 | 40 |
| 2021 | 25.70 | 0.917 | 7.8 | 41 |
| 2022 | 92.43 | 0.533 | 70.9 | 42 |
| 2023 | 21.03 | 0.752 | 6.7 | 43 |
| 2024 | 22.94 | 0.680 | 10.8 | 43 |

From here on, skill means 1 − MAE / raw MAE of the same year. Splits:
- `year`: train on the other five years.
- `forward`: train on the earlier years only, so 2019 is not scored.

### Within-year splits overstate the gain: two years (2019 + 2023)

Experiments E19–E28 on two years, suffixed `_2y` in `results.csv`.

| experiment | raw MAE | lgbm MAE |
|---|---|---|
| month split (E28) | 16.28 | 13.89 |
| year split (E19) | 16.28 | 20.53 |

- The correction that looks best within a year is worse than raw on an unseen year.
- Seeds alone move the year split MAE between 20.3 and 22.2.

### Six years (E19, E20, E23, E24, E27)

Skill per year of `lgbm` with the default features:

| split | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | mean | worst |
|---|---|---|---|---|---|---|---|---|
| year (E19) | +0.01 | +0.14 | +0.20 | +0.11 | −0.02 | +0.17 | +0.10 | −0.02 |
| forward (E20) | – | +0.23 | −0.09 | +0.16 | −0.10 | +0.17 | +0.07 | −0.10 |

- With five training years, the year split helps four of six years clearly: median MAE 20.94 → 19.30, r 0.729 → 0.768, bottom-5 % MAE 39.5 → 25.0.
  `bus_bias` (20.53) and `linear` (21.89) do not transfer.
- The forward split fails at the regime changes. In 2021, the start of the gas crisis, MAE rises 25.7 → 28.0. In 2023, after the crisis, it rises 21.0 → 23.2.
  In 2023 the year split correction overshoots: r improves 0.752 → 0.791, but |bias| 6.7 → 10.3 and std ratio 0.81 → 1.06.
  The model learns the hourly shape of the error, not its level.
- 2022 keeps a median |bias| of 61 after the correction (71 raw). The level of the crisis year is not learnable from the other years.
- Fuel cost features (E23) do not help: 20.98 vs 19.30. Price and margins alone (E27): 21.40, worse than raw.

### Feature ablation on six years (report of 2026-10-29, year split)

Three seeds give a noise band of ±0.15 MAE (2σ).

| removed | Δ MAE | | added | Δ MAE |
|---|---|---|---|---|
| hour | +0.70 | | bus | −0.69 |
| margin_with_imports | +0.45 | | margin_domestic_mw | −0.39 |
| vres_share | +0.43 | | margin_with_imports_mw | −0.27 |
| firm_utilisation | +0.39 | | price_to_gas | −0.20 |
| storage_share | +0.22 | | price_daily_mean | −0.14 |
| margin_domestic | +0.17 | | price_deviation | +0.04 |
| dayofweek | +0.17 | | month | +0.22 |
| net_import_share | +0.17 | | cost_gas | +1.66 |
| curtailment_share | +0.03 | | cost_coal | +2.40 |
| model_price | **−0.53** | | | |

- **Absolute price levels hurt across years.** Removing `model_price` improves MAE by 0.53, and gas and coal costs add 1.7 and 2.4.
  Trees cannot extrapolate a level they have not seen. Scale-free features (shares, ratios, margins) carry over.
- **The capacity margins matter across years** (+0.45 and +0.17 when removed), although they were at the noise level within 2023.
- **Bus identity helps across years (−0.69)**, unlike within 2023. The zone-specific shape of the error persists, though `bus_bias` alone does not.
- **Month hurts across years (+0.22)**, as expected from its leak within a year.
- Split gain ranks `model_price` first (30 %), although removing it helps. Gain is a poor guide again.

### Next

Round 3 follows the top of [ROADMAP.md](ROADMAP.md):
- local time and holidays;
- the target scaled by the gas plant cost (`lgbm_gas`);
- scale-free features without `model_price`, with and without bus identity;
- the mixed week split for comparison.

## Round 3: roadmap items on six years (2026-10-29)

Each row is a mean / worst skill over held-out years (1 − MAE / raw MAE).
`year` trains on the other five years, `forward` on the earlier years only.

| id | change | year: mean / worst | forward: mean / worst |
|---|---|---|---|
| E19/E20 | default (whole correction) | 0.101 / −0.021 | 0.074 / −0.104 |
| E29/E30 | local hour, day of week, holidays | 0.101 / −0.035 | 0.071 / −0.108 |
| E31/E32 | error relative to the gas plant cost (`lgbm_gas`) | 0.114 / 0.002 | −0.031 / −0.722 |
| E33/E34 | the same with local time | 0.117 / −0.004 | −0.028 / −0.695 |
| E40/E41 | `lgbm_gas` with the gas cost clipped to the training range | 0.109 / 0.002 | 0.067 / −0.175 |
| E36/E37 | without `model_price`, with `price_to_gas` | 0.058 / −0.060 | 0.038 / −0.108 |
| E38/E39 | the same plus bus identity | 0.065 / −0.025 | 0.066 / −0.078 |
| **E42/E43** | **70 % of the correction (`lgbm` now)** | **0.133 / 0.042** | **0.092 / −0.074** |
| E35 | week split, 5 folds, all years mixed | 0.286 / 0.178 | |

- **Applying 70 % of the correction is the largest and simplest gain.** It is better than raw PyPSA in every held-out year of the year split, and better than the whole correction on both splits.
  The share was chosen from the out-of-sample predictions of the report.
  Chosen on the other five years only, it lands between 0.5 and 0.8 for every held-out year, and gives a mean skill of 0.111 with a worst of +0.025.
  The whole correction overshoots in years unlike those in training: 2019 skill 0.01 at 100 %, 0.10 at 70 %; 2023 −0.02 at 100 %, +0.04 at 70 %.
- **Local time and holidays change nothing.** UTC hour and day of week already capture the daily shape.
- **The gas-scaled target is robust on the year split but fails going forward.** In 2021, trained on 2019 and 2020 only, the relative error learnt at low gas costs is multiplied by crisis gas costs.
  Clipping the gas cost to the training range limits the damage (2021: −0.72 → −0.18), but stays worse than the default. Not worth the extra concept.
- **Dropping `model_price` hurts mean skill (0.058),** although the ablation showed a lower median MAE without it.
  The median over zone-years and the mean skill over years can disagree. Use mean and worst skill to decide.
- **The mixed week split scores almost three times the year split (0.286 vs 0.101).** Within-year or mixed splits must not be used to choose models.

### Defaults changed

`lgbm` applies 70 % of its correction (`SHRINK` in `models.py`). `lgbm_full` keeps the whole correction.
Experiments E01–E41 used the whole correction: rerun them with `lgbm_full` to reproduce.

### Next

- Price-setting technology and congestion features (roadmap 5).
- Seed bagging and sample weights per zone-year, both simple.
- Tails: quantile or L2 models for the spike hours, which still carry the largest errors.

## Round 4: simpler or zone-aware feature sets (2026-10-29)

All rows apply 70 % of the correction. Values are mean / worst skill per held-out year.

| id | features | year | forward |
|---|---|---|---|
| E42/E43 | default (10) | 0.133 / 0.042 | 0.092 / −0.074 |
| E44/E45 | without `margin_domestic` and `vres_share` (8) | 0.119 / 0.012 | 0.074 / −0.083 |
| E46/E47 | default plus `bus` | 0.128 / 0.039 | 0.106 / −0.051 |

- **Keep the domestic margin and the renewable share.** The median-MAE ablation suggested dropping them, but mean skill falls on both splits without them.
- **Bus identity does not justify itself.** It is even on the year split and +0.014 on the forward split, about the seed noise, and it cannot handle zones unseen in training.
- **The defaults stay as they are.**

## Round 5: price setter, congestion and 2022 (2026-10-29)

New columns from the solved network (`network.py`), re-extracted for all six years:
- **Price setter:** the technology of the partly loaded generators whose marginal cost, quadratic term included, matches the zone price. If none does, it is storage or import.
- **Congestion:** the share of import or export capacity on links at their limit.

All rows apply 70 % of the correction. Values are mean / worst skill per held-out year.

| id | features | year | forward |
|---|---|---|---|
| E42/E43 | default (10) | 0.133 / 0.042 | 0.092 / −0.074 |
| E48/E49 | + `price_setter` | 0.148 / 0.082 | 0.100 / −0.067 |
| E50/E51 | + `imports_congested`, `exports_congested` | 0.149 / 0.047 | 0.100 / −0.062 |
| **E54/E55** | **+ both (13, now the default)** | **0.157 / 0.084** | **0.104 / −0.069** |
| E52/E53 | default, 2022 left out | 0.124 / 0.035 | 0.076 / −0.064 |

- **Both network features help on both splits and are explainable:** which technology sets the model's price, and how full the interconnectors are.
- **The price setter explains part of 2022.** In the model, gas sets the price in 26 % of zone-hours in 2022, against 45–66 % in the other years. Coal and lignite are marginal instead.
  The market priced gas in most hours, so the backcast's merit order is off in the crisis. The correction cannot learn this from other years.
- **Keep 2022 in training.** Over the same five years, mean skill falls from 0.139 to 0.124 without it, mostly in 2021 (0.176 → 0.104).
  The dashboard shows the mean skill with and without 2022, as its error is mostly an upstream level problem.

### Defaults changed

`DEFAULT_FEATURES` adds `price_setter`, `imports_congested` and `exports_congested`. The linear baseline one-hot encodes categorical features.

## Diagnostic: is the tail a feature problem or a backcast problem? (2026-10-29)

Checked directly, no training needed: in the worst-error hours of hard years, does the
*solved network itself* show any sign of scarcity that a feature could expose, or does
it look comfortable while the market was not?

| year/zone | model price max (whole year) | worst 100–200 hours: model mean | observed mean | margin_domestic | price_setter |
|---|---|---|---|---|---|
| 2022 DE00 | 306 | 195 | 661 | 1.39 (comfortable) | 100% coal |
| 2022 FI00 | 306 | 28 | 648 | 0.72 (comfortable) | 100% nuclear |
| 2021 DE00 | 233 | 203 | 418 | 0.99 | 100% coal |
| 2023 DE00 | 202 | 124 | 213 | 1.04 | 84% lignite |

**In the worst hours of the crisis years, the model's own state shows no scarcity
signal at all** — domestic margin is comfortable (>0.7, often >1), thermal plants are
far from their limit, and a cheap technology (coal, lignite, even nuclear in Finland)
sets the price. There is nothing in the solved network for any feature to find. The
gap is a wrong *input* to the backcast (fuel/merit-order assumptions for 2022
specifically), not a gap in what the correction model can see. This matches what two
sibling backcast-repo branches already found independently: `entsoe-unavailability`
(spare fossil capacity means outage data changes nothing) and `daily-gas-prices`
("2022 baseline is broken (gas underdispatch)"). 2023 shows the same pattern but much
smaller (124 vs 213, not 195 vs 661), consistent with 2023 being a normal year where
the correction actually helps.

**Conclusion: no amount of feature engineering in this repo will close this gap for
2022-like hours.** Effort here should go into the tail generally (quantile/weighted
models, round 6) and the regime-change problem (`forward` split), not into trying to
patch the crisis year specifically.

## Round 6: models aimed at the tail (2026-10-29)

All three apply the whole correction (no shrink). Mean / worst skill per held-out year, and pooled MAE in the top 5 % hours (raw PyPSA: 187 year split, 206 forward split).

| id | model | year | forward | top-5 % MAE, year / forward |
|---|---|---|---|---|
| E54/E55 | `lgbm` (default, 70 % shrink) | 0.157 / 0.084 | 0.104 / −0.069 | 174 / – |
| E56/E57 | `lgbm_spike_weighted`: rows weighted by the size of their correction | −0.200 / −0.511 | −0.102 / −0.348 | 148 / 170 |
| E58/E59 | `lgbm_asinh`: fitted on asinh of the correction | 0.113 / 0.029 | 0.095 / −0.043 | 172 / 186 |
| E60/E61 | `lgbm_q90`: 90th percentile of the correction | −0.454 / −1.144 | −0.345 / −1.069 | 129 / 134 |

- **Every point model that gets closer to spikes pays for it in normal hours.** Spike weighting cuts the top-5 % error by a fifth but raises the MAE of 2023 from 21 to 30. The 90th percentile is, as expected, not a point forecast.
- **asinh is a weaker version of the shrink**: it compresses large corrections, but less well than applying 70 % of the correction.
- **None replaces the default.** The tail is better served by bands around the point price (round 7) than by bending the point price towards spikes.

## Round 7: two cheap features and price bands (2026-10-29)

The two features ranked first by a data check (partial correlation with the residual, per year, see HANDOVER.md) that need no re-extraction:
- `spread_europe`: model price minus the median model price over all zones in the same hour. If the model has a zone cheap next to its neighbours, coupling pulls the real price up.
- `vres_share_week`: mean renewable share of the past 168 hours. Long low-wind spells drain storage and hydro.

| id | features | year | forward |
|---|---|---|---|
| E54/E55 | default (13) | 0.157 / 0.084 | 0.104 / −0.069 |
| E62/E63 | + `spread_europe` | 0.157 / 0.092 | **0.114 / −0.010** |
| E64/E65 | + `vres_share_week` | 0.156 / 0.069 | 0.111 / −0.045 |

- **`spread_europe` is a candidate default**: no worse on the year split and better forward, where the worst year goes from −6.9 % to −1 %.
- `vres_share_week` is mixed: slightly better forward, slightly worse on the year split.

### Price bands

`Banded` takes the point price of `lgbm` plus quantiles (5 % to 95 %) of its error in years it did not see: the point model is refitted with each training year left out once. Scored per held-out year by the share of hours inside the 10–90 % band (0.8 if calibrated), the mean pinball loss over the quantiles against the point price used as every quantile, and the Brier score of the probability of a price above 150 EUR/MWh.

E66, year split, errors per zone and level of the predicted price in EUR/MWh:

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| inside 10–90 % | 0.91 | 0.92 | 0.79 | 0.61 | 0.78 | 0.69 |
| width 10–90 %, EUR/MWh | 35 | 40 | 79 | 137 | 91 | 55 |
| pinball bands / point | 3.8 / 3.3 | 3.2 / 3.6 | 8.3 / 10.4 | 32.3 / 40.1 | 9.2 / 9.6 | 7.7 / 9.8 |
| Brier > 150, bands / point | – | – | 0.043 / 0.060 | 0.128 / 0.168 | 0.084 / 0.091 | 0.042 / 0.053 |

Pinball skill 0.108 mean, −0.131 worst (year); 0.084 / −0.007 forward (E67, 2021–2024: 2020 has a single training year). The bands are too wide in calm years and too narrow in 2022.

A post-hoc comparison on the E66 predictions (leave one year out over the out-of-sample errors, the same small leak for every variant) favours errors **relative to the predicted price, per zone, without price levels**: pinball skill 0.151 mean and +0.038 worst, against 0.069 / −0.067 for the E66 method. The levels cut the data too thin. A relative band carries over from cheap to expensive years. This is now the `Banded` default, tested properly in E78/E79.

## Round 8: the daily level and seed bagging (2026-10-29)

An oracle check on `report/prices.parquet` (cheating with the observed price) bounds what a level correction could reach on top of `lgbm`, as mean skill over the six years: perfect zone-year level 0.23, zone-month 0.36, zone-day 0.54. Most of the remaining error is in the level of each day, so this round tries to give the model the day.

| id | change | year | forward |
|---|---|---|---|
| E54/E55 | default (13) | 0.157 / 0.084 | 0.104 / −0.069 |
| E68/E69 | + daily mean of margin with imports, renewable share, firm utilisation, gas price-setter share | 0.161 / 0.074 | 0.103 / −0.074 |
| E70/E71 | + share of zones where gas sets the price that hour | 0.144 / 0.061 | 0.078 / −0.093 |
| E72/E73 | `lgbm_bag`, the mean of five seeds | 0.158 / 0.069 | 0.106 / −0.066 |

- **Daily summaries of the network do not carry the daily level.** The model already sees the hours of the day; averaging them adds nothing.
- **The Europe gas-setter share hurts**: it mostly encodes the fuel regime of each year.
- **Seed bagging is within the noise** at five times the cost.
- Nothing changes the defaults. With the oracle bounds above, a 50 % cut of the MAE is out of reach by correcting the backcast as it is; the backcast itself has to get closer first (round 9).

## Round 9: 2022 from the backcast with the fixes of issue #26 (2026-10-29)

The 2022 backcast was rerun upstream (pypsa-eur-backcast `feat/backcast-2022`, commit 492689d6: monthly coal, lignite, biomass and waste availability, daily gas, reservoir hold-back). Its raw median MAE falls from 92.4 to 59.4 and gas sets the model price in 61 % of hours instead of 28 %. The table is in `data/tables-issue26/`, the other five years are unchanged.

Skill is given against the raw price of the same tables ("own") and against the original raw backcast, which shows the upstream and the correction gain together.

| id | 2022 | features | year, own | year, vs original | forward, own | forward, vs original |
|---|---|---|---|---|---|---|
| E54/E55 | old | default | 0.157 / 0.084 | 0.157 / 0.084 | 0.104 / −0.069 | 0.104 / −0.069 |
| E62/E63 | old | + `spread_europe` | 0.157 / 0.092 | 0.157 / 0.092 | 0.114 / −0.010 | 0.114 / −0.010 |
| E74/E75 | issue #26 | default | 0.106 / −0.042 | **0.168** / 0.088 | 0.048 / −0.131 | **0.129** / −0.069 |
| E76/E77 | issue #26 | + `spread_europe` | 0.097 / −0.072 | 0.161 / 0.070 | 0.042 / −0.229 | 0.130 / −0.010 |

MAE of the corrected price per held-out year, year split: old 2022 `6.6, 7.2, 20.8, 80.2, 19.3, 19.6`; issue #26 `7.2, 7.0, 23.4, 61.9, 18.5, 20.0`.

- **The upstream fix is worth more than any correction tried here**: the 2022 MAE drops from 80 (corrected) to 59 (raw).
- **The correction now makes 2022 worse** (59.4 raw, 61.9 corrected, 67.2 trained on 2019–2021 only): the other years still teach it to raise prices that the fixed 2022 no longer underprices.
- **2021 loses most of its gain** (corrected 20.8 → 23.4). Its fourth quarter underprices like the old 2022, which used to teach the correction. Fixed and unfixed years in one training set contradict each other.
- **`spread_europe` only helped through the old 2022** and does not become a default.
- Next: retrain once every year is rerun with the same upstream fixes, so the correction learns from consistent backcasts.

## Round 10: relative price bands (2026-10-30)

`Banded` now takes the errors relative to the predicted price (at least 20 EUR/MWh), per zone, with no price levels, as the post-hoc comparison of round 7 suggested.

| | year (E66 → E78) | forward (E67 → E79) |
|---|---|---|
| pinball skill, mean / worst | 0.108 / −0.131 → **0.151 / 0.031** | 0.084 / −0.007 → **0.169 / 0.155** |

E78, year split:

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| inside 10–90 % | 0.94 | 0.83 | 0.85 | 0.57 | 0.82 | 0.78 |
| width 10–90 %, EUR/MWh | 38 | 29 | 80 | 126 | 79 | 61 |
| Brier > 150, bands / point | – | – | 0.051 / 0.060 | 0.132 / 0.168 | 0.065 / 0.091 | 0.040 / 0.053 |

E79, forward split (2021–2024): inside 0.64, 0.61, 0.80, 0.78; Brier 0.063 / 0.091, 0.116 / 0.151, 0.071 / 0.092, 0.040 / 0.053.

- **The bands beat the point price in every held-out year on both splits.** Coverage is near the nominal 80 % in normal years.
- **They miss upwards in regime changes**: 2022 (34 % of hours above the band) and 2021 going forward (29 %), the same years where the point correction fails.
- **The probability of a price above 150 EUR/MWh from the bands beats the yes/no of the point price in every year.**
- `ppf report` now writes the 10 and 90 % quantiles to `prices.parquet` and their scores to `band_scores.csv`, and the dashboard shows the band per zone and its calibration per year.

## Round 11: simpler trees (2026-10-30)

Round 1 picked the tree size on the month split. On held-out years:

| id | trees | year | forward |
|---|---|---|---|
| E80/E81 | 15 leaves, ≥ 500 rows per leaf, 300 trees | 0.121 / 0.037 | 0.079 / −0.081 |
| E82/E83 | 31 leaves, ≥ 200 rows per leaf, 500 trees | 0.142 / 0.057 | 0.092 / −0.065 |
| E54/E55 | default: 63 leaves, ≥ 50 rows per leaf, 800 trees | **0.157 / 0.084** | **0.104 / −0.069** |

Simpler trees carry over to unseen years worse, not better: the 70 % shrink already hedges, and the trees need the detail. Round 12 tries larger ones.

## Round 12: larger trees (2026-10-30)

| id | trees | year | forward |
|---|---|---|---|
| E84/E85 | 127 leaves, ≥ 25 rows per leaf, 1200 trees | 0.150 / 0.067 | 0.106 / −0.070 |
| E54/E55 | default | **0.157 / 0.084** | 0.104 / −0.069 |

Worse on the year split, even forward. **The default tree size is the best of the four sizes** on held-out years.

## Round 13: the backcast runs of 2026-11-02 (2026-11-02)

All six years were rerun upstream on the same code (pypsa-eur-backcast with the issue #26 branch). The switches of issue #26 are only on for 2022: monthly coal, lignite, biomass and waste availability, daily gas and country-specific coal prices. The other five years are unchanged within 0.05 EUR/MWh of raw median MAE. 2022 falls from 92.4 to 57.6.

From this round on, `data/tables` holds these runs, extracted with `experiments/extract_set.sh` against the backcast's own `day_ahead_prices.csv` (identical to the ENTSO-E prices of the earlier tables). The tables of rounds 2–12 are in `data/tables-2026-10-29`.

### Point correction

| id | training | year | year, vs original raw | forward |
|---|---|---|---|---|
| E54/E55 | old tables | 0.157 / 0.084 | 0.157 / 0.084 | 0.104 / −0.069 |
| E86/E87 | new tables | 0.090 / −0.117 | 0.161 / 0.072 | 0.037 / −0.145 |
| E90/E91 | new tables without 2022 | 0.150 / 0.068 | – | 0.082 / −0.077 |

Corrected MAE per held-out year, E86: `7.0, 7.0, 22.5, 64.3, 19.5, 20.7` (raw `7.9, 9.4, 25.7, 57.6, 21.0, 22.9`).

Skill on the five other years, year split:

| training | 2019 | 2020 | 2021 | 2023 | 2024 | mean |
|---|---|---|---|---|---|---|
| old (unfixed) 2022 | 0.159 | 0.230 | 0.190 | 0.084 | 0.144 | 0.162 |
| fixed 2022 | 0.117 | 0.249 | 0.123 | 0.070 | 0.097 | 0.131 |
| no 2022 | 0.192 | 0.277 | 0.100 | 0.068 | 0.111 | 0.150 |

On the forward split the four other years score 0.082 with and without the fixed 2022.

- **The correction makes the fixed 2022 worse** (57.6 raw, 64.3 corrected): the other years still teach it to raise prices that 2022 no longer underprices.
- **The fixed 2022 also hurts the other years** in training (0.150 → 0.131 on the year split). The old 2022 helped them, as it shares the underpricing of 2021.
- **2021 is the inconsistent year now**: in its fourth quarter the model makes 40 TWh of gas against 130 in ENTSO-E and prices 37 EUR/MWh too low, the pattern the 2022 fix removed. 2024 makes 0.74 of the ENTSO-E gas.
- **The report leaves 2022 out** (`ppf report --exclude-year 2022`, in the `report` task) until the issue #26 switches cover the other years; then 2022 goes back in.

### Price bands

| id | tables | year | forward |
|---|---|---|---|
| E78/E79 | old tables | 0.151 / 0.031 | 0.169 / 0.155 |
| E88/E89 | new tables | **0.190 / 0.066** | **0.190 / 0.140** |

Pinball skill, mean / worst. With the fixed 2022, its band holds 71 % of hours (57 % before) and 10 % fall above it (34 %). The bands gain where the point correction loses, as the relative band follows the level of the new 2022.

## Round 14: does the correction need more years? (2026-11-02)

`experiments/learning_curve.py` predicts each of 2019, 2020, 2021, 2023 and 2024 (the round 13 tables, 2022 left out) from every subset of one to four of the other years: 75 fits with the default `lgbm`. Results per fit in `experiments/results_learning_curve.csv`.

Skill per held-out year, mean over the training subsets of each size:

| training years | 2019 | 2020 | 2021 | 2023 | 2024 | mean | worst |
|---|---|---|---|---|---|---|---|
| 1 | −0.107 | 0.079 | 0.048 | 0.010 | 0.064 | 0.019 | −0.107 |
| 2 | 0.057 | 0.206 | 0.084 | 0.040 | 0.092 | 0.096 | 0.040 |
| 3 | 0.104 | 0.238 | 0.100 | 0.060 | 0.131 | 0.127 | 0.060 |
| 4 | 0.178 | 0.267 | 0.088 | 0.060 | 0.130 | 0.145 | 0.060 |

- **The gain per added year shrinks**: +0.077, +0.031, +0.018 on the mean, nothing on the worst year from three to four. Extrapolated, a fifth and sixth year of the same quality add about 0.01–0.02 each.
- **Which years matters more than how many.** The best three-year subset beats all four years for 2019 (0.192), 2021 (0.149), 2023 (0.093) and 2024 (0.139). Adding 2023 to 2019+2020+2024 lowers 2021 from 0.149 to 0.088, and adding 2023 to 2020+2021+2024 lowers 2019 from 0.192 to 0.178. A year whose backcast errs differently teaches the wrong correction.
- 2021 and 2023 stay at 0.06–0.10 whatever the training: their errors are not shared by any other year.
- More years of the current backcast will not get near 0.5. Years with consistent upstream fixes (issue #26 on every year) should come first; 2025 is then the most useful new year, as the closest to future runs.

## Round 15: variables from outside the network (2026-11-02)

Does anything the network does not know track the error the correction leaves? `experiments/exogenous_check.py` first correlated the out-of-sample residual of the report (observed − `lgbm`, mean per zone and day) with candidate daily variables (Spearman):

| variable | daily | within zone-year | zone-month |
|---|---|---|---|
| temperature (population weighted, ERA5 cutout) | 0.215 | 0.167 | 0.292 |
| temperature anomaly vs the six years | −0.043 | −0.019 | −0.068 |
| runoff of the past 90 days / 1991–2020 normal | −0.116 | −0.142 | −0.136 |
| runoff of the past 365 days / normal | −0.017 | −0.079 | −0.003 |
| gas cost change over 30 days | 0.187 | 0.142 | 0.229 |
| gas volatility over 30 days | −0.056 | −0.039 | −0.097 |
| CO2 price change over 30 days | 0.080 | 0.054 | 0.086 |

Temperature works through the season (its anomaly carries nothing), the rest is weak but plausible: wet quarters lower prices beyond what PyPSA's hydro dispatch gives, rising gas lifts them. The candidates became features, with the daily data built by the new `ppf exogenous`:

- `temperature`, `hdd`: population-weighted daily mean temperature per bus region from the ERA5 cutout of the backcast.
- `runoff_90d`: ERA5 runoff per country over the past 90 days against its 1991–2020 normal (PyPSA-Eur's runoff series, extended to 2024 with the cutout), capped at 5 for countries with hardly any runoff (NL, DK, LU).
- `reservoir_anomaly`: weekly hydro reservoir filling of the country from ENTSO-E (16.1.D, 20 countries), as share of its 2015–2018 maximum minus the 2015–2018 median of the same week, lagged one week as known at the auction.
- `gas_change_30d`, `gas_vol_30d`: from the gas cost of the table.

Each was added alone to the 13 default features, 2022 left out. The baseline is the default rerun on the same tables (E116/E117 and seeds E102–E105): E90/E91 ran on an earlier 2024 table without AL00, ME00, MK00 and XK00, so their raw MAE differs (22.94 instead of 24.07) and they are not comparable.

| features | year mean / worst | forward mean / worst |
|---|---|---|
| default, seeds 0–2 (E116, E102, E104 / E117, E103, E105) | 0.144 / 0.060 | 0.087 / −0.074 |
| + gas change (E92/E93) | 0.139 / 0.045 | 0.089 / −0.013 |
| + gas volatility (E94/E95) | 0.136 / 0.067 | 0.085 / −0.042 |
| + temperature (E96/E97) | 0.160 / 0.042 | 0.095 / −0.066 |
| + runoff (E98/E99) | 0.150 / 0.073 | 0.083 / −0.027 |
| + reservoir, 2015–2024 normal, seeds 0–2 (E100, E106, E108 / E101, E107, E109) | 0.175 / 0.069 | 0.091 / −0.025 |
| **+ reservoir, 2015–2018 normal (E112/E113)** | **0.176 / 0.073** | **0.092 / −0.029** |
| + reservoir, runoff, temperature (E114/E115) | 0.172 / 0.063 | 0.106 / −0.017 |

Skill per held-out year, year split:

| | 2019 | 2020 | 2021 | 2023 | 2024 |
|---|---|---|---|---|---|
| default (E116) | 0.178 | 0.267 | 0.088 | 0.060 | 0.130 |
| + reservoir (E112) | 0.272 | 0.281 | 0.110 | 0.073 | 0.144 |

- **Reservoir filling is the one clear gain**: +0.031 on the year mean, better in every held-out year, the same over three seeds (spread ±0.002 with the 2015–2024 normal), and it more than halves the worst forward year (2021, −0.074 → −0.029). It is the water value PyPSA misses: with perfect foresight its reservoirs release at near-zero cost, while the market prices hydro by how full the reservoirs are. 2019 gains most, 0.178 → 0.272.
- The reservoir normal of E100–E111 included the held-out year; the 2015–2018 normal, before any backcast year, gives the same skill, so the gain is not a leak.
- Gas momentum and volatility do not carry over: the default features already see the gas cost through the model price.
- Temperature lifts the mean but lowers the worst year, as a season proxy it lets trees memorise years. Runoff is about the noise on its own and adds nothing on top of reservoirs.
- All three weather variables together are the best forward mean (0.106) but below reservoirs alone on the year split. Reservoirs alone is the simpler choice.
- For a scenario run the observed filling is unknown; `reservoir_anomaly` = 0 (a normal year) is the neutral input. The model's own hydro state of charge would be the scenario-proof version (needs a new table column).

### The day-ahead forecasts the market cleared on

`ppf forecasts` downloads the ENTSO-E day-ahead load forecast (6.1.B) and wind and solar forecast (14.1.D) per area, hourly, 2019–2024 (16 area-years unpublished). `load_forecast_error` = forecast / model load − 1 and `vres_forecast_error` = (forecast − available model wind and solar) / model load, per zone. `experiments/forecast_check.py` finds no correlation with the residual of the report: Spearman −0.002 / −0.007 hourly, 0.008 / −0.011 within zone-year, −0.005 / −0.009 per zone and day.

In training they still help, on top of reservoirs:

| features | year mean / worst | forward mean / worst |
|---|---|---|
| default + reservoir (E112/E113) | 0.176 / 0.073 | 0.092 / −0.029 |
| + both forecast errors, seeds 0–2 (E118, E120, E122 / E119, E121, E123) | 0.184 / 0.091 | 0.102 / −0.020 |
| + load forecast error (E124/E125) | 0.182 / 0.089 | 0.096 / −0.018 |
| + wind and solar forecast error (E126/E127) | 0.182 / 0.090 | 0.093 / −0.034 |
| **placebo**: each bus keeps its median forecast error of all years (E128/E129) | 0.184 / 0.090 | 0.102 / −0.028 |

- **The gain is the zone, not the forecast.** The placebo, a constant per bus, scores as well as the hourly forecasts in every held-out year. The trees use how far the model's load and wind and solar sit from ENTSO-E's in each zone (ITCA load +53 %, SE02 wind and solar +49 % of load), a numeric zone descriptor, which unlike the `bus` category (E46/E47) can rank zones.
- The hourly forecast errors carry nothing beyond that, so they are not worth a download that only exists for past years. The zone offsets point at upstream calibration (zone extents and capacities in the backcast) and are better fixed there.

### Round 15 conclusions

- Use `reservoir_anomaly` (+0.032 year mean, every held-out year better, worst forward year −0.074 → −0.029). It is not a default yet because it needs `ppf exogenous` data, which the report and dashboard would need too.
- Of the literature's usual price drivers, the weather (temperature, runoff), the fuel momentum and the day-ahead forecasts add nothing once reservoirs are in, or only through zone identity.
- Still far from 0.5: the remaining error is mostly the level per zone and year that no outside variable here explains.

## Round 16: every year on one identical upstream config (2026-11-03)

The backcast reran 2019–2025 on one model config, keeping only per-year facts (branch `exp/backcast-uniform-config` of pypsa-eur-backcast), in two variants: `uniform`, the plain base model, and `recipe`, the 2022 recipe on every year (monthly availability caps, daily gas, PL coal price). `experiments/extract_set.sh` now takes a variant and extracted them into `data/tables-uniform` and `data/tables-recipe`; `data/tables-master` is the master set plus its 2025 run. Raw median MAE:

| set | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| master | 7.89 | 9.36 | 25.68 | 57.57 | 20.96 | 24.07 | 25.01 |
| uniform | 7.89 | 9.36 | 25.69 | 92.30 | 21.31 | 22.84 | 24.94 |
| recipe | 9.98 | 11.16 | 24.91 | 57.76 | 25.27 | 25.63 | 24.11 |

Default `lgbm`, skill against the raw MAE of the same set (E130–E141), and with reservoir filling (E142–E147):

| set | year mean / worst | forward mean / worst |
|---|---|---|
| master | 0.122 / −0.068 | 0.064 / −0.132 |
| uniform | 0.140 / 0.056 | 0.088 / −0.077 |
| recipe | 0.187 / −0.009 | 0.160 / 0.009 |
| recipe + reservoir | 0.203 / −0.005 | 0.168 / −0.021 |
| master, no 2022 | 0.146 / 0.079 | 0.097 / −0.077 |
| master, no 2022, + reservoir | 0.173 / 0.080 | 0.104 / −0.029 |
| uniform, no 2022 | 0.139 / 0.081 | 0.086 / −0.077 |
| recipe, no 2022 | 0.218 / 0.125 | 0.186 / 0.131 |
| **recipe, no 2022, + reservoir** | **0.233 / 0.127** | **0.202 / 0.127** |

Skill flatters the recipe, whose raw prices are too high in normal years: part of its skill is the correction taking that bias out again. The fair comparison is the corrected MAE itself. `ppf evaluate` per set (`results/round16/`), median over the buses all three sets score (master also scores PL00 in 2020 and AL00, ME00, MK00, XK00 in 2023–2024), mean over years:

| | year, all | forward, all | year, no 2022 | forward, no 2022 |
|---|---|---|---|---|
| master raw | 24.20 | 26.92 | 18.64 | 20.79 |
| master + correction | 22.72 | 27.04 | 16.40 | 19.57 |
| uniform + correction | 25.23 | 29.54 | 16.41 | 19.58 |
| recipe + correction | **22.21** | **24.84** | **16.31** | **18.46** |

- **The recipe set gives the best corrected prices** in every column. Its gain is in the crisis years: 2021 forward 21.32 against 27.66 on master, 2022 57.24 against 65.19. In normal years master + correction stays 0.3–0.8 better (2019, 2020, 2023).
- **With the recipe the correction carries over**: every held-out year improves in both splits, the forward worst is +0.009 with 2022 and +0.131 without, where master never got above −0.077. A correction trained on years that err alike is the consistency round 14 asked for.
- On master, the correction learned on earlier years makes the forward split worse overall (27.04 against raw 26.92), because nothing before 2021 teaches it a gas crisis.
- `uniform` is master with a broken 2022 (gas under-dispatch, 92.30): no gain, as the backcast's own comparison found.
- Reservoir filling adds +0.015 to +0.027 on the year mean of every set. With 2022 on the forward split it costs 2022 (57.24 → 58.99, trained on 2019–2021 only).
- Best so far: recipe tables with reservoir filling, 0.233 / 0.202 mean skill without 2022. Against master's raw MAE the corrected recipe prices (default features) are 8 % (year) and 8 % (forward) better with 2022, 12.5 % and 11 % without.
