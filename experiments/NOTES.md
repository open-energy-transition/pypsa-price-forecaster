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
