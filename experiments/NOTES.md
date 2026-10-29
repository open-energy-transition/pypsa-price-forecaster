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
