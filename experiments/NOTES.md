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
