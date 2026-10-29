# Roadmap

Ideas to improve the correction, drafted on 2026-10-29 after Round 2 (six years, 2019–2024).
Judge every idea on the `year` and `forward` splits, never on `month` or `week` alone.
The status of each idea is in [NOTES.md](NOTES.md).

## Diagnosis

The model learns the hourly shape of the error but not its level.

- In 2023 it raises r from 0.752 to 0.791. But |bias| rises from 6.7 to 10.3 and the std ratio from 0.81 to 1.06, so MAE ends slightly worse (21.03 → 21.47).
- On the forward split it hurts 2021 (MAE 25.7 → 28.0, |bias| 7.8 → 17.3) and 2023.

A correction in EUR/MWh, learnt in one price regime, does not carry over to the next.

## Top five, by expected gain per effort

1. **Target scaled by the gas plant cost** (`lgbm_gas`).
   Fuel costs as features fail across years (20.98 vs 19.30), because trees cannot extrapolate to 2022 gas costs. Scaling the error turns extrapolation into a multiplication. A bid markup plausibly scales with fuel cost.
2. **Evaluation:**
   - mean and worst skill over years (1 − MAE / raw MAE) instead of the median over zone-years, which hides 2022;
   - 3 or more seeds;
   - forward-2024 kept as a final holdout that is not used to choose between models.
3. **Shrinkage towards no correction outside the training range, plus monotone constraints.** These target the forward split, which is worse than raw (24.23 vs 22.94).
4. **Local time and public holidays per zone.** Hour is the strongest feature (+1.19 MAE when removed), and it is in UTC.
5. **Price-setting technology and congestion.** The carrier that sets the price, import links at capacity, model price spread to neighbours, link shadow prices.
   These target the structural errors: German exports, Finland too cheap. Needs changes to `network.py` and re-extracting every year.

## Further ideas

- **Tails:** quantile LightGBM (0.1, 0.5, 0.9) or an L2 model for high-price hours, plus conformal intervals. Top-5 % MAE barely moves (187 → 174).
- **Seed bagging and an ensemble with a linear model on the scaled target.** This reduces the seed spread, and the linear model extrapolates where trees flatten out.
- **Intraday shape and duration features:** rank of residual load within the day, ramps, rolling 24 h and 168 h model price, hours since the margin was last tight, hydro state of charge.
- **Sample weights:** each zone-year equal, or 1 / raw MAE of its year, so 2022 does not dominate the gradients.
- **More years:** 2015–2018 for other fuel and carbon regimes, and 2025 as a true forward test. With only six regimes, this is the biggest lever.
- **Upstream fixes** in the backcast: daily gas prices, negative prices, exports. The correction cannot learn 2022's level (+71 EUR/MWh bias) from the other years.

## Not now

- **CatBoost or XGBoost:** expect results within noise of LightGBM. Use them only as ensemble members.
- **MLP, TCN, transformer or graph networks:** there are 2.2M rows but only six independent price regimes, and the error that hurts is the level of a year. Revisit with 10 or more years.
- **A separate crisis model:** there is a single crisis year, so it cannot be validated.
- **Bus identity and month:** zone biases do not persist across years (`bus_bias` is worse than raw in 2019 and 2021).

## Sources

- Uniejewski, Weron, Ziel (2018), variance stabilizing transformations for electricity spot price forecasting.
- Uniejewski (2025), VSTs in periods of increased volatility, arXiv 2511.13603.
- Ghelasi, Ziel (2025), a data-driven merit order, arXiv 2501.02963.
