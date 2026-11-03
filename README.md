# pypsa-price-forecaster

> [!WARNING]
> Work in progress. Interfaces may change. Picking this up fresh? Read [HANDOVER.md](HANDOVER.md) first.

Corrects the marginal prices of a solved PyPSA network (e.g. from [pypsa-eur-backcast](https://github.com/open-energy-transition/pypsa-eur-backcast)) to day-ahead prices.
A model learns the gap between observed and model price from hourly features of the network.

## Install

```bash
pixi install
pixi run test
pixi run ppf --help
```

## Usage

```bash
export ENTSOE_API_TOKEN=...   # only for downloading observed prices

# 1. extract one table per solved network, with observed prices from ENTSO-E
pixi run ppf extract path/to/solved_2019.nc --entsoe --run backcast-2019 -o data/tables/backcast-2019.parquet
pixi run ppf extract path/to/solved_2023.nc --entsoe --run backcast-2023 -o data/tables/backcast-2023.parquet

# 2. score out of sample
pixi run ppf evaluate data/tables/*.parquet -o results/eval                       # hold out each year
pixi run ppf evaluate data/tables/*.parquet -o results/eval-week --split week     # hold out every 4th week

# 3. train on everything and predict any solved network, including future ones
pixi run ppf train data/tables/*.parquet -o models/lgbm.joblib
pixi run ppf predict path/to/solved_2030.nc -m models/lgbm.joblib -o results/prices_2030.csv
```

- `extract --prices file.csv` uses a local CSV instead of ENTSO-E (naive UTC hours × buses, like the backcast's `day_ahead_prices.csv`).
- `evaluate` prints and writes `summary.csv` (median over buses of MAE, r and |bias| per model and year), `bus_metrics.csv` and `predictions.parquet`.
- `predict` writes a CSV of naive UTC hours × buses.
- Choose models with `-m` and features with `-f` (repeatable). See `--help` of each command.

## Observed prices

- Downloaded per ENTSO-E bidding zone and year, cached in `data/entsoe`, averaged to hourly. Gaps of up to 3 h are interpolated; longer gaps stay missing.
- Zones without prices or not in EUR (GB, and PL/RO/BG in 2019) are skipped and get a `.skip` file in the cache. Delete it to retry.
- Buses map to zones via the backcast's `entsoe_area_to_bus.csv`, which ships with the package; override it with `--area-to-bus`. FR15 (Corsica) is excluded.

## Features and models

| feature | definition |
|---|---|
| `model_price` | marginal price of the bus |
| `margin_domestic` | (available supply − load) / load |
| `margin_with_imports` | (available supply + import capacity − load) / load |
| `vres_share` | available wind, solar and run of river / load |
| `curtailment_share` | (available − dispatched variable renewables) / load |
| `firm_utilisation` | dispatched / available other generators |
| `net_import_share` | solved net import / load |
| `storage_share` | storage unit dispatch / load |
| `price_setter` | technology of the partly loaded generators whose marginal cost sets the zone price, else storage or import |
| `imports_congested`, `exports_congested` | share of the import or export capacity on links running at their limit |
| `hour`, `dayofweek` | UTC calendar |

These are the defaults, chosen in [experiments/NOTES.md](experiments/NOTES.md).
Also available, but not used by default: `month`, `bus`, `cost_gas`, `cost_coal`, `price_to_gas`, `price_daily_mean`, `price_deviation`, `hour_local`, `dayofweek_local`, `holiday`, and the margins in MW.

Available supply is `p_nom × p_max_pu` of generators (without load shedding) and storage units.
Import capacity is the hourly capacity of the NTC links into the zone.
Copperplated buses (e.g. DE00 + LUG1) count as one zone.
All features come from the network alone, so the model also works for future runs.

| model | prediction |
|---|---|
| `pypsa` | raw marginal price (baseline) |
| `bus_bias` | marginal price + median error per bus (baseline) |
| `linear` | marginal price + Huber regression on features |
| `lgbm` | marginal price + 70 % of a LightGBM (absolute error) correction on features |
| `lgbm_full` | marginal price + the whole LightGBM correction |
| `lgbm_gas` | as `lgbm_full`, with the error learnt relative to the gas plant cost |

Applying 70 % of the correction hedges against corrections that do not carry over to an unseen year: it beats raw PyPSA in every held-out year, while the whole correction is worse in 2023 ([experiments/NOTES.md](experiments/NOTES.md)).

### Price bands

`Banded` (`models.py`) adds a 10 to 90 % band, and every 5 % quantile, around the `lgbm` price. The band comes from the errors of `lgbm` in years it did not see, relative to its predicted price, per zone. It beats the point price on pinball loss in every held-out year, and its probability of a price above 150 EUR/MWh beats the point price ([experiments/NOTES.md](experiments/NOTES.md), rounds 7, 10 and 13). It misses upwards in regime changes such as 2022.

To add a feature, add an entry to `FEATURES` in `dataset.py`, plus a column in `network.py` if it needs new data from the network.

### Variables from outside the network

Some features need daily or hourly data the network does not hold. Build them once, then pass them with `--variables` to `evaluate`, `train`, `predict` and `report`. The default features include `reservoir_anomaly`, so the default model needs `--variables data/exogenous.parquet`:

```bash
# daily temperature and runoff from the backcast's ERA5 cutouts, hydro reservoir filling from ENTSO-E
pixi run ppf exogenous /path/to/pypsa-eur-backcast --year 2019 --year 2024 -o data/exogenous.parquet
# hourly day-ahead load and wind and solar forecasts from ENTSO-E
pixi run ppf forecasts --year 2019 --year 2024 -o data/forecasts.parquet

pixi run ppf evaluate data/tables/*.parquet -o results/eval --variables data/exogenous.parquet \
    -f model_price ... -f reservoir_anomaly
```

| feature | definition | source |
|---|---|---|
| `reservoir_anomaly` | hydro reservoir filling of the country as share of its 2015–2018 maximum, minus the 2015–2018 median of the same week, lagged a week | ENTSO-E 16.1.D |
| `runoff_90d` | ERA5 runoff of the country over the past 90 days / its 1991–2020 normal, at most 5 | backcast `data/country_runoff`, cutouts |
| `temperature`, `hdd` | population-weighted daily mean temperature of the bus region, heating degrees below 15.5 °C | backcast cutout |
| `gas_change_30d`, `gas_vol_30d` | change and volatility of the daily gas cost over 30 days | table |
| `load_forecast_error` | day-ahead load forecast of the zone / model load − 1 | ENTSO-E 6.1.B |
| `vres_forecast_error` | (day-ahead wind and solar forecast − available model wind and solar) / model load | ENTSO-E 14.1.D |

`reservoir_anomaly` is the one that helps on held-out years, and is a default feature; the forecast errors help only as a constant per zone (rounds 15 and 16 in [experiments/NOTES.md](experiments/NOTES.md)). Reservoir filling and the forecasts are only known for past years: for a scenario run, take `reservoir_anomaly` = 0, a normal year.

## Evaluation splits

- `--split year`: train on the other years. This is the real test, and it becomes meaningful once 3 or more years are available.
- `--split month`: leave one calendar month out. This is the most honest test within a single year.
- `--split week --folds K`: every K-th week of all years is one fold. It scores optimistically, because neighbouring weeks are in training.

Experiments and their results are logged in [experiments/](experiments/).

## Report and dashboard

`ppf report` scores the default models on all tables and writes small files to `report/`, which is committed:

- scores per model and year, per zone and per month;
- feature impact: the default LightGBM refitted without each of its features and with each other feature, plus two seed repeats that give the noise level;
- split gain per feature;
- hourly market, PyPSA and corrected prices, with the 10 and 90 % quantiles of the price band, and the band's calibration per year;
- `history.csv`, with one row per report.

It uses the year split once there are several years, otherwise the month split.

```bash
pixi run report                    # ppf report data/tables/*.parquet --variables data/exogenous.parquet, about 2 h
pixi run report --no-ablation      # quick check without the feature impact
pixi run report --no-bands         # without the price bands, which refit lgbm per pair of years
pixi run -e dashboard dashboard    # streamlit run dashboard/app.py
```

The dashboard (`dashboard/app.py`) shows `report/` and the experiment log.
Deploy it on Streamlit Community Cloud from the branch `master` with main file `dashboard/app.py`. It installs only `dashboard/requirements.txt`.
It then always shows the report committed on master. Rerun `pixi run report` and commit `report/` whenever the code, defaults or tables change.
The tables are the backcast runs of 2019 to 2025 on one identical configuration, the 2022 recipe of pypsa-eur-backcast (monthly availability and the other issue #26 fixes for every year). With every year on the same fixes, 2022 is back in training and scoring ([experiments/NOTES.md](experiments/NOTES.md), round 16).

To refresh the tables and the reservoir filling from a backcast checkout, with the observed prices the backcast scores itself against:

```bash
experiments/extract_set.sh ../pypsa-eur-backcast data/tables "$(seq -s ' ' 2019 2025)" recipe
pixi run ppf exogenous ../pypsa-eur-backcast $(printf -- '--year %s ' $(seq 2019 2025))
```
A test fails if `report/` was written with other default features or LightGBM parameters than the code has.
Set `PPF_REPORT` to show another report directory.

## License

MIT
