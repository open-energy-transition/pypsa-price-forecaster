# pypsa-price-forecaster

> [!WARNING]
> Work in progress. Interfaces, features and models may change without notice.

Converts the marginal prices of a solved [PyPSA](https://pypsa.org) network into day-ahead electricity prices with a machine learning correction.

A cost-based dispatch model such as [pypsa-eur-backcast](https://github.com/open-energy-transition/pypsa-eur-backcast) gets the merit order right but misses parts of the market price.
It produces no bid markups at price spikes, few hours at or below zero and level biases per bidding zone.
This package learns that gap, the observed minus the model price, from the state of the modelled system in each hour.
It is trained on historical backcast runs and scored on years it has not seen.

## Install

```bash
pixi install
pixi run test
```

The CLI is `ppf` (long name `pypsa-price-forecaster`). Run it with `pixi run ppf --help`.

## Workflow

The input is a solved network from pypsa-eur-backcast and, for training, the observed prices it builds:

- `results/backcast-<year>/networks/solved_<year>.nc`
- `resources/backcast-<year>/day_ahead_prices.csv`: naive UTC hours by bus, from ENTSO-E

```bash
# 1. one table per run, with observed prices for training years
ppf extract ../pypsa-eur-backcast/results/backcast-2019/networks/solved_2019.nc \
    --prices ../pypsa-eur-backcast/resources/backcast-2019/day_ahead_prices.csv \
    --out data/tables/backcast-2019.parquet --run backcast-2019
ppf extract ...2023... --out data/tables/backcast-2023.parquet --run backcast-2023

# 2. out-of-sample score: every year predicted by models fitted on the other years
ppf evaluate data/tables/*.parquet --out-dir results/evaluation

# 3. fit on all years and predict any solved network, also a future one without observed prices
ppf train data/tables/*.parquet --out models/lgbm.joblib
ppf predict path/to/solved_2030.nc --model models/lgbm.joblib --out results/prices_2030.csv
```

`evaluate` writes `summary.csv` (median over buses per model and year of MAE, Pearson r and absolute bias, as in the backcast README), `bus_metrics.csv` and `predictions.parquet`.
`predict` writes the same wide CSV format as `day_ahead_prices.csv`, so the backcast evaluation can score it directly.

The same steps are available in Python:

```python
import pypsa_price_forecaster as ppf

table = ppf.build_table("solved_2023.nc", "day_ahead_prices.csv")
model = ppf.make_model("lgbm").fit(train_table)
prices = model.predict(table)
```

## Inputs from the network

`extract` reads, per hour and AC bus:

| column | meaning |
|---|---|
| `model_price` | marginal price of the bus |
| `load` | load `p_set` |
| `supply_firm` | `p_nom × p_max_pu` of generators that are not variable renewables, without load shedding |
| `supply_vres` | the same for wind, solar and run of river |
| `supply_storage` | `p_nom × p_max_pu` of storage units |
| `import_capacity` | hourly capacity of lines and links into the zone, both link directions from `p_max_pu` and `p_min_pu` |

Quantities are summed per market zone and every bus gets the values of its zone.
Buses joined by branches of infinite capacity (the copperplate regions of PyPSA-Eur, such as DE00 and LUG1) form one zone.

The table only holds quantities from the network.
No observed data enters as feature, so the models also apply to future and scenario runs.

## Features

| feature | definition |
|---|---|
| `model_price` | marginal price of the bus |
| `margin_domestic` | (supply − load) / load |
| `margin_with_imports` | (supply + import capacity − load) / load |
| `margin_domestic_mw`, `margin_with_imports_mw` | the same margins in MW |
| `hour`, `dayofweek`, `month` | calendar of the UTC hour |

The defaults are all except the MW margins. Select features with repeated `--feature`.
To add a feature, add a column in `network.py` if needed and an entry in `FEATURES` in `dataset.py`.

## Models

All models predict the model price plus a correction.

| model | correction |
|---|---|
| `pypsa` | none, the raw marginal price |
| `bus_bias` | median error per bus in the training years |
| `linear` | Huber regression on the features |
| `lgbm` | LightGBM with absolute error loss on the features |

A learned model is worth its complexity only if it beats `bus_bias` out of sample.

## Evaluation

The score is leave-one-year-out: every year is predicted by models fitted on all other years.
The years differ in fuel and carbon prices, so this measures how well the correction carries over to an unseen market regime.
With only two years this is one fit per direction, so the score gets more reliable with every year added.

## Roadmap

- Temperature per bidding zone from the atlite cutout (ERA5) of the run
- More backcast years, including the crisis years 2021 and 2022
- Features scaled by fuel and carbon cost, so the correction transfers across price levels
- Local time calendar features, holidays

## License

MIT
