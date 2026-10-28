# pypsa-price-forecaster

> [!WARNING]
> Work in progress. Interfaces may change.

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
| `hour`, `dayofweek` | UTC calendar |

These are the defaults, chosen in [experiments/NOTES.md](experiments/NOTES.md).
Also available, but not used by default: `month`, `bus`, `cost_gas`, `cost_coal`, `price_to_gas`, `price_daily_mean`, `price_deviation`, and the margins in MW.

Available supply is `p_nom × p_max_pu` of generators (without load shedding) and storage units.
Import capacity is the hourly capacity of the NTC links into the zone.
Copperplated buses (e.g. DE00 + LUG1) count as one zone.
All features come from the network alone, so the model also works for future runs.

| model | prediction |
|---|---|
| `pypsa` | raw marginal price (baseline) |
| `bus_bias` | marginal price + median error per bus (baseline) |
| `linear` | marginal price + Huber regression on features |
| `lgbm` | marginal price + LightGBM (absolute error) on features |

To add a feature, add an entry to `FEATURES` in `dataset.py`, plus a column in `network.py` if it needs new data from the network.

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
- hourly market, PyPSA and corrected prices;
- `history.csv`, with one row per report.

It uses the year split once there are several years, otherwise the month split.

```bash
pixi run report                    # ppf report data/tables/*.parquet, ~45 min per year of tables
pixi run report --no-ablation      # quick check without the feature impact
pixi run -e dashboard dashboard    # streamlit run dashboard/app.py
```

The dashboard (`dashboard/app.py`) shows `report/` and the experiment log.
Deploy it on Streamlit Community Cloud from the branch `master` with main file `dashboard/app.py`. It installs only `dashboard/requirements.txt`.
It then always shows the report committed on master. Rerun `pixi run report` and commit `report/` whenever the code, defaults or tables change.
A test fails if `report/` was written with other default features or LightGBM parameters than the code has.
Set `PPF_REPORT` to show another report directory.

## License

MIT
