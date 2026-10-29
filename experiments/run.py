"""
Runs cross-validation experiments on the extracted tables and appends the scores to ``experiments/results.csv``.

Usage: ``pixi run python experiments/run.py <experiment> [<experiment> ...]``

The scores per held-out year go to ``experiments/results_by_year.csv``.

Besides the median over buses of MAE, r and |bias|, the scores include the
median MAE in the hours of each bus above its 95th (``mae_top5``) and below its
5th (``mae_low5``) percentile of the observed price, where the backcast misses
spikes and negative prices.
"""

import sys
import time
from pathlib import Path

import pandas as pd

from pypsa_price_forecaster.dataset import DEFAULT_FEATURES, read_tables
from pypsa_price_forecaster.evaluate import cross_validate, scores

TABLES = sorted(Path("data/tables").glob("*.parquet"))
RESULTS = Path("experiments/results.csv")
NO_MONTH = [f for f in DEFAULT_FEATURES if f != "month"]

EXPERIMENTS = {
    "E01_week4_default": dict(split="week", features=DEFAULT_FEATURES),
    "E02_month_default": dict(split="month", features=DEFAULT_FEATURES),
    "E03_week4_no_month": dict(split="week", features=NO_MONTH),
    "E04_month_no_month": dict(split="month", features=NO_MONTH),
}
SYSTEM = [
    "vres_share",
    "curtailment_share",
    "firm_utilisation",
    "net_import_share",
    "storage_share",
]
COSTS = ["cost_gas", "cost_coal", "price_to_gas"]
SHAPE = ["price_daily_mean", "price_deviation"]
LGBM = ["pypsa", "lgbm"]
EXPERIMENTS |= {
    "E05_month_system": dict(split="month", features=NO_MONTH + SYSTEM, models=LGBM),
    "E06_month_system_costs": dict(
        split="month", features=NO_MONTH + SYSTEM + COSTS, models=LGBM
    ),
    "E07_month_system_costs_shape": dict(
        split="month", features=NO_MONTH + SYSTEM + COSTS + SHAPE, models=LGBM
    ),
    "E08_month_all_bus": dict(
        split="month", features=NO_MONTH + SYSTEM + COSTS + SHAPE + ["bus"], models=LGBM
    ),
    "E09_month_bus_only": dict(split="month", features=NO_MONTH + ["bus"], models=LGBM),
}
E05 = NO_MONTH + SYSTEM
EXPERIMENTS |= {
    # seed noise of E05
    "E10_E05_seed1": dict(
        split="month", features=E05, models=["lgbm"], params={"random_state": 1}
    ),
    "E11_E05_seed2": dict(
        split="month", features=E05, models=["lgbm"], params={"random_state": 2}
    ),
    # E05 plus bus identity
    "E12_E05_bus": dict(split="month", features=E05 + ["bus"], models=["lgbm"]),
    # regularisation and objectives on E05
    "E13_E05_small": dict(
        split="month",
        features=E05,
        models=["lgbm"],
        params={"num_leaves": 15, "min_child_samples": 500, "n_estimators": 300},
    ),
    "E14_E05_large": dict(
        split="month",
        features=E05,
        models=["lgbm"],
        params={
            "num_leaves": 63,
            "min_child_samples": 50,
            "n_estimators": 800,
            "learning_rate": 0.03,
        },
    ),
    "E15_E05_huber": dict(
        split="month",
        features=E05,
        models=["lgbm"],
        params={"objective": "huber", "alpha": 20.0},
    ),
    "E16_E05_l2": dict(
        split="month", features=E05, models=["lgbm"], params={"objective": "l2"}
    ),
}
LARGE = {
    "num_leaves": 63,
    "min_child_samples": 50,
    "n_estimators": 800,
    "learning_rate": 0.03,
}
MARGINS = ["margin_domestic", "margin_with_imports"]
EXPERIMENTS |= {
    # E14 without the capacity margins
    "E17_E14_no_margins": dict(
        split="month",
        features=[f for f in E05 if f not in MARGINS],
        models=["lgbm"],
        params=LARGE,
    ),
    # the initial idea alone: model price and capacity margins
    "E18_price_margins_only": dict(
        split="month", features=["model_price"] + MARGINS, models=["lgbm"], params=LARGE
    ),
}
# Round 2: several years
EXPERIMENTS |= {
    "E19_year_default": dict(split="year", features=DEFAULT_FEATURES),
    "E20_forward_default": dict(split="forward", features=DEFAULT_FEATURES),
    # seed noise of E19
    "E21_year_seed1": dict(
        split="year",
        features=DEFAULT_FEATURES,
        models=["lgbm"],
        params={"random_state": 1},
    ),
    "E22_year_seed2": dict(
        split="year",
        features=DEFAULT_FEATURES,
        models=["lgbm"],
        params={"random_state": 2},
    ),
    # features that leaked or extrapolated within 2023 alone
    "E23_year_costs": dict(
        split="year", features=DEFAULT_FEATURES + COSTS, models=LGBM
    ),
    "E24_forward_costs": dict(
        split="forward", features=DEFAULT_FEATURES + COSTS, models=LGBM
    ),
    "E25_year_bus": dict(
        split="year", features=DEFAULT_FEATURES + ["bus"], models=LGBM
    ),
    "E26_year_month": dict(
        split="year", features=DEFAULT_FEATURES + ["month"], models=LGBM
    ),
    "E27_year_price_margins_only": dict(
        split="year", features=["model_price"] + MARGINS, models=LGBM
    ),
    # the round 1 split on all years, to compare with the year split
    "E28_month_default": dict(split="month", features=DEFAULT_FEATURES, models=LGBM),
}
# Round 3: six years, roadmap in experiments/ROADMAP.md
LOCAL = [
    {"hour": "hour_local", "dayofweek": "dayofweek_local"}.get(f, f)
    for f in DEFAULT_FEATURES
] + ["holiday"]
GAS = ["pypsa", "lgbm_gas"]
SCALE_FREE = [f for f in DEFAULT_FEATURES if f != "model_price"] + ["price_to_gas"]
EXPERIMENTS |= {
    "E29_year_local": dict(split="year", features=LOCAL, models=LGBM),
    "E30_forward_local": dict(split="forward", features=LOCAL, models=LGBM),
    "E31_year_gas": dict(split="year", features=DEFAULT_FEATURES, models=GAS),
    "E32_forward_gas": dict(split="forward", features=DEFAULT_FEATURES, models=GAS),
    "E33_year_gas_local": dict(split="year", features=LOCAL, models=GAS),
    "E34_forward_gas_local": dict(split="forward", features=LOCAL, models=GAS),
    # without model price, which carries the price level of a year (ablation: -0.53 MAE when removed)
    "E36_year_scalefree": dict(split="year", features=SCALE_FREE, models=LGBM),
    "E37_forward_scalefree": dict(split="forward", features=SCALE_FREE, models=LGBM),
    "E38_year_scalefree_bus": dict(
        split="year", features=SCALE_FREE + ["bus"], models=LGBM
    ),
    "E39_forward_scalefree_bus": dict(
        split="forward", features=SCALE_FREE + ["bus"], models=LGBM
    ),
    # every 5th week of all years held out, as a mixed split to compare with the year split
    "E35_week5_default": dict(
        split="week", folds=5, features=DEFAULT_FEATURES, models=LGBM
    ),
}
MODELS = ["pypsa", "bus_bias", "linear", "lgbm"]
RESULTS_BY_YEAR = Path("experiments/results_by_year.csv")
COLUMNS = ["mae", "r", "abs_bias", "mae_top5", "mae_low5"]


def run(name: str, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the scores of experiment `name` over all years and per year."""
    spec = {"folds": 4, "models": MODELS} | EXPERIMENTS[name]
    start = time.time()
    predictions = cross_validate(
        table,
        spec["models"],
        spec["features"],
        spec["split"],
        spec["folds"],
        {"lgbm": spec.get("params", {})},
    )
    seconds = round(time.time() - start)
    predicted = predictions.notna().any(axis=1)
    by_year = {
        year: scores(table[rows], predictions[rows])[COLUMNS]
        for year in sorted(table.year.unique())
        if (rows := predicted & (table.year == year)).any()
    }
    labels = dict(
        experiment=name,
        split=spec["split"],
        features=",".join(spec["features"]),
        params=str(spec.get("params", {})),
        seconds=seconds,
    )
    result = scores(table, predictions)[COLUMNS].round(3).reset_index(names="model")
    by_year = (
        pd.concat(by_year, names=["year", "model"])
        .round(3)
        .reset_index()
        .assign(experiment=name, split=spec["split"])
    )
    return result.assign(**labels), by_year


if __name__ == "__main__":
    table = read_tables(TABLES)
    for name in sys.argv[1:]:
        result, by_year = run(name, table)
        print(f"\n== {name}")
        print(result.drop(columns=["experiment", "features"]).to_string(index=False))
        print(by_year.drop(columns="experiment").to_string(index=False))
        mae = by_year.pivot(index="year", columns="model", values="mae")
        if "pypsa" in mae:
            skill = 1 - mae.drop(columns="pypsa").div(mae.pypsa, axis=0)
            print("skill (1 - MAE / MAE pypsa):")
            print(skill.round(3).to_string())
            print("mean:", skill.mean().round(3).to_dict())
            print("worst:", skill.min().round(3).to_dict())
        result.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
        by_year.to_csv(
            RESULTS_BY_YEAR, mode="a", header=not RESULTS_BY_YEAR.exists(), index=False
        )
