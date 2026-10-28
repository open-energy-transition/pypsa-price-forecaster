"""
Runs cross-validation experiments on the extracted tables and appends the scores to ``experiments/results.csv``.

Usage: ``pixi run python experiments/run.py <experiment> [<experiment> ...]``

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
MODELS = ["pypsa", "bus_bias", "linear", "lgbm"]


def run(name: str, table: pd.DataFrame) -> pd.DataFrame:
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
    result = scores(table, predictions)[
        ["mae", "r", "abs_bias", "mae_top5", "mae_low5"]
    ]
    return (
        result.round(3)
        .reset_index(names="model")
        .assign(
            experiment=name,
            split=spec["split"],
            features=",".join(spec["features"]),
            params=str(spec.get("params", {})),
            seconds=round(time.time() - start),
        )
    )


if __name__ == "__main__":
    table = read_tables(TABLES)
    for name in sys.argv[1:]:
        result = run(name, table)
        print(f"\n== {name}")
        print(result.drop(columns=["experiment", "features"]).to_string(index=False))
        result.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
