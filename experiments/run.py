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

from pypsa_price_forecaster.dataset import (
    DEFAULT_FEATURES,
    TARGET,
    attach_exogenous,
    read_tables,
)
from pypsa_price_forecaster.evaluate import (
    band_scores,
    cross_validate,
    cross_validate_bands,
    scores,
)

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
    # gas-scaled target with the gas cost clipped to the training range (E32 overshot 2021)
    "E40_year_gas_clip": dict(
        split="year", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_gas_clip"]
    ),
    "E41_forward_gas_clip": dict(
        split="forward", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_gas_clip"]
    ),
    # 70 % of the lgbm correction, the share chosen on the other years in 0.5 to 0.8
    "E42_year_shrink": dict(
        split="year", features=DEFAULT_FEATURES, models=["pypsa", "lgbm"]
    ),
    "E43_forward_shrink": dict(
        split="forward", features=DEFAULT_FEATURES, models=["pypsa", "lgbm"]
    ),
    # every 5th week of all years held out, as a mixed split to compare with the year split
    "E35_week5_default": dict(
        split="week", folds=5, features=DEFAULT_FEATURES, models=LGBM
    ),
}
# Round 4: simpler feature sets with lgbm applying 70 % of its correction
LEAN = [f for f in DEFAULT_FEATURES if f not in ("margin_domestic", "vres_share")]
EXPERIMENTS |= {
    "E44_year_lean": dict(split="year", features=LEAN, models=LGBM),
    "E45_forward_lean": dict(split="forward", features=LEAN, models=LGBM),
    "E46_year_bus": dict(
        split="year", features=DEFAULT_FEATURES + ["bus"], models=LGBM
    ),
    "E47_forward_bus": dict(
        split="forward", features=DEFAULT_FEATURES + ["bus"], models=LGBM
    ),
}
# Round 5: price setter and congestion from the solved network, and 2022 left out
SETTER = DEFAULT_FEATURES + ["price_setter"]
CONGESTION = DEFAULT_FEATURES + ["imports_congested", "exports_congested"]
EXPERIMENTS |= {
    "E48_year_setter": dict(split="year", features=SETTER, models=LGBM),
    "E49_forward_setter": dict(split="forward", features=SETTER, models=LGBM),
    "E50_year_congestion": dict(split="year", features=CONGESTION, models=LGBM),
    "E51_forward_congestion": dict(split="forward", features=CONGESTION, models=LGBM),
    "E52_year_no2022": dict(
        split="year", features=DEFAULT_FEATURES, models=LGBM, exclude_years=[2022]
    ),
    "E53_forward_no2022": dict(
        split="forward", features=DEFAULT_FEATURES, models=LGBM, exclude_years=[2022]
    ),
}
NETWORK = DEFAULT_FEATURES + ["price_setter", "imports_congested", "exports_congested"]
EXPERIMENTS |= {
    "E54_year_setter_congestion": dict(split="year", features=NETWORK, models=LGBM),
    "E55_forward_setter_congestion": dict(
        split="forward", features=NETWORK, models=LGBM
    ),
}
# Round 6: literature-grounded attempts at the tail (HANDOVER.md, experiments/NOTES.md)
# all unshrunk (shrink=1.0), so the shrink factor can be swept for free afterwards
# from the stored full correction, the same trick used to pick SHRINK in round 3
EXPERIMENTS |= {
    "E56_year_spike_weighted": dict(
        split="year", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_spike_weighted"]
    ),
    "E57_forward_spike_weighted": dict(
        split="forward",
        features=DEFAULT_FEATURES,
        models=["pypsa", "lgbm_spike_weighted"],
    ),
    "E58_year_asinh": dict(
        split="year", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_asinh"]
    ),
    "E59_forward_asinh": dict(
        split="forward", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_asinh"]
    ),
    "E60_year_q90": dict(
        split="year", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_q90"]
    ),
    "E61_forward_q90": dict(
        split="forward", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_q90"]
    ),
}
# Round 7: the two features the data check ranked first (no re-extraction), and
# price bands from the out-of-sample error quantiles of lgbm
SPREAD = DEFAULT_FEATURES + ["spread_europe"]
ABSOLUTE = dict(relative=False, by_level=True)
VRES_WEEK = DEFAULT_FEATURES + ["vres_share_week"]
EXPERIMENTS |= {
    "E62_year_spread": dict(split="year", features=SPREAD, models=LGBM),
    "E63_forward_spread": dict(split="forward", features=SPREAD, models=LGBM),
    "E64_year_vres_week": dict(split="year", features=VRES_WEEK, models=LGBM),
    "E65_forward_vres_week": dict(split="forward", features=VRES_WEEK, models=LGBM),
    # errors per zone and level of the predicted price, in EUR/MWh
    "E66_year_bands": dict(split="year", features=DEFAULT_FEATURES, bands=ABSOLUTE),
    "E67_forward_bands": dict(
        split="forward", features=DEFAULT_FEATURES, bands=ABSOLUTE
    ),
}
# Round 8: the daily level, where an oracle check puts most of the remaining error
DAILY = DEFAULT_FEATURES + [
    "margin_with_imports_day",
    "vres_share_day",
    "firm_utilisation_day",
    "gas_setter_day",
]
GAS_EUROPE = DEFAULT_FEATURES + ["gas_setter_europe"]
EXPERIMENTS |= {
    "E68_year_daily": dict(split="year", features=DAILY, models=LGBM),
    "E69_forward_daily": dict(split="forward", features=DAILY, models=LGBM),
    "E70_year_gas_europe": dict(split="year", features=GAS_EUROPE, models=LGBM),
    "E71_forward_gas_europe": dict(split="forward", features=GAS_EUROPE, models=LGBM),
    "E72_year_bag": dict(
        split="year", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_bag"]
    ),
    "E73_forward_bag": dict(
        split="forward", features=DEFAULT_FEATURES, models=["pypsa", "lgbm_bag"]
    ),
}
# Round 9: 2022 from the backcast run with the fixes of issue #26 (monthly fossil
# availability, daily gas, reservoir hold-back), other years unchanged
ISSUE26 = {"backcast-2022": Path("data/tables-issue26/backcast-2022.parquet")}
EXPERIMENTS |= {
    "E74_year_issue26": dict(
        split="year", features=DEFAULT_FEATURES, models=LGBM, tables=ISSUE26
    ),
    "E75_forward_issue26": dict(
        split="forward", features=DEFAULT_FEATURES, models=LGBM, tables=ISSUE26
    ),
    "E76_year_issue26_spread": dict(
        split="year", features=SPREAD, models=LGBM, tables=ISSUE26
    ),
    "E77_forward_issue26_spread": dict(
        split="forward", features=SPREAD, models=LGBM, tables=ISSUE26
    ),
}
# errors per zone relative to the predicted price, best in a post-hoc comparison
# on the E66 predictions (experiments/NOTES.md)
EXPERIMENTS |= {
    "E78_year_bands_relative": dict(split="year", features=DEFAULT_FEATURES, bands={}),
    "E79_forward_bands_relative": dict(
        split="forward", features=DEFAULT_FEATURES, bands={}
    ),
}
# Round 11: simpler trees, never tested on held-out years (E13 used the month split)
SMALL = {
    "num_leaves": 15,
    "min_child_samples": 500,
    "n_estimators": 300,
    "learning_rate": 0.05,
}
MEDIUM = {"num_leaves": 31, "min_child_samples": 200, "n_estimators": 500}
EXPERIMENTS |= {
    "E80_year_small": dict(
        split="year", features=DEFAULT_FEATURES, models=LGBM, params=SMALL
    ),
    "E81_forward_small": dict(
        split="forward", features=DEFAULT_FEATURES, models=LGBM, params=SMALL
    ),
    "E82_year_medium": dict(
        split="year", features=DEFAULT_FEATURES, models=LGBM, params=MEDIUM
    ),
    "E83_forward_medium": dict(
        split="forward", features=DEFAULT_FEATURES, models=LGBM, params=MEDIUM
    ),
}
# Round 12: larger trees, as small (E80) < medium (E82) < default (E54) on held-out years
LARGER = {"num_leaves": 127, "min_child_samples": 25, "n_estimators": 1200}
EXPERIMENTS |= {
    "E84_year_larger": dict(
        split="year", features=DEFAULT_FEATURES, models=LGBM, params=LARGER
    ),
    "E85_forward_larger": dict(
        split="forward", features=DEFAULT_FEATURES, models=LGBM, params=LARGER
    ),
}
# Round 13: all six years rerun upstream with the same fixes (issue #26 switches),
# extracted with experiments/extract_set.sh into data/tables-consistent
CONSISTENT = "data/tables-consistent"
EXPERIMENTS |= {
    "E86_year_consistent": dict(
        split="year", features=DEFAULT_FEATURES, table_dir=CONSISTENT
    ),
    "E87_forward_consistent": dict(
        split="forward", features=DEFAULT_FEATURES, table_dir=CONSISTENT
    ),
    "E88_year_bands_consistent": dict(
        split="year", features=DEFAULT_FEATURES, bands={}, table_dir=CONSISTENT
    ),
    "E89_forward_bands_consistent": dict(
        split="forward", features=DEFAULT_FEATURES, bands={}, table_dir=CONSISTENT
    ),
}
# without 2022, now the only year with the issue #26 fixes upstream
EXPERIMENTS |= {
    "E90_year_consistent_no2022": dict(
        split="year",
        features=DEFAULT_FEATURES,
        table_dir=CONSISTENT,
        exclude_years=[2022],
    ),
    "E91_forward_consistent_no2022": dict(
        split="forward",
        features=DEFAULT_FEATURES,
        table_dir=CONSISTENT,
        exclude_years=[2022],
    ),
}
# Round 15: variables from outside the network, built with `ppf exogenous` into
# data/exogenous.parquet (temperature, runoff, reservoir filling) or computed
# from the gas cost of the table (gas momentum); 2022 left out as in E90/E91
EXOGENOUS = "data/exogenous.parquet"
ROUND15 = {
    "gas_change": ["gas_change_30d"],
    "gas_vol": ["gas_vol_30d"],
    "temperature": ["temperature"],
    "runoff": ["runoff_90d"],
    "reservoir": ["reservoir_anomaly"],
}
for i, (key, extra) in enumerate(ROUND15.items()):
    for j, split in enumerate(["year", "forward"]):
        EXPERIMENTS[f"E{92 + 2 * i + j}_{split}_{key}"] = dict(
            split=split,
            features=DEFAULT_FEATURES + extra,
            models=LGBM,
            exogenous=EXOGENOUS,
            exclude_years=[2022],
        )
# seed noise of round 15, and the weather variables together
ROUND15B = {
    "consistent": ([], None),
    "reservoir": (["reservoir_anomaly"], EXOGENOUS),
}
for i, (key, (extra, exogenous)) in enumerate(ROUND15B.items()):
    for j, seed in enumerate([1, 2]):
        for k, split in enumerate(["year", "forward"]):
            EXPERIMENTS[f"E{102 + 4 * i + 2 * j + k}_{split}_{key}_seed{seed}"] = dict(
                split=split,
                features=DEFAULT_FEATURES + extra,
                models=LGBM,
                exclude_years=[2022],
                params={"random_state": seed},
            ) | ({"exogenous": exogenous} if exogenous else {"table_dir": CONSISTENT})
for k, split in enumerate(["year", "forward"]):
    EXPERIMENTS[f"E{110 + k}_{split}_weather"] = dict(
        split=split,
        features=DEFAULT_FEATURES + ["reservoir_anomaly", "runoff_90d", "temperature"],
        models=LGBM,
        exogenous=EXOGENOUS,
        exclude_years=[2022],
    )
# E92-E111 used a reservoir normal of 2015-2024, which includes the held-out
# year (data/exogenous_allyears_normal.parquet); from here on the normal is
# 2015-2018, before any backcast year
for k, split in enumerate(["year", "forward"]):
    EXPERIMENTS[f"E{112 + k}_{split}_reservoir_past_normal"] = dict(
        split=split,
        features=DEFAULT_FEATURES + ["reservoir_anomaly"],
        models=LGBM,
        exogenous=EXOGENOUS,
        exclude_years=[2022],
    )
    EXPERIMENTS[f"E{114 + k}_{split}_weather_past_normal"] = dict(
        split=split,
        features=DEFAULT_FEATURES + ["reservoir_anomaly", "runoff_90d", "temperature"],
        models=LGBM,
        exogenous=EXOGENOUS,
        exclude_years=[2022],
    )
# E90/E91 again: they ran before the 2024 table was extracted anew, so the
# baseline of round 15 is this and E102-E105 on the same tables
for k, split in enumerate(["year", "forward"]):
    EXPERIMENTS[f"E{116 + k}_{split}_consistent_seed0"] = dict(
        split=split,
        features=DEFAULT_FEATURES,
        models=LGBM,
        table_dir=CONSISTENT,
        exclude_years=[2022],
    )
# the day-ahead forecasts the market cleared on (`ppf forecasts`) on top of reservoirs
FORECASTS = "data/forecasts.parquet"
for k, split in enumerate(["year", "forward"]):
    EXPERIMENTS[f"E{118 + k}_{split}_reservoir_forecasts"] = dict(
        split=split,
        features=DEFAULT_FEATURES
        + ["reservoir_anomaly", "load_forecast_error", "vres_forecast_error"],
        models=LGBM,
        exogenous=[EXOGENOUS, FORECASTS],
        exclude_years=[2022],
    )
for i, seed in enumerate([1, 2]):
    for k, split in enumerate(["year", "forward"]):
        EXPERIMENTS[f"E{120 + 2 * i + k}_{split}_reservoir_forecasts_seed{seed}"] = (
            EXPERIMENTS[f"E{118 + k}_{split}_reservoir_forecasts"]
            | {"params": {"random_state": seed}}
        )
for i, forecast in enumerate(["load_forecast_error", "vres_forecast_error"]):
    for k, split in enumerate(["year", "forward"]):
        EXPERIMENTS[f"E{124 + 2 * i + k}_{split}_reservoir_{forecast[:4]}"] = dict(
            split=split,
            features=DEFAULT_FEATURES + ["reservoir_anomaly", forecast],
            models=LGBM,
            exogenous=[EXOGENOUS, FORECASTS],
            exclude_years=[2022],
        )
# placebo: each bus keeps only its median forecast error over all years, so
# the forecasts can only tell zones apart (data/forecasts_placebo.parquet)
for k, split in enumerate(["year", "forward"]):
    EXPERIMENTS[f"E{128 + k}_{split}_reservoir_forecasts_placebo"] = EXPERIMENTS[
        f"E{118 + k}_{split}_reservoir_forecasts"
    ] | {"exogenous": [EXOGENOUS, "data/forecasts_placebo.parquet"]}
# Round 16: every year 2019-2025 rerun upstream on one identical config, as
# the plain base model (uniform) and with the 2022 recipe on every year
# (recipe), against master with its 2025 run (exp/backcast-uniform-config)
TABLE_SETS = {
    "master": "data/tables-master",
    "uniform": "data/tables-uniform",
    "recipe": "data/tables-recipe",
}
for i, (key, table_dir) in enumerate(TABLE_SETS.items()):
    for j, excluded in enumerate([[], [2022]]):
        for k, split in enumerate(["year", "forward"]):
            name = f"{split}_{key}" + ("_no2022" if excluded else "")
            EXPERIMENTS[f"E{130 + 4 * i + 2 * j + k}_{name}"] = dict(
                split=split,
                features=DEFAULT_FEATURES,
                models=LGBM,
                table_dir=table_dir,
                exclude_years=excluded,
            )
# reservoir filling on the round 16 sets (data/exogenous.parquet now 2019-2025)
for i, (key, excluded) in enumerate(
    [("master", [2022]), ("recipe", []), ("recipe", [2022])]
):
    for k, split in enumerate(["year", "forward"]):
        name = f"{split}_{key}" + ("_no2022" if excluded else "") + "_reservoir"
        EXPERIMENTS[f"E{142 + 2 * i + k}_{name}"] = dict(
            split=split,
            features=DEFAULT_FEATURES + ["reservoir_anomaly"],
            models=LGBM,
            table_dir=TABLE_SETS[key],
            exogenous=EXOGENOUS,
            exclude_years=excluded,
        )
# Round 17: the model's own hydro reservoirs instead of the observed filling, on the
# recipe tables with 2022, against E138/E139 (none) and E144/E145 (observed)
MODEL_RESERVOIRS = "data/reservoirs_model.parquet"
for i, added in enumerate(
    [
        ["reservoir_share_model"],
        ["reservoir_anomaly_model"],
        ["inflow_90d_model"],
        ["reservoir_anomaly_model", "inflow_90d_model"],
    ]
):
    for k, split in enumerate(["year", "forward"]):
        name = f"{split}_recipe_" + "_".join(a.removesuffix("_model") for a in added)
        EXPERIMENTS[f"E{148 + 2 * i + k}_{name}"] = dict(
            split=split,
            features=DEFAULT_FEATURES + added,
            models=LGBM,
            table_dir=TABLE_SETS["recipe"],
            exogenous=MODEL_RESERVOIRS,
        )
# Round 18: on the recipe tables with 2022 (data/tables, the report), against
# E138/E139: temperature, the gas-scaled corrections, and bands per price level
for k, split in enumerate(["year", "forward"]):
    EXPERIMENTS[f"E{156 + k}_{split}_recipe_temperature"] = dict(
        split=split,
        features=DEFAULT_FEATURES + ["temperature"],
        models=LGBM,
        exogenous=EXOGENOUS,
    )
    for i, model in enumerate(["lgbm_gas", "lgbm_gas_clip"]):
        EXPERIMENTS[f"E{158 + 2 * i + k}_{split}_recipe_{model}"] = dict(
            split=split, features=DEFAULT_FEATURES, models=["pypsa", model]
        )
EXPERIMENTS |= {
    "E162_year_bands_recipe_level": dict(
        split="year", features=DEFAULT_FEATURES, bands={"by_level": True}
    ),
    "E163_forward_bands_recipe": dict(
        split="forward", features=DEFAULT_FEATURES, bands={}
    ),
    "E164_forward_bands_recipe_level": dict(
        split="forward", features=DEFAULT_FEATURES, bands={"by_level": True}
    ),
}
# seed noise of E138/E139 and E156/E157: is the temperature gain real?
for i, (key, extra) in enumerate(
    {"recipe": [], "recipe_temperature": ["temperature"]}.items()
):
    for j, seed in enumerate([1, 2]):
        for k, split in enumerate(["year", "forward"]):
            EXPERIMENTS[f"E{165 + 4 * i + 2 * j + k}_{split}_{key}_seed{seed}"] = dict(
                split=split,
                features=DEFAULT_FEATURES + extra,
                models=LGBM,
                params={"random_state": seed},
            ) | ({"exogenous": EXOGENOUS} if extra else {})
MODELS = ["pypsa", "bus_bias", "linear", "lgbm"]
RESULTS_BY_YEAR = Path("experiments/results_by_year.csv")
RESULTS_BANDS = Path("experiments/results_bands.csv")
COLUMNS = ["mae", "r", "abs_bias", "mae_top5", "mae_low5"]


def spec_table(spec: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Return the table of an experiment: all runs of `table_dir`, or `table` with runs replaced by `tables`."""
    if "table_dir" in spec:
        return read_tables(sorted(Path(spec["table_dir"]).glob("*.parquet")))
    if "tables" in spec:
        return read_tables([spec["tables"].get(path.stem, path) for path in TABLES])
    return table


def run(name: str, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the scores of experiment `name` over all years and per year."""
    spec = {"folds": 4, "models": MODELS} | EXPERIMENTS[name]
    table = spec_table(spec, table)
    table = table[~table.year.isin(spec.get("exclude_years", []))]
    exogenous = spec.get("exogenous", [])
    for path in [exogenous] if isinstance(exogenous, str) else exogenous:
        table = attach_exogenous(table, pd.read_parquet(path))
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


def run_bands(name: str, table: pd.DataFrame) -> pd.DataFrame:
    """Return the band scores per held-out year of experiment `name`, keeping the bands in ``results/``."""
    spec = EXPERIMENTS[name]
    table = spec_table(spec, table)
    start = time.time()
    bands = cross_validate_bands(
        table, "lgbm", spec["features"], spec["split"], **spec["bands"]
    )
    seconds = round(time.time() - start)
    Path("results").mkdir(exist_ok=True)
    table[["run", "year", "snapshot", "bus", TARGET]].join(bands).dropna(
        subset=["point"]
    ).to_parquet(f"results/bands_{name}.parquet")
    return (
        band_scores(table, bands)
        .round(3)
        .reset_index()
        .assign(experiment=name, split=spec["split"], seconds=seconds)
    )


if __name__ == "__main__":
    table = read_tables(TABLES)
    for name in sys.argv[1:]:
        if "bands" in EXPERIMENTS[name]:
            result = run_bands(name, table)
            print(f"\n== {name}")
            print(result.drop(columns="experiment").to_string(index=False))
            skill = 1 - result.pinball / result.pinball_point
            print("mean:", {"pinball_skill": round(skill.mean(), 3)})
            print("worst:", {"pinball_skill": round(skill.min(), 3)})
            result.to_csv(
                RESULTS_BANDS, mode="a", header=not RESULTS_BANDS.exists(), index=False
            )
            continue
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
