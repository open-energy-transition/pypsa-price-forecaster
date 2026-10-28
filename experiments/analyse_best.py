"""
Analyses the best configuration so far (E14: E05 features, larger LightGBM) on 2023.

Prints per-bus and per-month improvements of the month split, a forward split
(train January to August, test September to December), the recovered share of
hours at or below zero and spread, and the feature importance. Writes the month
split predictions to ``results/E14_predictions.parquet``.
"""

import pandas as pd

from pypsa_price_forecaster.dataset import TARGET, read_tables
from pypsa_price_forecaster.evaluate import bus_metrics, cross_validate
from pypsa_price_forecaster.models import make_model

FEATURES = [
    "model_price",
    "margin_domestic",
    "margin_with_imports",
    "hour",
    "dayofweek",
    "vres_share",
    "curtailment_share",
    "firm_utilisation",
    "net_import_share",
    "storage_share",
]
PARAMS = {
    "num_leaves": 63,
    "min_child_samples": 50,
    "n_estimators": 800,
    "learning_rate": 0.03,
}
MODELS = ["pypsa", "bus_bias", "lgbm"]


def describe(table: pd.DataFrame, predictions: pd.DataFrame, title: str) -> None:
    metrics = bus_metrics(table, predictions)
    rows = table[[TARGET]].join(predictions).dropna(subset=[TARGET])
    summary = metrics.groupby("model", sort=False).agg(
        mae=("mae", "median"), r=("r", "median")
    )
    summary["std_ratio"] = [rows[m].std() / rows[TARGET].std() for m in summary.index]
    summary["share_le_0"] = [(rows[m] <= 0).mean() for m in summary.index]
    print(f"\n== {title} (observed share <= 0: {(rows[TARGET] <= 0).mean():.3f})")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    table = read_tables(["data/tables/backcast-2023.parquet"])
    params = {"lgbm": PARAMS}

    predictions = cross_validate(table, MODELS, FEATURES, "month", params=params)
    table[["snapshot", "bus", TARGET]].join(predictions).to_parquet(
        "results/E14_predictions.parquet"
    )
    describe(table, predictions, "month split")

    mae = bus_metrics(table, predictions).mae.unstack("model").droplevel("year")
    change = (mae.lgbm - mae.pypsa).sort_values()
    print(f"\nbuses improved by lgbm: {(change < 0).sum()} of {len(change)}")
    print("largest gains:", change.head(5).round(1).to_dict())
    print("largest losses:", change.tail(5).round(1).to_dict())

    rows = table[["snapshot", "bus", TARGET]].join(predictions).dropna(subset=[TARGET])
    monthly = (
        rows.assign(
            **{m: (rows[m] - rows[TARGET]).abs() for m in MODELS},
            month=rows.snapshot.dt.month,
        )
        .groupby(["month", "bus"])[MODELS]
        .mean()
        .groupby("month")
        .median()
    )
    print("\nmedian MAE over buses per held-out month:")
    print(monthly.round(1).T.to_string())

    train = table.snapshot.dt.month <= 8
    forward = pd.DataFrame(index=table.index[~train])
    for name in MODELS:
        model = make_model(name, FEATURES, params.get(name)).fit(table[train])
        forward[name] = model.predict(table[~train])
    describe(table[~train], forward, "forward split: train Jan-Aug, test Sep-Dec")

    full = make_model("lgbm", FEATURES, PARAMS).fit(table)
    booster = full.estimator.booster_
    gain = pd.Series(
        booster.feature_importance("gain"), index=booster.feature_name()
    ).sort_values(ascending=False)
    print("\nfeature importance (share of gain):")
    print((gain / gain.sum()).round(3).to_string())
