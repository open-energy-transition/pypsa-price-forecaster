"""
Sanity checks of the table of one backcast year against the metrics of the backcast.

Usage: ``pixi run python experiments/sanity.py <year>``, with the backcast repository
at ``../pypsa-eur-backcast`` or in the environment variable ``BACKCAST``.
"""

import os
import sys

import pandas as pd

from pypsa_price_forecaster.dataset import FEATURES
from pypsa_price_forecaster.evaluate import bus_metrics

year = sys.argv[1]
backcast = os.environ.get("BACKCAST", "../pypsa-eur-backcast")
t = pd.read_parquet(f"data/tables/backcast-{year}.parquet")
m = (
    bus_metrics(t, t[["model_price"]].rename(columns={"model_price": "pypsa"}))
    .loc["pypsa"]
    .droplevel("year")
)
ref = pd.read_csv(
    f"{backcast}/results/backcast-{year}/backcast/price_metrics.csv",
    index_col="bus",
)
cmp = pd.DataFrame(
    {
        "mae": m.mae,
        "mae_ref": ref.mae,
        "bias": m.bias,
        "bias_ref": ref.mean_bias,
        "r": m.r,
        "r_ref": ref.pearson_r,
    }
)
print(
    "max abs diff:",
    (cmp.mae - cmp.mae_ref).abs().max(),
    (cmp.bias - cmp.bias_ref).abs().max(),
    (cmp.r - cmp.r_ref).abs().max(),
)
print(
    "median MAE / r / |bias|:",
    round(m.mae.median(), 2),
    round(m.r.median(), 3),
    round(m.bias.abs().median(), 2),
)

print("\nzones with >1 bus:", sorted(z for z in t.zone.unique() if "+" in z))
print(
    "buses without observed price:",
    sorted(set(t.bus) - set(t.dropna(subset=["observed_price"]).bus)),
)
print(
    "\nmodel price quantiles:",
    t.model_price.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(1).to_dict(),
)
print("rows with model price >= 1000:", (t.model_price >= 1000).sum())
print(
    "observed quantiles:",
    t.observed_price.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(1).to_dict(),
)
for f in ["margin_domestic", "margin_with_imports"]:
    print(f, FEATURES[f](t).describe().round(2).to_dict())
print("\nper-zone mean MW:")
print(
    t.groupby("zone")[
        ["load", "supply_firm", "supply_vres", "supply_storage", "import_capacity"]
    ]
    .mean()
    .round(0)
    .sort_values("load", ascending=False)
    .head(12)
)
