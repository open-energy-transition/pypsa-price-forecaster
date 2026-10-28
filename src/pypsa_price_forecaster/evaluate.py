"""
Out-of-sample evaluation of the models by cross-validation over years or weeks.

The rows are split into folds and every fold is predicted by models fitted on
all other folds, so every row gets an out-of-sample prediction:

- ``year``: one fold per year. No hour of the predicted year is seen in
  training, which measures how the correction carries over to an unseen year
  with its own fuel and carbon prices.
- ``forward``: one fold per year, fitted only on the earlier years, as when the
  model corrects a future run. The first year is not predicted and not scored.
- ``week``: every ``folds``-th week of the year (weeks counted from 1 January)
  forms a fold across all years. Every season and year is in training and test,
  which works with a single year but is optimistic, as the fitted correction has
  seen the price level of the neighbouring weeks.
- ``month``: one fold per calendar month across all years. Contiguous blocks
  leak less of the price level than interleaved weeks, but the fitted
  correction has still seen the rest of the year.

The metrics per bus follow pypsa-eur-backcast (mean absolute error, root mean
squared error, Pearson correlation and mean bias over hours with observed price)
and the summary takes the median over buses per model and year. The scores add
the error in the tails, where the backcast misses spikes and negative prices,
and the spread of the predicted prices.
"""

import logging

import numpy as np
import pandas as pd

from pypsa_price_forecaster.dataset import TARGET
from pypsa_price_forecaster.models import make_model

logger = logging.getLogger(__name__)


SPLITS = ["year", "forward", "week", "month"]


def fold_labels(table: pd.DataFrame, split: str, folds: int = 4) -> pd.Series:
    """Return the fold of every row of `table` for the `split`, see the module docstring."""
    if split in ("year", "forward"):
        return table.year
    if split == "week":
        week = (table.snapshot.dt.day_of_year - 1) // 7
        return week % folds
    if split == "month":
        return table.snapshot.dt.month
    raise ValueError(f"Unknown split {split}, available: {SPLITS}")


def cross_validate(
    table: pd.DataFrame,
    models: list[str],
    features: list[str] | None = None,
    split: str = "year",
    folds: int = 4,
    params: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Return the out-of-sample price of every row of `table`, one column per model, with estimator `params` per model.

    Rows of the first year of the ``forward`` split have no prediction.
    """
    labels = fold_labels(table, split, folds)
    if labels.nunique() < 2:
        raise ValueError(
            f"Splitting by {split} needs at least two folds, got {sorted(labels.unique())}. "
            "With a single year, split by week."
        )
    predictions = pd.DataFrame(np.nan, index=table.index, columns=models)
    fold_values = sorted(labels.unique())
    if split == "forward":
        fold_values = fold_values[1:]
    for label in fold_values:
        test = labels == label
        train = labels < label if split == "forward" else ~test
        for name in models:
            logger.info(f"Predicting {split} fold {label} with {name}.")
            model = make_model(name, features, (params or {}).get(name))
            model = model.fit(table[train])
            predictions.loc[test, name] = model.predict(table[test])
    return predictions


def scored_rows(
    table: pd.DataFrame, columns: list[str], predictions: pd.DataFrame
) -> pd.DataFrame:
    """Return `columns` of `table` joined with `predictions` over the rows with observed and predicted price."""
    return (
        table[columns]
        .join(predictions)
        .dropna(subset=[TARGET])
        .dropna(subset=list(predictions), how="all")
    )


def bus_metrics(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the metrics per model, year and bus over the rows with observed and predicted price."""
    rows = scored_rows(table, ["year", "bus", TARGET], predictions)
    metrics = {}
    for model in predictions:
        error = rows[model] - rows[TARGET]
        grouped = rows.assign(
            predicted=rows[model],
            error=error,
            abs_error=error.abs(),
            sq_error=error**2,
        ).groupby(["year", "bus"])
        metrics[model] = pd.DataFrame(
            {
                "mae": grouped.abs_error.mean(),
                "rmse": np.sqrt(grouped.sq_error.mean()),
                "r": grouped[["predicted", TARGET]].apply(
                    lambda g: g.predicted.corr(g[TARGET])
                ),
                "bias": grouped.error.mean(),
                "hours": grouped.error.size(),
            }
        )
    return pd.concat(metrics, names=["model"])


def summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the median over buses of MAE, correlation and absolute bias per model and year."""
    return (
        metrics.assign(abs_bias=metrics.bias.abs())
        .groupby(["model", "year"], sort=False)
        .agg(
            mae=("mae", "median"),
            r=("r", "median"),
            abs_bias=("abs_bias", "median"),
            buses=("mae", "size"),
        )
    )


def tail_mae(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the median over buses of the MAE in the top and bottom 5 % observed hours per model."""
    rows = scored_rows(table, ["bus", TARGET], predictions)
    high = rows[TARGET] >= rows.groupby("bus")[TARGET].transform("quantile", 0.95)
    low = rows[TARGET] <= rows.groupby("bus")[TARGET].transform("quantile", 0.05)
    result = {}
    for model in predictions:
        error = (rows[model] - rows[TARGET]).abs()
        result[model] = {
            "mae_top5": error[high].groupby(rows.bus[high]).mean().median(),
            "mae_low5": error[low].groupby(rows.bus[low]).mean().median(),
        }
    return pd.DataFrame(result).T


def scores(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row of scores per model over all years.

    Columns are the median over buses of MAE, r and |bias|, the tail MAEs of
    :func:`tail_mae`, the ratio of the standard deviation of predicted to
    observed prices and the share of hours at or below zero.
    """
    metrics = bus_metrics(table, predictions)
    rows = scored_rows(table, [TARGET], predictions)
    result = (
        metrics.assign(abs_bias=metrics.bias.abs())
        .groupby("model", sort=False)
        .agg(mae=("mae", "median"), r=("r", "median"), abs_bias=("abs_bias", "median"))
        .join(tail_mae(table, predictions))
    )
    result["std_ratio"] = [rows[m].std() / rows[TARGET].std() for m in result.index]
    result["share_le_0"] = [(rows[m] <= 0).mean() for m in result.index]
    return result
