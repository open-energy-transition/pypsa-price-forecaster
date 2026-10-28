"""
Out-of-sample evaluation of the models by cross-validation over years or weeks.

The rows are split into folds and every fold is predicted by models fitted on
all other folds, so every row gets an out-of-sample prediction:

- ``year``: one fold per year. No hour of the predicted year is seen in
  training, which measures how the correction carries over to an unseen year
  with its own fuel and carbon prices.
- ``week``: every ``folds``-th week of the year (weeks counted from 1 January)
  forms a fold across all years. Every season and year is in training and test,
  which works with a single year but is optimistic, as the fitted correction has
  seen the price level of the neighbouring weeks.
- ``month``: one fold per calendar month across all years. Contiguous blocks
  leak less of the price level than interleaved weeks, but the fitted
  correction has still seen the rest of the year.

The metrics per bus follow pypsa-eur-backcast (mean absolute error, root mean
squared error, Pearson correlation and mean bias over hours with observed price)
and the summary takes the median over buses per model and year.
"""

import logging

import numpy as np
import pandas as pd

from pypsa_price_forecaster.dataset import TARGET
from pypsa_price_forecaster.models import make_model

logger = logging.getLogger(__name__)


SPLITS = ["year", "week", "month"]


def fold_labels(table: pd.DataFrame, split: str, folds: int = 4) -> pd.Series:
    """Return the fold of every row of `table` for the `split`, see the module docstring."""
    if split == "year":
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
    """Return the out-of-sample price of every row of `table`, one column per model, with estimator `params` per model."""
    labels = fold_labels(table, split, folds)
    if labels.nunique() < 2:
        raise ValueError(
            f"Splitting by {split} needs at least two folds, got {sorted(labels.unique())}. "
            "With a single year, split by week."
        )
    predictions = pd.DataFrame(np.nan, index=table.index, columns=models)
    for label in sorted(labels.unique()):
        test = labels == label
        for name in models:
            logger.info(f"Predicting {split} fold {label} with {name}.")
            model = make_model(name, features, (params or {}).get(name))
            model = model.fit(table[~test])
            predictions.loc[test, name] = model.predict(table[test])
    return predictions


def bus_metrics(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the metrics per model, year and bus over the rows with observed price."""
    rows = table[["year", "bus", TARGET]].join(predictions).dropna(subset=[TARGET])
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
