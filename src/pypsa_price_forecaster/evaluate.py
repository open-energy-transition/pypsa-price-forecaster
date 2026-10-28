"""
Out-of-sample evaluation of the models by leaving one year out.

Every year in the table is predicted by models fitted on all other years, so
no hour of the predicted year is seen in training. The metrics per bus follow
pypsa-eur-backcast (mean absolute error, root mean squared error, Pearson
correlation and mean bias over hours with observed price) and the summary
takes the median over buses per model and year.
"""

import logging

import numpy as np
import pandas as pd

from pypsa_price_forecaster.dataset import TARGET
from pypsa_price_forecaster.models import make_model

logger = logging.getLogger(__name__)


def leave_one_year_out(
    table: pd.DataFrame, models: list[str], features: list[str] | None = None
) -> pd.DataFrame:
    """Return the out-of-sample price of every row of `table`, one column per model."""
    years = sorted(table.year.unique())
    if len(years) < 2:
        raise ValueError(f"Leaving one year out needs at least two years, got {years}")
    predictions = pd.DataFrame(np.nan, index=table.index, columns=models)
    for year in years:
        test = table.year == year
        for name in models:
            logger.info(f"Predicting {year} with {name} fitted on the other years.")
            model = make_model(name, features).fit(table[~test])
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
