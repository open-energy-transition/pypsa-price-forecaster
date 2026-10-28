"""
Builds the tables the models train and predict on, and the features derived from them.

A table holds one row per snapshot and bus of one or more runs (solved
networks) with the columns of :data:`pypsa_price_forecaster.network.COLUMNS`,
the run name, the weather year of the run and, for training, the observed
day-ahead price. Features are computed from these columns on the fly, so a new
feature only needs an entry in ``FEATURES``.
"""

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from pypsa_price_forecaster.network import extract, to_long

logger = logging.getLogger(__name__)

TARGET = "observed_price"


def supply(table: pd.DataFrame) -> pd.Series:
    """Available domestic supply of generators and storage units in MW."""
    return table.supply_firm + table.supply_vres + table.supply_storage


def daily_mean_price(table: pd.DataFrame) -> pd.Series:
    """Mean model price of the bus over the UTC day of each row."""
    days = [table.run, table.bus, table.snapshot.dt.date]
    return table.groupby(days).model_price.transform("mean")


FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "model_price": lambda t: t.model_price,
    "margin_domestic": lambda t: (supply(t) - t.load) / t.load,
    "margin_with_imports": lambda t: (supply(t) + t.import_capacity - t.load) / t.load,
    "margin_domestic_mw": lambda t: supply(t) - t.load,
    "margin_with_imports_mw": lambda t: supply(t) + t.import_capacity - t.load,
    "vres_share": lambda t: t.supply_vres / t.load,
    "curtailment_share": lambda t: (t.supply_vres - t.dispatch_vres) / t.load,
    "firm_utilisation": lambda t: t.dispatch_firm / t.supply_firm,
    "net_import_share": lambda t: t.net_import / t.load,
    "storage_share": lambda t: t.storage_dispatch / t.load,
    "cost_gas": lambda t: t.cost_gas,
    "cost_coal": lambda t: t.cost_coal,
    "price_to_gas": lambda t: t.model_price / t.cost_gas,
    "price_daily_mean": daily_mean_price,
    "price_deviation": lambda t: t.model_price - daily_mean_price(t),
    "bus": lambda t: t.bus.astype("category"),
    "hour": lambda t: t.snapshot.dt.hour,
    "dayofweek": lambda t: t.snapshot.dt.day_of_week,
    "month": lambda t: t.snapshot.dt.month,
}
# best set on 2023 (experiments/NOTES.md); month leaks the price level of the year
DEFAULT_FEATURES = [
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


def feature_matrix(table: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Return the features of every table row, with infinite values (zero load) as missing."""
    unknown = set(features).difference(FEATURES)
    if unknown:
        raise ValueError(
            f"Unknown features {sorted(unknown)}, available: {list(FEATURES)}"
        )
    return pd.DataFrame({f: FEATURES[f](table) for f in features}).replace(
        [np.inf, -np.inf], np.nan
    )


def read_observed_prices(path: str | Path) -> pd.DataFrame:
    """
    Return observed hourly prices per bus as long table with columns ``snapshot``, ``bus`` and ``observed_price``.

    The CSV is wide with naive UTC timestamps as index and one column per bus,
    as ``resources/day_ahead_prices.csv`` of pypsa-eur-backcast.
    """
    wide = pd.read_csv(path, index_col=0, parse_dates=True)
    return to_long(wide, "bus", TARGET)


def build_table(
    network: str | Path | pypsa.Network,
    prices: str | Path | pd.DataFrame | None = None,
    run: str | None = None,
) -> pd.DataFrame:
    """
    Return the table of a solved network, with observed prices if `prices` is given.

    Parameters
    ----------
    network : str | Path | pypsa.Network
        Solved network or its path.
    prices : str | Path | pd.DataFrame | None
        Observed prices per bus, wide with naive UTC hours as index, or the path
        to such a CSV, see :func:`read_observed_prices`.
    run : str | None
        Name of the run, defaults to the file name of `network` without suffix.
    """
    if not isinstance(network, pypsa.Network):
        run = run or Path(network).stem
        network = pypsa.Network(network)
    if run is None:
        raise ValueError("A run name is needed for networks not read from a file")
    table = extract(network)
    table.insert(0, "run", run)
    table.insert(1, "year", int(table.snapshot.dt.year.mode().iat[0]))
    if prices is not None:
        observed = (
            to_long(prices, "bus", TARGET)
            if isinstance(prices, pd.DataFrame)
            else read_observed_prices(prices)
        )
        table = table.merge(observed, on=["snapshot", "bus"], how="left")
        matched = table[TARGET].notna()
        logger.info(
            f"Observed prices for {table.bus[matched].nunique()} of {table.bus.nunique()} buses "
            f"and {matched.mean():.1%} of rows."
        )
        if not matched.any():
            raise ValueError(
                f"No observed price matches the snapshots and buses of run {run}"
            )
    return table


def read_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Return the concatenated tables of several runs."""
    table = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    duplicated = table.duplicated(["run", "snapshot", "bus"])
    assert not duplicated.any(), (
        f"Rows duplicated across tables, runs need distinct names: {table.run[duplicated].unique().tolist()}"
    )
    return table
