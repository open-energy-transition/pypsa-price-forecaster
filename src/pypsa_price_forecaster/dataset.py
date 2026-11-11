"""
Builds the tables the models train and predict on, and the features derived from them.

A table holds one row per snapshot and bus of one or more runs (solved
networks) with the columns of :data:`pypsa_price_forecaster.network.COLUMNS`,
the run name, the weather year of the run and, for training, the observed
day-ahead price. Features are computed from these columns on the fly, so a new
feature only needs an entry in ``FEATURES``.
"""

import functools
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pypsa

from pypsa_price_forecaster.network import PRICE_SETTERS, extract, to_long

logger = logging.getLogger(__name__)

TARGET = "observed_price"
SETTERS = [*PRICE_SETTERS, "storage", "import", "other"]


def supply(table: pd.DataFrame) -> pd.Series:
    """Available domestic supply of generators and storage units in MW."""
    return table.supply_firm + table.supply_vres + table.supply_storage


# time zone of each country code, the first two letters of a bus name
TIME_ZONES = {
    **dict.fromkeys(["GB", "IE", "PT"], "Europe/London"),
    **dict.fromkeys(["BG", "EE", "FI", "GR", "LT", "LV", "RO"], "Europe/Helsinki"),
}
DEFAULT_TIME_ZONE = "Europe/Berlin"


def local_time(table: pd.DataFrame) -> pd.Series:
    """Naive local time of every row in the time zone of the country of its bus."""
    country = table.bus.str[:2]
    local = pd.Series(pd.NaT, index=table.index, dtype=table.snapshot.dtype)
    for zone, rows in country.groupby(
        country.map(TIME_ZONES).fillna(DEFAULT_TIME_ZONE)
    ).groups.items():
        local[rows] = (
            table.snapshot[rows]
            .dt.tz_localize("UTC")
            .dt.tz_convert(zone)
            .dt.tz_localize(None)
        )
    return local


@functools.cache
def public_holidays(country: str, years: tuple[int, ...]) -> frozenset:
    """Public holiday dates of a country, empty for countries the holidays package lacks."""
    try:
        return frozenset(holidays.country_holidays(country, years=years))
    except NotImplementedError:
        logger.warning(f"No public holidays for country {country}.")
        return frozenset()


def holiday(table: pd.DataFrame) -> pd.Series:
    """Whether the local date of every row is a public holiday in the country of its bus."""
    date = local_time(table).dt.date
    country = table.bus.str[:2]
    years = tuple(sorted(table.snapshot.dt.year.unique().tolist()))
    result = pd.Series(False, index=table.index)
    for code, rows in country.groupby(country).groups.items():
        result[rows] = date[rows].isin(public_holidays(code, years))
    return result.astype(int)


def daily_mean_price(table: pd.DataFrame) -> pd.Series:
    """Mean model price of the bus over the UTC day of each row."""
    days = [table.run, table.bus, table.snapshot.dt.date]
    return table.groupby(days).model_price.transform("mean")


def europe_spread(table: pd.DataFrame) -> pd.Series:
    """Model price minus the median model price over all buses of the run in the same hour."""
    median = table.groupby([table.run, table.snapshot]).model_price.transform("median")
    return table.model_price - median


def weekly_vres_share(table: pd.DataFrame) -> pd.Series:
    """Mean available renewable share of load of the bus over the past 168 hours."""
    share = (table.supply_vres / table.load).replace([np.inf, -np.inf], np.nan)
    ordered = (
        table[["run", "bus", "snapshot"]]
        .assign(share=share)
        .sort_values("snapshot", kind="stable")
    )
    weekly = ordered.groupby(["run", "bus"]).share.transform(
        lambda s: s.rolling(168, min_periods=24).mean()
    )
    return weekly.reindex(table.index)


def daily_mean(table: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Mean of `values` over the UTC day of the bus of each row."""
    days = [table.run, table.bus, table.snapshot.dt.date]
    return values.replace([np.inf, -np.inf], np.nan).groupby(days).transform("mean")


def rank_in_day(table: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Rank of `values` among the hours of the UTC day of the bus, as share from 0 to 1."""
    days = [table.run, table.bus, table.snapshot.dt.date]
    return values.groupby(days).rank(pct=True)


def residual_ramp(table: pd.DataFrame, hours: int) -> pd.Series:
    """
    Change of the residual load over the past `hours`, as share of the mean load of the day.

    The residual load is the load minus the available renewables, what storage,
    imports and thermal plants have to follow. Negative `hours` give the change
    over the coming hours instead.
    """
    residual = table.load - table.supply_vres
    other = table.snapshot - pd.Timedelta(hours=hours)
    now = pd.MultiIndex.from_arrays([table.run, table.bus, table.snapshot])
    there = pd.MultiIndex.from_arrays([table.run, table.bus, other])
    before = pd.Series(residual.to_numpy(), index=now).reindex(there).to_numpy()
    return np.sign(hours) * (residual - before) / daily_mean(table, table.load)


def europe_total(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sum of `columns` over all zones of the run in the same hour, for every row."""
    # the columns are zone totals repeated for every bus of a zone: count each zone once
    zones = table.drop_duplicates(["run", "snapshot", "zone"])
    total = zones.groupby(["run", "snapshot"])[columns].sum()
    hours = pd.MultiIndex.from_arrays([table.run, table.snapshot])
    return total.reindex(hours).set_axis(table.index)


def europe_vres_share(table: pd.DataFrame) -> pd.Series:
    total = europe_total(table, ["supply_vres", "load"])
    return total.supply_vres / total.load


def europe_margin(table: pd.DataFrame) -> pd.Series:
    columns = ["supply_firm", "supply_vres", "supply_storage", "load"]
    total = europe_total(table, columns)
    return supply(total) / total.load - 1


def europe_price_ratio(table: pd.DataFrame) -> pd.Series:
    """Model price over the median model price of all buses of the run in the same hour."""
    median = table.groupby([table.run, table.snapshot]).model_price.transform("median")
    return table.model_price / median.clip(lower=1.0)


def gas_sets_price(table: pd.DataFrame) -> pd.Series:
    return (table.price_setter == "gas").astype(float)


def daily_gas(
    table: pd.DataFrame, transform: Callable[[pd.Series], pd.Series]
) -> pd.Series:
    """`transform` of the daily mean gas cost per run and bus, mapped back to the rows."""
    day = table.snapshot.dt.floor("D")
    gas = table.cost_gas.groupby([table.run, table.bus, day]).mean()
    result = gas.groupby(level=["run", "bus"], group_keys=False).apply(transform)
    keys = pd.MultiIndex.from_arrays([table.run, table.bus, day])
    return pd.Series(result.reindex(keys).to_numpy(), index=table.index)


def attach_exogenous(table: pd.DataFrame, variables: pd.DataFrame) -> pd.DataFrame:
    """
    Return `table` with variables per bus from outside the network.

    `variables` are daily with a ``date`` column (:func:`pypsa_price_forecaster.exogenous.build`)
    or hourly with a ``snapshot`` column (:func:`pypsa_price_forecaster.forecasts.build`).
    """
    if "snapshot" in variables:
        time = table.snapshot
    else:
        time = table.snapshot.dt.floor("D").rename("date")
    joined = pd.concat([table[["bus"]], time], axis=1).merge(
        variables, on=["bus", time.name], how="left"
    )
    columns = variables.columns.difference(["bus", time.name])
    return table.drop(columns=columns, errors="ignore").assign(
        **{c: joined[c].to_numpy() for c in columns}
    )


def exogenous(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    def feature(table: pd.DataFrame) -> pd.Series:
        if name not in table:
            raise ValueError(
                f"Feature {name} needs the variables of `ppf exogenous` or `ppf forecasts`"
            )
        return table[name]

    return feature


FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "model_price": lambda t: t.model_price,
    "margin_domestic": lambda t: (supply(t) - t.load) / t.load,
    "margin_with_imports": lambda t: (supply(t) + t.import_capacity - t.load) / t.load,
    "margin_domestic_mw": lambda t: supply(t) - t.load,
    "margin_with_imports_mw": lambda t: supply(t) + t.import_capacity - t.load,
    "vres_share": lambda t: t.supply_vres / t.load,
    "vres_share_week": weekly_vres_share,
    "spread_europe": europe_spread,
    "margin_with_imports_day": lambda t: daily_mean(
        t, (supply(t) + t.import_capacity - t.load) / t.load
    ),
    "vres_share_day": lambda t: daily_mean(t, t.supply_vres / t.load),
    "firm_utilisation_day": lambda t: daily_mean(t, t.dispatch_firm / t.supply_firm),
    "gas_setter_day": lambda t: daily_mean(t, gas_sets_price(t)),
    "gas_setter_europe": lambda t: (
        gas_sets_price(t).groupby([t.run, t.snapshot]).transform("mean")
    ),
    "curtailment_share": lambda t: (t.supply_vres - t.dispatch_vres) / t.load,
    "firm_utilisation": lambda t: t.dispatch_firm / t.supply_firm,
    "net_import_share": lambda t: t.net_import / t.load,
    "storage_share": lambda t: t.storage_dispatch / t.load,
    "cost_gas": lambda t: t.cost_gas,
    "cost_coal": lambda t: t.cost_coal,
    "price_to_gas": lambda t: t.model_price / t.cost_gas,
    "price_daily_mean": daily_mean_price,
    "price_deviation": lambda t: t.model_price - daily_mean_price(t),
    # where an hour sits in the day of its bus and in Europe as a whole (round 21)
    "price_rel_day": lambda t: t.model_price / daily_mean_price(t).clip(lower=1.0),
    "price_rank_day": lambda t: rank_in_day(t, t.model_price),
    "residual_rank_day": lambda t: rank_in_day(t, t.load - t.supply_vres),
    "ramp_3h": lambda t: residual_ramp(t, 3),
    "ramp_next_3h": lambda t: residual_ramp(t, -3),
    "price_rel_europe": europe_price_ratio,
    "vres_share_europe": europe_vres_share,
    "margin_europe": europe_margin,
    "bus": lambda t: t.bus.astype("category"),
    "price_setter": lambda t: pd.Categorical(t.price_setter, categories=SETTERS),
    "imports_congested": lambda t: t.import_congested / t.import_capacity,
    "exports_congested": lambda t: t.export_congested / t.export_capacity,
    "hour": lambda t: t.snapshot.dt.hour,
    "dayofweek": lambda t: t.snapshot.dt.day_of_week,
    "hour_local": lambda t: local_time(t).dt.hour,
    "dayofweek_local": lambda t: local_time(t).dt.day_of_week,
    "holiday": holiday,
    "month": lambda t: t.snapshot.dt.month,
    # smooth season, 1 in mid-January, -1 in mid-July, the same for every bus (round 20)
    "winter": lambda t: np.cos(2 * np.pi * (t.snapshot.dt.dayofyear - 15) / 365.25),
    "gas_change_30d": lambda t: daily_gas(t, lambda s: s / s.shift(30) - 1),
    "gas_vol_30d": lambda t: daily_gas(t, lambda s: np.log(s).diff().rolling(30).std()),
    "temperature": exogenous("temperature"),
    "hdd": lambda t: (15.5 - exogenous("temperature")(t)).clip(lower=0),
    "runoff_90d": exogenous("runoff_90d"),
    "reservoir_anomaly": exogenous("reservoir_anomaly"),
    # the model's own hydro reservoirs, experiments/model_reservoirs.py (round 17)
    "reservoir_share_model": exogenous("reservoir_share_model"),
    "reservoir_anomaly_model": exogenous("reservoir_anomaly_model"),
    "inflow_90d_model": exogenous("inflow_90d_model"),
    # load and supply of the table are zone totals, as the ENTSO-E forecasts
    "load_forecast_error": lambda t: exogenous("load_forecast")(t) / t.load - 1,
    "vres_forecast_error": lambda t: (
        (exogenous("vres_forecast")(t) - t.supply_vres) / t.load
    ),
}
# column from outside the network each feature needs, see attach_exogenous
VARIABLES = {
    "temperature": "temperature",
    "hdd": "temperature",
    "runoff_90d": "runoff_90d",
    "reservoir_anomaly": "reservoir_anomaly",
    "reservoir_share_model": "reservoir_share_model",
    "reservoir_anomaly_model": "reservoir_anomaly_model",
    "inflow_90d_model": "inflow_90d_model",
    "load_forecast_error": "load_forecast",
    "vres_forecast_error": "vres_forecast",
}
# best set on held-out years from the network alone (experiments/NOTES.md, round 5)
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
    "price_setter",
    "imports_congested",
    "exports_congested",
]

# the hours around a row and the other zones, from the network alone (round 21)
CONTEXT_FEATURES = [
    "price_rel_day",
    "price_rank_day",
    "residual_rank_day",
    "ramp_3h",
    "ramp_next_3h",
    "price_rel_europe",
    "vres_share_europe",
    "margin_europe",
]


def has_variables(table: pd.DataFrame, feature: str) -> bool:
    """Whether `table` holds the variables from outside the network that `feature` needs."""
    return feature not in VARIABLES or VARIABLES[feature] in table


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
