"""
Downloads observed day-ahead prices from the ENTSO-E Transparency Platform and maps them to buses.

The prices are queried per ENTSO-E bidding zone area and calendar year in UTC
with the API token in ``ENTSOE_API_TOKEN``, and cached as CSV with UTC
timestamps in their native resolution (hourly, or 15 minutes since October
2025). Areas without published prices or with prices not only in EUR (e.g. GB,
or PL, RO and BG in 2019) are skipped with a warning and a ``.skip`` file in the
cache, so they are not queried again. The prices are averaged to naive UTC
hours, gaps of up to ``MAX_INTERPOLATE_GAP_HOURS`` are interpolated and longer
gaps are kept as missing values.

Every bus gets the price of its area. The default map from buses to areas is the
one of pypsa-eur-backcast (``data/entsoe_area_to_bus.csv``). Buses in
``EXCLUDED_BUSES`` get no price.
"""

import functools
import logging
import os
import re
from importlib.resources import files
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.exceptions import NoMatchingDataError

logger = logging.getLogger(__name__)

MAX_INTERPOLATE_GAP_HOURS = 3
EXCLUDED_BUSES = ["FR15"]  # Corsica is not coupled to the French day-ahead market


class Unavailable(Exception):
    """No usable day-ahead prices are published for an area and year."""


def read_area_to_bus(path: str | Path | None = None) -> pd.Series:
    """Read the many-to-one map from buses to ENTSO-E areas, by default the packaged one."""
    path = path or files("pypsa_price_forecaster") / "data" / "entsoe_area_to_bus.csv"
    return pd.read_csv(path, index_col="bus")["entsoe_area"]


@functools.cache
def entsoe_client() -> EntsoePandasClient:
    """Create an ENTSO-E client with the API token from ``ENTSOE_API_TOKEN``."""
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Set ENTSOE_API_TOKEN to download prices from the ENTSO-E Transparency Platform"
        )
    return EntsoePandasClient(api_key=token, timeout=120)


def query_area_year(area: str, year: int) -> pd.Series:
    """Return the day-ahead prices of `area` in the UTC calendar `year` with UTC timestamps, in EUR/MWh."""
    client = entsoe_client()
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    try:
        xml = EntsoeRawClient.query_day_ahead_prices(client, area, start=start, end=end)
        currencies = set(re.findall(r"<currency_Unit\.name>(\w+)<", xml))
        if currencies != {"EUR"}:
            raise Unavailable(f"prices are given in {sorted(currencies)}, not only EUR")
        prices = client.query_day_ahead_prices(area, start=start, end=end)
    except NoMatchingDataError as e:
        raise Unavailable("no day-ahead prices are published") from e
    return prices.tz_convert("UTC").loc[lambda s: s.index < end].rename(area)


def area_prices(area: str, year: int, cache_dir: Path) -> pd.Series | None:
    """Return the cached or queried prices of `area` in `year`, None if unavailable."""
    csv = cache_dir / f"day_ahead_prices_{area}_{year}.csv"
    skip = csv.with_suffix(".skip")
    if not csv.exists() and not skip.exists():
        logger.info(f"Querying day-ahead prices of {area} in {year}.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            query_area_year(area, year).rename_axis("utc_timestamp").to_csv(csv)
        except Unavailable as e:
            skip.write_text(str(e))
    if skip.exists():
        logger.warning(f"No observed prices for {area} in {year}: {skip.read_text()}")
        return None
    prices = pd.read_csv(csv, index_col=0).squeeze("columns")
    return prices.set_axis(pd.to_datetime(prices.index, utc=True))


def fill_short_gaps(prices: pd.DataFrame, max_hours: int) -> pd.DataFrame:
    """Interpolate gaps of up to `max_hours` hours inside each column, keep longer gaps missing."""
    gap = prices.isna()
    length = gap.apply(lambda c: c.groupby((c != c.shift()).cumsum()).transform("size"))
    short = gap & (length <= max_hours)
    return prices.where(~short, prices.interpolate(limit_area="inside"))


def observed_prices(
    buses: pd.Index,
    snapshots: pd.DatetimeIndex,
    cache_dir: str | Path,
    area_to_bus: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Return the observed hourly day-ahead prices per bus.

    Parameters
    ----------
    buses : pd.Index
        Buses to return prices for.
    snapshots : pd.DatetimeIndex
        Naive UTC hours to return prices for.
    cache_dir : str | Path
        Directory of the cached downloads.
    area_to_bus : pd.Series | None
        ENTSO-E area per bus, by default :func:`read_area_to_bus`.

    Returns
    -------
    pd.DataFrame
        Prices in EUR/MWh with `snapshots` as index and one column per bus with
        an available area, missing values in long gaps.
    """
    area_to_bus = read_area_to_bus() if area_to_bus is None else area_to_bus
    buses = buses.difference(EXCLUDED_BUSES)
    unmapped = buses.difference(area_to_bus.index)
    if not unmapped.empty:
        logger.warning(f"Buses without ENTSO-E area get no price: {unmapped.tolist()}")
    bus_area = area_to_bus.reindex(buses.intersection(area_to_bus.index))

    hourly = {}
    for area in sorted(bus_area.unique()):
        series = [
            s
            for year in sorted(set(snapshots.year))
            if (s := area_prices(area, year, Path(cache_dir))) is not None
        ]
        if series:
            prices = pd.concat(series).sort_index()
            hourly[area] = prices.resample("h").mean().tz_localize(None)
    prices = fill_short_gaps(
        pd.DataFrame(hourly).reindex(snapshots), MAX_INTERPOLATE_GAP_HOURS
    )
    bus_area = bus_area[bus_area.isin(prices.columns)]
    return (
        prices[bus_area.to_numpy()]
        .set_axis(bus_area.index, axis=1)
        .rename_axis(index="snapshot", columns=None)
    )
