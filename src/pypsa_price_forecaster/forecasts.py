"""
Downloads the day-ahead load and wind and solar forecasts of the ENTSO-E Transparency Platform.

These are what the market knew when it cleared the day-ahead auction, while
the backcast is solved with the realised load and ERA5 weather. The forecasts
are queried per ENTSO-E area and calendar year in UTC with the API token in
``ENTSOE_API_TOKEN``, averaged to hours and cached as CSV. A forecast that is
not published for an area is kept missing, an area with neither gets a
``.skip`` file. Unlike the weather variables of
:mod:`pypsa_price_forecaster.exogenous`, they only exist for past years.

The result has one row per bus and hour with the forecasts of its area, joined
to a table with :func:`pypsa_price_forecaster.dataset.attach_exogenous`.
"""

import logging
from pathlib import Path

import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from pypsa_price_forecaster.entsoe import entsoe_client, read_area_to_bus

logger = logging.getLogger(__name__)

COLUMNS = ["load_forecast", "vres_forecast"]


def query_forecasts(area: str, year: int) -> pd.DataFrame:
    """Hourly day-ahead load and wind and solar forecasts of `area` in `year` in MW, UTC timestamps."""
    client = entsoe_client()
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    queries = {
        "load_forecast": lambda: client.query_load_forecast(area, start=start, end=end),
        "vres_forecast": lambda: client.query_wind_and_solar_forecast(
            area, start=start, end=end
        ),
    }
    result = {}
    for name, query in queries.items():
        try:
            frame = query()
        except NoMatchingDataError:
            continue
        hourly = frame.tz_convert("UTC").resample("h").mean()
        result[name] = hourly.sum(axis=1, min_count=1).loc[lambda s: s.index < end]
    return pd.DataFrame(result, columns=COLUMNS)


def area_forecasts(area: str, year: int, cache_dir: Path) -> pd.DataFrame | None:
    """Return the cached or queried forecasts of `area` in `year`, None if none are published."""
    csv = cache_dir / f"day_ahead_forecasts_{area}_{year}.csv"
    skip = csv.with_suffix(".skip")
    if not csv.exists() and not skip.exists():
        logger.info(f"Querying day-ahead forecasts of {area} in {year}.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        forecasts = query_forecasts(area, year)
        if forecasts.dropna(how="all").empty:
            skip.write_text("no day-ahead forecasts published")
        else:
            forecasts.rename_axis("utc_timestamp").to_csv(csv)
    if skip.exists():
        return None
    forecasts = pd.read_csv(csv, index_col=0)
    return forecasts.set_axis(
        pd.to_datetime(forecasts.index, utc=True).tz_localize(None)
    )


def build(
    years: list[int], cache_dir: str | Path, area_to_bus: pd.Series | None = None
) -> pd.DataFrame:
    """Return the forecasts of :data:`COLUMNS` per bus and naive UTC hour of `years`."""
    area_to_bus = read_area_to_bus() if area_to_bus is None else area_to_bus
    frames = []
    for area, buses in area_to_bus.groupby(area_to_bus).groups.items():
        parts = [
            f
            for year in sorted(years)
            if (f := area_forecasts(area, year, Path(cache_dir))) is not None
        ]
        if not parts:
            continue
        forecasts = pd.concat(parts).rename_axis("snapshot").reset_index()
        frames += [forecasts.assign(bus=bus) for bus in buses]
    return pd.concat(frames, ignore_index=True)[["snapshot", "bus", *COLUMNS]]
