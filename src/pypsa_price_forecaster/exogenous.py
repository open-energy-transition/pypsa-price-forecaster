"""
Builds daily variables per bus from data outside the solved network.

Weather comes from the inputs of a pypsa-eur-backcast checkout: the
population-weighted daily mean temperature of each bus region from the ERA5
cutout of the year, and the runoff of the country over the past 90 days against
its 1991-2020 mean. Hydro reservoir filling per country is downloaded from the
ENTSO-E Transparency Platform (weekly, article 16.1.D) with the API token in
``ENTSOE_API_TOKEN``, cached as CSV, and compared with its 2015-2018 normal,
before any backcast year. Temperature and runoff are also known for a scenario
run of a weather year; reservoir filling is observed, so a scenario run takes
an anomaly of 0, a normal year.

The result has one row per bus and day, joined to a table with
:func:`pypsa_price_forecaster.dataset.attach_exogenous`.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from entsoe.exceptions import NoMatchingDataError

from pypsa_price_forecaster.entsoe import entsoe_client

logger = logging.getLogger(__name__)

COLUMNS = ["temperature", "runoff_90d", "reservoir_anomaly"]
RUNOFF = "data/country_runoff/archive/2025-08-13/era5-runoff-per-country.csv"
RUNOFF_DAYS = 90
RUNOFF_CAP = (
    5  # countries with hardly any runoff (NL, DK, LU) give ratios far beyond this
)
CLIMATE = slice("1991", "2020")
RESERVOIR_YEARS = range(2015, 2025)
RESERVOIR_CLIMATE = range(
    2015, 2019
)  # before the backcast years, so no test year sets the normal


def cutout_path(root: Path, year: int) -> Path:
    return root / f"data/cutout/archive/v1.0/europe-{year}-sarah3-era5.nc"


def cell_weights(
    shapes: gpd.GeoSeries, weight: xr.DataArray, normalise: bool = True
) -> xr.DataArray:
    """Weight of every grid cell of `weight` per shape, summing to one per shape if `normalise`."""
    nx, ny = weight.sizes["x"], weight.sizes["y"]
    cells = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(np.tile(weight.x, ny), np.repeat(weight.y, nx)),
        crs=shapes.crs,
    )
    numbered = gpd.GeoDataFrame(
        {"position": range(len(shapes))}, geometry=shapes.values
    )
    joined = gpd.sjoin(cells, numbered.set_crs(shapes.crs), predicate="within")
    position = joined.position.to_numpy()
    w = np.zeros((len(shapes), nx * ny))
    w[position, joined.index] = weight.values.ravel()[joined.index]
    if normalise:
        total = w.sum(axis=1, keepdims=True)
        w = w / np.where(total > 0, total, 1)
    return xr.DataArray(
        w.reshape(len(shapes), ny, nx),
        coords={
            "shape": shapes.index.to_numpy(),
            "y": weight.y.values,
            "x": weight.x.values,
        },
        dims=["shape", "y", "x"],
    )


def temperature(root: Path, year: int) -> pd.DataFrame:
    """Population-weighted daily mean temperature in degC, one column per bus."""
    resources = root / f"resources/backcast-{year}"
    regions = (
        gpd.read_file(resources / "onshore_regions.geojson").set_index("name").geometry
    )
    weights = cell_weights(
        regions, xr.open_dataarray(resources / "pop_layout_total.nc")
    )
    cutout = xr.open_dataset(cutout_path(root, year))
    daily = (
        cutout.temperature.astype("float32").resample(time="1D").mean().load() - 273.15
    )
    return xr.dot(weights, daily, dim=["y", "x"]).to_pandas().T


def cutout_runoff(root: Path, year: int) -> pd.DataFrame:
    """Daily runoff summed over the cells of each country, in cutout units."""
    shapes = gpd.read_file(root / f"resources/backcast-{year}/country_shapes.geojson")
    shapes = shapes.set_index("name").geometry
    cutout = xr.open_dataset(cutout_path(root, year))
    weights = cell_weights(shapes, xr.ones_like(cutout.height), normalise=False)
    daily = cutout.runoff.resample(time="1D").sum().load()
    return xr.dot(weights, daily, dim=["y", "x"]).to_pandas().T


def country_runoff(root: Path, years: list[int]) -> pd.DataFrame:
    """
    ERA5 daily runoff per country, from the PyPSA-Eur series and extended past it with the cutouts.

    Cutout years after the series are scaled per country to the series in its
    last year, so both are on the same scale.
    """
    series = pd.read_csv(root / RUNOFF, index_col=0, parse_dates=True)
    last = series.index.max().year
    extra = [y for y in years if y > last]
    if extra:
        overlap = cutout_runoff(root, last)
        scale = series.loc[str(last)].sum() / overlap.sum()
        later = (
            pd.concat(cutout_runoff(root, y) for y in extra).mul(scale).dropna(axis=1)
        )
        series = pd.concat([series, later[later.columns.intersection(series.columns)]])
    return series


def runoff_anomaly(runoff: pd.DataFrame, days: int = RUNOFF_DAYS) -> pd.DataFrame:
    """Runoff of the past `days` days over its 1991-2020 mean for the same day of year, at most :data:`RUNOFF_CAP`."""
    rolling = runoff.rolling(days, min_periods=days).sum()
    climate = rolling.loc[CLIMATE]
    mean = climate.groupby(climate.index.day_of_year).mean()
    ratio = rolling / mean.reindex(rolling.index.day_of_year).set_axis(rolling.index)
    return ratio.replace(np.inf, np.nan).clip(upper=RUNOFF_CAP)


def reservoir_filling(country: str, cache_dir: Path) -> pd.Series | None:
    """Weekly stored energy of hydro reservoirs of `country` in MWh, None if not published."""
    csv = cache_dir / f"reservoir_filling_{country}.csv"
    skip = csv.with_suffix(".skip")
    if not csv.exists() and not skip.exists():
        logger.info(f"Querying hydro reservoir filling of {country}.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for year in RESERVOIR_YEARS:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            try:
                parts.append(
                    entsoe_client().query_aggregate_water_reservoirs_and_hydro_storage(
                        country, start=start, end=end
                    )
                )
            except NoMatchingDataError:
                continue
        if not parts:
            skip.write_text("no reservoir filling published")
        else:
            filling = pd.concat(parts)
            filling = filling[~filling.index.duplicated()].sort_index()
            filling.tz_convert("UTC").rename("filling").rename_axis(
                "utc_timestamp"
            ).to_csv(csv)
    if skip.exists():
        return None
    filling = pd.read_csv(csv, index_col=0).squeeze("columns")
    return filling.set_axis(pd.to_datetime(filling.index, utc=True).tz_localize(None))


def reservoir_anomaly(
    countries: list[str],
    days: pd.DatetimeIndex,
    cache_dir: Path,
    climate: range = RESERVOIR_CLIMATE,
) -> pd.DataFrame:
    """
    Reservoir filling as share of its highest value in `climate` years, minus the median share of the same week in them.

    Each day gets the value of the week published before it, as known at the
    day-ahead auction.
    """
    result = {}
    for country in countries:
        filling = reservoir_filling(country, cache_dir)
        past = None if filling is None else filling[filling.index.year.isin(climate)]
        if past is None or past.max() <= 0:
            continue
        share = filling / past.max()
        week = pd.Series(share.index.isocalendar().week.to_numpy(), index=share.index)
        normal = share[past.index].groupby(week).median()
        anomaly = share - normal.reindex(week).to_numpy()
        published = anomaly.set_axis(anomaly.index + pd.Timedelta(days=7))
        result[country] = (
            published.reindex(published.index.union(days)).ffill().reindex(days)
        )
    return pd.DataFrame(result, index=days)


def bus_country(buses: pd.Index) -> pd.Series:
    return pd.Series(buses.str[:2], index=buses)


def build(root: str | Path, years: list[int], cache_dir: str | Path) -> pd.DataFrame:
    """Return the daily variables of :data:`COLUMNS` per bus and day of `years`."""
    root = Path(root)
    years = sorted(years)
    temp = pd.concat(temperature(root, y) for y in years)
    days = temp.index
    country = bus_country(temp.columns)
    runoff = runoff_anomaly(country_runoff(root, years)).reindex(days)
    reservoirs = reservoir_anomaly(sorted(country.unique()), days, Path(cache_dir))
    frame = (
        temp.stack().rename("temperature").rename_axis(["date", "bus"]).reset_index()
    )
    frame["country"] = frame.bus.map(country)
    for name, wide in [("runoff_90d", runoff), ("reservoir_anomaly", reservoirs)]:
        long = wide.stack().rename(name).rename_axis(["date", "country"]).reset_index()
        frame = frame.merge(long, on=["date", "country"], how="left")
    return frame.drop(columns="country")
