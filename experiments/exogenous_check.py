"""
Do outside variables track the error the correction leaves?

Correlates the out-of-sample residual of report/prices.parquet (market minus
corrected price, mean per zone and day) with daily variables per zone built
from the backcast inputs: population-weighted temperature from the ERA5
cutout, ERA5 runoff per country against its 1991-2020 climate, and the
momentum of the gas and CO2 prices the backcast uses.

    pixi run python experiments/exogenous_check.py /home/cesare/pypsa-eur-backcast
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

OUT = Path("results/exogenous_daily.parquet")


def zone_weights(root: Path, year: int) -> xr.DataArray:
    """Population weight of every cutout cell per zone, summing to one per zone."""
    resources = root / f"resources/backcast-{year}"
    regions = gpd.read_file(resources / "onshore_regions.geojson").set_index("name")
    pop = xr.open_dataarray(resources / "pop_layout_total.nc")
    cells = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            np.tile(pop.x, pop.sizes["y"]), np.repeat(pop.y, pop.sizes["x"])
        ),
        crs=regions.crs,
    )
    joined = gpd.sjoin(cells, regions[["geometry"]], predicate="within")
    w = np.zeros((len(regions), pop.size))
    pos = {name: i for i, name in enumerate(regions.index)}
    flat = pop.values.ravel()
    for cell, name in joined["name"].items():
        w[pos[name], cell] = flat[cell]
    w = w / np.where(w.sum(axis=1, keepdims=True) > 0, w.sum(axis=1, keepdims=True), 1)
    return xr.DataArray(
        w.reshape(len(regions), pop.sizes["y"], pop.sizes["x"]),
        coords={"bus": regions.index.to_numpy(), "y": pop.y.values, "x": pop.x.values},
        dims=["bus", "y", "x"],
    )


def temperature(root: Path, year: int) -> pd.DataFrame:
    cutout = xr.open_dataset(
        root / f"data/cutout/archive/v1.0/europe-{year}-sarah3-era5.nc"
    )
    daily = (
        cutout.temperature.astype("float32").resample(time="1D").mean().load() - 273.15
    )
    w = zone_weights(root, year)
    t = xr.dot(w, daily, dims=["y", "x"]).to_pandas().T  # time x bus
    return t.stack().rename("temperature").rename_axis(["date", "bus"]).reset_index()


def runoff(root: Path) -> pd.DataFrame:
    """ERA5 runoff per country: past 90 and 365 days against the 1991-2020 mean of the same window."""
    r = pd.read_csv(
        root / "data/country_runoff/archive/2025-08-13/era5-runoff-per-country.csv",
        index_col=0,
        parse_dates=True,
    )
    result = {}
    for days in (90, 365):
        rolling = r.rolling(days, min_periods=days).sum()
        clim = (
            rolling["1991":"2020"]
            .groupby(rolling["1991":"2020"].index.dayofyear)
            .mean()
        )
        result[f"runoff_{days}d"] = rolling / clim.reindex(
            rolling.index.dayofyear
        ).set_axis(rolling.index)
    frame = pd.concat({k: v.stack() for k, v in result.items()}, axis=1)
    return frame.rename_axis(["date", "country"]).reset_index()


def prices(root: Path, year: int) -> pd.DataFrame:
    resources = root / f"resources/backcast-{year}"
    co2 = pd.read_csv(resources / "co2_price.csv", index_col=0, parse_dates=True).price
    co2 = co2.resample("1D").last().ffill()
    frame = pd.DataFrame({"co2": co2})
    frame["co2_change_30d"] = co2 / co2.shift(30) - 1
    frame["co2_vol_30d"] = np.log(co2).diff().rolling(30).std()
    return frame.loc[str(year)].rename_axis("date").reset_index()


def main(root: Path) -> None:
    p = pd.read_parquet("report/prices.parquet")
    p["date"] = p.snapshot.dt.floor("D")
    p["residual"] = p.observed_price - p.lgbm
    p["raw_residual"] = p.observed_price - p.pypsa
    daily = (
        p.groupby(["run", "bus", "date"])[["residual", "raw_residual"]]
        .mean()
        .reset_index()
    )
    daily["year"] = daily.date.dt.year

    tables = pd.concat(
        pd.read_parquet(f) for f in sorted(Path("data/tables").glob("*.parquet"))
    )
    tables["date"] = tables.snapshot.dt.floor("D")
    gas = tables.groupby(["bus", "date"]).cost_gas.mean().reset_index()
    gas = gas.sort_values("date")
    gas["gas_change_30d"] = gas.groupby("bus").cost_gas.transform(
        lambda s: s / s.shift(30) - 1
    )
    gas["gas_vol_30d"] = gas.groupby("bus").cost_gas.transform(
        lambda s: np.log(s).diff().rolling(30).std()
    )

    years = sorted(daily.year.unique())
    temp = pd.concat(temperature(root, y) for y in years)
    temp["doy"] = temp.date.dt.dayofyear
    clim = temp.groupby(["bus", "doy"]).temperature.transform("mean")
    temp["temp_anomaly"] = temp.temperature - clim
    temp["hdd"] = (15.5 - temp.temperature).clip(lower=0)
    temp["cdd"] = (temp.temperature - 22).clip(lower=0)

    frame = (
        daily.merge(temp.drop(columns="doy"), on=["bus", "date"], how="left")
        .merge(gas, on=["bus", "date"], how="left")
        .merge(pd.concat(prices(root, y) for y in years), on="date", how="left")
    )
    frame["country"] = frame.bus.str[:2]
    frame = frame.merge(runoff(root), on=["date", "country"], how="left")
    OUT.parent.mkdir(exist_ok=True)
    frame.to_parquet(OUT)

    variables = [
        "temperature",
        "temp_anomaly",
        "hdd",
        "cdd",
        "runoff_90d",
        "runoff_365d",
        "cost_gas",
        "gas_change_30d",
        "gas_vol_30d",
        "co2",
        "co2_change_30d",
        "co2_vol_30d",
    ]
    rows = {}
    for v in variables:
        f = frame.dropna(subset=[v, "residual"])
        within = f.groupby(["bus", "year"])
        demeaned = (f[v] - within[v].transform("mean")).corr(
            f.residual - within.residual.transform("mean"), method="spearman"
        )
        month = (
            f.assign(month=f.date.dt.to_period("M"))
            .groupby(["bus", "month"])[[v, "residual"]]
            .mean()
        )
        rows[v] = {
            "rows": len(f),
            "daily": f[v].corr(f.residual, method="spearman"),
            "daily, within zone-year": demeaned,
            "zone-month": month[v].corr(month.residual, method="spearman"),
            "daily vs raw error": f[v].corr(f.raw_residual, method="spearman"),
        }
    print(
        "Spearman correlation with the residual (market - corrected), mean per zone and day:"
    )
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
