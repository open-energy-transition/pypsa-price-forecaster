"""
Builds daily hydro reservoir variables per bus from the solved networks, the model's
own counterpart of ``reservoir_anomaly`` from ENTSO-E (round 17).

Usage: ``pixi run python experiments/model_reservoirs.py <backcast checkout> <variant> <out> <year> ...``

Per market zone and day, from the hydro storage units:

- ``reservoir_share_model``: state of charge / energy capacity (p_nom * max_hours)
- ``reservoir_anomaly_model``: the share minus the median over all given years of the
  same ISO week
- ``inflow_90d_model``: inflow over the past 90 days / the median over all given years
  of the same day of year; the years are chained, so the first 89 days of the first
  year are missing

The normals use every given year, model output only, no observed data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from pypsa_price_forecaster.network import market_zones, nominal, zonal_sum


def daily_hydro(network: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the daily mean state of charge, energy capacity and inflow per bus in MWh and MW."""
    n = pypsa.Network(network)
    zone = market_zones(n)
    hydro = n.storage_units[
        (n.storage_units.carrier == "hydro") & n.storage_units.bus.isin(zone.index)
    ]
    soc = n.storage_units_t.state_of_charge.reindex(columns=hydro.index, fill_value=0)
    inflow = n.storage_units_t.inflow.reindex(columns=hydro.index, fill_value=0)
    capacity = pd.DataFrame(
        [nominal(hydro, "p_nom") * hydro.max_hours], index=n.snapshots[:1]
    )
    by_zone = [
        zonal_sum(values, hydro.bus, zone).resample("D").mean()
        for values in [soc, capacity, inflow]
    ]
    # every bus gets the values of its zone
    return tuple(z[zone.to_numpy()].set_axis(zone.index, axis=1) for z in by_zone)


def build(root: Path, variant: str, years: list[int]) -> pd.DataFrame:
    levels, capacities, inflows = [], {}, []
    for year in years:
        network = root / f"results/backcast-{year}-{variant}/networks/solved_{year}.nc"
        level, capacity, inflow = daily_hydro(network)
        levels.append(level)
        capacities[year] = capacity.iloc[0]
        inflows.append(inflow)
    level = pd.concat(levels)
    capacity = (
        pd.DataFrame(capacities).T.reindex(level.index.year).set_axis(level.index)
    )
    share = (level / capacity.where(capacity > 0)).astype(float)
    week = share.index.isocalendar().week.to_numpy()
    anomaly = share - share.groupby(week).median().reindex(week).to_numpy()

    # without the empty 29 February of networks of 8760 hours in leap years
    inflow = pd.concat(inflows).dropna(how="all")
    past = inflow.rolling(90, min_periods=90).sum()
    day = np.minimum(past.index.dayofyear, 365)
    normal = past.groupby(day).median().reindex(day).set_axis(past.index)
    ratio = (past / normal.where(normal > 0)).astype(float)

    columns = {
        "reservoir_share_model": share,
        "reservoir_anomaly_model": anomaly,
        "inflow_90d_model": ratio,
    }
    long = [
        wide.rename_axis(index="date", columns="bus")
        .stack(future_stack=True)
        .rename(name)
        for name, wide in columns.items()
    ]
    return pd.concat(long, axis=1).reset_index()


if __name__ == "__main__":
    root, variant, out, *years = sys.argv[1:]
    daily = build(Path(root), variant, [int(y) for y in years])
    daily.to_parquet(out, index=False)
    print(daily.describe().round(3).to_string())
    print(f"Wrote {len(daily)} rows ({daily.bus.nunique()} buses) to {out}")
