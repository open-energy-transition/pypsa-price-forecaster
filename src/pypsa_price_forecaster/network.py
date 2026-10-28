"""
Extracts hourly quantities per bus from a solved PyPSA network.

The model price is the marginal price of each AC bus. Load, available supply and
import capacity are summed per market zone and every bus gets the values of its
zone. Buses joined by branches of infinite capacity (the copperplate links of
PyPSA-Eur ``clustering: copperplate_regions``) form one zone, every other bus is
its own zone. Import capacity counts the hourly capacity of all finite branches
into the zone from other zones.
"""

import numpy as np
import pandas as pd
import pypsa
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

VRES_CARRIERS = [
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "onwind",
    "ror",
    "solar",
    "solar rooftop",
    "solar-hsat",
]
LOAD_SHEDDING_CARRIER = "load"

COLUMNS = [
    "snapshot",
    "bus",
    "zone",
    "model_price",
    "load",
    "supply_firm",
    "supply_vres",
    "supply_storage",
    "import_capacity",
]


def market_zones(n: pypsa.Network) -> pd.Series:
    """Return the market zone per AC bus, named by its buses joined with ``+``."""
    buses = n.buses.index[n.buses.carrier == "AC"]
    copper = pd.concat(
        [
            n.lines.loc[np.isinf(n.lines.s_nom), ["bus0", "bus1"]],
            n.links.loc[np.isinf(n.links.p_nom), ["bus0", "bus1"]],
        ]
    ).query("bus0 in @buses and bus1 in @buses")
    position = pd.Series(np.arange(len(buses)), index=buses)
    graph = coo_matrix(
        (np.ones(len(copper)), (position[copper.bus0], position[copper.bus1])),
        shape=(len(buses), len(buses)),
    )
    _, labels = connected_components(graph, directed=False)
    return (
        pd.Series(buses, index=buses)
        .groupby(labels)
        .transform(lambda b: "+".join(sorted(b)))
        .rename("zone")
    )


def nominal(df: pd.DataFrame, attr: str) -> pd.Series:
    """Return the optimised nominal capacity of extendable and the fixed one of other components."""
    return df[f"{attr}_opt"].where(df[f"{attr}_extendable"], df[attr])


def zonal_sum(values: pd.DataFrame, bus: pd.Series, zone: pd.Series) -> pd.DataFrame:
    """Sum the columns of `values` (one per component at `bus`) to market zones, one column per zone."""
    bus = bus[bus.isin(zone.index)]
    return (
        values[bus.index]
        .T.groupby(bus.map(zone))
        .sum()
        .T.reindex(columns=zone.unique(), fill_value=0.0)
    )


def between_zones(branches: pd.DataFrame, zone: pd.Series) -> pd.DataFrame:
    """Return the branches connecting buses of two different zones."""
    zone0, zone1 = branches.bus0.map(zone), branches.bus1.map(zone)
    return branches[zone0.notna() & zone1.notna() & (zone0 != zone1)]


def import_capacity(n: pypsa.Network, zone: pd.Series) -> pd.DataFrame:
    """Return the hourly capacity of finite branches into each zone from other zones."""
    lines = between_zones(n.lines, zone)
    capacity = n.get_switchable_as_dense("Line", "s_max_pu")[lines.index] * nominal(
        lines, "s_nom"
    )
    total = zonal_sum(capacity, lines.bus1, zone) + zonal_sum(
        capacity, lines.bus0, zone
    )

    links = between_zones(n.links, zone)
    p_nom = nominal(links, "p_nom")
    forward = n.get_switchable_as_dense("Link", "p_max_pu")[links.index] * p_nom
    reverse = -n.get_switchable_as_dense("Link", "p_min_pu")[links.index] * p_nom
    total += zonal_sum(forward.clip(lower=0.0), links.bus1, zone) + zonal_sum(
        reverse.clip(lower=0.0), links.bus0, zone
    )
    return total


def zonal_quantities(n: pypsa.Network, zone: pd.Series) -> dict[str, pd.DataFrame]:
    """Return the hourly load, available supply by type and import capacity per zone in MW."""
    generators = n.generators[n.generators.carrier != LOAD_SHEDDING_CARRIER]
    available = n.get_switchable_as_dense("Generator", "p_max_pu")[
        generators.index
    ] * nominal(generators, "p_nom")
    vres = generators.carrier.isin(VRES_CARRIERS)
    storage = n.get_switchable_as_dense("StorageUnit", "p_max_pu") * nominal(
        n.storage_units, "p_nom"
    )
    return {
        "load": zonal_sum(
            n.get_switchable_as_dense("Load", "p_set"), n.loads.bus, zone
        ),
        "supply_firm": zonal_sum(available.loc[:, ~vres], generators.bus[~vres], zone),
        "supply_vres": zonal_sum(available.loc[:, vres], generators.bus[vres], zone),
        "supply_storage": zonal_sum(storage, n.storage_units.bus, zone),
        "import_capacity": import_capacity(n, zone),
    }


def to_long(values: pd.DataFrame, columns: str, name: str) -> pd.DataFrame:
    """Return a snapshot by `columns` frame as long table with columns ``snapshot``, `columns` and `name`."""
    return (
        values.rename_axis(index="snapshot", columns=columns)
        .reset_index()
        .melt(id_vars="snapshot", var_name=columns, value_name=name)
    )


def extract(n: pypsa.Network) -> pd.DataFrame:
    """
    Return the hourly model price and zonal quantities per AC bus of a solved network.

    Parameters
    ----------
    n : pypsa.Network
        Solved single period network with hourly datetime snapshots.

    Returns
    -------
    pd.DataFrame
        One row per snapshot and bus with the columns in ``COLUMNS``. Quantities are in MW.
    """
    assert isinstance(n.snapshots, pd.DatetimeIndex), (
        "Only single period networks with datetime snapshots are supported"
    )
    zone = market_zones(n)
    table = to_long(n.buses_t.marginal_price[zone.index], "bus", "model_price")
    table["zone"] = table.bus.map(zone)
    for name, values in zonal_quantities(n, zone).items():
        table = table.merge(
            to_long(values, "zone", name), on=["snapshot", "zone"], how="left"
        )
    return table[COLUMNS]
