import numpy as np
import pandas as pd
import pytest

from pypsa_price_forecaster.dataset import (
    DEFAULT_FEATURES,
    build_table,
    feature_matrix,
    has_variables,
    holiday,
    local_time,
)
from pypsa_price_forecaster.network import COLUMNS, extract, market_zones


def test_copperplated_buses_share_a_zone(network):
    assert market_zones(network).to_dict() == {"A": "A", "B1": "B1+B2", "B2": "B1+B2"}


def test_extract_sums_zonal_quantities(network):
    table = extract(network)
    assert list(table.columns) == COLUMNS
    assert len(table) == len(network.snapshots) * 3

    a = table[table.bus == "A"].set_index("snapshot")
    load = network.loads_t.p_set
    solar = network.generators_t.p_max_pu["A solar"] * 50
    assert np.allclose(a.load, load["A"])
    assert np.allclose(a.supply_firm, 100)
    assert np.allclose(a.supply_vres, solar)
    assert np.allclose(a.supply_storage, 20)
    assert np.allclose(a.import_capacity, 40 * 0.25)
    assert np.allclose(a.model_price, network.buses_t.marginal_price["A"])

    b2 = table[table.bus == "B2"].set_index("snapshot")
    wind = network.generators_t.p_max_pu["B2 onwind"] * 40
    assert np.allclose(b2.load, load["B1"] + load["B2"])
    assert np.allclose(b2.supply_firm, 50)
    assert np.allclose(b2.supply_vres, wind)
    assert np.allclose(b2.import_capacity, 40 * 0.5)
    assert np.allclose(b2.model_price, network.buses_t.marginal_price["B2"])


def test_build_table_reads_network_and_prices(run_files):
    network, prices = run_files[-1]
    table = build_table(network, prices)
    assert set(table.run) == {"solved_2023"}
    assert set(table.year) == {2023}
    assert table.observed_price.notna().all()
    network = [f for f in DEFAULT_FEATURES if has_variables(table, f)]
    assert feature_matrix(table, network).notna().all().all()


def test_unknown_feature_raises(network):
    with pytest.raises(ValueError, match="Unknown features"):
        feature_matrix(extract(network), ["moon_phase"])


def test_extract_reads_solved_dispatch_flows_and_costs(network):
    n = network
    n.generators.loc["A gas", "marginal_cost"] = 70.0
    n.generators_t["p"] = n.get_switchable_as_dense("Generator", "p_max_pu")[
        ["A solar", "A gas"]
    ] * [25, 60]
    n.links_t["p0"] = pd.DataFrame({"A->B1": 10.0}, index=n.snapshots)
    n.links_t["p1"] = -n.links_t["p0"]
    table = extract(n)

    a = table[table.bus == "A"].set_index("snapshot")
    assert np.allclose(a.dispatch_vres, n.generators_t.p_max_pu["A solar"] * 25)
    assert np.allclose(a.dispatch_firm, 60)
    assert np.allclose(a.net_import, -10)
    assert np.allclose(table[table.bus == "B2"].net_import, 10)
    assert np.allclose(table.cost_gas, 70)
    assert np.allclose(table.storage_dispatch, 0)


def test_extract_finds_price_setter_and_congested_links(network):
    n = network
    n.generators.loc["A gas", "marginal_cost"] = 70.0
    n.generators_t["p"] = pd.DataFrame({"A gas": 60.0}, index=n.snapshots)
    n.buses_t.marginal_price["A"] = 70.0
    n.buses_t.marginal_price.iloc[:24, 0] = 90.0
    n.links_t["p0"] = pd.DataFrame({"A->B1": 20.0}, index=n.snapshots)
    n.links_t["p0"].iloc[:24] = 5.0
    a = extract(n).query("bus == 'A'").set_index("snapshot")

    assert (a.price_setter.iloc[24:] == "gas").all()
    assert (a.price_setter.iloc[:24] != "gas").all()
    # the link carries up to 0.5 * 40 MW from A, and 0.25 * 40 MW back
    assert np.allclose(a.export_capacity, 20)
    assert np.allclose(a.export_congested.iloc[24:], 20)
    assert np.allclose(a.export_congested.iloc[:24], 0)
    assert np.allclose(a.import_congested, 0)


def test_local_time_and_holidays_follow_the_country_of_the_bus():
    table = pd.DataFrame(
        {
            "snapshot": pd.to_datetime(["2023-07-01 10:00"] * 3 + ["2023-12-25 12:00"]),
            "bus": ["DE00", "FI00", "PT00", "GB00"],
        }
    )
    assert local_time(table).dt.hour.tolist() == [12, 13, 11, 12]
    assert holiday(table).tolist() == [0, 0, 0, 1]
