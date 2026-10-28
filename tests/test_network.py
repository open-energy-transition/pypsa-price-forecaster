import numpy as np
import pytest

from pypsa_price_forecaster.dataset import DEFAULT_FEATURES, build_table, feature_matrix
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
    assert feature_matrix(table, DEFAULT_FEATURES).notna().all().all()


def test_unknown_feature_raises(network):
    with pytest.raises(ValueError, match="Unknown features"):
        feature_matrix(extract(network), ["temperature"])
