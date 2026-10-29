"""
Synthetic networks shaped like the pypsa-eur-backcast networks.

Bus A trades with the copperplated zone B1+B2 over one NTC link. The marginal
prices are set directly instead of solving. The observed price adds a markup to
the model price that rises as the margin with imports gets tight, which the
learned models should pick up.
"""

import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa_price_forecaster.dataset import FEATURES

HOURS = 24 * 28


def make_network(year: int, seed: int) -> pypsa.Network:
    rng = np.random.default_rng(seed)
    snapshots = pd.date_range(f"{year}-01-01", periods=HOURS, freq="h")
    hour = snapshots.hour.to_numpy()
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", ["A", "B1", "B2"], carrier="AC")

    n.add(
        "Load",
        ["A", "B1", "B2"],
        bus=["A", "B1", "B2"],
        p_set=pd.DataFrame(
            {
                "A": 110
                + 40 * np.sin(2 * np.pi * (hour - 6) / 24)
                + rng.normal(0, 10, HOURS),
                "B1": 30 + rng.normal(0, 5, HOURS),
                "B2": 20 + rng.normal(0, 5, HOURS),
            },
            index=snapshots,
        ),
    )
    n.add("Generator", "A gas", bus="A", carrier="CCGT", p_nom=100, marginal_cost=60)
    n.add(
        "Generator",
        "A solar",
        bus="A",
        carrier="solar",
        p_nom=50,
        p_max_pu=np.clip(np.sin(2 * np.pi * (hour - 6) / 24), 0, None),
    )
    n.add("Generator", "B1 coal", bus="B1", carrier="coal", p_nom=50)
    n.add(
        "Generator",
        "B2 onwind",
        bus="B2",
        carrier="onwind",
        p_nom=40,
        p_max_pu=rng.uniform(0, 1, HOURS),
    )
    n.add(
        "Generator",
        ["A load shedding", "B1 load shedding", "B2 load shedding"],
        bus=["A", "B1", "B2"],
        carrier="load",
        p_nom=np.inf,
        marginal_cost=4000,
    )
    n.add("StorageUnit", "A PHS", bus="A", carrier="PHS", p_nom=20, max_hours=6)
    n.add(
        "Link",
        "A->B1",
        bus0="A",
        bus1="B1",
        carrier="DC",
        p_nom=40,
        p_max_pu=0.5,
        p_min_pu=-0.25,
    )
    n.add(
        "Link",
        "copper_B1_B2_0",
        bus0="B1",
        bus1="B2",
        carrier="copper",
        p_nom=np.inf,
        p_min_pu=-1,
    )
    n.buses_t["marginal_price"] = pd.DataFrame(
        rng.normal(60 + year % 10, 15, (HOURS, 3)),
        index=snapshots,
        columns=["A", "B1", "B2"],
    )
    return n


def observed_prices(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return wide observed prices with a scarcity markup on the model price."""
    rng = np.random.default_rng(seed)
    margin = FEATURES["margin_with_imports"](table)
    price = (
        table.model_price + 80 * np.exp(-4 * margin) - 10 + rng.normal(0, 3, len(table))
    )
    return table.assign(price=price).pivot(
        index="snapshot", columns="bus", values="price"
    )


@pytest.fixture
def network() -> pypsa.Network:
    return make_network(2023, seed=0)


@pytest.fixture
def run_files(tmp_path) -> list[tuple]:
    """Write solved networks and observed price CSVs of three years, return their paths."""
    from pypsa_price_forecaster.network import extract

    files = []
    for seed, year in enumerate([2019, 2021, 2023]):
        n = make_network(year, seed)
        network = tmp_path / f"solved_{year}.nc"
        prices = tmp_path / f"day_ahead_prices_{year}.csv"
        n.export_to_netcdf(network)
        observed_prices(extract(n), seed).to_csv(prices)
        files.append((network, prices))
    return files


@pytest.fixture
def table(run_files) -> pd.DataFrame:
    """Tables of the three runs with observed prices."""
    from pypsa_price_forecaster.dataset import build_table

    return pd.concat([build_table(*f) for f in run_files], ignore_index=True)


@pytest.fixture
def small_lgbm(monkeypatch):
    """Use fewer trees, as the report refits lgbm many times."""
    from pypsa_price_forecaster import models

    base_model = models.base_model

    def small(name, features):
        model = base_model(name, features)
        if name == "lgbm":
            model.estimator.set_params(n_estimators=100, learning_rate=0.1)
        return model

    monkeypatch.setattr(models, "base_model", small)
