import numpy as np
import pandas as pd
import pytest

from pypsa_price_forecaster import entsoe
from pypsa_price_forecaster.entsoe import Unavailable, observed_prices, read_area_to_bus

SNAPSHOTS = pd.date_range("2023-01-01", periods=48, freq="h")


@pytest.fixture
def queries(monkeypatch) -> list:
    """Replace the ENTSO-E query by quarter-hourly prices of DE_LU with a gap, record the queries."""
    calls = []

    def query(area: str, year: int) -> pd.Series:
        calls.append((area, year))
        if area != "DE_LU":
            raise Unavailable("prices are given in ['GBP'], not only EUR")
        index = pd.date_range(f"{year}-01-01", periods=48 * 4, freq="15min", tz="UTC")
        prices = pd.Series(np.repeat(np.arange(48.0), 4), index=index, name=area)
        prices.iloc[4 * 10 : 4 * 12] = np.nan  # 2 hour gap, interpolated
        prices.iloc[4 * 20 : 4 * 25] = np.nan  # 5 hour gap, kept missing
        return prices.dropna()

    monkeypatch.setattr(entsoe, "query_area_year", query)
    return calls


def test_observed_prices_per_bus(queries, tmp_path):
    buses = pd.Index(["DE00", "LUG1", "GB00", "FR15", "ZZ00"])
    prices = observed_prices(buses, SNAPSHOTS, tmp_path)

    assert list(prices.columns) == ["DE00", "LUG1"]
    assert prices.index.equals(SNAPSHOTS)
    pd.testing.assert_series_equal(prices.DE00, prices.LUG1, check_names=False)
    expected = pd.Series(np.arange(48.0), index=SNAPSHOTS)
    expected.iloc[20:25] = np.nan
    np.testing.assert_allclose(prices.DE00, expected)
    assert sorted(queries) == [("DE_LU", 2023), ("GB", 2023)]


def test_downloads_are_cached(queries, tmp_path):
    buses = pd.Index(["DE00", "GB00"])
    first = observed_prices(buses, SNAPSHOTS, tmp_path)
    second = observed_prices(buses, SNAPSHOTS, tmp_path)

    pd.testing.assert_frame_equal(first, second)
    assert len(queries) == 2
    assert (tmp_path / "day_ahead_prices_GB_2023.skip").exists()


def test_packaged_area_map():
    area_to_bus = read_area_to_bus()
    assert area_to_bus["DE00"] == "DE_LU"
    assert area_to_bus["GBNI"] == "IE_SEM"
