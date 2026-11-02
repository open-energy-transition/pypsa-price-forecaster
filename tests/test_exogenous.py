import numpy as np
import pandas as pd

from pypsa_price_forecaster.exogenous import (
    RUNOFF_CAP,
    reservoir_anomaly,
    runoff_anomaly,
)


def test_runoff_anomaly_compares_with_the_climate_and_is_capped():
    days = pd.date_range("1991-01-01", "2021-12-31", freq="D")
    runoff = pd.DataFrame({"wet": 1.0, "dry": 0.001, "none": 0.0}, index=days)
    runoff.loc["2021", "wet"] = 2.0
    runoff.loc["2021-12-31", ["dry", "none"]] = 1.0
    anomaly = runoff_anomaly(runoff, days=10)
    assert anomaly.loc["2020-06-01", "wet"] == 1
    assert np.isclose(anomaly.loc["2021-06-01", "wet"], 2, rtol=0.05)
    assert anomaly.loc["2021-12-31", "dry"] == RUNOFF_CAP
    assert np.isnan(anomaly.loc["2021-12-31", "none"])


def test_reservoir_anomaly_uses_the_week_published_before(tmp_path):
    weeks = pd.date_range("2015-01-05", "2024-12-30", freq="7D")
    filling = (
        pd.Series(100.0, index=weeks).rename("filling").rename_axis("utc_timestamp")
    )
    filling.loc["2024-06-03"] = 50.0
    filling.tz_localize("UTC").to_csv(tmp_path / "reservoir_filling_NO.csv")
    days = pd.date_range("2024-06-01", "2024-06-20", freq="D")
    anomaly = reservoir_anomaly(["NO"], days, tmp_path).NO
    assert anomaly.loc["2024-06-09"] == 0
    assert anomaly.loc["2024-06-10"] == -0.5
    assert anomaly.loc["2024-06-17"] == 0
