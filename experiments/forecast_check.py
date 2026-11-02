"""
Do the day-ahead forecasts the market cleared on track the error the correction leaves?

Correlates the out-of-sample residual of report/prices.parquet (market minus
corrected price, hourly) with the gap between the ENTSO-E day-ahead load and
wind and solar forecasts and the model's load and available wind and solar per
zone (`ppf forecasts`).

    pixi run python experiments/forecast_check.py
"""

from pathlib import Path

import pandas as pd

from pypsa_price_forecaster.dataset import attach_exogenous, feature_matrix, read_tables

FEATURES = ["load_forecast_error", "vres_forecast_error"]


def main() -> None:
    table = read_tables(sorted(Path("data/tables").glob("*.parquet")))
    table = attach_exogenous(table, pd.read_parquet("data/forecasts.parquet"))
    table = pd.concat([table, feature_matrix(table, FEATURES)], axis=1)
    prices = pd.read_parquet("report/prices.parquet")
    frame = table[["run", "snapshot", "bus", *FEATURES]].merge(
        prices, on=["run", "snapshot", "bus"]
    )
    frame["residual"] = frame.observed_price - frame.lgbm
    frame["raw_residual"] = frame.observed_price - frame.pypsa
    frame["year"] = frame.snapshot.dt.year
    frame["day"] = frame.snapshot.dt.floor("D")
    print(frame[FEATURES].describe().round(3).to_string())
    rows = {}
    for v in FEATURES:
        f = frame.dropna(subset=[v, "residual"])
        f = f[f[v].abs() < 1]  # forecasts of a different zone extent
        within = f.groupby(["bus", "year"])
        daily = f.groupby(["bus", "day"])[[v, "residual"]].mean()
        rows[v] = {
            "rows": len(f),
            "hourly": f[v].corr(f.residual, method="spearman"),
            "hourly, within zone-year": (f[v] - within[v].transform("mean")).corr(
                f.residual - within.residual.transform("mean"), method="spearman"
            ),
            "zone-day": daily[v].corr(daily.residual, method="spearman"),
            "hourly vs raw error": f[v].corr(f.raw_residual, method="spearman"),
        }
    print("Spearman correlation with the residual (market - corrected):")
    print(pd.DataFrame(rows).T.round(3).to_string())
    by_bus = frame.dropna(subset=FEATURES).groupby("bus")[FEATURES].median()
    print("Median per bus, largest gaps (zone extent or capacity differs):")
    print(by_bus.abs().sort_values("vres_forecast_error").tail(8).round(2).to_string())


if __name__ == "__main__":
    main()
