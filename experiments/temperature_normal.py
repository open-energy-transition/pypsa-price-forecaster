"""
Replaces the daily temperature by its normal for the day of the year (round 20).

If the correction keeps its temperature gain on the normal, temperature works as a
season signal and needs no weather forecast; if not, the actual weather matters.

    pixi run python experiments/temperature_normal.py data/exogenous.parquet data/temperature_normal.parquet

Per bus and day of the year, the mean over the other years only (leave one year
out, so no year sees its own weather), smoothed with a centred 31-day window
around the year. 29 February takes the normal of 28 February.
"""

import sys

import numpy as np
import pandas as pd

WINDOW = 31


def normal(daily: pd.DataFrame) -> pd.DataFrame:
    """Return the smoothed mean temperature per day of the year (rows 1-365) and bus (columns)."""
    day = np.minimum(daily.index.dayofyear.to_numpy(), 365)
    # 29 February and 1 March share day 60 in leap years, close enough for a normal
    mean = daily.groupby(day).mean().reindex(range(1, 366))
    wrapped = pd.concat([mean.iloc[-WINDOW:], mean, mean.iloc[:WINDOW]])
    smooth = wrapped.rolling(WINDOW, center=True, min_periods=1).mean()
    return smooth.iloc[WINDOW:-WINDOW]


if __name__ == "__main__":
    source, out = sys.argv[1:]
    variables = pd.read_parquet(source, columns=["date", "bus", "temperature"])
    variables["date"] = pd.to_datetime(variables.date)
    wide = variables.pivot(index="date", columns="bus", values="temperature")
    years = sorted(wide.index.year.unique())
    parts = []
    for year in years:
        days = wide.index[wide.index.year == year]
        normals = normal(wide[wide.index.year != year])
        values = normals.loc[np.minimum(days.dayofyear.to_numpy(), 365)].set_axis(days)
        parts.append(values)
    result = (
        pd.concat(parts)
        .rename_axis(index="date", columns="bus")
        .stack()
        .rename("temperature")
        .reset_index()
    )
    print(
        "r with the actual temperature:",
        round(
            np.corrcoef(
                result.temperature,
                wide.stack()
                .reindex(pd.MultiIndex.from_frame(result[["date", "bus"]]))
                .to_numpy(),
            )[0, 1],
            3,
        ),
    )
    result.to_parquet(out, index=False)
    print(f"Wrote {len(result)} rows ({result.bus.nunique()} buses) to {out}")
