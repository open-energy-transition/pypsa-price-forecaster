"""
Where the gain of the context features comes from, on the corrections kept by ``experiments/lab.py``.

Usage: ``PYTHONPATH=experiments pixi run -e lab python experiments/lab_context.py``

Prints, for ``lgbm`` and ``lgbm_context``: the error of the daily mean price
and of the hours around it per held-out year, the hourly correlation with the
observed price, the zones that gain, and the skill with the share of the
correction chosen on the other held-out years.
"""

import numpy as np
import pandas as pd
from lab import OUT, SHARE, load, score

from pypsa_price_forecaster.dataset import TARGET

CANDIDATES = ["lgbm", "lgbm_context"]
SHARES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

data = load()
t = data.table[data.labelled]
observed = t[TARGET]
day = [t.bus, t.snapshot.dt.floor("D")]


def price(name: str | None, split: str = "year") -> pd.Series:
    """Model price with `SHARE` of the correction of candidate `name`, the raw price without name."""
    if name is None:
        return t.model_price
    file = OUT / f"{name}_{split}_s0.parquet"
    correction = pd.read_parquet(file).correction.to_numpy()[data.labelled]
    return t.model_price + SHARE * correction


def level_and_shape(predicted: pd.Series) -> pd.DataFrame:
    """Median over buses of the MAE per year, of the daily mean error and of the rest."""
    error = predicted - observed
    level = error.groupby(day).transform("mean")
    parts = pd.DataFrame(
        {"mae": error.abs(), "level": level.abs(), "shape": (error - level).abs()}
    )
    return parts.groupby([t.year, t.bus]).mean().groupby("year").median()


def correlation(predicted: pd.Series) -> pd.Series:
    """Median over buses of the hourly correlation with the observed price per year."""
    both = pd.DataFrame({"predicted": predicted, "observed": observed})
    by_bus = both.groupby([t.year, t.bus]).apply(lambda g: g.predicted.corr(g.observed))
    return by_bus.groupby("year").median()


def bus_mae(predicted: pd.Series) -> pd.DataFrame:
    return (predicted - observed).abs().groupby([t.bus, t.year]).mean().unstack()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    names = {"raw": None} | {name: name for name in CANDIDATES}
    for split in ("year", "forward"):
        parts = {label: level_and_shape(price(n, split)) for label, n in names.items()}
        print(f"\n{split} split, MAE, of the daily mean and of the hours around it")
        print(pd.concat(parts, axis=1).dropna().round(2).to_string())
    print("\nyear split, hourly correlation with the observed price")
    print(pd.DataFrame({k: correlation(price(n)) for k, n in names.items()}).round(3).T)

    gain = (bus_mae(price("lgbm")) - bus_mae(price("lgbm_context"))) / bus_mae(
        price(None)
    )
    print(
        f"\nyear split: {(gain > 0).sum().sum()} of {gain.notna().sum().sum()} bus-years "
        f"and {(gain.mean(axis=1) > 0).sum()} of {len(gain)} buses gain with the context features"
    )
    print(gain.mean(axis=1).sort_values().round(3).to_string())

    print(
        "\nmean skill by share of the correction, and with the share chosen on the other years"
    )
    for split in ("year", "forward"):
        for name in CANDIDATES:
            file = OUT / f"{name}_{split}_s0.parquet"
            correction = pd.read_parquet(file).correction.to_numpy()
            skill = pd.DataFrame({s: score(data, correction, s).skill for s in SHARES})
            nested = [
                skill.loc[y, skill.drop(index=y).mean().idxmax()] for y in skill.index
            ]
            print(
                split,
                name,
                skill.mean().round(3).to_dict(),
                "nested",
                np.mean(nested).round(3),
            )
