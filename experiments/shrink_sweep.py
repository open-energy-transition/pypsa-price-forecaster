"""
Sweeps the share of the correction that is applied, for free from stored predictions.

The band runs keep the corrected price at SHRINK (``point``) per held-out hour, so
the full correction is (point - model price) / SHRINK and any other share follows.
Skill per held-out year is 1 - median MAE over buses / that of raw PyPSA, as in run.py.

    pixi run python experiments/shrink_sweep.py results/bands_E162_year_bands_recipe_level.parquet ...
"""

import sys
from pathlib import Path

import pandas as pd

from pypsa_price_forecaster.dataset import TARGET, read_tables
from pypsa_price_forecaster.models import SHRINK

SHARES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if __name__ == "__main__":
    tables = read_tables(sorted(map(str, Path("data/tables").glob("*.parquet"))))
    keys = ["run", "snapshot", "bus"]
    for path in sys.argv[1:]:
        bands = pd.read_parquet(path, columns=[*keys, "year", TARGET, "point"])
        rows = bands.merge(tables[[*keys, "model_price"]], on=keys)
        full = (rows.point - rows.model_price) / SHRINK
        skill = {}
        for share in SHARES:
            error = (rows.model_price + share * full - rows[TARGET]).abs()
            raw = (rows.model_price - rows[TARGET]).abs()
            mae = (
                pd.DataFrame({"year": rows.year, "bus": rows.bus, "c": error, "r": raw})
                .groupby(["year", "bus"])
                .mean()
                .groupby("year")
                .median()
            )
            skill[share] = 1 - mae.c / mae.r
        skill = pd.DataFrame(skill)
        print(f"\n== {path}")
        print(skill.round(3).to_string())
        print("mean :", skill.mean().round(3).to_dict())
        print("worst:", skill.min().round(3).to_dict())
