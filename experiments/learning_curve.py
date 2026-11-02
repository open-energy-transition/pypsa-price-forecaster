"""
Does the correction keep improving with more training years?

Every year is predicted by models trained on every subset of 1 to N-1 of the
other years. Skill is 1 - median MAE over buses / that of raw PyPSA, as in run.py.

    pixi run python experiments/learning_curve.py data/tables/*.parquet --exclude-year 2022
"""

import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd

from pypsa_price_forecaster.dataset import read_tables
from pypsa_price_forecaster.evaluate import bus_metrics, summary
from pypsa_price_forecaster.models import make_model

RESULTS = Path(__file__).parent / "results_learning_curve.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", nargs="+")
    parser.add_argument("--exclude-year", type=int, action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    table = read_tables(args.tables)
    table = table[~table.year.isin(args.exclude_year)]
    years = sorted(table.year.unique())
    rows = []
    for test_year in years:
        test = table[table.year == test_year]
        others = [y for y in years if y != test_year]
        for k in range(1, len(others) + 1):
            for train_years in itertools.combinations(others, k):
                logging.info(f"Predicting {test_year} from {train_years}.")
                model = make_model("lgbm").fit(table[table.year.isin(train_years)])
                predictions = pd.DataFrame(
                    {"pypsa": test.model_price, "lgbm": model.predict(test)}, index=test.index
                )
                mae = summary(bus_metrics(test, predictions)).mae.droplevel("year")
                rows.append(
                    {
                        "test_year": test_year,
                        "train_years": "+".join(map(str, train_years)),
                        "n_train": k,
                        "mae_pypsa": mae["pypsa"],
                        "mae_lgbm": mae["lgbm"],
                        "skill": 1 - mae["lgbm"] / mae["pypsa"],
                    }
                )
                pd.DataFrame(rows).to_csv(RESULTS, index=False)

    result = pd.DataFrame(rows)
    per_year = result.groupby(["n_train", "test_year"]).skill.mean().unstack()
    print("Mean skill over training subsets, per held-out year:")
    print(per_year.round(3).to_string())
    print("\nMean / worst over held-out years:")
    print(pd.DataFrame({"mean": per_year.mean(axis=1), "worst": per_year.min(axis=1)}).round(3))
    print("=== DONE")


if __name__ == "__main__":
    main()
