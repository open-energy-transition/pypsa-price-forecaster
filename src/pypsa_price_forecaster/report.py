"""
Writes the report of the default model that the dashboard shows.

The report scores the default models out of sample on all tables and measures
the impact of every feature by ablation: the default ``lgbm`` refitted without
each default feature and with each other feature. Repeats with other random
seeds give the noise level below which a change in MAE means nothing. The
report is a directory of small files, committed to git so the dashboard on
the branch ``master`` always shows the latest results:

- ``meta.json``: commit, runs, split, features and LightGBM parameters
- ``scores.csv``: scores per model, see :func:`pypsa_price_forecaster.evaluate.scores`
- ``scores_by_year.csv``: the same per model and year
- ``bus_metrics.csv``: metrics per model, year and bus
- ``monthly.csv``: median over buses of the MAE per model, year and month
- ``ablation.csv``: scores of ``lgbm`` without or with each feature
- ``importance.csv``: share of the split gain per feature of ``lgbm`` fitted on all rows
- ``prices.parquet``: observed, PyPSA and out-of-sample ``lgbm`` price per hour and bus
- ``history.csv``: headline scores of every report written, one row appended per report
"""

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

from pypsa_price_forecaster import __version__
from pypsa_price_forecaster.dataset import DEFAULT_FEATURES, FEATURES, TARGET
from pypsa_price_forecaster.evaluate import bus_metrics, cross_validate, scores
from pypsa_price_forecaster.models import DEFAULT_MODELS, labelled, make_model

logger = logging.getLogger(__name__)

SEEDS = [1, 2]


def default_split(table: pd.DataFrame) -> str:
    """Split by year if the tables span several years, else by month, else by week."""
    if table.year.nunique() > 1:
        return "year"
    return "month" if table.snapshot.dt.month.nunique() > 1 else "week"


def git_commit() -> tuple[str, bool]:
    """Return the short commit of the working directory and whether the source has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return commit, bool(status.strip())


def lgbm_params() -> dict:
    """Return the parameters of the default ``lgbm`` that differ from those of LightGBM."""
    params = make_model("lgbm").estimator.get_params()
    defaults = LGBMRegressor().get_params()
    return {k: v for k, v in params.items() if v != defaults.get(k)}


def by_year(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the scores per model and year."""
    return (
        pd.concat(
            {
                year: scores(table.loc[rows], predictions.loc[rows])
                for year, rows in table.groupby("year").groups.items()
            },
            names=["year", "model"],
        )
        .swaplevel()
        .loc[list(predictions.columns)]
    )


def monthly(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the median over buses of the MAE per model, year and month."""
    rows = table[["year", "bus", "snapshot", TARGET]].join(predictions)
    rows = rows.dropna(subset=[TARGET])
    errors = (rows[predictions.columns].sub(rows[TARGET], axis=0)).abs()
    groups = [rows.year, rows.snapshot.dt.month.rename("month"), rows.bus]
    return (
        errors.groupby(groups)
        .mean()
        .groupby(["year", "month"])
        .median()
        .rename_axis(columns="model")
        .stack()
        .rename("mae")
        .reorder_levels(["model", "year", "month"])
        .sort_index()
    )


def ablation(
    table: pd.DataFrame, split: str, folds: int, features: list[str]
) -> pd.DataFrame:
    """Return the scores of ``lgbm`` without each of `features` and with each other feature."""
    variants = [("without", f, [g for g in features if g != f]) for f in features]
    variants += [("with", f, [*features, f]) for f in FEATURES if f not in features]
    result = {}
    for change, feature, used in variants:
        logger.info(f"Ablation: lgbm {change} {feature}.")
        predictions = cross_validate(table, ["lgbm"], used, split, folds)
        result[(change, feature)] = scores(table, predictions).loc["lgbm"]
    return pd.DataFrame(result).T.rename_axis(["change", "feature"])


def importance(table: pd.DataFrame, features: list[str]) -> pd.Series:
    """Return the share of the split gain per feature of ``lgbm`` fitted on all rows."""
    booster = make_model("lgbm", features).fit(table).estimator.booster_
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    return (gain / gain.sum()).sort_values(ascending=False).rename("gain_share")


def prices(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Return the observed, PyPSA and ``lgbm`` price of the rows with observed price.

    Rounded to 0.1 EUR/MWh, which shrinks the compressed file to a size fit for git.
    """
    rows = table[["run", "snapshot", "bus", TARGET]].join(
        predictions[["pypsa", "lgbm"]]
    )
    rows = rows.dropna(subset=[TARGET]).reset_index(drop=True)
    values = [TARGET, "pypsa", "lgbm"]
    rows[values] = rows[values].round(1).astype("float32")
    return rows.astype({"run": "category", "bus": "category"})


def write_report(
    table: pd.DataFrame,
    out: str | Path,
    split: str | None = None,
    folds: int = 4,
    features: list[str] | None = None,
    with_ablation: bool = True,
) -> pd.DataFrame:
    """
    Write the report of the default models on `table` to the directory `out` and return the scores.

    Parameters
    ----------
    table : pd.DataFrame
        Tables of all runs with observed prices.
    out : str | Path
        Report directory, see the module docstring.
    split : str | None
        Cross-validation split, defaults to :func:`default_split`.
    folds : int
        Number of folds of the week split.
    features : list[str] | None
        Features of the learned models, defaults to ``DEFAULT_FEATURES``.
    with_ablation : bool
        Whether to run the feature ablation and the seed repeats, which refit
        ``lgbm`` once per feature.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = labelled(table).reset_index(drop=True)
    split = split or default_split(table)
    features = list(DEFAULT_FEATURES if features is None else features)

    predictions = cross_validate(table, DEFAULT_MODELS, features, split, folds)
    result = scores(table, predictions)
    result.to_csv(out / "scores.csv", index_label="model")
    by_year(table, predictions).to_csv(out / "scores_by_year.csv")
    bus_metrics(table, predictions).to_csv(out / "bus_metrics.csv")
    monthly(table, predictions).to_csv(out / "monthly.csv")
    prices(table, predictions).to_parquet(
        out / "prices.parquet", compression="zstd", compression_level=19
    )
    importance(table, features).to_csv(out / "importance.csv", index_label="feature")

    seed_mae = [result.mae["lgbm"]]
    if with_ablation:
        ablation(table, split, folds, features).to_csv(out / "ablation.csv")
        for seed in SEEDS:
            logger.info(f"Repeating lgbm with seed {seed}.")
            repeat = cross_validate(
                table,
                ["lgbm"],
                features,
                split,
                folds,
                {"lgbm": {"random_state": seed}},
            )
            seed_mae.append(scores(table, repeat).mae["lgbm"])

    commit, dirty = git_commit()
    created = datetime.now(UTC).isoformat(timespec="seconds")
    meta = {
        "created": created,
        "commit": commit,
        "dirty": dirty,
        "version": __version__,
        "runs": sorted(table.run.unique()),
        "years": sorted(int(y) for y in table.year.unique()),
        "split": split,
        "folds": int(table.snapshot.dt.month.nunique()) if split == "month" else folds,
        "buses": int(table.bus.nunique()),
        "hours": len(table),
        "features": features,
        "lgbm_params": lgbm_params(),
        "seed_mae": [round(float(m), 4) for m in seed_mae],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    headline = result.loc[["pypsa", "lgbm"], ["mae", "r", "mae_top5", "mae_low5"]]
    entry = pd.DataFrame(
        [
            {"created": created, "commit": commit, "split": split}
            | {"runs": " ".join(meta["runs"])}
            | {f"{k}_{m}": v for (m, k), v in headline.stack().round(3).items()}
        ]
    )
    history = out / "history.csv"
    entry.to_csv(history, mode="a", header=not history.exists(), index=False)
    return result
