"""
Model lab: other model families and training routines for the correction.

Usage: ``pixi run -e lab python experiments/lab.py <candidate> [...] [--split year forward] [--seed 0] [--threads 8]``

A candidate maps the rows of the training years to the whole correction
(observed minus model price) of a held-out year. The corrections are kept per
row in ``results/lab/<candidate>_<split>_s<seed>.parquet``, so that correction
shares and averages of candidates are scored without refitting
(``experiments/lab_scores.py``), and the scores are appended to
``experiments/results_lab.csv``. The scores are those of ``run.py``: per
held-out year the median over buses of the MAE, and the skill 1 - MAE / raw MAE,
with ``SHARE`` of the correction applied as ``lgbm`` does.

The last fold of the forward split (the last year from all earlier ones) is the
fold of that year in the year split, and is copied from it when it exists.
"""

import argparse
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pyarrow.parquet import read_schema

from pypsa_price_forecaster.dataset import (
    CONTEXT_FEATURES,
    DEFAULT_FEATURES,
    SETTERS,
    TARGET,
    europe_total,
    feature_matrix,
    read_tables,
)
from pypsa_price_forecaster.models import SHRINK as SHARE

TABLES = sorted(Path("data/tables").glob("*.parquet"))
OUT = Path("results/lab")
RESULTS = Path("experiments/results_lab.csv")
THREADS = 8
# EUR/MWh, lowest price level a relative correction is taken relative to
RELATIVE_FLOOR = 20.0
# the lgbm of the package and the larger trees of round 19
BASE = dict(
    objective="l1",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    verbose=-1,
)
LARGER = dict(n_estimators=1200, num_leaves=127, min_child_samples=25)


def more_context(t: pd.DataFrame) -> pd.DataFrame:
    """
    Second set of context features, tried after ``CONTEXT_FEATURES`` of the package helped.

    More of the day and of Europe, and where an hour sits in its week and its
    year. None of them added to the first set, so they are not in the package.
    """
    series = [t.run, t.bus]
    day = [t.run, t.bus, t.snapshot.dt.floor("D")]
    hour = [t.run, t.snapshot]
    residual = t.load - t.supply_vres
    price = t.model_price
    price_day = price.groupby(day).transform("mean").clip(lower=1.0)
    price_range = price.groupby(day).transform("max") - price.groupby(day).transform(
        "min"
    )
    # rows are ordered by hour within every run and bus
    price_week = price.groupby(series).transform(
        lambda p: p.rolling(168, center=True, min_periods=24).mean()
    )
    total = europe_total(t, ["load", "supply_vres", "dispatch_vres"])
    firm = europe_total(t, ["dispatch_firm", "supply_firm"])
    return pd.DataFrame(
        {
            "price_range_day": price_range / price_day,
            "load_rel_day": t.load / t.load.groupby(day).transform("mean"),
            "price_rank_europe": price.groupby(hour).rank(pct=True),
            "curtailment_europe": (total.supply_vres - total.dispatch_vres)
            / total.load,
            "firm_utilisation_europe": firm.dispatch_firm / firm.supply_firm,
            "price_rel_week": price / price_week.clip(lower=1.0),
            "price_rel_year": price
            / price.groupby(series).transform("median").clip(lower=1.0),
            "residual_rel_year": residual / residual.groupby(series).transform("max"),
        }
    ).replace([np.inf, -np.inf], np.nan)


# first set, fixed before any run: where an hour sits in its day, and Europe as a whole
CONTEXT = CONTEXT_FEATURES
DAY, EUROPE = CONTEXT[:5], CONTEXT[5:]
# second set, fixed after the first one helped
DAY2 = ["price_range_day", "load_rel_day"]
EUROPE2 = ["price_rank_europe", "curtailment_europe", "firm_utilisation_europe"]
LONG = ["price_rel_week", "price_rel_year", "residual_rel_year"]
CACHED = DEFAULT_FEATURES + CONTEXT + DAY2 + EUROPE2 + LONG


@dataclass
class Data:
    """The table, the features of every row and the correction to learn."""

    table: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    labelled: np.ndarray

    @property
    def day(self) -> list[pd.Series]:
        t = self.table
        return [t.run, t.bus, t.snapshot.dt.floor("D")]

    def price_level(self) -> np.ndarray:
        """Mean model price of the day of the bus, at least `RELATIVE_FLOOR`."""
        level = self.table.model_price.groupby(self.day).transform("mean")
        return level.clip(lower=RELATIVE_FLOOR).to_numpy()


def load() -> Data:
    table = read_tables(TABLES)
    # rows are ordered by snapshot within every run and bus, which more_context needs
    ordered = table.groupby(["run", "bus"]).snapshot.is_monotonic_increasing
    assert ordered.all()
    cache = OUT / "features.parquet"
    if cache.exists() and set(CACHED) <= set(read_schema(cache).names):
        X = pd.read_parquet(cache)
    else:
        X = feature_matrix(table, DEFAULT_FEATURES + CONTEXT).join(more_context(table))
        OUT.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache)
    assert len(X) == len(table)
    X["price_setter"] = pd.Categorical(
        X.price_setter.astype(object), categories=SETTERS
    )
    y = (table[TARGET] - table.model_price).to_numpy()
    return Data(table, X, y, ~np.isnan(y))


def yearly_mae(data: Data, price: np.ndarray) -> pd.Series:
    """Median over buses of the MAE of `price` per year, over the rows with observed and predicted price."""
    rows = data.labelled & ~np.isnan(price)
    t = data.table[rows]
    error = pd.Series(np.abs(price[rows] - t[TARGET].to_numpy()), index=t.index)
    return error.groupby([t.year, t.bus]).mean().groupby("year").median()


def score(data: Data, correction: np.ndarray, share: float = SHARE) -> pd.DataFrame:
    """Raw and corrected MAE and the skill per held-out year, with `share` of `correction` applied."""
    price = data.table.model_price.to_numpy()
    raw = yearly_mae(data, np.where(np.isnan(correction), np.nan, price))
    mae = yearly_mae(data, price + share * correction)
    return pd.DataFrame({"raw": raw, "mae": mae, "skill": 1 - mae / raw})


Candidate = Callable[[Data, np.ndarray, np.ndarray, int], "np.ndarray | dict"]


def lgbm(
    features: list[str] = DEFAULT_FEATURES,
    weight: str | None = None,
    relative: bool = False,
    **params,
) -> Candidate:
    """
    LightGBM on the correction, as the ``lgbm`` of the package without its share.

    `weight` ``zone_year`` weights every row by 1 / raw MAE of its bus and year,
    so that every bus and year counts alike, as in the skill, ``shift`` by how
    much it looks like the rows to predict (:func:`shift_weights`). With
    `relative`, the correction is learnt as a share of the price level of the day.
    """

    def run(data, train, test, seed):
        fit = train & data.labelled
        scale = data.price_level() if relative else np.ones(len(data.y))
        model = LGBMRegressor(**(BASE | params), random_state=seed, n_jobs=THREADS)
        rows = np.flatnonzero(fit)
        weights = None
        if weight == "zone_year":
            t = data.table[fit]
            error = pd.Series(np.abs(data.y[fit]), index=t.index)
            weights = 1 / error.groupby([t.year, t.bus]).transform("mean").to_numpy()
        if weight == "shift":
            weights = shift_weights(data.X[features], fit, test, seed)
        if weights is not None:
            # rows drawn by their weight: LightGBM is very slow with weights and the L1 loss
            rng = np.random.default_rng(seed)
            rows = np.sort(rng.choice(rows, len(rows), p=weights / weights.sum()))
        model.fit(data.X.iloc[rows][features], (data.y / scale)[rows])
        return model.predict(data.X.loc[test, features]) * scale[test]

    return run


def shift_weights(
    X: pd.DataFrame, fit: np.ndarray, test: np.ndarray, seed: int, sample: int = 300_000
) -> np.ndarray:
    """
    Weight of every `fit` row by how much its features look like those of the `test` rows.

    A small classifier tells rows of the year to predict from training rows by
    their features alone (no observed price is involved, so this also works for
    a future run). Its odds are the density ratio between the two, clipped to
    0.2-5: the correction is learnt mostly from hours like those it will correct.
    """
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(seed)
    rows = [
        rng.choice(np.flatnonzero(mask), min(sample, mask.sum()), replace=False)
        for mask in (fit, test)
    ]
    classifier = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=500,
        class_weight="balanced",
        random_state=seed,
        n_jobs=THREADS,
        verbose=-1,
    )
    labels = np.repeat([0, 1], [len(r) for r in rows])
    classifier.fit(X.iloc[np.concatenate(rows)], labels)
    odds = classifier.predict_proba(X[fit])[:, 1]
    weights = np.clip(odds / (1 - odds), 0.2, 5.0)
    return weights / weights.mean()


def staged(
    stages: tuple[int, ...] = (200, 400, 800, 1200, 2000, 3000), **params
) -> Candidate:
    """One LightGBM fit with the most trees of `stages`, predicted with each number of trees."""

    def run(data, train, test, seed):
        fit = train & data.labelled
        model = LGBMRegressor(
            **(BASE | params | dict(n_estimators=max(stages))),
            random_state=seed,
            n_jobs=THREADS,
        )
        model.fit(data.X.loc[fit, DEFAULT_FEATURES], data.y[fit])
        X = data.X.loc[test, DEFAULT_FEATURES]
        return {str(n): model.predict(X, num_iteration=n) for n in stages}

    return run


def jackknife(candidate: Candidate) -> Candidate:
    """
    `candidate` fitted with each training year left out once: the mean and the median correction.

    The median keeps what the fits agree on, so a correction that hangs on a
    single training year does not pass.
    """

    def run(data, train, test, seed):
        years = sorted(data.table.year[train].unique())
        if len(years) < 2:
            single = candidate(data, train, test, seed)
            return {"mean": single, "median": single}
        fits = np.array(
            [
                candidate(data, train & (data.table.year != y).to_numpy(), test, seed)
                for y in years
            ]
        )
        return {"mean": fits.mean(axis=0), "median": np.median(fits, axis=0)}

    return run


def per_year(candidate: Candidate) -> Candidate:
    """`candidate` fitted on each training year alone: the mean and the median correction."""

    def run(data, train, test, seed):
        years = sorted(data.table.year[train].unique())
        fits = np.array(
            [
                candidate(data, train & (data.table.year == y).to_numpy(), test, seed)
                for y in years
            ]
        )
        return {"mean": fits.mean(axis=0), "median": np.median(fits, axis=0)}

    return run


def level_shape(features: list[str] = DEFAULT_FEATURES, **params) -> Candidate:
    """
    The correction split into the mean of the day and the hours around it.

    The level model sees one row per bus and day, the daily means of the numeric
    features, and learns the mean correction of the day. The shape model sees
    the hours and learns the correction minus its mean of the day.
    """
    numeric = [f for f in features if f not in ("price_setter", "hour")]

    def run(data, train, test, seed):
        fit = train & data.labelled
        y = pd.Series(data.y)
        keys = [k.to_numpy() for k in data.day]
        level = y.groupby(keys).transform("mean").to_numpy()
        shape = LGBMRegressor(**(BASE | params), random_state=seed, n_jobs=THREADS)
        shape.fit(data.X.loc[fit, features], (data.y - level)[fit])
        daily = data.X[numeric].groupby(keys).transform("mean")
        first = (
            ~pd.DataFrame(dict(zip("rbd", keys, strict=True))).duplicated().to_numpy()
        )
        days = first & ~np.isnan(level)
        model = LGBMRegressor(
            **(BASE | params | dict(min_child_samples=20)),
            random_state=seed,
            n_jobs=THREADS,
        )
        model.fit(daily[train & days], level[train & days])
        parts = {
            "level": model.predict(daily[test]),
            "shape": shape.predict(data.X.loc[test, features]),
        }
        return parts | {"sum": parts["level"] + parts["shape"]}

    return run


def catboost(features: list[str] = DEFAULT_FEATURES, **params) -> Candidate:
    def run(data, train, test, seed):
        from catboost import CatBoostRegressor

        fit = train & data.labelled
        X = data.X[features].assign(
            price_setter=data.X.price_setter.astype(object).fillna("none")
        )
        model = CatBoostRegressor(
            **(
                dict(loss_function="MAE", iterations=1500, learning_rate=0.06, depth=8)
                | params
            ),
            cat_features=["price_setter"],
            random_seed=seed,
            thread_count=THREADS,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X[fit], data.y[fit])
        return model.predict(X[test])

    return run


def xgboost(features: list[str] = DEFAULT_FEATURES, **params) -> Candidate:
    def run(data, train, test, seed):
        from xgboost import XGBRegressor

        fit = train & data.labelled
        defaults = dict(
            objective="reg:absoluteerror",
            n_estimators=800,
            learning_rate=0.03,
            max_leaves=63,
            max_depth=0,
            grow_policy="lossguide",
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.9,
            tree_method="hist",
            enable_categorical=True,
        )
        model = XGBRegressor(**(defaults | params), random_state=seed, n_jobs=THREADS)
        model.fit(data.X.loc[fit, features], data.y[fit])
        return model.predict(data.X.loc[test, features])

    return run


def neural(kind: str, **options) -> Candidate:
    def run(data, train, test, seed):
        import lab_nn

        return lab_nn.fit_predict(kind, data, train, test, seed, THREADS, **options)

    return run


ALL = DEFAULT_FEATURES + CONTEXT
CANDIDATES: dict[str, Candidate] = {
    # references: the package defaults and the larger trees of round 19
    "lgbm": lgbm(),
    "lgbm_larger": lgbm(**LARGER),
    # training routines
    "lgbm_weighted": lgbm(weight="zone_year"),
    "lgbm_shift": lgbm(weight="shift"),
    "lgbm_staged": staged(),
    "lgbm_slow": lgbm(learning_rate=0.01, n_estimators=2400),
    "lgbm_relative": lgbm(relative=True),
    "lgbm_huber": lgbm(objective="huber", alpha=20.0),
    "lgbm_linear_tree": lgbm(
        objective="huber", alpha=20.0, linear_tree=True, linear_lambda=1.0
    ),
    "lgbm_half_features": lgbm(colsample_bytree=0.5),
    "lgbm_extra_trees": lgbm(extra_trees=True),
    "lgbm_smooth": lgbm(path_smooth=100.0, reg_lambda=10.0),
    "lgbm_dart": lgbm(boosting_type="dart", n_estimators=400, learning_rate=0.1),
    "lgbm_jackknife": jackknife(lgbm()),
    "lgbm_per_year": per_year(lgbm()),
    "lgbm_level_shape": level_shape(),
    # hours around a row and the other zones
    "lgbm_context": lgbm(ALL),
    "lgbm_day": lgbm(DEFAULT_FEATURES + DAY),
    "lgbm_europe": lgbm(DEFAULT_FEATURES + EUROPE),
    "lgbm_larger_context": lgbm(ALL, **LARGER),
    "lgbm_context_day2": lgbm(ALL + DAY2),
    "lgbm_context_europe2": lgbm(ALL + EUROPE2),
    "lgbm_context_long": lgbm(ALL + LONG),
    "lgbm_context_all": lgbm(ALL + DAY2 + EUROPE2 + LONG),
    # the first context set without each feature, and each feature alone
    **{f"lgbm_context_no_{f}": lgbm([g for g in ALL if g != f]) for f in CONTEXT},
    **{f"lgbm_plus_{f}": lgbm(DEFAULT_FEATURES + [f]) for f in CONTEXT},
    "lgbm_context_jackknife": jackknife(lgbm(ALL)),
    "lgbm_context_weighted": lgbm(ALL, weight="zone_year"),
    "lgbm_context_shift": lgbm(ALL, weight="shift"),
    # other tree libraries
    "catboost": catboost(),
    "xgboost": xgboost(),
    # neural networks, experiments/lab_nn.py
    "mlp": neural("mlp"),
    "tcn": neural("tcn"),
    "tcn_europe": neural("tcn", europe=True),
    "mlp_context": neural("mlp", features=ALL),
    "tcn_context": neural("tcn", features=ALL),
}


def path(name: str, split: str, seed: int) -> Path:
    return OUT / f"{name}_{split}_s{seed}.parquet"


def cross_validate(
    data: Data, name: str, split: str, seed: int
) -> dict[str, np.ndarray]:
    """Return the held-out correction of every row per output of candidate `name`."""
    year = data.table.year.to_numpy()
    years = sorted(np.unique(year))
    corrections: dict[str, np.ndarray] = {}
    for held_out in years[1:] if split == "forward" else years:
        test = year == held_out
        train = year < held_out if split == "forward" else ~test
        start = time.time()
        result = None
        if split == "forward" and held_out == years[-1]:
            suffix = f"_year_s{seed}.parquet"
            stored = {
                p.name.removesuffix(suffix): p for p in OUT.glob(f"{name}*{suffix}")
            }
            result = {
                key.removeprefix(name).removeprefix("."): pd.read_parquet(
                    p
                ).correction.to_numpy()[test]
                for key, p in stored.items()
                if key == name or key.startswith(name + ".")
            } or None
        if result is None:
            result = CANDIDATES[name](data, train, test, seed)
        if not isinstance(result, dict):
            result = {"": result}
        for key, values in result.items():
            corrections.setdefault(key, np.full(len(year), np.nan))[test] = values
        print(f"  {split} {held_out}: {time.time() - start:.0f} s", flush=True)
    return {f"{name}.{k}" if k else name: v for k, v in corrections.items()}


def report(data: Data, name: str, split: str, seed: int, correction, seconds) -> None:
    scores = score(data, correction)
    print(f"\n== {name} {split} seed {seed}")
    print(scores.round(3).T.to_string())
    print(f"mean {scores.skill.mean():.3f} worst {scores.skill.min():.3f}", flush=True)
    row = {
        "candidate": name,
        "split": split,
        "seed": seed,
        "share": SHARE,
        "mean": scores.skill.mean(),
        "worst": scores.skill.min(),
        **{f"skill_{y}": s for y, s in scores.skill.items()},
        **{f"mae_{y}": m for y, m in scores.mae.items()},
        "seconds": round(seconds),
    }
    columns = ["candidate", "split", "seed", "share", "mean", "worst"]
    columns += [
        f"{k}_{y}" for k in ("skill", "mae") for y in sorted(data.table.year.unique())
    ]
    row = pd.DataFrame([row]).reindex(columns=[*columns, "seconds"]).round(4)
    row.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", choices=list(CANDIDATES))
    parser.add_argument("--split", nargs="+", default=["year", "forward"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    THREADS = args.threads
    data = load()
    for name in args.candidates:
        for split in args.split:
            start = time.time()
            try:
                corrections = cross_validate(data, name, split, args.seed)
            except Exception:
                # keep the queue of candidates going
                traceback.print_exc()
                continue
            seconds = time.time() - start
            for key, correction in corrections.items():
                pd.DataFrame({"correction": correction.astype("float32")}).to_parquet(
                    path(key, split, args.seed)
                )
                report(data, key, split, args.seed, correction, seconds)
