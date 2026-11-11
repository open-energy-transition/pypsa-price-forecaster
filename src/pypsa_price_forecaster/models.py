"""
Models mapping a table of PyPSA quantities to day-ahead prices.

Every model predicts the model price plus a correction. The baselines are the
raw PyPSA price (``pypsa``) and a constant correction per bus (``bus_bias``).
The learned models (``linear``, ``lgbm``) predict the correction, the observed
minus the model price, from the features of each row. The corrections are fitted
with robust losses (median for ``bus_bias``, absolute error for ``lgbm``, Huber for
``linear``), which match the mean absolute error used to score the backcast and
are robust to load shedding hours in the model price.
"""

from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pypsa_price_forecaster.dataset import DEFAULT_FEATURES, TARGET, feature_matrix

DEFAULT_MODELS = ["pypsa", "bus_bias", "linear", "lgbm"]
# lgbm_gas learns the error relative to the marginal cost of gas plants,
# lgbm_gas_clip clips that cost to its range in training when predicting,
# lgbm_full applies the whole correction instead of the SHRINK share of lgbm,
# lgbm_spike_weighted weights training rows by the size of their correction,
# lgbm_asinh fits the correction on an asinh-transformed (variance-stabilised) scale,
# lgbm_q90 fits the 90th percentile of the correction instead of its median,
# lgbm_bag averages the lgbm correction over several random seeds
MODELS = [
    *DEFAULT_MODELS,
    "lgbm_full",
    "lgbm_gas",
    "lgbm_gas_clip",
    "lgbm_spike_weighted",
    "lgbm_asinh",
    "lgbm_q90",
    "lgbm_bag",
]
# share of the correction applied by lgbm, chosen on held-out years (experiments/NOTES.md)
SHRINK = 0.7
WEIGHTS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # 1 EUR/MWh floor, so small-correction rows still count a little
    "spike": lambda rows: (rows[TARGET] - rows.model_price).abs().clip(lower=1.0),
}
# a fixed, data-independent transform of the correction and its inverse: asinh
# compresses large corrections before the L1 loss sees them (unlike scaling by
# gas cost, which failed because gas cost is itself non-stationary across years)
TRANSFORMS: dict[str, tuple[Callable, Callable]] = {"asinh": (np.arcsinh, np.sinh)}


class SeedBag:
    """Mean prediction of copies of `estimator` fitted with different random seeds."""

    def __init__(self, estimator, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
        self.estimator = estimator
        self.seeds = seeds

    def set_params(self, **params) -> "SeedBag":
        self.estimator.set_params(**params)
        return self

    def fit(self, X, y, **kwargs) -> "SeedBag":
        self.estimators_ = [
            clone(self.estimator).set_params(random_state=seed).fit(X, y, **kwargs)
            for seed in self.seeds
        ]
        return self

    def predict(self, X) -> np.ndarray:
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)


def labelled(table: pd.DataFrame) -> pd.DataFrame:
    """Return the rows with observed and model price."""
    return table.dropna(subset=[TARGET, "model_price"])


class Forecaster:
    """Base model predicting the model price without correction."""

    features: list[str] = []

    def fit(self, table: pd.DataFrame) -> "Forecaster":
        """Fit the correction to the rows of `table` with observed price."""
        rows = labelled(table)
        if rows.empty:
            raise ValueError("No rows with observed and model price to fit on")
        self.trained_on = sorted(rows.run.unique())
        self._fit(table, rows)
        return self

    def _fit(self, table: pd.DataFrame, rows: pd.DataFrame) -> None:
        pass

    def correction(self, table: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=table.index)

    def predict(self, table: pd.DataFrame) -> pd.Series:
        """Return the predicted day-ahead price of every row of `table`."""
        return (table.model_price + self.correction(table)).rename("price")


class BusBias(Forecaster):
    """Model price plus the median error of the bus in training, zero for unseen buses."""

    def _fit(self, table: pd.DataFrame, rows: pd.DataFrame) -> None:
        self.bias = (rows[TARGET] - rows.model_price).groupby(rows.bus).median()

    def correction(self, table: pd.DataFrame) -> pd.Series:
        return table.bus.map(self.bias).fillna(0.0)


class Residual(Forecaster):
    """
    Model price plus a correction regressed on features by `estimator`.

    With a `scale` column, the estimator learns the error divided by that
    column, e.g. the marginal cost of gas plants, and the correction is
    multiplied back, so it carries over to years with other fuel prices.
    With `clip`, the scale of predicted rows is clipped to its range in
    training, so the correction does not grow beyond what was learnt.
    `shrink` is the share of the learnt correction that is applied: below 1
    it hedges against corrections that do not carry over to an unseen year.
    With `weight`, a function of the training rows gives the sample weight of
    the fit, e.g. to make large corrections (spikes) count more than the many
    rows with a small one, which the default absolute-error loss otherwise
    treats as equally important as a large one.
    With `transform`, a name of `TRANSFORMS`, the estimator is fitted on a
    fixed, monotonic transform of the correction instead of the correction
    itself, and its prediction is transformed back. Unlike `scale`, this
    compresses large corrections without depending on another column.
    """

    def __init__(
        self,
        estimator,
        features: list[str],
        scale: str | None = None,
        clip: bool = False,
        shrink: float = 1.0,
        weight: Callable[[pd.DataFrame], pd.Series] | None = None,
        transform: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.features = list(features)
        self.scale = scale
        self.clip = clip
        self.shrink = shrink
        self.weight = weight
        self.transform = transform

    def _scale(self, table: pd.DataFrame) -> pd.Series | float:
        if self.scale is None:
            return 1.0
        if self.clip and hasattr(self, "scale_range"):
            return table[self.scale].clip(*self.scale_range)
        return table[self.scale]

    def _fit(self, table: pd.DataFrame, rows: pd.DataFrame) -> None:
        scale = 1.0 if self.scale is None else rows[self.scale]
        if self.scale is not None:
            if not (scale > 0).all():
                raise ValueError(f"Scale {self.scale} must be positive in every row")
            self.scale_range = (scale.min(), scale.max())
        y = (rows[TARGET] - rows.model_price) / scale
        if self.transform is not None:
            y = TRANSFORMS[self.transform][0](y)
        # features of the whole table, as some compare a row with other buses or hours
        X = feature_matrix(table, self.features).loc[rows.index]
        if self.weight is None:
            self.estimator.fit(X, y)
        else:
            self.estimator.fit(X, y, sample_weight=self.weight(rows))

    def correction(self, table: pd.DataFrame) -> pd.Series:
        predicted = self.estimator.predict(feature_matrix(table, self.features))
        if self.transform is not None:
            predicted = TRANSFORMS[self.transform][1](predicted)
        return (
            self.shrink * pd.Series(predicted, index=table.index) * self._scale(table)
        )


# quantiles of the price bands, every 5 % from 5 % to 95 %
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]
# levels of the predicted price in EUR/MWh within which the band errors are taken
LEVELS = [-np.inf, 30, 60, 100, 150, 250, np.inf]
# EUR/MWh, lowest price the relative band errors are taken relative to
RELATIVE_FLOOR = 20.0


def price_level(price: pd.Series) -> pd.Series:
    """Index of the interval of `LEVELS` of every price."""
    return pd.Series(pd.cut(price, LEVELS, labels=False), index=price.index)


def error_quantiles(errors: pd.DataFrame, keys: list[str], min_rows: int):
    """Return the `QUANTILES` of the error per group of `keys` with at least `min_rows` rows."""
    grouped = errors.groupby(keys).error
    quantiles = grouped.quantile(QUANTILES).unstack()
    return quantiles[grouped.size() >= min_rows]


class Banded(Forecaster):
    """
    A point model plus price bands from the quantiles of its error out of sample.

    The errors (observed minus predicted price) come from refitting the point
    model with each training year left out once and predicting that year, so
    they measure how wrong the model is in a year it has not seen. Their
    quantiles are taken per bus (and with `by_level` per level of the predicted
    price, else per level alone), else over all rows, wherever a group has
    fewer than `min_rows` rows. With `relative`, the errors are taken relative
    to the predicted price (at least `RELATIVE_FLOOR`), so the bands of a bus
    widen and narrow with its price level, e.g. from a cheap to a crisis year.
    """

    def __init__(
        self,
        base: str = "lgbm",
        features: list[str] | None = None,
        params: dict | None = None,
        min_rows: int = 500,
        relative: bool = True,
        by_level: bool = False,
    ) -> None:
        self.base = base
        self.features = list(DEFAULT_FEATURES if features is None else features)
        self.params = params
        self.min_rows = min_rows
        self.relative = relative
        self.by_level = by_level

    def _point_model(self) -> Forecaster:
        return make_model(self.base, self.features, self.params)

    def _scale(self, point: pd.Series) -> pd.Series | float:
        return point.abs().clip(lower=RELATIVE_FLOOR) if self.relative else 1.0

    def _keys(self) -> list[str]:
        return ["bus", "level"] if self.by_level else ["bus"]

    def _fit(self, table: pd.DataFrame, rows: pd.DataFrame) -> None:
        years = sorted(rows.year.unique())
        if len(years) < 2:
            raise ValueError(f"Bands need at least two training years, got {years}")
        errors = []
        for year in years:
            held_out = table.year == year
            model = self._point_model().fit(table[~held_out])
            scored = rows[rows.year == year]
            predicted = model.predict(table[held_out]).loc[scored.index]
            errors.append(
                pd.DataFrame(
                    {
                        "bus": scored.bus.astype(str),
                        "level": price_level(predicted),
                        "error": (scored[TARGET] - predicted) / self._scale(predicted),
                    }
                )
            )
        errors = pd.concat(errors)
        self.by_bus = error_quantiles(errors, self._keys(), self.min_rows)
        self.by_price = error_quantiles(errors, ["level"], self.min_rows)
        self.overall = errors.error.quantile(QUANTILES)
        self.model = self._point_model().fit(table)

    def predict(self, table: pd.DataFrame) -> pd.Series:
        return self.model.predict(table)

    def quantiles(self, table: pd.DataFrame) -> pd.DataFrame:
        """Return the predicted price quantiles of every row, one column per `QUANTILES`."""
        point = self.predict(table)
        keys = pd.DataFrame(
            {"bus": table.bus.astype(str), "level": price_level(point)},
            index=table.index,
        )
        offsets = keys.join(self.by_bus, on=self._keys())[QUANTILES]
        if self.by_level:
            offsets = offsets.fillna(keys.join(self.by_price, on="level")[QUANTILES])
        offsets = offsets.fillna(self.overall)
        return offsets.mul(self._scale(point), axis=0).add(point, axis=0)


def make_model(
    name: str, features: list[str] | None = None, params: dict | None = None
) -> Forecaster:
    """
    Return the unfitted model `name` using `features` (``DEFAULT_FEATURES`` if None).

    `params` override the parameters of the estimator of a learned model, e.g.
    ``{"num_leaves": 15}`` for ``lgbm``.
    """
    model = base_model(name, DEFAULT_FEATURES if features is None else features)
    if params:
        if not isinstance(model, Residual):
            raise ValueError(f"Model {name} has no parameters")
        model.estimator.set_params(**params)
    return model


def base_model(name: str, features: list[str]) -> Forecaster:
    """Return the unfitted model `name` with its default parameters."""
    if name == "pypsa":
        return Forecaster()
    if name == "bus_bias":
        return BusBias()
    if name == "linear":
        return Residual(
            make_pipeline(
                make_column_transformer(
                    (
                        OneHotEncoder(handle_unknown="ignore"),
                        make_column_selector(dtype_include="category"),
                    ),
                    remainder=make_pipeline(SimpleImputer(), StandardScaler()),
                ),
                HuberRegressor(max_iter=1000),
            ),
            features,
        )
    if name in (
        "lgbm",
        "lgbm_full",
        "lgbm_gas",
        "lgbm_gas_clip",
        "lgbm_spike_weighted",
        "lgbm_asinh",
        "lgbm_q90",
        "lgbm_bag",
    ):
        objective = {"lgbm_q90": "quantile"}.get(name, "l1")
        estimator = LGBMRegressor(
            objective=objective,
            **({"alpha": 0.9} if objective == "quantile" else {}),
            # larger trees than rounds 5 to 20 (63 leaves, 50 rows, 800 trees): with the
            # context features they gain on held-out years (rounds 19 and 21)
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=127,
            min_child_samples=25,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=0,
            verbose=-1,
        )
        return Residual(
            SeedBag(estimator) if name == "lgbm_bag" else estimator,
            features,
            scale="cost_gas" if name.startswith("lgbm_gas") else None,
            clip=name == "lgbm_gas_clip",
            shrink=SHRINK if name in ("lgbm", "lgbm_bag") else 1.0,
            weight=WEIGHTS["spike"] if name == "lgbm_spike_weighted" else None,
            transform="asinh" if name == "lgbm_asinh" else None,
        )
    raise ValueError(f"Unknown model {name}, available: {MODELS}")


def save(model: Forecaster, path: str | Path) -> None:
    """Write a fitted model to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: str | Path) -> Forecaster:
    """Read a fitted model written by :func:`save`."""
    return joblib.load(path)
