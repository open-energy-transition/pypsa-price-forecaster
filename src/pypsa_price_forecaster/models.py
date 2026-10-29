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

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pypsa_price_forecaster.dataset import DEFAULT_FEATURES, TARGET, feature_matrix

DEFAULT_MODELS = ["pypsa", "bus_bias", "linear", "lgbm"]
# lgbm_gas learns the error relative to the marginal cost of gas plants,
# lgbm_gas_clip clips that cost to its range in training when predicting,
# lgbm_full applies the whole correction instead of the SHRINK share of lgbm
MODELS = [*DEFAULT_MODELS, "lgbm_full", "lgbm_gas", "lgbm_gas_clip"]
# share of the correction applied by lgbm, chosen on held-out years (experiments/NOTES.md)
SHRINK = 0.7


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
        self._fit(rows)
        return self

    def _fit(self, rows: pd.DataFrame) -> None:
        pass

    def correction(self, table: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=table.index)

    def predict(self, table: pd.DataFrame) -> pd.Series:
        """Return the predicted day-ahead price of every row of `table`."""
        return (table.model_price + self.correction(table)).rename("price")


class BusBias(Forecaster):
    """Model price plus the median error of the bus in training, zero for unseen buses."""

    def _fit(self, rows: pd.DataFrame) -> None:
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
    """

    def __init__(
        self,
        estimator,
        features: list[str],
        scale: str | None = None,
        clip: bool = False,
        shrink: float = 1.0,
    ) -> None:
        self.estimator = estimator
        self.features = list(features)
        self.scale = scale
        self.clip = clip
        self.shrink = shrink

    def _scale(self, table: pd.DataFrame) -> pd.Series | float:
        if self.scale is None:
            return 1.0
        if self.clip and hasattr(self, "scale_range"):
            return table[self.scale].clip(*self.scale_range)
        return table[self.scale]

    def _fit(self, rows: pd.DataFrame) -> None:
        scale = 1.0 if self.scale is None else rows[self.scale]
        if self.scale is not None:
            if not (scale > 0).all():
                raise ValueError(f"Scale {self.scale} must be positive in every row")
            self.scale_range = (scale.min(), scale.max())
        self.estimator.fit(
            feature_matrix(rows, self.features),
            (rows[TARGET] - rows.model_price) / scale,
        )

    def correction(self, table: pd.DataFrame) -> pd.Series:
        predicted = self.estimator.predict(feature_matrix(table, self.features))
        return (
            self.shrink * pd.Series(predicted, index=table.index) * self._scale(table)
        )


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
                SimpleImputer(),
                StandardScaler(),
                HuberRegressor(max_iter=1000),
            ),
            features,
        )
    if name in ("lgbm", "lgbm_full", "lgbm_gas", "lgbm_gas_clip"):
        return Residual(
            LGBMRegressor(
                objective="l1",
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.9,
                random_state=0,
                verbose=-1,
            ),
            features,
            scale="cost_gas" if name.startswith("lgbm_gas") else None,
            clip=name == "lgbm_gas_clip",
            shrink=SHRINK if name == "lgbm" else 1.0,
        )
    raise ValueError(f"Unknown model {name}, available: {MODELS}")


def save(model: Forecaster, path: str | Path) -> None:
    """Write a fitted model to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: str | Path) -> Forecaster:
    """Read a fitted model written by :func:`save`."""
    return joblib.load(path)
