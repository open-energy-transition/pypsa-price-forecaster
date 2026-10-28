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
    """Model price plus a correction regressed on features by `estimator`."""

    def __init__(self, estimator, features: list[str]) -> None:
        self.estimator = estimator
        self.features = list(features)

    def _fit(self, rows: pd.DataFrame) -> None:
        self.estimator.fit(
            feature_matrix(rows, self.features), rows[TARGET] - rows.model_price
        )

    def correction(self, table: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.estimator.predict(feature_matrix(table, self.features)),
            index=table.index,
        )


def make_model(name: str, features: list[str] | None = None) -> Forecaster:
    """Return the unfitted model `name` using `features` (``DEFAULT_FEATURES`` if None)."""
    features = DEFAULT_FEATURES if features is None else features
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
    if name == "lgbm":
        return Residual(
            LGBMRegressor(
                objective="l1",
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=100,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.9,
                random_state=0,
                verbose=-1,
            ),
            features,
        )
    raise ValueError(f"Unknown model {name}, available: {DEFAULT_MODELS}")


def save(model: Forecaster, path: str | Path) -> None:
    """Write a fitted model to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: str | Path) -> Forecaster:
    """Read a fitted model written by :func:`save`."""
    return joblib.load(path)
