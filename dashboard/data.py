"""
Reads the report written by ``ppf report`` and the experiment log for the dashboard.

Only pandas and pyarrow are needed, so the app runs without the package and its
modelling dependencies on Streamlit Community Cloud.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
EXPERIMENTS = ROOT / "experiments"
AREA_TO_BUS = ROOT / "src/pypsa_price_forecaster/data/entsoe_area_to_bus.csv"

MODEL_NAMES = {
    "pypsa": "PyPSA",
    "bus_bias": "PyPSA + bias per zone",
    "linear": "PyPSA + linear correction",
    "lgbm": "PyPSA + LightGBM correction",
}
SPLIT_TEXT = {
    "year": "Every year is predicted by a model trained on the other years only.",
    "month": "Every calendar month is predicted by a model trained on the other months only.",
    "week": "Every fourth week is predicted by a model trained on the other weeks only.",
}
FEATURE_TEXT = {
    "model_price": "PyPSA marginal price",
    "margin_domestic": "Capacity margin, domestic",
    "margin_with_imports": "Capacity margin, with imports",
    "margin_domestic_mw": "Capacity margin, domestic (MW)",
    "margin_with_imports_mw": "Capacity margin, with imports (MW)",
    "vres_share": "Wind, solar and run of river available / load",
    "curtailment_share": "Curtailment / load",
    "firm_utilisation": "Thermal utilisation (dispatch / available)",
    "net_import_share": "Net imports / load",
    "storage_share": "Storage dispatch / load",
    "cost_gas": "Gas plant marginal cost",
    "cost_coal": "Coal plant marginal cost",
    "price_to_gas": "Model price / gas plant cost",
    "price_daily_mean": "Daily mean model price",
    "price_deviation": "Model price minus daily mean",
    "bus": "Bidding zone identity",
    "hour": "Hour of day (UTC)",
    "dayofweek": "Day of week",
    "month": "Month",
}
AREA_NAMES = {
    "DE_LU": "Germany",
    "FR": "France",
    "BE": "Belgium",
    "NL": "Netherlands",
    "AT": "Austria",
    "CH": "Switzerland",
    "CZ": "Czechia",
    "PL": "Poland",
    "SK": "Slovakia",
    "HU": "Hungary",
    "SI": "Slovenia",
    "HR": "Croatia",
    "RO": "Romania",
    "BG": "Bulgaria",
    "RS": "Serbia",
    "GR": "Greece",
    "ES": "Spain",
    "PT": "Portugal",
    "IT_NORD": "Italy North",
    "IT_CNOR": "Italy Centre-North",
    "IT_CSUD": "Italy Centre-South",
    "IT_SUD": "Italy South",
    "IT_CALA": "Italy Calabria",
    "IT_SICI": "Sicily",
    "IT_SARD": "Sardinia",
    "NO_1": "Norway South-East",
    "NO_2": "Norway South-West",
    "NO_3": "Norway Centre",
    "NO_4": "Norway North",
    "NO_5": "Norway West",
    "SE_1": "Sweden North",
    "SE_2": "Sweden North-Centre",
    "SE_3": "Sweden South-Centre",
    "SE_4": "Sweden South",
    "FI": "Finland",
    "DK_1": "Denmark West",
    "DK_2": "Denmark East",
    "EE": "Estonia",
    "LV": "Latvia",
    "LT": "Lithuania",
    "IE_SEM": "Ireland",
    "GB": "Great Britain",
    "AL": "Albania",
    "BA": "Bosnia and Herzegovina",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "XK": "Kosovo",
    "MD": "Moldova",
    "UA_IPS": "Ukraine",
}
BUS_NAMES = {"LUG1": "Luxembourg", "GBNI": "Northern Ireland"}


@dataclass(frozen=True)
class Report:
    """Files of one report, see ``pypsa_price_forecaster.report``."""

    meta: dict
    scores: pd.DataFrame
    by_year: pd.DataFrame
    bus_metrics: pd.DataFrame
    monthly: pd.DataFrame
    importance: pd.Series
    ablation: pd.DataFrame | None
    prices: pd.DataFrame
    history: pd.DataFrame
    price_setters: pd.DataFrame | None

    @property
    def noise(self) -> float:
        """Two standard deviations of the LightGBM MAE over random seeds, 0 without repeats."""
        seeds = pd.Series(self.meta["seed_mae"])
        return 2 * seeds.std() if len(seeds) > 1 else 0.0

    def bus_mae(self) -> pd.DataFrame:
        """MAE per run year and bus (rows) and model (columns)."""
        return self.bus_metrics.mae.unstack("model")


def read_report(path: Path | None = None) -> Report | None:
    """Return the report in `path` (default: $PPF_REPORT or report/), or None if there is none."""
    path = Path(path or os.environ.get("PPF_REPORT", REPORT))
    if not (path / "meta.json").exists():
        return None
    ablation = path / "ablation.csv"
    return Report(
        meta=json.loads((path / "meta.json").read_text()),
        scores=pd.read_csv(path / "scores.csv", index_col="model"),
        by_year=pd.read_csv(path / "scores_by_year.csv", index_col=["model", "year"]),
        bus_metrics=pd.read_csv(
            path / "bus_metrics.csv", index_col=["model", "year", "bus"]
        ),
        monthly=pd.read_csv(path / "monthly.csv"),
        importance=pd.read_csv(path / "importance.csv", index_col="feature").gain_share,
        ablation=pd.read_csv(ablation, index_col=["change", "feature"])
        if ablation.exists()
        else None,
        prices=pd.read_parquet(path / "prices.parquet"),
        history=pd.read_csv(path / "history.csv", parse_dates=["created"]),
        price_setters=pd.read_csv(path / "price_setters.csv")
        if (path / "price_setters.csv").exists()
        else None,
    )


def read_experiments() -> pd.DataFrame:
    """Return the experiment log of ``experiments/results.csv``."""
    return pd.read_csv(EXPERIMENTS / "results.csv")


def read_experiments_by_year() -> pd.DataFrame | None:
    """Return the per-year experiment scores of ``experiments/results_by_year.csv``, if any."""
    path = EXPERIMENTS / "results_by_year.csv"
    return pd.read_csv(path) if path.exists() else None


def read_notes() -> str:
    """Return the experiment notes, with links relative to the repository."""
    return (EXPERIMENTS / "NOTES.md").read_text()


def bus_names() -> dict[str, str]:
    """Plain name of every bus with an ENTSO-E area."""
    areas = pd.read_csv(AREA_TO_BUS, index_col="bus").entsoe_area
    names = areas.map(AREA_NAMES).dropna().to_dict()
    return names | BUS_NAMES


def label(bus: str, names: dict[str, str]) -> str:
    """Plain name of a bus with its code."""
    return f"{names[bus]} ({bus})" if bus in names else bus


def feature_label(feature: str) -> str:
    """Plain description of a feature."""
    return FEATURE_TEXT.get(feature, feature)
