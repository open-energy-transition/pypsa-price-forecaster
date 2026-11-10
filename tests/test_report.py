import json
from pathlib import Path

import pandas as pd
import pytest

from pypsa_price_forecaster import report
from pypsa_price_forecaster.dataset import DEFAULT_FEATURES, FEATURES

REPORT = Path(__file__).parent.parent / "report"


def test_write_report(table, tmp_path, monkeypatch, small_lgbm):
    candidates = ["model_price", "margin_with_imports", "hour"]
    monkeypatch.setattr(report, "FEATURES", {f: FEATURES[f] for f in candidates})
    report.write_report(table, tmp_path, features=candidates[:2], with_ablation=False)
    scores = report.write_report(table, tmp_path, features=candidates[:2])

    assert scores.mae["lgbm"] < scores.mae["pypsa"]
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["split"] == "year"
    assert meta["years"] == [2019, 2021, 2023]
    assert len(meta["seed_mae"]) == 3

    ablation = pd.read_csv(tmp_path / "ablation.csv", index_col=[0, 1])
    assert ablation.index.tolist() == [
        ("without", "model_price"),
        ("without", "margin_with_imports"),
        ("with", "hour"),
    ]
    # the markup of the observed prices depends on the margin
    assert ablation.mae["without", "margin_with_imports"] > scores.mae["lgbm"]

    prices = pd.read_parquet(tmp_path / "prices.parquet")
    assert len(prices) == table.observed_price.notna().sum()
    # bands for every year with two other training years
    assert prices[report.BAND_COLUMNS].notna().all().all()
    assert (prices.lgbm_q10 <= prices.lgbm_q90).all()
    bands = pd.read_csv(tmp_path / "band_scores.csv", index_col="year")
    assert bands.index.tolist() == [2019, 2021, 2023]
    assert meta["bands"]
    assert set(pd.read_csv(tmp_path / "monthly.csv").model) == set(scores.index)
    assert len(pd.read_csv(tmp_path / "history.csv")) == 2
    assert (tmp_path / "price_setters.csv").exists()
    by_year = pd.read_csv(tmp_path / "scores_by_year.csv", index_col=[0, 1])
    assert by_year.index.get_level_values("year").unique().tolist() == [
        2019,
        2021,
        2023,
    ]


def test_report_without_bands_removes_old_band_scores(table, tmp_path, small_lgbm):
    features = ["model_price", "margin_with_imports"]
    (tmp_path / "band_scores.csv").write_text("year\n")
    report.write_report(
        table, tmp_path, features=features, with_ablation=False, with_bands=False
    )
    assert not (tmp_path / "band_scores.csv").exists()
    assert "lgbm_q10" not in pd.read_parquet(tmp_path / "prices.parquet")


def test_report_leaves_out_excluded_years(table, tmp_path, small_lgbm):
    features = ["model_price", "margin_with_imports"]
    report.write_report(
        table, tmp_path, features=features, with_ablation=False, exclude_years=[2021]
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["years"] == [2019, 2023]
    assert meta["excluded_years"] == [2021]
    prices = pd.read_parquet(tmp_path / "prices.parquet")
    assert not (prices.snapshot.dt.year == 2021).any()


def test_default_split_falls_back_to_month_and_week(table):
    one = table[table.year == 2023]
    assert report.default_split(table) == "year"
    assert report.default_split(one) == "week"
    two_months = one.assign(snapshot=one.snapshot + pd.Timedelta(days=20))
    assert report.default_split(two_months) == "month"


@pytest.mark.skipif(not (REPORT / "meta.json").exists(), reason="no report")
def test_committed_report_uses_the_defaults():
    meta = json.loads((REPORT / "meta.json").read_text())
    current = json.loads(json.dumps(report.lgbm_params()))
    assert (meta["features"], meta["lgbm_params"]) == (DEFAULT_FEATURES, current), (
        "The defaults changed since the report, rerun `pixi run report`"
    )
