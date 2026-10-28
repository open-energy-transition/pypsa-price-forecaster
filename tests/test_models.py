import pandas as pd
import pytest
from typer.testing import CliRunner

from pypsa_price_forecaster.cli import app
from pypsa_price_forecaster.evaluate import bus_metrics, cross_validate, summary
from pypsa_price_forecaster.models import DEFAULT_MODELS, load, make_model, save


def test_learned_models_beat_raw_price_out_of_sample(table):
    predictions = cross_validate(table, DEFAULT_MODELS)
    assert predictions.notna().all().all()
    mae = summary(bus_metrics(table, predictions)).mae.groupby("model").mean()
    assert mae["bus_bias"] < mae["pypsa"]
    assert mae["linear"] < mae["bus_bias"]
    assert mae["lgbm"] < mae["linear"]


def test_year_split_needs_two_years(table):
    with pytest.raises(ValueError, match="split by week"):
        cross_validate(table[table.year == 2023], ["pypsa"])


def test_week_split_predicts_held_out_weeks_of_one_year(table):
    one_year = table[table.year == 2023]
    predictions = cross_validate(one_year, ["pypsa", "lgbm"], split="week", folds=4)
    assert predictions.notna().all().all()
    mae = summary(bus_metrics(one_year, predictions)).mae
    assert mae["lgbm", 2023] < mae["pypsa", 2023]


def test_saved_model_predicts_the_same(table, tmp_path):
    model = make_model("lgbm").fit(table[table.year < 2023])
    save(model, tmp_path / "model.joblib")
    test = table[table.year == 2023]
    pd.testing.assert_series_equal(
        load(tmp_path / "model.joblib").predict(test), model.predict(test)
    )
    assert model.trained_on == ["solved_2019", "solved_2021"]


def test_cli_extract_evaluate_train_predict(run_files, tmp_path):
    runner = CliRunner()

    def invoke(*args):
        result = runner.invoke(app, [str(a) for a in args])
        assert result.exit_code == 0, result.output
        return result

    tables = []
    for network, prices in run_files:
        tables.append(tmp_path / "tables" / f"{network.stem}.parquet")
        invoke("extract", network, "--prices", prices, "--out", tables[-1])

    result = invoke(
        "evaluate", *tables, "--out-dir", tmp_path / "eval", "-m", "pypsa", "-m", "lgbm"
    )
    invoke(
        "evaluate", tables[0], "--out-dir", tmp_path / "eval_week", "--split", "week"
    )
    assert "lgbm" in result.output
    scores = pd.read_csv(tmp_path / "eval" / "summary.csv")
    assert set(scores.model) == {"pypsa", "lgbm"}
    assert set(scores.year) == {2019, 2021, 2023}

    invoke(
        "train",
        *tables[:2],
        "--out",
        tmp_path / "model.joblib",
        "-f",
        "model_price",
        "-f",
        "margin_with_imports",
    )
    future = run_files[-1][0]
    invoke(
        "predict",
        future,
        "--model",
        tmp_path / "model.joblib",
        "--out",
        tmp_path / "prices.csv",
    )
    prices = pd.read_csv(tmp_path / "prices.csv", index_col=0, parse_dates=True)
    assert list(prices.columns) == ["A", "B1", "B2"]
    assert len(prices) == 24 * 28 and prices.notna().all().all()
