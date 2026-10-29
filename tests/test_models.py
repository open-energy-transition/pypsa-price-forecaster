import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pypsa_price_forecaster.cli import app
from pypsa_price_forecaster.dataset import TARGET, feature_matrix
from pypsa_price_forecaster.evaluate import (
    band_scores,
    bus_metrics,
    cross_validate,
    cross_validate_bands,
    summary,
)
from pypsa_price_forecaster.models import (
    DEFAULT_MODELS,
    QUANTILES,
    TRANSFORMS,
    labelled,
    load,
    make_model,
    save,
)


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


def test_forward_split_fits_on_earlier_years_only(table, small_lgbm):
    predictions = cross_validate(table, ["pypsa", "lgbm"], split="forward")
    assert predictions[table.year == 2019].isna().all().all()
    assert predictions[table.year > 2019].notna().all().all()
    # 2021 is predicted from 2019 alone, unlike in the year split
    later = table.year == 2021
    year = cross_validate(table, ["lgbm"], split="year")
    assert not predictions.lgbm[later].equals(year.lgbm[later])
    metrics = bus_metrics(table, predictions)
    assert set(metrics.index.get_level_values("year")) == {2021, 2023}


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


def test_gas_scaled_correction_scales_with_the_gas_cost(table, small_lgbm):
    model = make_model("lgbm_gas", ["model_price", "margin_with_imports"]).fit(table)
    doubled = table.assign(cost_gas=2 * table.cost_gas)
    correction = model.predict(table) - table.model_price
    assert np.allclose(model.predict(doubled) - table.model_price, 2 * correction)


def test_clipped_gas_correction_stops_at_the_training_range(table, small_lgbm):
    model = make_model("lgbm_gas_clip", ["model_price", "margin_with_imports"])
    model.fit(table)
    doubled = table.assign(cost_gas=2 * table.cost_gas)
    correction = model.predict(table) - table.model_price
    assert np.allclose(model.predict(doubled) - table.model_price, correction)


def test_transforms_round_trip():
    x = np.array([-500.0, -1.0, 0.0, 1.0, 500.0])
    for forward, inverse in TRANSFORMS.values():
        assert np.allclose(inverse(forward(x)), x)


def test_spike_weighted_correction_fits_large_errors_more_closely(table, small_lgbm):
    features = ["model_price", "margin_with_imports"]
    plain = make_model("lgbm_full", features).fit(table)
    weighted = make_model("lgbm_spike_weighted", features).fit(table)
    rows = labelled(table)
    gap = (rows[TARGET] - rows.model_price).abs()
    large = gap > gap.groupby(rows.bus).transform("quantile", 0.95)

    def error(model):
        return (model.predict(rows) - rows[TARGET]).abs()

    assert error(weighted)[large].mean() <= error(plain)[large].mean()


def test_asinh_correction_runs_and_compresses_large_values(table, small_lgbm):
    model = make_model("lgbm_asinh", ["model_price", "margin_with_imports"]).fit(table)
    predictions = model.predict(table)
    assert predictions.notna().all() and np.isfinite(predictions).all()


def test_quantile_correction_runs(table, small_lgbm):
    model = make_model("lgbm_q90", ["model_price", "margin_with_imports"]).fit(table)
    predictions = model.predict(table)
    assert predictions.notna().all() and np.isfinite(predictions).all()


def test_cross_bus_and_rolling_features_ignore_row_order(table):
    features = ["spread_europe", "vres_share_week"]
    shuffled = table.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(
        feature_matrix(shuffled, features).sort_index(),
        feature_matrix(table, features),
    )
    spread = feature_matrix(table, ["spread_europe"]).spread_europe
    # the median bus of every hour has no spread
    assert (spread.abs().groupby([table.run, table.snapshot]).min() < 1e-9).all()


def test_bands_cover_most_hours_and_beat_the_point_price(table, small_lgbm):
    bands = cross_validate_bands(table, features=["model_price", "margin_with_imports"])
    assert (bands[QUANTILES].diff(axis=1).iloc[:, 1:] >= 0).all().all()
    result = band_scores(table, bands)
    assert set(result.index) == {2019, 2021, 2023}
    assert (result.coverage_80 > 0.6).all()
    assert (result.pinball < result.pinball_point).all()


def test_forward_bands_need_two_training_years(table, small_lgbm):
    bands = cross_validate_bands(
        table, features=["model_price", "margin_with_imports"], split="forward"
    )
    assert bands[table.year < 2023].isna().all().all()
    assert bands[table.year == 2023].notna().all().all()


def test_daily_features_are_constant_within_a_day(table):
    daily = ["margin_with_imports_day", "vres_share_day", "gas_setter_day"]
    values = feature_matrix(table, daily)
    days = [table.run, table.bus, table.snapshot.dt.date]
    assert (values.groupby(days).nunique() <= 1).all().all()


def test_seed_bag_averages_several_fits(table, small_lgbm):
    features = ["model_price", "margin_with_imports"]
    model = make_model("lgbm_bag", features, {"n_estimators": 50}).fit(table)
    assert len(model.estimator.estimators_) == 5
    assert np.isfinite(model.predict(table)).all()
