import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from pypsa_price_forecaster import report  # noqa: E402
from pypsa_price_forecaster.dataset import FEATURES  # noqa: E402

PAGES = ["overview", "prices", "features", "zones", "experiments"]


@pytest.fixture
def report_dir(table, tmp_path, monkeypatch, small_lgbm):
    candidates = ["model_price", "margin_with_imports", "hour"]
    monkeypatch.setattr(report, "FEATURES", {f: FEATURES[f] for f in candidates})
    report.write_report(table, tmp_path, features=candidates[:2])
    monkeypatch.setenv("PPF_REPORT", str(tmp_path))
    return tmp_path


def test_every_page_renders(report_dir):
    app = AppTest.from_file("../dashboard/app.py", default_timeout=60).run()
    assert not app.exception, app.exception
    for page in PAGES:
        app = AppTest.from_string(
            f"from dashboard.views import {page}\n{page}()", default_timeout=60
        ).run()
        assert not app.exception, (page, app.exception)

    # the prices page opens on the latest run and the zone the correction helps most
    prices = pd.read_parquet(report_dir / "prices.parquet")
    latest = prices[prices.run == prices.run.cat.categories[-1]]
    error = latest[["pypsa", "lgbm"]].sub(latest.observed_price, axis=0).abs()
    mae = error.groupby(latest.bus, observed=True).mean()
    app = AppTest.from_string(
        "from dashboard.views import prices\nprices()", default_timeout=60
    ).run()
    assert app.selectbox(key="prices_zone").value == (mae.pypsa - mae.lgbm).idxmax()
    # mean per zone, duration curve, typical day, setter chart, 2 heatmaps
    assert len(app.get("plotly_chart")) == 6
