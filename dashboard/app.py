"""
Streamlit app showing the latest report of the price correction: ``pixi run -e dashboard dashboard``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.views import (
    cached_report,
    experiments,
    features,
    overview,
    prices,
    zones,
)

st.set_page_config(page_title="PyPSA price forecaster", layout="wide")
if cached_report() is None:
    st.error("No report in report/, run `pixi run report` and commit it.")
    st.stop()
page = st.navigation(
    [
        st.Page(overview, title="Does it help?", default=True),
        st.Page(prices, title="Prices: raw vs corrected", url_path="prices"),
        st.Page(features, title="Feature impact", url_path="features"),
        st.Page(zones, title="Zones and months", url_path="zones"),
        st.Page(experiments, title="Experiment log", url_path="experiments"),
    ]
)
page.run()
