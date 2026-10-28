"""
Pages of the dashboard, one function per page.
"""

import re

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.data import (
    MODEL_NAMES,
    SPLIT_TEXT,
    Report,
    bus_names,
    feature_label,
    label,
    read_experiments,
    read_notes,
    read_report,
)

SCORE_COLUMNS = {
    "mae": "MAE",
    "r": "r",
    "abs_bias": "|bias|",
    "mae_top5": "MAE top 5 % hours",
    "mae_low5": "MAE bottom 5 % hours",
    "std_ratio": "spread ratio",
    "share_le_0": "hours ≤ 0",
}
SCORE_HELP = (
    "Medians over zones of the hourly errors in EUR/MWh, out of sample. "
    "The tail columns score the hours above the 95th and below the 5th percentile "
    "of each zone's market price. Spread ratio is the standard deviation of the "
    "predicted over the market price, hours ≤ 0 the share of hours at or below zero."
)


@st.cache_data
def cached_report() -> Report | None:
    return read_report()


def provenance(report: Report) -> str:
    """One line on what the report was computed from."""
    meta = report.meta
    commit = meta["commit"] + (" with uncommitted changes" if meta["dirty"] else "")
    return (
        f"Report of {meta['created'][:10]} at commit {commit} · "
        f"runs {', '.join(meta['runs'])} · {meta['split']} split, {meta['folds']} folds · "
        f"{meta['buses']} zones with market prices"
    )


def score_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Scores with readable model and column names."""
    table = scores[list(SCORE_COLUMNS)].rename(columns=SCORE_COLUMNS)
    table.index = [MODEL_NAMES.get(m, m) for m in table.index.get_level_values("model")]
    return table


def overview() -> None:
    report = cached_report()
    st.title("Does the correction help?")
    st.caption(provenance(report))
    st.markdown(
        "The correction learns the gap between the market price and the PyPSA price "
        f"from hourly features of the solved network. {SPLIT_TEXT[report.meta['split']]}"
    )
    scores = report.scores
    raw, corrected = scores.loc["pypsa"], scores.loc["lgbm"]
    tiles = [
        ("Median MAE (EUR/MWh)", "mae", "{:.2f}", "inverse"),
        ("Median correlation r", "r", "{:.3f}", "normal"),
        ("MAE in top 5 % hours (EUR/MWh)", "mae_top5", "{:.1f}", "inverse"),
        ("MAE in bottom 5 % hours (EUR/MWh)", "mae_low5", "{:.1f}", "inverse"),
    ]
    for column, (name, key, template, direction) in zip(
        st.columns(4), tiles, strict=True
    ):
        column.metric(
            name,
            template.format(corrected[key]),
            delta=f"{corrected[key] - raw[key]:+.{3 if key == 'r' else 2}f} vs PyPSA",
            delta_color=direction,
            border=True,
        )

    mae = report.bus_mae()
    buses = (mae.lgbm < mae.pypsa).sum()
    unit = "zones" if len(report.meta["years"]) == 1 else "zone-years"
    monthly = report.monthly.pivot_table("mae", ["year", "month"], "model")
    months = (monthly.lgbm < monthly.pypsa).sum()
    st.markdown(
        f"Out of sample, the LightGBM correction lowers the median MAE from "
        f"**{raw.mae:.2f}** to **{corrected.mae:.2f} EUR/MWh** "
        f"({corrected.mae / raw.mae - 1:+.0%}). "
        f"It is better in **{buses} of {len(mae)}** {unit} and "
        f"**{months} of {len(monthly)}** months."
    )

    st.subheader("All models")
    st.dataframe(
        score_table(scores).style.format(
            {
                **dict.fromkeys(["MAE", "|bias|", "MAE top 5 % hours"], "{:.2f}"),
                "MAE bottom 5 % hours": "{:.2f}",
                "r": "{:.3f}",
                "spread ratio": "{:.2f}",
                "hours ≤ 0": "{:.1%}",
            }
        ),
        width="stretch",
    )
    observed = report.prices.observed_price
    st.caption(f"{SCORE_HELP} Market: {(observed <= 0).mean():.1%} of hours ≤ 0.")

    years = report.by_year.index.get_level_values("year").unique()
    if len(years) > 1:
        st.subheader("Per year")
        st.dataframe(
            report.by_year[["mae", "r", "mae_top5", "mae_low5"]]
            .rename(columns=SCORE_COLUMNS)
            .unstack("model")
            .round(2),
            width="stretch",
        )

    st.subheader("Progress over reports")
    if len(report.history) > 1:
        st.plotly_chart(charts.history_lines(report.history), width="stretch")
    st.caption(
        "Every `ppf report` adds a point, so this shows how the default model "
        "changed with the code and runs on master."
    )


def features() -> None:
    report = cached_report()
    st.title("Which features matter?")
    st.caption(provenance(report))
    if report.ablation is None:
        st.info("This report has no feature ablation, rerun `pixi run report`.")
        return
    reference = report.scores.mae["lgbm"]
    noise = report.noise
    changes = report.ablation.mae - reference
    st.markdown(
        f"The default LightGBM correction (median MAE **{reference:.2f}**) is refitted "
        "once without each of its features and once with each other feature. "
        f"The grey band (±{noise:.2f}) is twice the spread over random seeds: "
        "changes inside it are noise."
    )
    if report.meta["split"] != "year":
        st.warning(
            "All scores come from a single year. Features that track the season, "
            "such as the month and fuel costs, can only be judged with the year split."
        )
    left, right = st.columns(2)
    left.markdown("**Remove a default feature:** MAE rises if it helps")
    left.plotly_chart(
        charts.ablation_bars(changes["without"], noise, 1), width="stretch"
    )
    right.markdown("**Add another feature:** MAE falls if it helps")
    right.plotly_chart(
        charts.ablation_bars(changes["with"], noise, -1), width="stretch"
    )

    st.subheader("Split gain of the fitted model")
    st.plotly_chart(charts.importance_bars(report.importance), width="stretch")
    st.caption(
        "Share of the loss reduction from splits on each feature, of the model "
        "fitted on all hours. Gain shows what the trees use, not what improves "
        "out-of-sample scores: correlated features share it."
    )

    with st.expander("All ablation scores"):
        table = report.ablation[["mae", "r", "mae_top5", "mae_low5"]].copy()
        table.insert(1, "MAE change", changes)
        table.index = [
            f"{change} {feature_label(feature)}" for change, feature in table.index
        ]
        st.dataframe(table.round(3).rename(columns=SCORE_COLUMNS), width="stretch")


def zones() -> None:
    report = cached_report()
    names = bus_names()
    st.title("Where does it help?")
    st.caption(provenance(report))

    mae = report.bus_mae()
    years = mae.index.get_level_values("year").unique()
    year = (
        st.segmented_control("Year", years, default=years[-1])
        if len(years) > 1
        else years[0]
    )
    year = year or years[-1]
    labels = {b: label(b, names) for b in mae.loc[year].index}
    st.subheader("MAE per zone")
    st.plotly_chart(charts.bus_dumbbell(mae.loc[year], labels), width="stretch")
    st.caption("Sorted by the change of the corrected over the PyPSA price.")

    st.subheader("MAE per month")
    st.plotly_chart(charts.monthly_lines(report.monthly), width="stretch")
    st.caption("Median over zones of the MAE in every month.")

    st.subheader("Price replay")
    prices = report.prices
    runs = list(prices.run.cat.categories)
    left, middle, right = st.columns(3)
    run = left.selectbox("Run", runs, index=len(runs) - 1) if len(runs) > 1 else runs[0]
    rows = prices[prices.run == run]
    gains = mae.loc[year].lgbm - mae.loc[year].pypsa
    buses = sorted(rows.bus.unique(), key=lambda b: labels.get(b, b))
    bus = middle.selectbox(
        "Zone",
        buses,
        index=buses.index(gains.idxmin()) if gains.idxmin() in buses else 0,
        format_func=lambda b: label(b, names),
    )
    series = rows[rows.bus == bus].set_index("snapshot").sort_index()
    months = series.index.to_period("M").unique()
    periods = ["Whole year, daily means", *[m.strftime("%B") for m in months]]
    period = right.selectbox("Period", periods, index=1 if len(periods) > 1 else 0)
    columns = ["observed_price", "pypsa", "lgbm"]
    if period == periods[0]:
        shown = series[columns].resample("D").mean()
    else:
        shown = series.loc[series.index.strftime("%B") == period, columns]
    st.plotly_chart(charts.price_lines(shown), width="stretch")
    st.caption(
        "Out-of-sample predictions: every hour is predicted by a model that did not "
        "see its fold. Times in UTC."
    )


def experiments() -> None:
    st.title("Experiment log")
    log = read_experiments()
    st.markdown(
        "Every experiment run with `experiments/run.py`. The scores are not all "
        "comparable: the week split is optimistic, see the notes below."
    )
    st.plotly_chart(charts.experiment_dots(log), width="stretch")
    with st.expander("All scores"):
        st.dataframe(log.drop(columns="features"), width="stretch", hide_index=True)
    st.divider()
    # one heading level down, below the page title
    st.markdown(re.sub(r"^#", "##", read_notes(), flags=re.MULTILINE))
