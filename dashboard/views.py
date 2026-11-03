"""
Pages of the dashboard, one function per page.
"""

import re

import numpy as np
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
    read_experiments_by_year,
    read_notes,
    read_report,
)

MODELS = ["pypsa", "lgbm"]
PRICES = ["observed_price", *MODELS]
BANDS = ["lgbm_q10", "lgbm_q90"]
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
        + (
            f" · left out: {', '.join(map(str, meta['excluded_years']))}"
            if meta.get("excluded_years")
            else ""
        )
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
    if 2022 in report.meta.get("excluded_years", []):
        st.info(
            "2022 is left out of training and scoring. Details in the experiment log, "
            "rounds 13 and 16."
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
        st.subheader("Skill per held-out year")
        mae = report.by_year.mae.unstack("model")
        skill = 1 - mae.lgbm / mae.pypsa
        st.plotly_chart(charts.skill_bars(skill), width="stretch")
        typical = skill.drop(2022, errors="ignore")
        st.caption(
            f"Share by which the corrected price lowers the median MAE of PyPSA in "
            f"each year, predicted by a model that did not see that year. Mean "
            f"{skill.mean():+.0%}, worst year {skill.min():+.0%}"
            + (
                f"; without 2022, whose error is mostly the price level of the "
                f"gas crisis, mean {typical.mean():+.0%}."
                if 2022 in skill.index
                else "."
            )
        )
        st.subheader("Per year")
        st.dataframe(
            report.by_year[["mae", "r", "mae_top5", "mae_low5"]]
            .rename(columns=SCORE_COLUMNS)
            .unstack("model")
            .round(2),
            width="stretch",
        )

    if report.band_scores is not None:
        band_table(report.band_scores)

    st.subheader("Progress over reports")
    if len(report.history) > 1:
        st.plotly_chart(charts.history_lines(report.history), width="stretch")
    st.caption(
        "Every `ppf report` adds a point, so this shows how the default model "
        "changed with the code and runs on master."
    )


def band_table(scores: pd.DataFrame) -> None:
    """Calibration and skill of the price bands per held-out year."""
    st.subheader("How reliable are the price bands?")
    table = pd.DataFrame(
        {
            "inside 10-90 %": scores.coverage_80,
            "above": scores.above_90,
            "below": scores.below_10,
            "width (EUR/MWh)": scores.width_80,
            "pinball skill": 1 - scores.pinball / scores.pinball_point,
            "Brier > 150, bands": scores.brier,
            "Brier > 150, point": scores.brier_point,
        }
    )
    st.dataframe(
        table.style.format(
            {
                **dict.fromkeys(["inside 10-90 %", "above", "below"], "{:.0%}"),
                "width (EUR/MWh)": "{:.0f}",
                "pinball skill": "{:+.0%}",
                **dict.fromkeys(["Brier > 150, bands", "Brier > 150, point"], "{:.3f}"),
            }
        ),
        width="stretch",
    )
    st.caption(
        "Besides the corrected price, the model gives a band from its errors in years "
        "it did not see. Calibrated bands hold 80 % of hours, 10 % above and 10 % "
        "below. Pinball skill compares the band with the corrected price alone over "
        "every quantile, so above zero the band adds information. The Brier score "
        "rates the probability of a price above 150 EUR/MWh, lower is better. Medians "
        "over zones."
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


@st.cache_data
def zone_means(run: str) -> pd.DataFrame:
    """Mean price and MAE of PyPSA and the corrected price per zone in one run."""
    prices = cached_report().prices
    rows = prices[prices.run == run]
    errors = rows[MODELS].sub(rows.observed_price, axis=0).abs().add_prefix("mae_")
    return (
        pd.concat([rows[["bus", *PRICES]], errors], axis=1)
        .groupby("bus", observed=True)
        .mean()
    )


@st.cache_data
def zone_prices(run: str, bus: str) -> pd.DataFrame:
    """Hourly market, PyPSA and corrected price of one zone in one run."""
    prices = cached_report().prices
    rows = prices[(prices.run == run) & (prices.bus == bus)]
    columns = PRICES + [c for c in BANDS if c in prices]
    return rows.set_index("snapshot")[columns].sort_index()


def duration_table(prices: pd.DataFrame, points: int = 501) -> pd.DataFrame:
    """Every price column sorted from high to low, sampled at `points` shares of hours."""
    share = np.linspace(0, 1, points)
    return pd.DataFrame(
        {c: np.quantile(prices[c], 1 - share) for c in PRICES}, index=share
    )


def density_table(prices: pd.DataFrame, model: str, edges: np.ndarray) -> pd.DataFrame:
    """Hours per bin of model price (rows) and market price (columns)."""
    counts, _, _ = np.histogram2d(prices[model], prices.observed_price, [edges, edges])
    centres = (edges[1:] + edges[:-1]) / 2
    return pd.DataFrame(counts, index=centres, columns=centres)


def prices() -> None:
    report = cached_report()
    names = bus_names()
    st.title("How does the correction change prices?")
    st.caption(provenance(report))
    st.markdown(
        "PyPSA prices come from a cost-based model, so they miss part of what "
        "moves market prices: price spikes, negative prices and some of the price "
        "level. This page puts the market price next to the raw PyPSA price and "
        f"the corrected price. {SPLIT_TEXT[report.meta['split']]}"
    )

    runs = list(report.prices.run.cat.categories)
    left, right = st.columns([2, 1])
    run = (
        left.segmented_control(
            "Year",
            runs,
            default=runs[-1],
            format_func=lambda r: r.removeprefix("backcast-"),
        )
        if len(runs) > 1
        else runs[0]
    )
    run = run or runs[-1]
    means = zone_means(run)
    labels = {b: label(b, names) for b in means.index}
    buses = sorted(means.index, key=labels.get)
    zone = st.session_state.get("prices_zone")
    if zone not in buses:
        zone = (means.mae_pypsa - means.mae_lgbm).idxmax()
    bus = right.selectbox(
        "Zone",
        buses,
        index=buses.index(zone),
        format_func=labels.get,
        key="prices_zone",
    )

    hourly = zone_prices(run, bus)
    zone_mean = means.loc[bus]
    st.markdown(
        f"In **{labels[bus]}**, the market price averaged "
        f"**{zone_mean.observed_price:.1f} EUR/MWh**. PyPSA averaged "
        f"**{zone_mean.pypsa:.1f}** ({zone_mean.pypsa - zone_mean.observed_price:+.1f}) "
        f"and the corrected price **{zone_mean.lgbm:.1f}** "
        f"({zone_mean.lgbm - zone_mean.observed_price:+.1f}). The typical hourly "
        f"error (MAE) goes from **{zone_mean.mae_pypsa:.1f}** to "
        f"**{zone_mean.mae_lgbm:.1f} EUR/MWh**."
    )

    st.subheader("Mean price per zone")
    st.plotly_chart(charts.zone_mean_dots(means, labels, bus), width="stretch")
    st.caption(
        "Zones sorted by their market price, the selected zone shaded. Where the blue "
        "dots sit far from the black ones, PyPSA gets the price level wrong; orange "
        "dots closer to black mean the correction fixes it."
    )

    left, right = st.columns(2)
    left.subheader("Price duration curve")
    left.plotly_chart(charts.duration_curves(duration_table(hourly)), width="stretch")
    left.caption(
        "Every hour of the year sorted from the highest to the lowest price. Look at "
        "the two ends: the spikes on the left and the hours at or below zero on the right."
    )
    right.subheader("Typical day")
    right.plotly_chart(
        charts.typical_day(hourly[PRICES].groupby(hourly.index.hour).mean()),
        width="stretch",
    )
    right.caption(
        "Mean price at every hour of the day, in UTC. Look at the shape, such as the "
        "morning and evening peaks and the midday dip where there is much solar."
    )

    if set(BANDS) <= set(hourly):
        band_section(hourly, labels[bus])

    if report.price_setters is not None:
        st.subheader("What sets the model price")
        setters = report.price_setters[report.price_setters.bus == bus]
        shares = (
            setters.assign(
                year=setters.run.str.removeprefix("backcast-"),
                technology=setters.price_setter.where(
                    setters.price_setter.isin(charts.SETTER_COLORS), "other"
                ),
            )
            .pivot_table("share", "year", "technology", aggfunc="sum")
            .fillna(0.0)
        )
        st.plotly_chart(charts.setter_bars(shares), width="stretch")
        st.caption(
            "Share of hours in which each technology sets the PyPSA price of the zone: "
            "the partly loaded plants whose cost matches the price, else storage or "
            "imports. Where gas sets the market price but coal or lignite set the "
            "model price, as in 2022, PyPSA is too cheap."
        )

    st.subheader("Model against market, hour by hour")
    edges = np.linspace(*np.quantile(hourly[PRICES].to_numpy(), [0.005, 0.995]), 61)
    densities = {m: density_table(hourly, m, edges) for m in MODELS}
    zmax = max(d.to_numpy().max() for d in densities.values())
    for column, model in zip(st.columns(2), MODELS, strict=True):
        column.markdown(
            f"**{MODEL_NAMES[model]}**: MAE {zone_mean[f'mae_{model}']:.1f} EUR/MWh, "
            f"r {hourly[model].corr(hourly.observed_price):.2f}"
        )
        column.plotly_chart(
            charts.price_density(densities[model], model, zmax), width="stretch"
        )
    st.caption(
        "Every hour counts in the cell of its market price (across) and model price "
        "(up), darker cells hold more hours. The closer the cells hug the dashed "
        "line, the closer the model is to the market. The most extreme 1 % of "
        "prices falls outside the axes."
    )


def band_section(hourly: pd.DataFrame, zone: str) -> None:
    """Price bands of one zone and run: one month of hours and the share inside."""
    st.subheader("How sure is the corrected price?")
    observed = hourly.observed_price
    inside = (hourly.lgbm_q10 <= observed) & (observed <= hourly.lgbm_q90)
    above, below = (
        (observed > hourly.lgbm_q90).mean(),
        (observed < hourly.lgbm_q10).mean(),
    )
    months = hourly.index.to_period("M")
    names = [m.strftime("%B") for m in months.unique()]
    default = months[observed.groupby(months).transform("mean").argmax()].strftime("%B")
    month = st.selectbox("Month", names, index=names.index(default), key="band_month")
    shown = hourly[months.strftime("%B") == month]
    st.plotly_chart(charts.band_lines(shown), width="stretch")
    st.caption(
        "The band spans the 10 % to the 90 % quantile of the corrected price: the "
        "errors the model made in years it did not see, relative to its price, per "
        "zone. If the band is right, the market sits inside it in 80 % of hours. "
        f"In {zone} this year it was inside in **{inside.mean():.0%}**, above in "
        f"**{above:.0%}** and below in **{below:.0%}** of hours. The month shown "
        "first has the highest market price."
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
    by_year = read_experiments_by_year()
    if by_year is not None:
        with st.expander("Scores per held-out year"):
            st.dataframe(
                by_year.pivot_table(
                    "mae", ["experiment", "split", "model"], "year", sort=False
                ).round(2),
                width="stretch",
            )
    st.divider()
    # one heading level down, below the page title
    st.markdown(re.sub(r"^#", "##", read_notes(), flags=re.MULTILINE))
