"""
Plotly figures of the dashboard, built from the report tables alone.

Colours are the Okabe-Ito set of pypsa-eur-backcast: black for the market,
blue for PyPSA, orange for the corrected price and grey for the baselines.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.data import MODEL_NAMES, feature_label

OBSERVED_COLOR = "#000000"
MODEL_COLORS = {
    "pypsa": "#0072B2",
    "bus_bias": "#999999",
    "linear": "#999999",
    "lgbm": "#E69F00",
}
MODEL_DASH = {"bus_bias": "dot", "linear": "dash"}
HELPS_COLOR = "#009E73"
HURTS_COLOR = "#D55E00"
NOISE_COLOR = "#BBBBBB"
GAIN_COLOR = "#0072B2"
CONNECTOR_COLOR = "#D5DCE4"
ROW_HEIGHT = 22
PRICE_SERIES = [
    ("observed_price", "Market", OBSERVED_COLOR),
    ("pypsa", MODEL_NAMES["pypsa"], MODEL_COLORS["pypsa"]),
    ("lgbm", MODEL_NAMES["lgbm"], MODEL_COLORS["lgbm"]),
]


def layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    """Shared layout: white background, legend on top, compact margins, overridden by `kwargs`."""
    fig.update_layout(
        dict(
            height=height,
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="closest",
        )
        | kwargs
    )
    return fig


def impact_class(change: float, noise: float, sign: int) -> str:
    """Whether a change in MAE means the feature helps, hurts or is within noise."""
    if abs(change) <= noise:
        return "within noise"
    return "helps" if sign * change > 0 else "hurts"


def ablation_bars(changes: pd.Series, noise: float, sign: int) -> go.Figure:
    """
    Horizontal bars of the change in MAE per feature with the seed noise band.

    `sign` is +1 when a rise in MAE means the feature helps (removing a default
    feature) and -1 when a fall does (adding another feature).
    """
    changes = changes.sort_values(ascending=sign < 0)
    classes = [impact_class(c, noise, sign) for c in changes]
    colors = {"helps": HELPS_COLOR, "hurts": HURTS_COLOR, "within noise": NOISE_COLOR}
    labels = [feature_label(f) for f in changes.index]
    fig = go.Figure()
    for name, color in colors.items():
        mask = [c == name for c in classes]
        fig.add_bar(
            x=changes[mask],
            y=[label for label, m in zip(labels, mask, strict=True) if m],
            orientation="h",
            name=name,
            marker=dict(color=color, cornerradius=4),
            customdata=changes.index[mask],
            hovertemplate="%{y} (%{customdata}): %{x:+.2f} EUR/MWh<extra></extra>",
        )
    if noise > 0:
        fig.add_vrect(
            x0=-noise,
            x1=noise,
            fillcolor=NOISE_COLOR,
            opacity=0.25,
            line_width=0,
            layer="below",
        )
    fig.add_vline(x=0, line_width=1, line_color="#1F2933")
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
    fig.update_xaxes(title="change in median MAE (EUR/MWh)", zeroline=False)
    return layout(fig, ROW_HEIGHT * len(changes) + 110, bargap=0.25)


def importance_bars(gain: pd.Series) -> go.Figure:
    """Horizontal bars of the share of split gain per feature."""
    gain = gain.sort_values()
    fig = go.Figure(
        go.Bar(
            x=gain,
            y=[feature_label(f) for f in gain.index],
            orientation="h",
            marker=dict(color=GAIN_COLOR, cornerradius=4),
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title="share of split gain", tickformat=".0%")
    return layout(fig, ROW_HEIGHT * len(gain) + 90, bargap=0.25)


def bus_dumbbell(mae: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """MAE per bus of PyPSA and the corrected price, sorted by the change."""
    mae = mae.assign(change=mae.lgbm - mae.pypsa).sort_values("change")
    names = [labels.get(b, b) for b in mae.index]
    fig = go.Figure()
    fig.add_scatter(
        x=mae[["pypsa", "lgbm"]].assign(gap=None).to_numpy().ravel(),
        y=[n for n in names for _ in range(3)],
        mode="lines",
        line=dict(color=CONNECTOR_COLOR, width=2),
        hoverinfo="skip",
        showlegend=False,
    )
    for model in ["pypsa", "lgbm"]:
        fig.add_scatter(
            x=mae[model],
            y=names,
            mode="markers",
            name=MODEL_NAMES[model],
            marker=dict(color=MODEL_COLORS[model], size=9),
            customdata=mae.change,
            hovertemplate="%{y}: %{x:.1f} EUR/MWh (change %{customdata:+.1f})<extra></extra>",
        )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(title="MAE (EUR/MWh)", rangemode="tozero")
    return layout(fig, ROW_HEIGHT * len(mae) + 110)


def monthly_lines(monthly: pd.DataFrame) -> go.Figure:
    """Median over buses of the MAE per month and model."""
    monthly = monthly.assign(
        date=pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
    )
    fig = go.Figure()
    for model in MODEL_NAMES:
        rows = monthly[monthly.model == model]
        if rows.empty:
            continue
        fig.add_scatter(
            x=rows.date,
            y=rows.mae,
            name=MODEL_NAMES[model],
            mode="lines+markers",
            line=dict(
                color=MODEL_COLORS[model], width=2, dash=MODEL_DASH.get(model, "solid")
            ),
            marker=dict(size=8),
            hovertemplate="%{x|%b %Y}: %{y:.1f} EUR/MWh<extra>%{fullData.name}</extra>",
        )
    fig.update_yaxes(title="median MAE over zones (EUR/MWh)", rangemode="tozero")
    return layout(fig, hovermode="x unified")


def price_lines(prices: pd.DataFrame) -> go.Figure:
    """Market, PyPSA and corrected price over time, indexed by hour or day."""
    fig = go.Figure()
    for column, name, color in PRICE_SERIES:
        fig.add_scatter(
            x=prices.index,
            y=prices[column],
            name=name,
            mode="lines",
            line=dict(color=color, width=1.5),
            hovertemplate="%{y:.1f} EUR/MWh<extra>%{fullData.name}</extra>",
        )
    fig.update_yaxes(title="EUR/MWh")
    return layout(fig, hovermode="x unified")


def zone_mean_dots(
    means: pd.DataFrame, labels: dict[str, str], selected: str
) -> go.Figure:
    """Mean market, PyPSA and corrected price per zone, sorted by the market price."""
    means = means.sort_values("observed_price")
    fig = go.Figure()
    for column, name, color in PRICE_SERIES:
        fig.add_scatter(
            x=means.index,
            y=means[column],
            name=name,
            mode="markers",
            marker=dict(color=color, size=9),
            customdata=[labels.get(b, b) for b in means.index],
            hovertemplate="%{customdata}: %{y:.1f} EUR/MWh<extra>%{fullData.name}</extra>",
        )
    if selected in means.index:
        position = list(means.index).index(selected)
        fig.add_vrect(
            x0=position - 0.5,
            x1=position + 0.5,
            fillcolor=CONNECTOR_COLOR,
            opacity=0.6,
            line_width=0,
            layer="below",
        )
    fig.update_xaxes(type="category", tickangle=-90)
    fig.update_yaxes(title="mean price (EUR/MWh)")
    return layout(fig, 400)


def duration_curves(curves: pd.DataFrame) -> go.Figure:
    """Prices sorted from the highest to the lowest hour, indexed by the share of hours."""
    fig = price_lines(curves)
    fig.add_hline(y=0, line=dict(color="#1F2933", width=1))
    fig.update_xaxes(
        title="share of hours at a higher price", tickformat=".0%", hoverformat=".1%"
    )
    return fig


def typical_day(profile: pd.DataFrame) -> go.Figure:
    """Mean price per hour of day."""
    fig = price_lines(profile)
    fig.update_traces(mode="lines+markers", marker=dict(size=5))
    fig.update_xaxes(title="hour of day (UTC)", dtick=3, hoverformat="d")
    return fig


def price_density(counts: pd.DataFrame, model: str, zmax: float) -> go.Figure:
    """
    Hours per bin of market price (x) and model price (y), with the line where they agree.

    `counts` has the model price bins as index and the market price bins as columns,
    both at the bin centres in EUR/MWh. Colours are on a log scale up to `zmax` hours.
    """
    counts = counts.where(counts > 0)
    ticks = [t for t in [1, 10, 100, 1000, 10000] if t <= max(zmax, 1)]
    fig = go.Figure(
        go.Heatmap(
            x=counts.columns,
            y=counts.index,
            z=np.log10(counts),
            zmin=0,
            zmax=np.log10(max(zmax, 1)),
            customdata=counts,
            colorscale=[[0, "#DDE3EA"], [1, MODEL_COLORS[model]]],
            colorbar=dict(
                title="hours", thickness=12, tickvals=np.log10(ticks), ticktext=ticks
            ),
            hovertemplate="market %{x:.0f}, model %{y:.0f} EUR/MWh: "
            "%{customdata:.0f} hours<extra></extra>",
        )
    )
    ends = [counts.columns[0], counts.columns[-1]]
    fig.add_scatter(
        x=ends,
        y=ends,
        mode="lines",
        line=dict(color=OBSERVED_COLOR, width=1, dash="dash"),
        hoverinfo="skip",
    )
    fig.update_xaxes(title="market price (EUR/MWh)")
    fig.update_yaxes(title=f"{MODEL_NAMES[model]} (EUR/MWh)")
    return layout(fig, 420, showlegend=False)


def history_lines(history: pd.DataFrame) -> go.Figure:
    """Median MAE of PyPSA and the corrected price in every report."""
    fig = go.Figure()
    for model in ["pypsa", "lgbm"]:
        fig.add_scatter(
            x=history.created,
            y=history[f"mae_{model}"],
            name=MODEL_NAMES[model],
            mode="lines+markers",
            line=dict(color=MODEL_COLORS[model], width=2),
            marker=dict(size=8),
            customdata=history[["commit", "split", "runs"]].to_numpy(),
            hovertemplate="%{y:.2f} EUR/MWh<br>commit %{customdata[0]}, "
            "split %{customdata[1]}, runs %{customdata[2]}<extra>%{fullData.name}</extra>",
        )
    fig.update_yaxes(title="median MAE (EUR/MWh)", rangemode="tozero")
    return layout(fig, 320)


def experiment_dots(experiments: pd.DataFrame) -> go.Figure:
    """MAE of every LightGBM experiment, with the PyPSA MAE of its split."""
    lgbm = experiments[experiments.model == "lgbm"]
    raw = experiments[experiments.model == "pypsa"].mae
    fig = go.Figure()
    symbols = {"month": "circle", "week": "diamond", "year": "square"}
    for split, rows in lgbm.groupby("split"):
        fig.add_scatter(
            x=rows.experiment.str.split("_").str[0],
            y=rows.mae,
            mode="markers",
            name=f"LightGBM, {split} split",
            marker=dict(
                color=MODEL_COLORS["lgbm"], size=10, symbol=symbols.get(split, "circle")
            ),
            customdata=rows[["experiment", "r", "mae_top5"]].to_numpy(),
            hovertemplate="%{customdata[0]}: MAE %{y:.2f}, r %{customdata[1]:.3f}, "
            "top 5 % MAE %{customdata[2]:.1f}<extra></extra>",
        )
    # the raw price is the same in every split of the same runs
    for mae in raw.round(2).unique():
        fig.add_hline(
            y=mae,
            line=dict(color=MODEL_COLORS["pypsa"], width=1, dash="dash"),
            annotation_text=f"PyPSA {mae:.2f}",
            annotation_position="top left",
        )
    ids = lgbm.experiment.str.split("_").str[0].unique()
    fig.update_xaxes(categoryorder="array", categoryarray=list(ids))
    fig.update_yaxes(title="median MAE (EUR/MWh)")
    return layout(fig, 440)
