"""
Command line interface: extract tables from solved networks, evaluate, train and predict.
"""

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import pypsa
import typer

from pypsa_price_forecaster import models as model_io
from pypsa_price_forecaster.dataset import (
    DEFAULT_FEATURES,
    FEATURES,
    TARGET,
    attach_exogenous,
    build_table,
    read_tables,
)
from pypsa_price_forecaster.entsoe import observed_prices, read_area_to_bus
from pypsa_price_forecaster.evaluate import SPLITS, bus_metrics, cross_validate, summary
from pypsa_price_forecaster.models import DEFAULT_MODELS
from pypsa_price_forecaster.report import write_report

app = typer.Typer(
    help="Convert PyPSA marginal prices to day-ahead electricity prices.",
    no_args_is_help=True,
)

Tables = Annotated[
    list[Path], typer.Argument(help="Tables written by `extract`, one per run.")
]
Features = Annotated[
    list[str] | None,
    typer.Option(
        "--feature",
        "-f",
        help=f"Feature to use, repeatable. Default: {', '.join(DEFAULT_FEATURES)}. "
        f"Available: {', '.join(FEATURES)}.",
    ),
]
Variables = Annotated[
    list[Path] | None,
    typer.Option(
        "--variables",
        help="Variables from outside the network written by `exogenous` or `forecasts`, "
        "needed by their features, repeatable.",
    ),
]


def with_variables(table: pd.DataFrame, variables: list[Path] | None) -> pd.DataFrame:
    for path in variables or []:
        table = attach_exogenous(table, pd.read_parquet(path))
    return table


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def extract(
    network: Annotated[Path, typer.Argument(help="Solved PyPSA network (.nc).")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output table (.parquet).")],
    prices: Annotated[
        Path | None,
        typer.Option(
            help="Observed prices, wide CSV of naive UTC hours by bus "
            "(resources/day_ahead_prices.csv of pypsa-eur-backcast)."
        ),
    ] = None,
    entsoe: Annotated[
        bool,
        typer.Option(
            "--entsoe",
            help="Download the observed prices from the ENTSO-E Transparency Platform "
            "(token in ENTSOE_API_TOKEN).",
        ),
    ] = False,
    cache_dir: Annotated[
        Path, typer.Option(help="Cache of the ENTSO-E downloads.")
    ] = Path("data/entsoe"),
    area_to_bus: Annotated[
        Path | None,
        typer.Option(
            help="CSV mapping buses to ENTSO-E areas (columns bus, entsoe_area), "
            "default: the map of pypsa-eur-backcast."
        ),
    ] = None,
    run: Annotated[
        str | None, typer.Option(help="Run name, default: network file name.")
    ] = None,
) -> None:
    """Extract the hourly model price, load, supply and import capacity per bus."""
    if entsoe and prices is not None:
        raise typer.BadParameter("Give either --prices or --entsoe, not both.")
    n = pypsa.Network(network)
    if entsoe:
        prices = observed_prices(
            n.buses.index[n.buses.carrier == "AC"],
            n.snapshots,
            cache_dir,
            read_area_to_bus(area_to_bus),
        )
    table = build_table(n, prices, run or network.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    typer.echo(
        f"Wrote {len(table)} rows of run {table.run.iat[0]} ({table.year.iat[0]}, "
        f"{table.bus.nunique()} buses) to {out}"
    )


@app.command("exogenous")
def exogenous_command(
    backcast: Annotated[
        Path,
        typer.Argument(
            help="pypsa-eur-backcast checkout with the cutouts and resources."
        ),
    ],
    years: Annotated[
        list[int], typer.Option("--year", help="Weather year, repeatable.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output (.parquet).")] = Path(
        "data/exogenous.parquet"
    ),
    cache_dir: Annotated[
        Path, typer.Option(help="Cache of the ENTSO-E downloads.")
    ] = Path("data/entsoe"),
) -> None:
    """Build daily temperature, runoff and hydro reservoir filling per bus (token in ENTSOE_API_TOKEN)."""
    from pypsa_price_forecaster.exogenous import build

    daily = build(backcast, years, cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out, index=False)
    typer.echo(f"Wrote {len(daily)} rows ({daily.bus.nunique()} buses) to {out}")


@app.command()
def forecasts(
    years: Annotated[list[int], typer.Option("--year", help="Year, repeatable.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output (.parquet).")] = Path(
        "data/forecasts.parquet"
    ),
    cache_dir: Annotated[
        Path, typer.Option(help="Cache of the ENTSO-E downloads.")
    ] = Path("data/entsoe"),
) -> None:
    """Download the day-ahead load and wind and solar forecasts per bus (token in ENTSOE_API_TOKEN)."""
    from pypsa_price_forecaster.forecasts import build

    hourly = build(years, cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out, index=False)
    typer.echo(f"Wrote {len(hourly)} rows ({hourly.bus.nunique()} buses) to {out}")


@app.command()
def evaluate(
    tables: Tables,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")],
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help=f"Model, repeatable. Default: {', '.join(DEFAULT_MODELS)}.",
        ),
    ] = None,
    feature: Features = None,
    split: Annotated[
        str,
        typer.Option(
            help=f"Fold split, one of {', '.join(SPLITS)}: one fold per year, "
            "one fold per year fitted on the earlier years only, "
            "every n-th week of all years as fold n, or one fold per calendar month."
        ),
    ] = "year",
    folds: Annotated[
        int, typer.Option(help="Number of folds of the week split.", min=2)
    ] = 4,
    variables: Variables = None,
) -> None:
    """Predict every fold with models fitted on the other folds and score against observed prices."""
    table = with_variables(read_tables(tables), variables)
    predictions = cross_validate(table, model or DEFAULT_MODELS, feature, split, folds)
    metrics = bus_metrics(table, predictions)
    scores = summary(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    table[["run", "year", "snapshot", "bus", TARGET]].join(predictions).to_parquet(
        out_dir / "predictions.parquet", index=False
    )
    metrics.to_csv(out_dir / "bus_metrics.csv")
    scores.to_csv(out_dir / "summary.csv")
    typer.echo("Median over buses, out of sample:")
    typer.echo(scores.round(3).to_string())


@app.command()
def report(
    tables: Tables,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")] = Path("report"),
    split: Annotated[
        str | None,
        typer.Option(help="Fold split, default: year with several years, else month."),
    ] = None,
    ablation: Annotated[
        bool,
        typer.Option(
            help="Refit lgbm without and with each feature and with other seeds."
        ),
    ] = True,
    bands: Annotated[
        bool,
        typer.Option(
            help="Add price bands, refitting lgbm per pair of held-out years."
        ),
    ] = True,
    exclude_year: Annotated[
        list[int] | None,
        typer.Option(help="Leave a year out of training and scoring (repeatable)."),
    ] = None,
    variables: Variables = None,
) -> None:
    """Write the report of the default models that the dashboard shows."""
    scores = write_report(
        with_variables(read_tables(tables), variables),
        out_dir,
        split,
        with_ablation=ablation,
        with_bands=bands,
        exclude_years=exclude_year,
    )
    typer.echo(f"Wrote the report to {out_dir}. Median over buses, out of sample:")
    typer.echo(scores.round(3).to_string())


@app.command()
def train(
    tables: Tables,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output model (.joblib).")],
    model: Annotated[str, typer.Option("--model", "-m")] = "lgbm",
    feature: Features = None,
    variables: Variables = None,
) -> None:
    """Fit a model on all rows with observed price."""
    table = with_variables(read_tables(tables), variables)
    fitted = model_io.make_model(model, feature).fit(table)
    model_io.save(fitted, out)
    typer.echo(f"Wrote {model} fitted on {', '.join(fitted.trained_on)} to {out}")


@app.command()
def predict(
    source: Annotated[
        Path, typer.Argument(help="Solved PyPSA network (.nc) or table (.parquet).")
    ],
    model: Annotated[Path, typer.Option("--model", "-m", help="Model from `train`.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output CSV of naive UTC hours by bus.")
    ],
    variables: Variables = None,
) -> None:
    """Predict hourly day-ahead prices per bus of a solved network."""
    table = with_variables(
        pd.read_parquet(source) if source.suffix == ".parquet" else build_table(source),
        variables,
    )
    fitted = model_io.load(model)
    prices = (
        table.assign(price=fitted.predict(table))
        .pivot(index="snapshot", columns="bus", values="price")
        .rename_axis(index="snapshot", columns=None)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out)
    typer.echo(
        f"Wrote prices of {prices.shape[1]} buses and {len(prices)} hours to {out}"
    )
