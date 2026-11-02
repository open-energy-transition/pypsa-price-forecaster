#!/bin/bash
# Extracts the solved networks of every year of a backcast checkout into one table set,
# with the observed prices the backcast scores itself against (no download).
# Usage: experiments/extract_set.sh <backcast checkout> <table dir> [years]
# e.g.   experiments/extract_set.sh ../pypsa-eur-backcast data/tables-consistent
set -euo pipefail
root=$1
out=$2
years=${3:-"2019 2020 2021 2022 2023 2024"}
mkdir -p "$out"
for year in $years; do
    network="$root/results/backcast-$year/networks/solved_$year.nc"
    prices="$root/resources/backcast-$year/day_ahead_prices.csv"
    for file in "$network" "$prices"; do
        [ -f "$file" ] || { echo "missing $file"; exit 1; }
    done
    pixi run ppf extract "$network" --prices "$prices" --run "backcast-$year" -o "$out/backcast-$year.parquet"
done
echo "EXTRACTED $years"
