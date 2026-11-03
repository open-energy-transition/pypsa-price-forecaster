#!/bin/bash
# Extracts the solved networks of every year of a backcast checkout into one table set,
# with the observed prices the backcast scores itself against (no download).
# Usage: experiments/extract_set.sh <backcast checkout> <table dir> [years] [variant]
# e.g.   experiments/extract_set.sh ../pypsa-eur-backcast data/tables-consistent
#        experiments/extract_set.sh ../pypsa-eur-backcast data/tables "$(seq -s ' ' 2019 2025)" recipe
# A variant reads results/backcast-<year>-<variant> and names the runs the same.
set -euo pipefail
root=$1
out=$2
years=${3:-"2019 2020 2021 2022 2023 2024"}
suffix=${4:+-$4}
mkdir -p "$out"
for year in $years; do
    run="backcast-$year$suffix"
    network="$root/results/$run/networks/solved_$year.nc"
    prices="$root/resources/$run/day_ahead_prices.csv"
    for file in "$network" "$prices"; do
        [ -f "$file" ] || { echo "missing $file"; exit 1; }
    done
    pixi run ppf extract "$network" --prices "$prices" --run "$run" -o "$out/$run.parquet"
done
echo "EXTRACTED $years"
