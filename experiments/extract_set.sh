#!/bin/bash
# Extracts the solved networks of every year of a backcast checkout into one table set.
# Usage: experiments/extract_set.sh <backcast checkout> <table dir> [years]
# e.g.   experiments/extract_set.sh ../pypsa-eur-backcast/.claude/worktrees/x data/tables-consistent
set -euo pipefail
root=$1
out=$2
years=${3:-"2019 2020 2021 2022 2023 2024"}
mkdir -p "$out"
for year in $years; do
    network="$root/results/backcast-$year/networks/solved_$year.nc"
    [ -f "$network" ] || { echo "missing $network"; exit 1; }
    for attempt in 1 2 3 4 5; do
        # ENTSO-E answers with transient server errors; downloaded zones stay cached
        if pixi run ppf extract "$network" --entsoe --run "backcast-$year" -o "$out/backcast-$year.parquet"; then
            break
        fi
        [ "$attempt" = 5 ] && { echo "extraction of $year failed"; exit 1; }
        sleep 120
    done
done
echo "EXTRACTED $years"
