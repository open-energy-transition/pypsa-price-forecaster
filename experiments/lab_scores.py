"""
Scores the corrections kept by ``experiments/lab.py``, without refitting.

Usage: ``pixi run python experiments/lab_scores.py [--mean a b ...] [--years] [candidate ...]``

Prints the mean and worst skill per split of every stored candidate (or those
named), and of the mean correction of the candidates after ``--mean``. With
``--distinct``, buses of one zone, which hold the same prices, count once. The best
share of the correction is found on the held-out years themselves, so it only
shows how far the fixed share is from the best one; it is not a fair score.
"""

import argparse

import numpy as np
import pandas as pd
from lab import OUT, SHARE, load, score

SHARES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def stored(split: str) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for file in sorted(OUT.glob(f"*_{split}_s*.parquet")):
        name, seed = file.stem.rsplit(f"_{split}_s", 1)
        result[name, int(seed)] = file
    return result


def line(data, correction) -> dict:
    skill = score(data, correction).skill
    by_share = {s: score(data, correction, s).skill.mean() for s in SHARES}
    best = max(by_share, key=by_share.get)
    return {
        "mean": skill.mean(),
        "worst": skill.min(),
        "best share": best,
        "at best": by_share[best],
        **skill.to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="*")
    parser.add_argument("--mean", nargs="+", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--years", action="store_true")
    parser.add_argument("--distinct", action="store_true")
    args = parser.parse_args()
    data = load()
    # buses of one zone (DE00 and LUG1, GBNI and IE00) hold the same prices: keep one
    t = data.table
    repeated = (t.bus != t.groupby(["run", "zone"]).bus.transform("min")).to_numpy()

    def read(file) -> np.ndarray:
        correction = pd.read_parquet(file).correction.to_numpy()
        return np.where(repeated, np.nan, correction) if args.distinct else correction

    for split in ("year", "forward"):
        files = stored(split)
        rows = {}
        for (name, seed), file in files.items():
            if not args.candidates or name in args.candidates:
                rows[f"{name} s{seed}"] = line(data, read(file))
        if args.mean:
            members = [read(files[name, args.seed]) for name in args.mean]
            rows["mean of " + " + ".join(args.mean)] = line(
                data, np.mean(members, axis=0)
            )
        table = pd.DataFrame(rows).T
        if not args.years:
            table = table[["mean", "worst", "best share", "at best"]]
        print(f"\n{split} split, skill at {SHARE:.0%} of the correction")
        print(table.round(3).to_string())
