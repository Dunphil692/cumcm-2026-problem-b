#!/usr/bin/env python3
"""Shadow stats: could 2–3 existing stops drop a cover vertex on E15?

Does not change decisions. Same seed family as the 1000-game gate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from problem3_belief_struct import run_belief_struct
from problem3_simulation_model import generate_instance

SEED = 2027


def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)


def run_one(index: int, master_seed: int) -> dict:
    rng = random.Random(_trial_seed(master_seed, index))
    world = generate_instance(rng)
    shadow: list[dict] = []
    stats = run_belief_struct(world, fidelity="high", mode="e15", shadow=shadow)
    snaps = [row for row in shadow if row["n_needed"]]
    extra = [row["joint23_extra"] for row in snaps]
    return {
        "trial": index,
        "remaining": stats["remaining"],
        "time_s": stats["time_s"],
        "n_snaps": len(snaps),
        "any_joint23": any(value > 0 for value in extra),
        "sum_joint23": sum(extra),
        "max_joint23": max(extra) if extra else 0,
        "max_drop1": max((row["drop1"] for row in snaps), default=0),
        "max_drop2": max((row["drop2"] for row in snaps), default=0),
        "max_drop3": max((row["drop3"] for row in snaps), default=0),
        "max_drop_all": max((row["drop_all"] for row in snaps), default=0),
        "any_in_region": any(row["in_region"] for row in snaps),
        "sum_in_region": sum(row["in_region"] for row in snaps),
        "slack_mean": statistics.fmean(row["slack_mean"] for row in snaps)
        if snaps
        else 0.0,
        "helper_mean": statistics.fmean(row["n_helpers"] for row in snaps)
        if snaps
        else 0.0,
    }


def _worker(payload: tuple[int, int]) -> dict:
    return run_one(*payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "_figs"
        / "q3_e15_joint_shadow_n200"
        / "summary.json",
    )
    args = parser.parse_args()
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    payloads = [(index, args.seed) for index in range(args.n)]
    rows: list[dict] = []
    started = time.time()
    print(f"joint-shadow  n={args.n}  workers={workers}", flush=True)
    if workers <= 1:
        iterator = (_worker(payload) for payload in payloads)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_worker, payloads, chunksize=max(1, args.n // (workers * 8)))
    try:
        for row in iterator:
            rows.append(row)
            if len(rows) % max(1, args.n // 10) == 0 or len(rows) == args.n:
                print(f"{len(rows)}/{args.n}  wall={time.time()-started:.1f}s", flush=True)
    finally:
        if workers > 1:
            pool.shutdown()

    n = len(rows)
    leftover = [row["remaining"] for row in rows if row["remaining"]]
    summary = {
        "n": n,
        "seed": args.seed,
        "clear_rate": sum(row["remaining"] == 0 for row in rows) / n,
        "n_not_cleared": len(leftover),
        "time_mean": statistics.fmean(row["time_s"] for row in rows),
        "games_with_joint23": sum(row["any_joint23"] for row in rows) / n,
        "games_with_in_region": sum(row["any_in_region"] for row in rows) / n,
        "mean_sum_joint23": statistics.fmean(row["sum_joint23"] for row in rows),
        "mean_max_joint23": statistics.fmean(row["max_joint23"] for row in rows),
        "mean_max_drop1": statistics.fmean(row["max_drop1"] for row in rows),
        "mean_max_drop2": statistics.fmean(row["max_drop2"] for row in rows),
        "mean_max_drop3": statistics.fmean(row["max_drop3"] for row in rows),
        "mean_max_drop_all": statistics.fmean(row["max_drop_all"] for row in rows),
        "mean_sum_in_region": statistics.fmean(row["sum_in_region"] for row in rows),
        "mean_slack": statistics.fmean(row["slack_mean"] for row in rows),
        "mean_helpers": statistics.fmean(row["helper_mean"] for row in rows),
        "wall_s": time.time() - started,
        "workers": workers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
