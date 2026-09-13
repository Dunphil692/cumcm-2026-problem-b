#!/usr/bin/env python3
"""Shadow layer-2 fixed R on local virtual worlds. Decisions stay E15."""

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
    rlink_shadow: list[dict] = []
    stats = run_belief_struct(
        world, fidelity="high", mode="e15", rlink_shadow=rlink_shadow
    )
    return {
        "trial": index,
        "remaining": stats["remaining"],
        "time_s": stats["time_s"],
        "any_become_clear": any(row["become_clear"] for row in rlink_shadow),
        "sum_become_clear": sum(row["become_clear"] for row in rlink_shadow),
        "any_enter_probe": any(row["enter_probe"] for row in rlink_shadow),
        "sum_enter_probe": sum(row["enter_probe"] for row in rlink_shadow),
        "any_leave_unresolved": any(row["leave_unresolved"] for row in rlink_shadow),
        "sum_leave_unresolved": sum(row["leave_unresolved"] for row in rlink_shadow),
        "any_samples_dead": any(row["samples_all_dead"] for row in rlink_shadow),
        "sum_vertex_killed": sum(row["vertex_killed"] for row in rlink_shadow),
        "max_mec_shrink": max((row["mec_shrink_max"] for row in rlink_shadow), default=0.0),
        "any_heard_sil": any(row["n_heard_with_sil"] for row in rlink_shadow),
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
        / "q3_e15_rlink_shadow_n200"
        / "summary.json",
    )
    args = parser.parse_args()
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    payloads = [(index, args.seed) for index in range(args.n)]
    rows: list[dict] = []
    started = time.time()
    print(f"rlink-shadow  n={args.n}  workers={workers}", flush=True)
    pool = None
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
        if pool is not None:
            pool.shutdown()
    n = len(rows)
    summary = {
        "n": n,
        "seed": args.seed,
        "official_simulator": False,
        "virtual_world": True,
        "decisions": "e15_unchanged",
        "clear_rate": sum(row["remaining"] == 0 for row in rows) / n,
        "time_mean": statistics.fmean(row["time_s"] for row in rows),
        "games_with_heard_sil": sum(row["any_heard_sil"] for row in rows) / n,
        "games_become_clear": sum(row["any_become_clear"] for row in rows) / n,
        "mean_sum_become_clear": statistics.fmean(row["sum_become_clear"] for row in rows),
        "games_enter_probe": sum(row["any_enter_probe"] for row in rows) / n,
        "mean_sum_enter_probe": statistics.fmean(row["sum_enter_probe"] for row in rows),
        "games_leave_unresolved": sum(row["any_leave_unresolved"] for row in rows) / n,
        "mean_sum_leave_unresolved": statistics.fmean(
            row["sum_leave_unresolved"] for row in rows
        ),
        "games_samples_dead": sum(row["any_samples_dead"] for row in rows) / n,
        "mean_sum_vertex_killed": statistics.fmean(row["sum_vertex_killed"] for row in rows),
        "mean_max_mec_shrink": statistics.fmean(row["max_mec_shrink"] for row in rows),
        "wall_s": time.time() - started,
        "workers": workers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
