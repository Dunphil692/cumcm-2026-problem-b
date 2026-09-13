#!/usr/bin/env python3
"""Local 10000-trial run of the patched dynamic belief policy.

Same seed family as the original belief-vs-limited 10k (2027).
Never talks to the official simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from problem3_fusion_policies import run_policy
from problem3_simulation_model import generate_instance

SEED = 2027
N = 10_000
FIDELITY = "high"
OUT = Path(__file__).resolve().parents[1] / "_figs" / "q3_belief_patched_10k"


def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)


def run_one(index: int, master_seed: int, fidelity: str) -> dict:
    rng = random.Random(_trial_seed(master_seed, index))
    world = generate_instance(rng)
    t0 = time.perf_counter()
    stats = run_policy(world, "belief", fidelity=fidelity)
    return {
        "trial": index,
        "stats": stats,
        "wall_s": time.perf_counter() - t0,
    }


def _worker(payload: tuple[int, int, str]) -> dict:
    return run_one(*payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fidelity", choices=("fast", "high"), default=FIDELITY)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    workers = args.workers
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    payloads = [(index, args.seed, args.fidelity) for index in range(args.n)]
    rows: list[dict] = []
    started = time.time()
    print(
        f"belief patched  n={args.n}  fidelity={args.fidelity}  "
        f"seed={args.seed}  workers={workers}",
        flush=True,
    )

    mark = max(1, args.n // 20)
    pool = None
    if workers <= 1:
        iterator = (_worker(payload) for payload in payloads)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(
            _worker, payloads, chunksize=max(1, args.n // (workers * 20))
        )
    try:
        for row in iterator:
            rows.append(row)
            if len(rows) % mark == 0 or len(rows) == args.n:
                stats = row["stats"]
                print(
                    f"{len(rows)}/{args.n}  wall={time.time()-started:.1f}s  "
                    f"t={stats['time_s']:.0f}s  rem={stats['remaining']}  "
                    f"cover={stats.get('cover_visits')}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    times = [row["stats"]["time_s"] for row in rows]
    walks = [row["stats"]["walk_m"] for row in rows]
    measures = [row["stats"]["measures"] for row in rows]
    replans = [row["stats"]["replans"] for row in rows]
    covers = [row["stats"].get("cover_visits") or 0 for row in rows]
    ok = [row["stats"]["cleared"] == row["stats"]["n_sources"] for row in rows]
    leftover = [row["stats"]["remaining"] for row in rows if row["stats"]["remaining"]]
    ok_times = [row["stats"]["time_s"] for row, flag in zip(rows, ok) if flag]
    reasons = {}
    for row in rows:
        key = row["stats"].get("stop_reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    summary = {
        "n": args.n,
        "seed": args.seed,
        "fidelity": args.fidelity,
        "policy": "belief_v2_heard16_fakerings_omega_p1geom",
        "official_simulator": False,
        "stop_uses_oracle_remaining": False,
        "baseline_belief_10k_mean_s": 4339.94,
        "fullopt_godstop_10k_mean_s": 3797.54,
        "wall_s": time.time() - started,
        "workers": workers,
        "clear_rate": sum(ok) / len(rows),
        "n_not_cleared": int(sum(1 for flag in ok if not flag)),
        "remaining_if_fail": leftover,
        "stop_reason_counts": reasons,
        "time_mean": statistics.fmean(times),
        "time_p50": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "time_stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "time_mean_cleared": statistics.fmean(ok_times) if ok_times else None,
        "walk_mean": statistics.fmean(walks),
        "measure_mean": statistics.fmean(measures),
        "replan_mean": statistics.fmean(replans),
        "cover_mean": statistics.fmean(covers),
        "cover7_frac": sum(1 for c in covers if c >= 7) / len(rows),
    }
    csv_path = out / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial",
                "n_sources",
                "cleared",
                "remaining",
                "time_s",
                "walk_m",
                "measures",
                "replans",
                "wall_s",
                "cover_visits",
                "cover_visits_at_first_clear",
                "stop_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            stats = row["stats"]
            writer.writerow(
                {
                    "trial": row["trial"],
                    "n_sources": stats["n_sources"],
                    "cleared": stats["cleared"],
                    "remaining": stats["remaining"],
                    "time_s": f"{stats['time_s']:.4f}",
                    "walk_m": f"{stats['walk_m']:.4f}",
                    "measures": stats["measures"],
                    "replans": stats["replans"],
                    "wall_s": f"{row['wall_s']:.6f}",
                    "cover_visits": stats.get("cover_visits", ""),
                    "cover_visits_at_first_clear": stats.get(
                        "cover_visits_at_first_clear", ""
                    ),
                    "stop_reason": stats.get("stop_reason", ""),
                }
            )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n== belief patched ==")
    print(
        f"clear={summary['clear_rate']*100:.2f}%  "
        f"time={summary['time_mean']:.1f}s  p50={summary['time_p50']:.1f}  "
        f"cleared_mean={summary['time_mean_cleared']}  "
        f"walk={summary['walk_mean']:.0f}m  cover7={summary['cover7_frac']*100:.1f}%  "
        f"stop={summary.get('stop_reason_counts')}  "
        f"wall={summary['wall_s']:.1f}s"
    )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
