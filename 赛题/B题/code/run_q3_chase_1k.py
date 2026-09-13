#!/usr/bin/env python3
"""Local 1000-trial Monte Carlo of the new cautious-hunt policy.

Same seed family as the fullopt 10k. Never talks to the official simulator.
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

from problem3_chase import run_chase
from problem3_simulation_model import generate_instance

SEED = 2027
N = 1000
OUT = Path(__file__).resolve().parents[1] / "_figs" / "q3_chase_1k"


def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)


def run_one(index: int, master_seed: int) -> dict:
    rng = random.Random(_trial_seed(master_seed, index))
    world = generate_instance(rng)
    t0 = time.perf_counter()
    stats = run_chase(world)
    return {
        "trial": index,
        "stats": stats,
        "wall_s": time.perf_counter() - t0,
    }


def _worker(payload: tuple[int, int]) -> dict:
    return run_one(*payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    workers = args.workers
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    payloads = [(index, args.seed) for index in range(args.n)]
    rows: list[dict] = []
    started = time.time()
    print(
        f"chase  n={args.n}  seed={args.seed}  workers={workers}",
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
                    f"cover={stats.get('cover_visits')}  "
                    f"stop={stats.get('stop_reason')}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    times = [row["stats"]["time_s"] for row in rows]
    walks = [row["stats"]["walk_m"] for row in rows]
    measures = [row["stats"]["measures"] for row in rows]
    covers = [row["stats"].get("cover_visits") or 0 for row in rows]
    ok = [row["stats"]["cleared"] == row["stats"]["n_sources"] for row in rows]
    leftover = [row["stats"]["remaining"] for row in rows if row["stats"]["remaining"]]
    ok_times = [row["stats"]["time_s"] for row, flag in zip(rows, ok) if flag]
    reasons: dict[str, int] = {}
    for row in rows:
        key = row["stats"].get("stop_reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    per_src = [
        row["stats"]["time_s"] / row["stats"]["cleared"]
        for row in rows
        if row["stats"]["cleared"]
    ]
    summary = {
        "n": args.n,
        "seed": args.seed,
        "policy": "chase",
        "official_simulator": False,
        "fullopt_godstop_10k_mean_s": 3797.54,
        "fullopt_godstop_10k_s_per_source": 291.4,
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
        "s_per_source_mean": statistics.fmean(per_src) if per_src else None,
        "walk_mean": statistics.fmean(walks),
        "measure_mean": statistics.fmean(measures),
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
                "steps",
                "cover_visits",
                "stop_reason",
                "wall_s",
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
                    "time_s": f"{stats['time_s']:.3f}",
                    "walk_m": f"{stats['walk_m']:.3f}",
                    "measures": stats["measures"],
                    "replans": stats["replans"],
                    "steps": stats["steps"],
                    "cover_visits": stats.get("cover_visits"),
                    "stop_reason": stats.get("stop_reason"),
                    "wall_s": f"{row['wall_s']:.3f}",
                }
            )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
