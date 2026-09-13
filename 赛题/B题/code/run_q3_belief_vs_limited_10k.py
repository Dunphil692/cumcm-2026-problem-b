#!/usr/bin/env python3
"""Paired local Monte Carlo: static find-all (`limited`) vs dynamic (`belief`).

Never talks to the official simulator. Default: 10000 high-fidelity trials,
seed 2027, same world cloned to both policies.
"""

from __future__ import annotations

import argparse
import csv
import json
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
OUT = Path(__file__).resolve().parents[1] / "_figs" / "q3_belief_vs_limited_10k"


def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)


def run_pair(index: int, master_seed: int, fidelity: str) -> dict:
    rng = random.Random(_trial_seed(master_seed, index))
    world = generate_instance(rng)
    t0 = time.perf_counter()
    limited = run_policy(world.clone(), "limited", fidelity=fidelity)
    t1 = time.perf_counter()
    belief = run_policy(world.clone(), "belief", fidelity=fidelity)
    t2 = time.perf_counter()
    return {
        "trial": index,
        "limited": limited,
        "belief": belief,
        "limited_wall_s": t1 - t0,
        "belief_wall_s": t2 - t1,
    }


def _worker(payload: tuple[int, int, str]) -> dict:
    index, master_seed, fidelity = payload
    return run_pair(index, master_seed, fidelity)


def _policy_summary(rows: list[dict], name: str, wall_key: str) -> dict:
    stats = [row[name] for row in rows]
    times = [row["time_s"] for row in stats]
    walks = [row["walk_m"] for row in stats]
    measures = [row["measures"] for row in stats]
    replans = [row["replans"] for row in stats]
    cleared_all = [row["cleared"] == row["n_sources"] for row in stats]
    walls = [row[wall_key] for row in rows]
    n = len(stats)
    return {
        "n": n,
        "clear_rate": sum(cleared_all) / n,
        "n_not_cleared": int(sum(1 for ok in cleared_all if not ok)),
        "time_mean": statistics.fmean(times),
        "time_p50": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "time_stdev": statistics.stdev(times) if n > 1 else 0.0,
        "walk_mean": statistics.fmean(walks),
        "measure_mean": statistics.fmean(measures),
        "replan_mean": statistics.fmean(replans),
        "wall_s": sum(walls),
        "wall_mean_ms": 1000.0 * statistics.fmean(walls),
    }


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
        import os

        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    payloads = [(index, args.seed, args.fidelity) for index in range(args.n)]
    rows: list[dict] = []
    started = time.time()
    print(
        f"belief vs limited  n={args.n}  fidelity={args.fidelity}  "
        f"seed={args.seed}  workers={workers}",
        flush=True,
    )

    mark = max(1, args.n // 20)
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
                elapsed = time.time() - started
                print(
                    f"{len(rows)}/{args.n}  wall={elapsed:.1f}s  "
                    f"lim={row['limited']['time_s']:.0f}s  "
                    f"bel={row['belief']['time_s']:.0f}s  "
                    f"bel_rem={row['belief']['remaining']}",
                    flush=True,
                )
    finally:
        if workers > 1:
            pool.shutdown()

    wins_belief = sum(
        1 for row in rows if row["belief"]["time_s"] < row["limited"]["time_s"]
    )
    summary = {
        "n": args.n,
        "seed": args.seed,
        "fidelity": args.fidelity,
        "official_simulator": False,
        "wall_s": time.time() - started,
        "workers": workers,
        "limited": _policy_summary(rows, "limited", "limited_wall_s"),
        "belief": _policy_summary(rows, "belief", "belief_wall_s"),
        "belief_win_rate": wins_belief / max(1, len(rows)),
        "mean_delta_s": statistics.fmean(
            row["belief"]["time_s"] - row["limited"]["time_s"] for row in rows
        ),
    }

    csv_path = out / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial",
                "policy",
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
            ],
        )
        writer.writeheader()
        for row in rows:
            for name, wall_key in (
                ("limited", "limited_wall_s"),
                ("belief", "belief_wall_s"),
            ):
                stats = row[name]
                writer.writerow(
                    {
                        "trial": row["trial"],
                        "policy": name,
                        "n_sources": stats["n_sources"],
                        "cleared": stats["cleared"],
                        "remaining": stats["remaining"],
                        "time_s": f"{stats['time_s']:.4f}",
                        "walk_m": f"{stats['walk_m']:.4f}",
                        "measures": stats["measures"],
                        "replans": stats["replans"],
                        "wall_s": f"{row[wall_key]:.6f}",
                        "cover_visits": stats.get("cover_visits", ""),
                        "cover_visits_at_first_clear": stats.get(
                            "cover_visits_at_first_clear", ""
                        ),
                    }
                )

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n== belief vs limited ==")
    for name in ("limited", "belief"):
        row = summary[name]
        print(
            f"{name:8s}  clear={row['clear_rate']*100:.2f}%  "
            f"time={row['time_mean']:.1f}s  p50={row['time_p50']:.1f}  "
            f"walk={row['walk_mean']:.0f}m  meas={row['measure_mean']:.1f}  "
            f"cpu={row['wall_s']:.2f}s"
        )
    print(
        f"belief win={summary['belief_win_rate']*100:.1f}%  "
        f"mean Δt={summary['mean_delta_s']:+.1f}s  "
        f"wall={summary['wall_s']:.1f}s"
    )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
