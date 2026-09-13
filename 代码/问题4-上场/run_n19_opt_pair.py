#!/usr/bin/env python3
"""Paired local MC: N19 pre-opt vs sector-delay + mute-transit.

Same trial index => same world. Records n_dir so we can split the delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from problem4_belief_n19 import run_belief_q4
from problem4_simulation_model import generate_instance

SEED = 2027
FIGS = Path(__file__).resolve().parents[2] / "结果" / "仿真图"


def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)


def run_one(
    index: int,
    master_seed: int,
    p_dir: float,
    optimized: bool,
) -> dict:
    rng = random.Random(_trial_seed(master_seed, index))
    world = generate_instance(rng, p_dir=p_dir)
    t0 = time.perf_counter()
    stats = run_belief_q4(
        world,
        mute_transit_unheard=optimized,
        sector_sweep_delay=optimized,
    )
    stats.pop("trace", None)
    return {
        "trial": index,
        "optimized": optimized,
        "stats": stats,
        "wall_s": time.perf_counter() - t0,
    }


def _worker(payload: tuple) -> dict:
    return run_one(*payload)


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    times = [row["stats"]["time_s"] for row in rows]
    walks = [row["stats"]["walk_m"] for row in rows]
    measures = [row["stats"]["measures"] for row in rows]
    ok = [row["stats"]["cleared"] == row["stats"]["n_sources"] for row in rows]
    return {
        "n": len(rows),
        "clear_rate": sum(ok) / len(rows),
        "n_not_cleared": int(sum(1 for flag in ok if not flag)),
        "time_mean": statistics.fmean(times),
        "time_p50": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "walk_mean": statistics.fmean(walks),
        "measure_mean": statistics.fmean(measures),
        "n_dir_mean": statistics.fmean([row["stats"]["n_dir"] for row in rows]),
        "n_src_mean": statistics.fmean([row["stats"]["n_sources"] for row in rows]),
        "opp_unheard_mean": statistics.fmean(
            [row["stats"].get("opp_unheard") or 0 for row in rows]
        ),
    }


def _dir_bin(n_dir: int) -> str:
    if n_dir <= 3:
        return "0-3"
    if n_dir <= 6:
        return "4-6"
    if n_dir <= 9:
        return "7-9"
    return "10+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--p-dir", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tag", default="q4_n19_opt_pair")
    args = parser.parse_args()

    out = FIGS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    workers = args.workers
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    payloads = [
        (index, args.seed, args.p_dir, optimized)
        for optimized in (False, True)
        for index in range(args.n)
    ]
    started = time.time()
    print(
        f"{args.tag}  n={args.n} seed={args.seed} p_dir={args.p_dir} "
        f"jobs={len(payloads)} workers={workers}",
        flush=True,
    )

    rows: list[dict] = []
    mark = max(1, len(payloads) // 10)
    pool = None
    if workers <= 1:
        iterator = (_worker(payload) for payload in payloads)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_worker, payloads, chunksize=1)
    try:
        for row in iterator:
            rows.append(row)
            if len(rows) % mark == 0 or len(rows) == len(payloads):
                stats = row["stats"]
                print(
                    f"{len(rows)}/{len(payloads)} wall={time.time()-started:.0f}s "
                    f"opt={int(row['optimized'])} t={stats['time_s']:.0f} "
                    f"ok={stats['cleared']}/{stats['n_sources']} "
                    f"n_dir={stats['n_dir']}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    old_by = {row["trial"]: row for row in rows if not row["optimized"]}
    new_by = {row["trial"]: row for row in rows if row["optimized"]}
    pairs = []
    for index in range(args.n):
        old = old_by[index]["stats"]
        new = new_by[index]["stats"]
        pairs.append(
            {
                "trial": index,
                "n_sources": old["n_sources"],
                "n_dir": old["n_dir"],
                "dir_frac": old["n_dir"] / old["n_sources"] if old["n_sources"] else 0.0,
                "dir_bin": _dir_bin(old["n_dir"]),
                "old_ok": old["cleared"] == old["n_sources"],
                "new_ok": new["cleared"] == new["n_sources"],
                "old_time": old["time_s"],
                "new_time": new["time_s"],
                "d_time": old["time_s"] - new["time_s"],
                "old_walk": old["walk_m"],
                "new_walk": new["walk_m"],
                "d_walk": old["walk_m"] - new["walk_m"],
                "old_meas": old["measures"],
                "new_meas": new["measures"],
                "d_meas": old["measures"] - new["measures"],
                "old_opp": old.get("opp_unheard") or 0,
                "new_opp": new.get("opp_unheard") or 0,
            }
        )

    old_sum = _summarize([row for row in rows if not row["optimized"]])
    new_sum = _summarize([row for row in rows if row["optimized"]])
    d_times = [row["d_time"] for row in pairs]
    d_walks = [row["d_walk"] for row in pairs]
    first10 = [row for row in pairs if row["trial"] < 10]

    by_bin: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        by_bin[row["dir_bin"]].append(row)

    bin_table = []
    for label in ("0-3", "4-6", "7-9", "10+"):
        group = by_bin.get(label, [])
        if not group:
            continue
        bin_table.append(
            {
                "dir_bin": label,
                "n": len(group),
                "n_dir_mean": statistics.fmean([row["n_dir"] for row in group]),
                "old_time": statistics.fmean([row["old_time"] for row in group]),
                "new_time": statistics.fmean([row["new_time"] for row in group]),
                "d_time": statistics.fmean([row["d_time"] for row in group]),
                "d_walk": statistics.fmean([row["d_walk"] for row in group]),
                "old_clear": sum(row["old_ok"] for row in group) / len(group),
                "new_clear": sum(row["new_ok"] for row in group) / len(group),
                "win_new": sum(row["d_time"] > 0 for row in group),
                "win_old": sum(row["d_time"] < 0 for row in group),
            }
        )

    payload = {
        "tag": args.tag,
        "n": args.n,
        "seed": args.seed,
        "p_dir": args.p_dir,
        "official_simulator": False,
        "wall_s": time.time() - started,
        "workers": workers,
        "old": old_sum,
        "new": new_sum,
        "paired": {
            "d_time_mean": statistics.fmean(d_times),
            "d_time_p50": statistics.median(d_times),
            "d_time_min": min(d_times),
            "d_time_max": max(d_times),
            "d_walk_mean": statistics.fmean(d_walks),
            "new_faster": sum(d > 0 for d in d_times),
            "old_faster": sum(d < 0 for d in d_times),
            "tie": sum(d == 0 for d in d_times),
        },
        "first10": {
            "n": len(first10),
            "d_time_mean": statistics.fmean([row["d_time"] for row in first10])
            if first10
            else None,
            "d_walk_mean": statistics.fmean([row["d_walk"] for row in first10])
            if first10
            else None,
        },
        "by_dir_bin": bin_table,
        "pairs": pairs,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "trial",
        "optimized",
        "n_sources",
        "n_dir",
        "cleared",
        "remaining",
        "time_s",
        "walk_m",
        "measures",
        "opp_unheard",
        "stop_reason",
        "wall_s",
    ]
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = row["stats"]
            writer.writerow(
                {
                    "trial": row["trial"],
                    "optimized": int(row["optimized"]),
                    "n_sources": stats["n_sources"],
                    "n_dir": stats["n_dir"],
                    "cleared": stats["cleared"],
                    "remaining": stats["remaining"],
                    "time_s": f"{stats['time_s']:.2f}",
                    "walk_m": f"{stats['walk_m']:.1f}",
                    "measures": stats["measures"],
                    "opp_unheard": stats.get("opp_unheard", ""),
                    "stop_reason": stats.get("stop_reason", ""),
                    "wall_s": f"{row['wall_s']:.3f}",
                }
            )

    print(
        f"\nold  clear={old_sum['clear_rate']*100:.0f}%  "
        f"mean={old_sum['time_mean']:.1f}  walk={old_sum['walk_mean']:.0f}  "
        f"meas={old_sum['measure_mean']:.1f}",
        flush=True,
    )
    print(
        f"new  clear={new_sum['clear_rate']*100:.0f}%  "
        f"mean={new_sum['time_mean']:.1f}  walk={new_sum['walk_mean']:.0f}  "
        f"meas={new_sum['measure_mean']:.1f}",
        flush=True,
    )
    print(
        f"paired Δtime={payload['paired']['d_time_mean']:+.1f}  "
        f"Δwalk={payload['paired']['d_walk_mean']:+.0f}  "
        f"new_faster={payload['paired']['new_faster']}/{args.n}  "
        f"first10 Δt={payload['first10']['d_time_mean']}",
        flush=True,
    )
    for cell in bin_table:
        print(
            f"  n_dir {cell['dir_bin']:>4}  n={cell['n']:2d}  "
            f"old={cell['old_time']:.0f}  new={cell['new_time']:.0f}  "
            f"Δ={cell['d_time']:+.0f}  new_wins={cell['win_new']}/{cell['n']}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
