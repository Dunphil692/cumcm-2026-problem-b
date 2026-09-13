#!/usr/bin/env python3
"""Paired local MC: cover-path extra listens x unheard_spacing.

Does not change the N19 census. Factor A is how many extra listen stops
are inserted on a hop *to* a cover point. Factor B is UNHEARD_SPACING for
opportunistic empty-channel listens at non-census stops.
Same trial index => same world, so cells are paired.
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
    way_n: int,
    spacing: float,
    gate_inner: bool,
) -> dict:
    rng = random.Random(_trial_seed(master_seed, index))
    world = generate_instance(rng, p_dir=p_dir)
    t0 = time.perf_counter()
    stats = run_belief_q4(
        world,
        unheard_spacing=spacing,
        cover_way_n=way_n,
        gate_inner_unheard=gate_inner,
    )
    stats.pop("trace", None)
    return {
        "trial": index,
        "way_n": way_n,
        "spacing": spacing,
        "gate_inner": gate_inner,
        "stats": stats,
        "wall_s": time.perf_counter() - t0,
    }


def _worker(payload: tuple) -> dict:
    return run_one(*payload)


def _summarize(rows: list[dict]) -> dict:
    times = [row["stats"]["time_s"] for row in rows]
    walks = [row["stats"]["walk_m"] for row in rows]
    measures = [row["stats"]["measures"] for row in rows]
    ok = [row["stats"]["cleared"] == row["stats"]["n_sources"] for row in rows]
    ok_times = [time_s for time_s, flag in zip(times, ok) if flag]
    return {
        "n": len(rows),
        "clear_rate": sum(ok) / len(rows) if rows else None,
        "n_not_cleared": int(sum(1 for flag in ok if not flag)),
        "time_mean": statistics.fmean(times),
        "time_p50": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "time_mean_cleared": statistics.fmean(ok_times) if ok_times else None,
        "walk_mean": statistics.fmean(walks),
        "measure_mean": statistics.fmean(measures),
        "cover_mean": statistics.fmean([row["stats"].get("cover_visits") or 0 for row in rows]),
        "way_stops_mean": statistics.fmean([row["stats"].get("way_stops") or 0 for row in rows]),
        "opp_unheard_mean": statistics.fmean([row["stats"].get("opp_unheard") or 0 for row in rows]),
        "wall_mean": statistics.fmean([row["wall_s"] for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--p-dir", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ways", default="0,1,2")
    parser.add_argument("--spacings", default="700")
    parser.add_argument("--gate-inner", action="store_true")
    parser.add_argument("--tag", default="q4_n19_way_spacing")
    args = parser.parse_args()

    way_ns = [int(item) for item in args.ways.split(",") if item.strip() != ""]
    spacings = [float(item) for item in args.spacings.split(",") if item.strip() != ""]
    out = FIGS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    workers = args.workers
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    payloads = [
        (index, args.seed, args.p_dir, way_n, spacing, args.gate_inner)
        for way_n in way_ns
        for spacing in spacings
        for index in range(args.n)
    ]
    started = time.time()
    print(
        f"{args.tag}  n={args.n} seed={args.seed} p_dir={args.p_dir} "
        f"ways={way_ns} spacings={spacings} gate_inner={args.gate_inner} "
        f"cells={len(way_ns)*len(spacings)} "
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
                    f"way={row['way_n']} sp={row['spacing']:.0f} gate={int(row['gate_inner'])} "
                    f"t={stats['time_s']:.0f} rem={stats['remaining']} "
                    f"ok={stats['cleared']}/{stats['n_sources']}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    cells: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["way_n"], row["spacing"]), []).append(row)

    baseline_key = (0, 700.0)
    base_by_trial = {}
    if baseline_key in cells:
        base_by_trial = {row["trial"]: row["stats"]["time_s"] for row in cells[baseline_key]}

    table = []
    for way_n in way_ns:
        for spacing in spacings:
            key = (way_n, spacing)
            cell_rows = cells[key]
            summary = _summarize(cell_rows)
            paired_delta = None
            if base_by_trial and key != baseline_key:
                deltas = [
                    row["stats"]["time_s"] - base_by_trial[row["trial"]]
                    for row in cell_rows
                    if row["trial"] in base_by_trial
                ]
                if deltas:
                    paired_delta = statistics.fmean(deltas)
            cell = {
                "way_n": way_n,
                "spacing": spacing,
                "gate_inner": args.gate_inner,
                "paired_delta_vs_way0_sp700": paired_delta,
                **summary,
            }
            table.append(cell)
            delta_txt = (
                f"Δ={paired_delta:+.0f}"
                if paired_delta is not None
                else "baseline"
            )
            print(
                f"way={way_n}  sp={spacing:.0f}  gate={int(args.gate_inner)}  "
                f"clear={summary['clear_rate']*100:.0f}%  "
                f"mean={summary['time_mean']:.0f}  p50={summary['time_p50']:.0f}  "
                f"walk={summary['walk_mean']:.0f}  meas={summary['measure_mean']:.0f}  "
                f"way_stops={summary['way_stops_mean']:.1f}  "
                f"opp={summary['opp_unheard_mean']:.1f}  {delta_txt}",
                flush=True,
            )

    fastest = min(table, key=lambda row: (row["n_not_cleared"], row["time_mean"]))
    payload = {
        "tag": args.tag,
        "n": args.n,
        "seed": args.seed,
        "p_dir": args.p_dir,
        "official_simulator": False,
        "ways": way_ns,
        "spacings": spacings,
        "gate_inner": args.gate_inner,
        "wall_s": time.time() - started,
        "workers": workers,
        "cells": table,
        "fastest": {
            "way_n": fastest["way_n"],
            "spacing": fastest["spacing"],
            "time_mean": fastest["time_mean"],
            "clear_rate": fastest["clear_rate"],
        },
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "trial", "way_n", "spacing", "gate_inner", "n_sources", "cleared", "remaining",
        "time_s", "walk_m", "measures", "cover_visits", "way_stops",
        "opp_unheard", "stop_reason", "wall_s",
    ]
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = row["stats"]
            writer.writerow(
                {
                    "trial": row["trial"],
                    "way_n": row["way_n"],
                    "spacing": row["spacing"],
                    "gate_inner": int(row["gate_inner"]),
                    "n_sources": stats["n_sources"],
                    "cleared": stats["cleared"],
                    "remaining": stats["remaining"],
                    "time_s": f"{stats['time_s']:.2f}",
                    "walk_m": f"{stats['walk_m']:.1f}",
                    "measures": stats["measures"],
                    "cover_visits": stats.get("cover_visits", ""),
                    "way_stops": stats.get("way_stops", ""),
                    "opp_unheard": stats.get("opp_unheard", ""),
                    "stop_reason": stats.get("stop_reason", ""),
                    "wall_s": f"{row['wall_s']:.3f}",
                }
            )
    print(
        f"\nfastest: way={fastest['way_n']} sp={fastest['spacing']:.0f} "
        f"mean={fastest['time_mean']:.0f} clear={fastest['clear_rate']*100:.0f}%  "
        f"wall={payload['wall_s']:.0f}s",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
