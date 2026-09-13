#!/usr/bin/env python3
"""问题四 正式版运行器（v14 combo_opk，已采纳 2026-09-13）。

正式策略 = problem4_belief_v14.run_belief(optical_cover=True,
order_mode="optimal", cover_mode="polygon", enable_k4=True)；
基线对照 = v13-N2（回退入口）。同世界配对。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from problem4_belief_v13 import run_belief as run_v13
from problem4_belief_v14 import run_belief as run_v14
from problem4_simulation_model import generate_instance

SEED = 2027
N = 1000
FIDELITY = "high"
OUT = Path(__file__).resolve().parents[1] / "results"

COMBO_OPK = {"order_mode": "optimal", "cover_mode": "polygon", "enable_k4": True}


def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)


def run_pair(index, seed, p_dir):
    rng = random.Random(_trial_seed(seed, index))
    world = generate_instance(rng, p_dir=p_dir)
    w13 = world.clone()
    w14 = world.clone()
    t0 = time.perf_counter()
    r13 = run_v13(w13, fidelity=FIDELITY, optical_cover=True)
    t1 = time.perf_counter()
    r14 = run_v14(w14, fidelity=FIDELITY, optical_cover=True, **COMBO_OPK)
    t2 = time.perf_counter()
    return {
        "trial": index,
        "v13": r13,
        "v14": r14,
        "v13_wall": t1 - t0,
        "v14_wall": t2 - t1,
    }


def _worker(payload):
    index, seed, p_dir = payload
    return run_pair(index, seed, p_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pdir", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tag", type=str, default="v14combo_final")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    workers = args.workers or max(1, min(16, (__import__("os").cpu_count() or 2) - 1))
    payloads = [(i, args.seed, args.pdir) for i in range(args.n)]
    rows = []
    started = time.time()
    print(f"v14 combo_opk 正式版 n={args.n} pdir={args.pdir} seed={args.seed} workers={workers}",
          flush=True)
    mark = max(1, args.n // 10)
    if workers <= 1:
        it = (_worker(p) for p in payloads)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        it = pool.map(_worker, payloads, chunksize=max(1, args.n // (workers * 10)))
    try:
        for row in it:
            rows.append(row)
            if len(rows) % mark == 0 or len(rows) == args.n:
                print(f"{len(rows)}/{args.n} wall={time.time()-started:.1f}s", flush=True)
    finally:
        if workers > 1:
            pool.shutdown()
    t13 = [r["v13"]["time_s"] for r in rows]
    t14 = [r["v14"]["time_s"] for r in rows]
    c13 = sum(1 for r in rows if r["v13"]["cleared"] == r["v13"]["n_sources"])
    c14 = sum(1 for r in rows if r["v14"]["cleared"] == r["v14"]["n_sources"])
    deltas = [a - b for a, b in zip(t13, t14)]  # savings = v13 - v14，正=省时
    n = len(rows)
    se = statistics.stdev(deltas) / math.sqrt(n) if n > 1 else 0.0
    ci_lo = statistics.fmean(deltas) - 1.96 * se
    ci_hi = statistics.fmean(deltas) + 1.96 * se
    summary = {
        "n": n, "seed": args.seed, "p_dir": args.pdir,
        "official_simulator": False, "config": COMBO_OPK,
        "v13_n2": {"clear": c13, "time_mean": statistics.fmean(t13),
                   "time_p50": statistics.median(t13)},
        "v14_combo_opk": {"clear": c14, "time_mean": statistics.fmean(t14),
                          "time_p50": statistics.median(t14)},
        "savings_mean": statistics.fmean(deltas),
        "savings_ci95_normal_approx": [ci_lo, ci_hi],
        "improved_frac": sum(1 for d in deltas if d > 0) / n,
        "wall_s": time.time() - started,
        "workers": workers,
    }
    tag = f"pdir{args.pdir}_{args.tag}"
    (OUT / f"v14_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / f"v14_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "v13_time", "v14_time",
                                           "v13_clear", "v14_clear", "savings"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "trial": r["trial"],
                "v13_time": f"{r['v13']['time_s']:.4f}",
                "v14_time": f"{r['v14']['time_s']:.4f}",
                "v13_clear": r["v13"]["cleared"] == r["v13"]["n_sources"],
                "v14_clear": r["v14"]["cleared"] == r["v14"]["n_sources"],
                "savings": f"{r['v13']['time_s'] - r['v14']['time_s']:.4f}",
            })
    print(f"v13-N2: clear={c13}/{n} mean={summary['v13_n2']['time_mean']:.1f}s")
    print(f"v14 combo_opk: clear={c14}/{n} mean={summary['v14_combo_opk']['time_mean']:.1f}s")
    print(f"savings(配对): mean={summary['savings_mean']:+.1f}s  "
          f"CI95≈[{ci_lo:+.1f},{ci_hi:+.1f}]  改善局占比={summary['improved_frac']*100:.1f}%")
    print(f"wrote v14_{tag}.json/.csv")


if __name__ == "__main__":
    main()
