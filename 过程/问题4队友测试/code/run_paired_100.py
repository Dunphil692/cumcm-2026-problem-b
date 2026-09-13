#!/usr/bin/env python3
"""
严格配对蒙特卡洛仿真测试：Q4 基线版 vs 优化版（100 局同世界对照）
"""

import os
import sys
import time
import math
import json
import csv
import random
import argparse
import statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from problem4_simulation_model import generate_instance
from problem4_belief_n19_baseline import run_belief as run_belief_base
from problem4_belief_n19 import run_belief as run_belief_up

MASTER_SEED = 2027

def eval_pair(payload):
    index, seed, p_dir, fidelity = payload
    trial_seed = (seed + 100003 * (index + 1)) % (2**31 - 1)
    
    # 严格相同世界生成
    rng_base = random.Random(trial_seed)
    world_base = generate_instance(rng_base, p_dir=p_dir)
    world_base.remaining = lambda: None
    
    rng_up = random.Random(trial_seed)
    world_up = generate_instance(rng_up, p_dir=p_dir)
    world_up.remaining = lambda: None

    # 1. 跑基线版
    t0 = time.perf_counter()
    stats_base = run_belief_base(world_base, fidelity=fidelity)
    wall_base = time.perf_counter() - t0

    # 2. 跑优化版
    t1 = time.perf_counter()
    stats_up = run_belief_up(world_up, fidelity=fidelity)
    wall_up = time.perf_counter() - t1

    n_src = len(world_base.sources)
    n_dir = sum(1 for s in world_base.sources if getattr(s, "directional", False))

    ok_base = (stats_base["cleared"] == n_src)
    ok_up = (stats_up["cleared"] == n_src)

    return {
        "trial": index + 1,
        "seed": trial_seed,
        "n_sources": n_src,
        "n_directional": n_dir,
        "base_ok": ok_base,
        "up_ok": ok_up,
        "base_time": stats_base["time_s"],
        "up_time": stats_up["time_s"],
        "time_saved": stats_base["time_s"] - stats_up["time_s"],
        "base_walk": stats_base["walk_m"],
        "up_walk": stats_up["walk_m"],
        "walk_saved": stats_base["walk_m"] - stats_up["walk_m"],
        "base_measures": stats_base["measures"],
        "up_measures": stats_up["measures"],
    }

def main():
    parser = argparse.ArgumentParser(description="Q4 100局严格配对测试")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--p-dir", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fidelity", choices=("fast", "high"), default="fast")
    args = parser.parse_args()

    workers = args.workers
    if workers <= 0:
        workers = max(1, min(16, (os.cpu_count() or 4)))

    print("=" * 88, flush=True)
    print(f"★ 启动 Q4 严格同世界配对 100 局蒙特卡洛仿真", flush=True)
    print(f"  对比项目: [基线版 N19]  vs  [路径分层+防倒车 优化版]", flush=True)
    print(f"  测试局数: {args.n} 局 | 主种子: {args.seed} | 定向源比例: {args.p_dir*100:.0f}%", flush=True)
    print(f"  计算核心: {workers} 核并行", flush=True)
    print("=" * 88, flush=True)

    payloads = [(i, args.seed, args.p_dir, args.fidelity) for i in range(args.n)]
    t_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(eval_pair, p): p[0] for p in payloads}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done = len(results)
            if done % 10 == 0 or done == args.n:
                elapsed = time.time() - t_start
                delta_t = statistics.fmean([r["time_saved"] for r in results])
                delta_w = statistics.fmean([r["walk_saved"] for r in results])
                print(f"[{done:>3d}/{args.n}] 耗时 {elapsed:5.1f}s | 当前均省: {delta_t:+6.1f}s | 均少走: {delta_w:+6.1f}m", flush=True)

    results.sort(key=lambda r: r["trial"])

    # 统计总结
    base_times = [r["base_time"] for r in results]
    up_times = [r["up_time"] for r in results]
    time_saved = [r["time_saved"] for r in results]
    base_walks = [r["base_walk"] for r in results]
    up_walks = [r["up_walk"] for r in results]
    walk_saved = [r["walk_saved"] for r in results]

    base_clears = sum(1 for r in results if r["base_ok"])
    up_clears = sum(1 for r in results if r["up_ok"])

    print("\n" + "=" * 88)
    print("★ 100 局严格配对对比结果汇报")
    print("=" * 88)
    print(f"1. 满额全清率: 基线版 {base_clears}/{len(results)} ({base_clears/len(results)*100:.1f}%) | 优化版 {up_clears}/{len(results)} ({up_clears/len(results)*100:.1f}%)")
    print(f"2. 平均搜索用时: 基线版 {statistics.fmean(base_times):.1f}s  ->  优化版 {statistics.fmean(up_times):.1f}s  (净省 {statistics.fmean(time_saved):+.1f}s, 降幅 {statistics.fmean(time_saved)/statistics.fmean(base_times)*100:.2f}%)")
    print(f"3. 中位数用时:   基线版 {statistics.median(base_times):.1f}s  ->  优化版 {statistics.median(up_times):.1f}s  (中位净省 {statistics.median(time_saved):+.1f}s)")
    print(f"4. 平均移动路程: 基线版 {statistics.fmean(base_walks):.1f}m  ->  优化版 {statistics.fmean(up_walks):.1f}m  (少走 {statistics.fmean(walk_saved):+.1f}m)")
    print(f"5. 平均测向次数: 基线版 {statistics.fmean([r['base_measures'] for r in results]):.1f}次 -> 优化版 {statistics.fmean([r['up_measures'] for r in results]):.1f}次")
    print("=" * 88)

if __name__ == "__main__":
    main()
