#!/usr/bin/env python3
"""
2026 高教社杯全国大学生数学建模竞赛 - B题 问题4
本地蒙特卡洛仿真测试脚本（默认 1000 轮，快速引擎）

算法模型：N19 扇区单向延迟收割 + 奔袭静默免听双优化架构
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
from problem4_belief_n19 import run_belief

MASTER_SEED = 2027

def eval_single_trial(payload):
    index, seed, p_dir, fidelity = payload
    trial_seed = (seed + 100003 * (index + 1)) % (2**31 - 1)
    rng = random.Random(trial_seed)
    
    # Generate mixed omni / directional world
    world = generate_instance(rng, p_dir=p_dir)
    world.remaining = lambda: None
    
    t0 = time.perf_counter()
    stats = run_belief(world, fidelity=fidelity)
    wall_s = time.perf_counter() - t0
    
    n_src = len(world.sources)
    n_dir = sum(1 for s in world.sources if getattr(s, "directional", False))
    cleared_total = stats["cleared"]
    cleared_dir = sum(1 for s in world.sources if getattr(s, "directional", False) and s.cleared)
    is_full = (cleared_total == n_src)
    
    return {
        "trial": index + 1,
        "seed": trial_seed,
        "n_sources": n_src,
        "n_directional": n_dir,
        "cleared_total": cleared_total,
        "cleared_dir": cleared_dir,
        "is_full_cleared": is_full,
        "time_s": stats["time_s"],
        "walk_m": stats["walk_m"],
        "measures": stats["measures"],
        "stops": stats.get("stops", stats.get("n_replan", 0)),
        "stop_reason": stats.get("stop_reason", ""),
        "wall_s": wall_s,
    }


def main():
    parser = argparse.ArgumentParser(description="问题4 N19模型 蒙特卡洛仿真")
    parser.add_argument("--n", type=int, default=1000, help="仿真轮数 (默认 1000)")
    parser.add_argument("--seed", type=int, default=MASTER_SEED, help="主随机种子")
    parser.add_argument("--p-dir", type=float, default=0.5, help="定向源概率 (题面默认 0.5)")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数 (0=自动设为CPU核心数)")
    parser.add_argument("--fidelity", choices=("fast", "high"), default="fast", help="保真度 (默认 fast, 每轮约7秒; high约30秒)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="结果输出目录")
    args = parser.parse_args()

    workers = args.workers
    if workers <= 0:
        workers = max(1, os.cpu_count() or 4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = args.out_dir / f"q4_n19_mc_{args.n}_trials.csv"
    json_file = args.out_dir / f"q4_n19_mc_{args.n}_summary.json"

    print("=" * 88, flush=True)
    print("算法已冻结：N19 + 逆时针 + 1300m扇区延迟 + 奔袭静默。禁止改任何 .py。", flush=True)
    print(f"★ 2026 国赛 B 题【问题4】本地快速蒙特卡洛（1000局）", flush=True)
    print(f"  仿真总轮数: {args.n:,} 轮 | 主种子: {args.seed} | 定向源比例: {args.p_dir*100:.0f}%", flush=True)
    print(f"  计算引擎: {'快速射线交会 (推荐)' if args.fidelity == 'fast' else '高精度Welzl最小覆盖圆'}", flush=True)
    print(f"  并发进程: {workers} 核并行运算", flush=True)
    print(f"  输出文件: {csv_file}", flush=True)
    print("=" * 88, flush=True)

    payloads = [(i, args.seed, args.p_dir, args.fidelity) for i in range(args.n)]
    
    t_start = time.time()
    results = []
    
    # Determine print interval
    if args.n <= 50:
        log_interval = 5
    elif args.n <= 500:
        log_interval = 25
    elif args.n <= 2000:
        log_interval = 100
    else:
        log_interval = 250

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(eval_single_trial, p) for p in payloads]
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done = len(results)
            
            if done % log_interval == 0 or done == args.n:
                elapsed = time.time() - t_start
                rate = done / elapsed
                rem_seconds = (args.n - done) / rate if rate > 0 else 0
                rem_str = f"{int(rem_seconds // 60)}分{int(rem_seconds % 60):02d}秒" if rem_seconds > 60 else f"{int(rem_seconds)}秒"
                
                curr_full = sum(1 for r in results if r["is_full_cleared"])
                curr_time = statistics.mean(r["time_s"] for r in results)
                curr_walk = statistics.mean(r["walk_m"] for r in results)
                
                print(
                    f"[{done:>5}/{args.n} ({done/args.n*100:5.1f}%)] "
                    f"全清率: {curr_full/done*100:5.1f}% | "
                    f"均时: {curr_time:6.1f}s | "
                    f"均走: {curr_walk:5.0f}m | "
                    f"速度: {rate:4.1f}局/s | 剩余约: {rem_str}",
                    flush=True,
                )

    wall_total = time.time() - t_start
    results.sort(key=lambda r: r["trial"])

    # Write CSV
    fieldnames = [
        "trial", "seed", "n_sources", "n_directional", "cleared_total", "cleared_dir",
        "is_full_cleared", "time_s", "walk_m", "measures", "stops", "stop_reason", "wall_s"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute Statistics
    times = [r["time_s"] for r in results]
    walks = [r["walk_m"] for r in results]
    measures = [r["measures"] for r in results]
    sources_tot = sum(r["n_sources"] for r in results)
    cleared_tot = sum(r["cleared_total"] for r in results)
    dir_sources_tot = sum(r["n_directional"] for r in results)
    dir_cleared_tot = sum(r["cleared_dir"] for r in results)
    full_cleared_trials = sum(1 for r in results if r["is_full_cleared"])

    mean_t = statistics.mean(times)
    median_t = statistics.median(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    mean_w = statistics.mean(walks)
    std_w = statistics.stdev(walks) if len(walks) > 1 else 0.0
    mean_m = statistics.mean(measures)
    
    per_source_t = mean_t / (cleared_tot / args.n)

    summary_data = {
        "n_trials": args.n,
        "master_seed": args.seed,
        "p_dir": args.p_dir,
        "fidelity": args.fidelity,
        "wall_clock_seconds": round(wall_total, 1),
        "full_clear_rate": round(full_cleared_trials / args.n * 100, 2),
        "total_sources": sources_tot,
        "total_cleared": cleared_tot,
        "overall_clear_rate": round(cleared_tot / sources_tot * 100, 2),
        "total_dir_sources": dir_sources_tot,
        "total_dir_cleared": dir_cleared_tot,
        "dir_clear_rate": round(dir_cleared_tot / dir_sources_tot * 100, 2),
        "time_s": {
            "mean": round(mean_t, 1),
            "median": round(median_t, 1),
            "std": round(std_t, 1),
            "min": round(min(times), 1),
            "max": round(max(times), 1),
            "per_source": round(per_source_t, 1),
        },
        "walk_m": {
            "mean": round(mean_w, 0),
            "median": round(statistics.median(walks), 0),
            "std": round(std_w, 0),
            "min": round(min(walks), 0),
            "max": round(max(walks), 0),
        },
        "measures": {
            "mean": round(mean_m, 1),
            "median": round(statistics.median(measures), 1),
        }
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Print Final Paper-ready Table
    print("\n" + "=" * 88, flush=True)
    print(f"★ 仿真圆满完成！总墙钟耗时: {wall_total/60:.1f} 分钟 ({wall_total:.1f} 秒)", flush=True)
    print("=" * 88, flush=True)
    print("\n【可直接复制进论文的正式统计表】\n")
    print("| 统计维度 | 评测指标数值 | 备注说明 |")
    print("| :--- | :--- | :--- |")
    print(f"| **仿真实验规模** | **{args.n:,} 轮** | 随机生成 10~16 源场景，覆盖各类空间分布 |")
    print(f"| **满额全清率** | **{full_cleared_trials}/{args.n} ({full_cleared_trials/args.n*100:.2f}%)** | 所有干扰源全部清除、零残留局数占比 |")
    print(f"| **干扰源总清除率** | **{cleared_tot:,} / {sources_tot:,} ({cleared_tot/sources_tot*100:.2f}%)** | 全向与定向混合干扰源的宏观清除比例 |")
    print(f"| **定向源专项清除率** | **{dir_cleared_tot:,} / {dir_sources_tot:,} ({dir_cleared_tot/dir_sources_tot*100:.2f}%)** | 180° 半平面波束隐蔽定向源的捕获清除率 |")
    print(f"| **平均总搜索用时** | **{mean_t:.1f} ± {std_t:.1f} 秒** | 涵盖巡航、测向与清除的全过程虚拟时间 |")
    print(f"| **中位数搜索用时** | **{median_t:.1f} 秒** | 消除极端长尾分布影响的中位性能水平 |")
    print(f"| **单源平均清除时间** | **{per_source_t:.1f} 秒 / 源** | 平均清除单个干扰源所需的折合时间 |")
    print(f"| **平均移动路程** | **{mean_w:.0f} ± {std_w:.0f} 米** | 机器狗全流程地面运动轨迹总长度 |")
    print(f"| **平均测向次数** | **{mean_m:.1f} 次** | 沿途测向切频与定位交会采样总频次 |")
    print("-" * 88, flush=True)
    print(f"数据已保存至:\n  - 明细: {csv_file}\n  - 汇总: {json_file}\n", flush=True)

if __name__ == "__main__":
    main()
