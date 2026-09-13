#!/usr/bin/env python3
"""Run 1,000 Monte-Carlo trials with the latest optimal Problem 4 algorithm:
N19 + CCW + 1300m Sector-Delay Sweep + Dash Silence.
Uses all 8 CPU threads.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Set up paths to 问题4上场包
WINPACK_DIR = Path(__file__).resolve().parents[3] / "代码" / "问题4-上场"
sys.path.insert(0, str(WINPACK_DIR))

from problem4_belief_n19 import run_belief_q4
from problem4_simulation_model import generate_instance

MASTER_SEED = 2027
TOTAL_TRIALS = 1000
P_DIR = 0.5
FIELDS = [
    "trial", "seed", "n_sources", "n_dir", "cleared", "dir_cleared", "remaining",
    "is_full_cleared", "time_s", "walk_m", "measures", "replans", "steps",
    "cover_visits", "stop_reason", "final_mean_mass", "wall_s"
]

def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)

def run_one(index: int, master_seed: int, p_dir: float) -> dict:
    trial_seed = _trial_seed(master_seed, index)
    rng = random.Random(trial_seed)
    world = generate_instance(rng, p_dir=p_dir)
    world.remaining = lambda: None
    
    t0 = time.perf_counter()
    stats = run_belief_q4(
        world,
        mute_transit_unheard=True,
        sector_sweep_delay=True,
    )
    wall_s = time.perf_counter() - t0
    
    n_src = stats["n_sources"]
    n_dir = sum(1 for s in world.sources if getattr(s, "directional", False))
    cleared = stats["cleared"]
    dir_cleared = sum(1 for s in world.sources if getattr(s, "directional", False) and s.cleared)
    is_full = (cleared == n_src)
    
    return {
        "trial": index,
        "seed": trial_seed,
        "n_sources": n_src,
        "n_dir": n_dir,
        "cleared": cleared,
        "dir_cleared": dir_cleared,
        "remaining": stats.get("remaining", n_src - cleared),
        "is_full_cleared": is_full,
        "time_s": stats["time_s"],
        "walk_m": stats["walk_m"],
        "measures": stats["measures"],
        "replans": stats.get("replans", 0),
        "steps": stats.get("steps", 0),
        "cover_visits": stats.get("cover_visits", 0),
        "stop_reason": stats.get("stop_reason", ""),
        "final_mean_mass": stats.get("final_mean_mass", 0.0),
        "wall_s": wall_s,
    }

def _worker(payload: tuple) -> dict:
    return run_one(*payload)

def save_summary(out_dir: Path, rows: list[dict], wall_total: float, workers: int) -> dict:
    times = [r["time_s"] for r in rows]
    walks = [r["walk_m"] for r in rows]
    measures = [r["measures"] for r in rows]
    walls = [r["wall_s"] for r in rows]
    
    n_src = sum(r["n_sources"] for r in rows)
    cleared = sum(r["cleared"] for r in rows)
    n_dir = sum(r["n_dir"] for r in rows)
    dir_cleared = sum(r["dir_cleared"] for r in rows)
    full_cleared_count = sum(1 for r in rows if r["is_full_cleared"])
    
    reasons: dict[str, int] = {}
    for r in rows:
        key = r.get("stop_reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
        
    summary = {
        "tag": "q4_optimal_n1000",
        "algorithm": "N19 CCW + 1300m Sector-Delay Sweep + Dash Silence (Latest Optimal)",
        "n": len(rows),
        "target_n": TOTAL_TRIALS,
        "seed": MASTER_SEED,
        "p_dir": P_DIR,
        "workers": workers,
        "wall_s": round(wall_total, 1),
        "wall_per_run_s": round(statistics.fmean(walls), 2) if walls else 0.0,
        "full_clear_rate": round(full_cleared_count / len(rows) * 100, 2),
        "n_not_cleared": int(len(rows) - full_cleared_count),
        "total_sources": n_src,
        "total_cleared": cleared,
        "overall_clear_rate": round(cleared / n_src * 100, 2) if n_src else 100.0,
        "total_dir_sources": n_dir,
        "total_dir_cleared": dir_cleared,
        "dir_clear_rate": round(dir_cleared / n_dir * 100, 2) if n_dir else 100.0,
        "omni_clear_rate": round((cleared - dir_cleared) / (n_src - n_dir) * 100, 2) if n_src > n_dir else 100.0,
        "stop_reason_counts": reasons,
        "time_s": {
            "mean": round(statistics.fmean(times), 1),
            "median": round(statistics.median(times), 1),
            "std": round(statistics.stdev(times), 1) if len(times) > 1 else 0.0,
            "min": round(min(times), 1),
            "max": round(max(times), 1),
            "per_source": round(statistics.fmean(times) / (cleared / len(rows)), 1),
        },
        "walk_m": {
            "mean": round(statistics.fmean(walks), 0),
            "median": round(statistics.median(walks), 0),
            "std": round(statistics.stdev(walks), 0) if len(walks) > 1 else 0.0,
            "min": round(min(walks), 0),
            "max": round(max(walks), 0),
        },
        "measures": {
            "mean": round(statistics.fmean(measures), 1),
            "median": round(statistics.median(measures), 1),
        }
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary

def main():
    figs_dir = Path(__file__).resolve().parents[1] / "_figs"
    out_dir = figs_dir / "q4_optimal_n1000"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "trials.csv"

    completed_rows: dict[int, dict] = {}
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row["trial"])
                    completed_rows[idx] = {
                        "trial": idx,
                        "seed": int(row["seed"]),
                        "n_sources": int(row["n_sources"]),
                        "n_dir": int(row["n_dir"]),
                        "cleared": int(row["cleared"]),
                        "dir_cleared": int(row["dir_cleared"]),
                        "remaining": int(row["remaining"]),
                        "is_full_cleared": row["is_full_cleared"].lower() == "true",
                        "time_s": float(row["time_s"]),
                        "walk_m": float(row["walk_m"]),
                        "measures": int(row["measures"]),
                        "replans": int(row.get("replans", 0)),
                        "steps": int(row.get("steps", 0)),
                        "cover_visits": int(row.get("cover_visits", 0)),
                        "stop_reason": row.get("stop_reason", ""),
                        "final_mean_mass": float(row.get("final_mean_mass", 0.0)),
                        "wall_s": float(row.get("wall_s", 0.0)),
                    }
                except Exception:
                    pass

    workers = max(1, os.cpu_count() or 8)

    print("=" * 88, flush=True)
    print("★ 2026 国赛 B 题【问题4】最新最优定稿算法 1,000 轮本地蒙特卡洛仿真", flush=True)
    print("  算法模型: N19 逆时针单向流 + 1300m扇区延迟收割 + 奔袭静默免听（最新最优定稿版）", flush=True)
    print(f"  硬件加速: 启用全部 {workers} 个 CPU 核心/线程并发运算", flush=True)
    print(f"  仿真轮数: {TOTAL_TRIALS:,} 轮 | 主种子: {MASTER_SEED} | 定向源比例: {P_DIR*100:.0f}%", flush=True)
    print(f"  已完成数: {len(completed_rows)} / {TOTAL_TRIALS}", flush=True)
    print(f"  输出目录: {out_dir}", flush=True)
    print("=" * 88, flush=True)

    # Open CSV for appending
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    csv_file = out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    missing_indices = [i for i in range(TOTAL_TRIALS) if i not in completed_rows]
    payloads = [(i, MASTER_SEED, P_DIR) for i in missing_indices]

    t_start = time.time()
    done_count = len(completed_rows)

    if not payloads:
        print("所有 1,000 轮已全部完成！", flush=True)
    else:
        log_interval = 25
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(_worker, p): p[0] for p in payloads}
            
            for future in as_completed(future_to_idx):
                res = future.result()
                completed_rows[res["trial"]] = res
                
                # Write to CSV
                writer.writerow({
                    "trial": res["trial"],
                    "seed": res["seed"],
                    "n_sources": res["n_sources"],
                    "n_dir": res["n_dir"],
                    "cleared": res["cleared"],
                    "dir_cleared": res["dir_cleared"],
                    "remaining": res["remaining"],
                    "is_full_cleared": res["is_full_cleared"],
                    "time_s": f"{res['time_s']:.2f}",
                    "walk_m": f"{res['walk_m']:.1f}",
                    "measures": res["measures"],
                    "replans": res["replans"],
                    "steps": res["steps"],
                    "cover_visits": res["cover_visits"],
                    "stop_reason": res["stop_reason"],
                    "final_mean_mass": f"{res['final_mean_mass']:.5f}",
                    "wall_s": f"{res['wall_s']:.3f}",
                })
                csv_file.flush()
                
                done_count += 1
                recent_done = done_count - (TOTAL_TRIALS - len(payloads))
                elapsed = time.time() - t_start
                rate = recent_done / elapsed if elapsed > 0 else 0
                rem_trials = TOTAL_TRIALS - done_count
                rem_s = rem_trials / rate if rate > 0 else 0
                rem_str = f"{int(rem_s // 60)}分{int(rem_s % 60):02d}秒" if rem_s > 60 else f"{int(rem_s)}秒"
                
                if done_count % log_interval == 0 or done_count == TOTAL_TRIALS:
                    cur_rows = list(completed_rows.values())
                    cur_full = sum(1 for r in cur_rows if r["is_full_cleared"])
                    cur_time = statistics.fmean(r["time_s"] for r in cur_rows)
                    cur_walk = statistics.fmean(r["walk_m"] for r in cur_rows)
                    print(
                        f"[{done_count:>4}/{TOTAL_TRIALS} ({done_count/TOTAL_TRIALS*100:5.1f}%)] "
                        f"全清率: {cur_full/done_count*100:5.1f}% | "
                        f"均时: {cur_time:6.1f}s | "
                        f"均走: {cur_walk:5.0f}m | "
                        f"速度: {rate:4.1f}局/s | 剩余约: {rem_str}",
                        flush=True,
                    )
                    save_summary(out_dir, cur_rows, time.time() - t_start, workers)

    csv_file.close()

    # Rewrite sorted clean CSV
    sorted_rows = [completed_rows[i] for i in sorted(completed_rows.keys())]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted_rows:
            w.writerow({
                "trial": r["trial"],
                "seed": r["seed"],
                "n_sources": r["n_sources"],
                "n_dir": r["n_dir"],
                "cleared": r["cleared"],
                "dir_cleared": r["dir_cleared"],
                "remaining": r["remaining"],
                "is_full_cleared": r["is_full_cleared"],
                "time_s": f"{r['time_s']:.2f}",
                "walk_m": f"{r['walk_m']:.1f}",
                "measures": r["measures"],
                "replans": r["replans"],
                "steps": r["steps"],
                "cover_visits": r["cover_visits"],
                "stop_reason": r["stop_reason"],
                "final_mean_mass": f"{r['final_mean_mass']:.5f}",
                "wall_s": f"{r['wall_s']:.3f}",
            })

    wall_total = time.time() - t_start
    summary = save_summary(out_dir, sorted_rows, wall_total, workers)

    print("\n" + "=" * 88, flush=True)
    print(f"★ 最新最优 Q4 算法 1,000 轮仿真圆满完成！总墙钟耗时: {wall_total/60:.1f} 分钟 ({wall_total:.1f} 秒)", flush=True)
    print("=" * 88, flush=True)
    print("\n【论文正文核心主表：问题4 最新最优算法 1000 局宏观评测】\n")
    print("| 统计指标项 | 1000 局评测数值 | 备注说明 |")
    print("| :--- | :---: | :--- |")
    print(f"| **仿真实验规模** | **{summary['n']:,} 局** | 随机生成 10~16 源场景，覆盖各类未知空间拓扑与朝向 |")
    print(f"| **场景满额全清率** | **{summary['full_clear_rate']:.2f}%** | 所有干扰源全部肃清、零残留局数占比 |")
    print(f"| **干扰源总清除率** | **{summary['overall_clear_rate']:.2f}%** | 混合全向与定向源的宏观捕获清除率 |")
    print(f"| **全向源专项清除率** | **{summary['omni_clear_rate']:.2f}%** | 全向源绝对肃清率 |")
    print(f"| **定向源专项清除率** | **{summary['dir_clear_rate']:.2f}%** | 180° 半平面隐蔽定向源专项捕获率 |")
    print(f"| **平均虚拟搜索用时** | **{summary['time_s']['mean']:.1f} ± {summary['time_s']['std']:.1f} 秒** | 涵盖巡航、测向与清除的全流程时间 |")
    print(f"| **中位数虚拟用时** | **{summary['time_s']['median']:.1f} 秒** | 消除极端长尾分布影响的中位性能水平 |")
    print(f"| **单源平均清除用时** | **{summary['time_s']['per_source']:.1f} 秒 / 源** | 平均清除单个干扰源折合耗时 |")
    print(f"| **平均地面行进路程** | **{summary['walk_m']['mean']:.0f} ± {summary['walk_m']['std']:.0f} 米** | 机器狗地面运动总路程 |")
    print(f"| **平均测向切频次数** | **{summary['measures']['mean']:.1f} 次** | 沿途测向采样总频次 |")
    print("-" * 88, flush=True)
    print(f"数据已保存至:\n  - 逐局明细: {out_csv}\n  - 统计汇总: {out_dir / 'summary.json'}\n", flush=True)

if __name__ == "__main__":
    main()
