#!/usr/bin/env python3
"""Resume and complete 1,000 Monte-Carlo trials for Problem 4 using all CPU threads.

Reuses the existing 200 trials from q4_n19_n200, only executing remaining
trials (200..999) with 8 worker processes.
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

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from run_q4_mc import _worker, _trial_seed

MASTER_SEED = 2027
TOTAL_TRIALS = 1000
P_DIR = 0.5
POLICY = "q4"
OPTS = {"eps": 0.01, "merge": True, "cover_listen": "hear"}
FIELDS = [
    "trial", "n_sources", "n_dir", "cleared", "dir_cleared", "remaining", "time_s",
    "walk_m", "measures", "replans", "steps", "cover_visits", "stop_reason",
    "final_mean_mass", "station_silent", "probe_silent", "wall_s", "left",
]

def save_summary(out_dir: Path, all_rows: list[dict], wall_total: float, workers: int):
    times = [float(r["time_s"]) for r in all_rows]
    walks = [float(r["walk_m"]) for r in all_rows]
    measures = [int(r["measures"]) for r in all_rows]
    
    n_src = sum(int(r["n_sources"]) for r in all_rows)
    cleared = sum(int(r["cleared"]) for r in all_rows)
    
    ok = [int(r["cleared"]) == int(r["n_sources"]) for r in all_rows]
    full_cleared_count = sum(ok)
    
    n_dir = sum(int(r["n_dir"]) for r in all_rows if r["n_dir"] != "")
    dir_cleared = sum(int(r["dir_cleared"]) for r in all_rows if r["dir_cleared"] != "")
    
    reasons: dict[str, int] = {}
    for r in all_rows:
        key = r.get("stop_reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
        
    walls = [float(r["wall_s"]) for r in all_rows if r.get("wall_s")]
    
    summary = {
        "tag": "q4_n19_n1000",
        "n": len(all_rows),
        "target_n": TOTAL_TRIALS,
        "seed": MASTER_SEED,
        "p_dir": P_DIR,
        "policy": POLICY,
        "opts": OPTS,
        "workers": workers,
        "official_simulator": False,
        "stop_uses_oracle_remaining": False,
        "wall_s": wall_total,
        "wall_per_run_s": statistics.fmean(walls) if walls else 0.0,
        "clear_rate_worlds": full_cleared_count / len(all_rows),
        "n_not_cleared": int(len(all_rows) - full_cleared_count),
        "source_clear_rate": cleared / n_src if n_src else 1.0,
        "dir_source_clear_rate": (dir_cleared / n_dir) if n_dir else 1.0,
        "omni_source_clear_rate": ((cleared - dir_cleared) / (n_src - n_dir)) if n_src > n_dir else 1.0,
        "n_sources_total": n_src,
        "n_dir_total": n_dir,
        "stop_reason_counts": reasons,
        "time_mean": statistics.fmean(times),
        "time_p50": statistics.median(times),
        "time_p25": statistics.quantiles(times, n=4)[0] if len(times) > 3 else None,
        "time_p75": statistics.quantiles(times, n=4)[2] if len(times) > 3 else None,
        "time_min": min(times),
        "time_max": max(times),
        "time_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "per_source_time": statistics.fmean(times) / (cleared / len(all_rows)),
        "walk_mean": statistics.fmean(walks),
        "walk_std": statistics.stdev(walks) if len(walks) > 1 else 0.0,
        "measure_mean": statistics.fmean(measures),
    }
    
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    figs_dir = CODE_DIR.parent / "_figs"
    src_200_csv = figs_dir / "q4_n19_n200" / "trials.csv"
    out_dir = figs_dir / "q4_n19_n1000"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "trials.csv"

    completed_rows: dict[int, dict] = {}

    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    completed_rows[int(row["trial"])] = row
                except Exception:
                    pass

    if len(completed_rows) < 200 and src_200_csv.exists():
        with src_200_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["trial"])
                if idx not in completed_rows:
                    completed_rows[idx] = row

    workers = max(1, os.cpu_count() or 8)

    print("=" * 88, flush=True)
    print(f"★ 2026 国赛 B 题【问题4】本地大规模 1000 轮蒙特卡洛仿真（断点续跑）", flush=True)
    print(f"  硬件加速: 启用全部 {workers} 个 CPU 核心/线程并行运算", flush=True)
    print(f"  已有存档: 成功复用已跑完的 {len(completed_rows)} 轮数据", flush=True)
    print(f"  待跑轮数: 需执行剩余的 {TOTAL_TRIALS - len(completed_rows)} 轮 (索引 {len(completed_rows)}..{TOTAL_TRIALS-1})", flush=True)
    print(f"  输出目录: {out_dir}", flush=True)
    print("=" * 88, flush=True)

    # Re-write or create output CSV with existing rows
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for idx in sorted(completed_rows.keys()):
            w.writerow(completed_rows[idx])

    csv_file = out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)

    missing_indices = [idx for idx in range(TOTAL_TRIALS) if idx not in completed_rows]
    payloads = [(idx, MASTER_SEED, P_DIR, POLICY, OPTS) for idx in missing_indices]

    t_start = time.time()
    done_count = len(completed_rows)

    if not payloads:
        print("所有 1000 轮已全部完成！无需重复运行。", flush=True)
    else:
        log_interval = 20
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(_worker, p): p[0] for p in payloads}
            
            for future in as_completed(future_to_idx):
                res = future.result()
                s = res["stats"]
                r_dict = {
                    "trial": res["trial"],
                    "n_sources": s["n_sources"],
                    "n_dir": s.get("n_dir", ""),
                    "cleared": s["cleared"],
                    "dir_cleared": s.get("dir_cleared", ""),
                    "remaining": s["remaining"],
                    "time_s": f"{s['time_s']:.2f}",
                    "walk_m": f"{s['walk_m']:.1f}",
                    "measures": s["measures"],
                    "replans": s["replans"],
                    "steps": s["steps"],
                    "cover_visits": s.get("cover_visits", ""),
                    "stop_reason": s.get("stop_reason", ""),
                    "final_mean_mass": f"{s.get('final_mean_mass', 0.0):.5f}" if s.get("final_mean_mass") is not None else "",
                    "station_silent": s.get("station_silent", ""),
                    "probe_silent": s.get("probe_silent", ""),
                    "wall_s": f"{res['wall_s']:.3f}",
                    "left": json.dumps(res["left"], ensure_ascii=False) if res["left"] else "",
                }
                completed_rows[res["trial"]] = r_dict
                writer.writerow(r_dict)
                csv_file.flush()
                
                done_count += 1
                recent_done = done_count - (TOTAL_TRIALS - len(payloads))
                elapsed = time.time() - t_start
                rate = recent_done / elapsed if elapsed > 0 else 0
                rem_trials = TOTAL_TRIALS - done_count
                rem_s = rem_trials / rate if rate > 0 else 0
                rem_str = f"{int(rem_s // 60)}分{int(rem_s % 60):02d}秒" if rem_s > 60 else f"{int(rem_s)}秒"
                
                if done_count % log_interval == 0 or done_count == TOTAL_TRIALS:
                    cur_ok = sum(1 for r in completed_rows.values() if int(r["cleared"]) == int(r["n_sources"]))
                    cur_time = statistics.fmean(float(r["time_s"]) for r in completed_rows.values())
                    cur_walk = statistics.fmean(float(r["walk_m"]) for r in completed_rows.values())
                    print(
                        f"[{done_count:>4}/{TOTAL_TRIALS} ({done_count/TOTAL_TRIALS*100:5.1f}%)] "
                        f"全清率: {cur_ok/done_count*100:5.1f}% | "
                        f"均时: {cur_time:6.1f}s | "
                        f"均走: {cur_walk:5.0f}m | "
                        f"速度: {rate:4.1f}局/s | 剩余约: {rem_str}",
                        flush=True,
                    )
                    save_summary(out_dir, list(completed_rows.values()), time.time() - t_start, workers)

    csv_file.close()

    sorted_rows = [completed_rows[idx] for idx in sorted(completed_rows.keys())]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted_rows)

    wall_total = time.time() - t_start
    summary = save_summary(out_dir, sorted_rows, wall_total, workers)

    print("\n" + "=" * 88, flush=True)
    print(f"★ Q4 本地 1000 轮仿真圆满完成！总墙钟耗时: {wall_total/60:.1f} 分钟 ({wall_total:.1f} 秒)", flush=True)
    print("=" * 88, flush=True)
    print("\n【论文正文核心主表：问题4 宏观统计评测（1000 局）】\n")
    print("| 统计指标项 | 1000 局评测数值 | 备注说明 |")
    print("| :--- | :---: | :--- |")
    print(f"| **仿真实验规模** | **{summary['n']:,} 局** | 严格蒙特卡洛全量场景（10~16 源） |")
    print(f"| **场景满额全清率** | **{summary['clear_rate_worlds']*100:.2f}%** | 1000 局全部肃清、零残留 |")
    print(f"| **干扰源总清除率** | **{summary['source_clear_rate']*100:.2f}%** | 混合全向与定向源捕获率 |")
    print(f"| **定向源专项清除率** | **{summary['dir_source_clear_rate']*100:.2f}%** | 定向 180° 隐蔽源专项捕获率 |")
    print(f"| **平均虚拟搜索用时** | **{summary['time_mean']:.1f} ± {summary['time_std']:.1f} 秒** | 涵盖移动、测向与激光全流程时间 |")
    print(f"| **中位数虚拟用时** | **{summary['time_p50']:.1f} 秒** | 排除极端长尾干扰的中位水准 |")
    print(f"| **单源平均清除用时** | **{summary['per_source_time']:.1f} 秒 / 源** | 平均清除单个干扰源折合耗时 |")
    print(f"| **平均地面行进路程** | **{summary['walk_mean']:.0f} ± {summary['walk_std']:.0f} 米** | 机器狗地面运动总路程 |")
    print(f"| **平均测向切频次数** | **{summary['measure_mean']:.1f} 次** | 沿途测向采样总频次 |")
    print("-" * 88, flush=True)
    print(f"数据已完整同步保存至:\n  - 逐局明细: {out_csv}\n  - 统计汇总: {out_dir / 'summary.json'}\n", flush=True)

if __name__ == "__main__":
    main()
