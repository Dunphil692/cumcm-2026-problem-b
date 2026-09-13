#!/usr/bin/env python3
"""Paired local Monte Carlo for the three problem-3 fusion policies.

Never talks to the official simulator. Fast layer uses the cheap two-bearing
diameter approximation; high-fidelity layer uses problem-1 half-plane intersection.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import Counter
from pathlib import Path

from problem3_fusion_policies import run_policy_set
from problem3_simulation_model import generate_instance

POLICIES = ("strict", "full", "limited")
KIND_KEYS = (
    "near",
    "wedge",
    "unbounded",
    "too_large",
    "clearable",
    "empty_intersection",
    "cleared",
)


def _blank_kinds() -> dict[str, int]:
    return {key: 0 for key in KIND_KEYS}


def run_pair(world, fidelity: str) -> dict[str, dict]:
    return run_policy_set(world, fidelity=fidelity)


def summarize(trials: list[dict[str, dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in POLICIES:
        times = [row[name]["time_s"] for row in trials]
        walks = [row[name]["walk_m"] for row in trials]
        cleared = [row[name]["cleared"] == row[name]["n_sources"] for row in trials]
        replans = [row[name]["replans"] for row in trials]
        regrets = [row[name]["regret_s"] for row in trials]
        measures = [row[name]["measures"] for row in trials]
        kinds = _blank_kinds()
        for row in trials:
            for key, value in row[name]["census_kinds"].items():
                if key in kinds:
                    kinds[key] += value
        n = max(1, len(trials))
        out[name] = {
            "n": len(trials),
            "clear_rate": sum(cleared) / n,
            "time_mean": statistics.fmean(times),
            "time_p50": statistics.median(times),
            "walk_mean": statistics.fmean(walks),
            "replan_mean": statistics.fmean(replans),
            "regret_mean": statistics.fmean(regrets),
            "measure_mean": statistics.fmean(measures),
            "kinds": {key: kinds[key] / n for key in KIND_KEYS},
        }
    wins = {name: 0 for name in POLICIES}
    for row in trials:
        best = min(POLICIES, key=lambda name: row[name]["time_s"])
        wins[best] += 1
    for name in POLICIES:
        out[name]["win_rate"] = wins[name] / max(1, len(trials))
    return out


def write_csv(path: Path, trials: list[dict[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "policy",
        "n_sources",
        "cleared",
        "time_s",
        "walk_m",
        "measures",
        "replans",
        "regret_s",
        "remaining",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(trials):
            for name in POLICIES:
                stats = row[name]
                writer.writerow(
                    {
                        "trial": index,
                        "policy": name,
                        "n_sources": stats["n_sources"],
                        "cleared": stats["cleared"],
                        "time_s": f"{stats['time_s']:.4f}",
                        "walk_m": f"{stats['walk_m']:.4f}",
                        "measures": stats["measures"],
                        "replans": stats["replans"],
                        "regret_s": f"{stats['regret_s']:.4f}",
                        "remaining": stats["remaining"],
                    }
                )


def plot_summary(path: Path, summary: dict[str, dict], title: str) -> None:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mplconfig"))
        path.parent.joinpath(".mplconfig").mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        names = list(POLICIES)
        times = [summary[name]["time_mean"] for name in names]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        axes[0].bar(names, times, color="#4c6b8a")
        axes[0].set_ylabel("mean virtual time (s)")
        axes[0].set_title(title)
        kinds = KIND_KEYS
        limited = summary["limited"]["kinds"]
        axes[1].bar(kinds, [limited[key] for key in kinds], color="#7a8f6a")
        axes[1].set_ylabel("mean channels / trial")
        axes[1].set_title("limited: census kinds (non-empty)")
        axes[1].tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception:
        return


def run_layer(
    n_trials: int,
    seed: int,
    fidelity: str,
    out_csv: Path,
    out_png: Path,
) -> dict[str, dict]:
    rng = __import__("random").Random(seed)
    trials: list[dict[str, dict]] = []
    started = time.time()
    for index in range(n_trials):
        world = generate_instance(rng)
        trials.append(run_pair(world, fidelity))
        if (index + 1) % max(1, n_trials // 10) == 0:
            elapsed = time.time() - started
            print(
                f"{fidelity} {index + 1}/{n_trials}  "
                f"{elapsed:.1f}s  last limited={trials[-1]['limited']['time_s']:.0f}s",
                flush=True,
            )
    summary = summarize(trials)
    write_csv(out_csv, trials)
    plot_summary(out_png, summary, f"{fidelity} mean time  n={n_trials}")
    return summary


def print_summary(label: str, summary: dict[str, dict]) -> None:
    print(f"\n== {label} ==")
    for name in POLICIES:
        row = summary[name]
        print(
            f"{name:8s}  clear={row['clear_rate']*100:5.1f}%  "
            f"time={row['time_mean']:7.1f}s  walk={row['walk_mean']:7.0f}m  "
            f"replan={row['replan_mean']:5.2f}  regret={row['regret_mean']:7.1f}s  "
            f"win={row['win_rate']*100:5.1f}%"
        )
    kinds = summary["limited"]["kinds"]
    print("limited census kinds:", {key: round(val, 3) for key, val in kinds.items() if val})


def main() -> None:
    parser = argparse.ArgumentParser(description="Problem 3 fusion policy Monte Carlo")
    parser.add_argument("--fast", type=int, default=100_000)
    parser.add_argument("--high", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "_figs",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fast_summary = None
    high_summary = None
    if args.fast > 0:
        fast_summary = run_layer(
            args.fast,
            args.seed,
            "fast",
            args.out_dir / "problem3_fusion_fast_100k.csv",
            args.out_dir / "problem3_fusion_fast.png",
        )
        print_summary("fast", fast_summary)
    if args.high > 0:
        high_summary = run_layer(
            args.high,
            args.seed + 1,
            "high",
            args.out_dir / "problem3_fusion_high_fidelity.csv",
            args.out_dir / "problem3_fusion_high.png",
        )
        print_summary("high", high_summary)

    note_path = args.out_dir.parent / "问题3-融合路线实验说明.md"
    lines = [
        "# 问题 3 融合路线本地实验",
        "",
        "全部为本地随机模拟，不连接官方模拟器、不占用正式测试次数。",
        "",
        f"- 随机种子：fast `{args.seed}`，high `{args.seed + 1}`",
        f"- 快速层：{args.fast} 次（两站交会近似直径）",
        f"- 高保真层：{args.high} 次（问题 1 半平面求交）",
        "",
        "三种策略：`strict` 先补测后清除；`full` 每步重规划；`limited` 有限触发融合（主方案）。",
        "",
    ]
    if fast_summary:
        lines.append("## 快速层")
        for name in POLICIES:
            row = fast_summary[name]
            lines.append(
                f"- {name}: 清完 {row['clear_rate']*100:.1f}%，均时 {row['time_mean']:.1f}s，"
                f"均走 {row['walk_mean']:.0f}m，重规划 {row['replan_mean']:.2f}，胜率 {row['win_rate']*100:.1f}%"
            )
        lines.append("")
    if high_summary:
        lines.append("## 高保真层")
        for name in POLICIES:
            row = high_summary[name]
            lines.append(
                f"- {name}: 清完 {row['clear_rate']*100:.1f}%，均时 {row['time_mean']:.1f}s，"
                f"均走 {row['walk_mean']:.0f}m，重规划 {row['replan_mean']:.2f}，胜率 {row['win_rate']*100:.1f}%"
            )
        lines.append("")
        kinds = high_summary["limited"]["kinds"]
        lines.append("7 点普查后（limited，非空频道均值）：")
        for key in KIND_KEYS:
            if kinds[key]:
                lines.append(f"- {key}: {kinds[key]:.3f}")
        lines.append("")
    note_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {note_path}")


if __name__ == "__main__":
    main()
