#!/usr/bin/env python3
"""100 local high-fidelity trials of 队员's find-all method.

7-point census -> Held-Karp + TSPN (`limited`: replan only after
supplementary stations / clear fail / near). Never talks to the official
simulator.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import time
from pathlib import Path

from problem3_fusion_policies import run_policy
from problem3_simulation_model import generate_instance

SEED = 2027
N = 100
POLICY = "limited"
FIDELITY = "high"
OUT = Path(__file__).resolve().parents[1] / "_figs" / "q3_findall_100"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows = []
    started = time.time()
    for index in range(N):
        world = generate_instance(rng)
        stats = run_policy(world, POLICY, fidelity=FIDELITY)
        rows.append(stats)
        if (index + 1) % 10 == 0 or index == 0:
            elapsed = time.time() - started
            print(
                f"{index + 1}/{N}  wall={elapsed:.1f}s  "
                f"time={stats['time_s']:.0f}s  "
                f"cleared={stats['cleared']}/{stats['n_sources']}  "
                f"remain={stats['remaining']}  "
                f"census={stats['census_time_s']:.0f}s",
                flush=True,
            )

    times = [row["time_s"] for row in rows]
    walks = [row["walk_m"] for row in rows]
    censuses = [row["census_time_s"] for row in rows]
    second = [row["time_s"] - row["census_time_s"] for row in rows]
    cleared_all = [row["cleared"] == row["n_sources"] for row in rows]
    replans = [row["replans"] for row in rows]
    measures = [row["measures"] for row in rows]
    n_sources = [row["n_sources"] for row in rows]
    remaining = [row["remaining"] for row in rows]

    summary = {
        "policy": POLICY,
        "fidelity": FIDELITY,
        "n": N,
        "seed": SEED,
        "official_simulator": False,
        "wall_s": time.time() - started,
        "clear_rate": sum(cleared_all) / N,
        "n_not_cleared": int(sum(1 for ok in cleared_all if not ok)),
        "time_mean": statistics.fmean(times),
        "time_p50": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "time_stdev": statistics.stdev(times),
        "census_mean": statistics.fmean(censuses),
        "second_stage_mean": statistics.fmean(second),
        "walk_mean": statistics.fmean(walks),
        "replan_mean": statistics.fmean(replans),
        "measure_mean": statistics.fmean(measures),
        "n_sources_mean": statistics.fmean(n_sources),
        "remaining_mean": statistics.fmean(remaining),
    }

    csv_path = OUT / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial",
                "n_sources",
                "cleared",
                "remaining",
                "time_s",
                "census_time_s",
                "second_stage_s",
                "walk_m",
                "measures",
                "replans",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "trial": index,
                    "n_sources": row["n_sources"],
                    "cleared": row["cleared"],
                    "remaining": row["remaining"],
                    "time_s": f"{row['time_s']:.4f}",
                    "census_time_s": f"{row['census_time_s']:.4f}",
                    "second_stage_s": f"{row['time_s'] - row['census_time_s']:.4f}",
                    "walk_m": f"{row['walk_m']:.4f}",
                    "measures": row["measures"],
                    "replans": row["replans"],
                }
            )

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n== 队员 find-all limited high 100 ==")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.3f}")
        else:
            print(f"{key}={value}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
