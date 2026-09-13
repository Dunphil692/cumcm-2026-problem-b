#!/usr/bin/env python3
"""Paired remaining-stop vs certificate-stop for fielded v14. Local only."""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "代码" / "问题4-上场"))

from problem4_belief_v14 import run_belief
from problem4_simulation_model import generate_instance

COMBO = {"order_mode": "optimal", "cover_mode": "polygon", "enable_k4": True}
MASTER = 2027


def trial_seed(index: int) -> int:
    return (MASTER + 100003 * (index + 1)) % (2**31 - 1)


def hide_remaining(world):
    world.remaining = lambda: None  # type: ignore[method-assign]
    return world


def slim(res: dict) -> dict:
    walk_kind = res.get("walk_kind") or {}
    return {
        "cleared": res["cleared"],
        "n_sources": res["n_sources"],
        "n_dir": res["n_dir"],
        "time_s": res["time_s"],
        "walk_m": res["walk_m"],
        "measures": res["measures"],
        "clear_fail": res["clear_fail"],
        "stop_reason": res["stop_reason"],
        "inner_visits": res["inner_visits"],
        "outer_visits": res["outer_visits"],
        "outer_triggered": res["outer_triggered"],
        "optical_fails_total": res.get("optical_fails_total", 0),
        "walk_cover": walk_kind.get("cover", 0.0),
        "walk_outer": walk_kind.get("outer", 0.0),
        "walk_station": walk_kind.get("station", 0.0),
        "walk_homing": walk_kind.get("homing", 0.0),
        "walk_optical": walk_kind.get("optical", 0.0),
        "walk_clear": walk_kind.get("clear", 0.0),
        "walk_probe": walk_kind.get("probe", 0.0),
    }


def run_pair(index: int) -> dict:
    rng = random.Random(trial_seed(index))
    world = generate_instance(rng, p_dir=0.5)
    rem = run_belief(world.clone(), fidelity="high", optical_cover=True, **COMBO)
    cert = run_belief(
        hide_remaining(world.clone()),
        fidelity="high",
        optical_cover=True,
        **COMBO,
    )
    return {"trial": index, "remaining": slim(rem), "certificate": slim(cert)}


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    t0 = time.time()
    rows = []
    print(f"q4 cert-vs-remaining n={n} workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(run_pair, range(n)), 1):
            rows.append(row)
            if i % 5 == 0 or i == n:
                print(f"{i}/{n} wall={time.time()-t0:.1f}s", flush=True)

    def col(mode, key):
        return [r[mode][key] for r in rows]

    deltas = [r["certificate"]["time_s"] - r["remaining"]["time_s"] for r in rows]
    walk_d = [r["certificate"]["walk_m"] - r["remaining"]["walk_m"] for r in rows]
    meas_d = [r["certificate"]["measures"] - r["remaining"]["measures"] for r in rows]
    out = {
        "n": n,
        "seed": MASTER,
        "p_dir": 0.5,
        "clear_remaining": sum(
            1 for r in rows if r["remaining"]["cleared"] == r["remaining"]["n_sources"]
        ),
        "clear_certificate": sum(
            1
            for r in rows
            if r["certificate"]["cleared"] == r["certificate"]["n_sources"]
        ),
        "time_remaining": mean(col("remaining", "time_s")),
        "time_certificate": mean(col("certificate", "time_s")),
        "delta_time_mean": mean(deltas),
        "delta_time_p50": statistics.median(deltas),
        "delta_walk_mean": mean(walk_d),
        "delta_measures_mean": mean(meas_d),
        "outer_visits_remaining": mean(col("remaining", "outer_visits")),
        "outer_visits_certificate": mean(col("certificate", "outer_visits")),
        "outer_trig_remaining": mean(
            [1.0 if r["remaining"]["outer_triggered"] else 0.0 for r in rows]
        ),
        "outer_trig_certificate": mean(
            [1.0 if r["certificate"]["outer_triggered"] else 0.0 for r in rows]
        ),
        "inner_remaining": mean(col("remaining", "inner_visits")),
        "inner_certificate": mean(col("certificate", "inner_visits")),
        "clear_fail_remaining": mean(col("remaining", "clear_fail")),
        "clear_fail_certificate": mean(col("certificate", "clear_fail")),
        "walk_kind_certificate": {
            k: mean(col("certificate", k))
            for k in (
                "walk_cover",
                "walk_outer",
                "walk_station",
                "walk_homing",
                "walk_optical",
                "walk_clear",
                "walk_probe",
            )
        },
        "walk_kind_remaining": {
            k: mean(col("remaining", k))
            for k in (
                "walk_cover",
                "walk_outer",
                "walk_station",
                "walk_homing",
                "walk_optical",
                "walk_clear",
                "walk_probe",
            )
        },
        "n_sources_mean": mean(col("remaining", "n_sources")),
        "n_lt_16": sum(1 for r in rows if r["remaining"]["n_sources"] < 16),
        "rows": rows,
    }
    dest = Path(__file__).with_name("q4_cert_vs_remaining_n40.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
