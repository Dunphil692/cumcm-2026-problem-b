#!/usr/bin/env python3
"""Gate for v14 orient_cert. Official-like remaining()=None."""

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
MASTER = 4242


def trial_seed(index: int, master: int = MASTER) -> int:
    return (master + 100003 * (index + 1)) % (2**31 - 1)


def hide_remaining(world):
    world.remaining = lambda: None  # type: ignore[method-assign]
    return world


def slim(res: dict) -> dict:
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
        "missed_dir": res["missed_dir"],
    }


def run_one(index: int, p_dir: float, orient: bool, hide: bool, master: int = MASTER):
    rng = random.Random(trial_seed(index, master))
    world = generate_instance(rng, p_dir=p_dir)
    if hide:
        hide_remaining(world)
    res = run_belief(
        world,
        fidelity="high",
        optical_cover=True,
        orient_cert=orient,
        **COMBO,
    )
    row = slim(res)
    row["ok"] = row["cleared"] == row["n_sources"]
    return row


def _worker(payload):
    return run_one(*payload)


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "clear": sum(1 for r in rows if r["ok"]),
        "missed_dir": sum(r["missed_dir"] for r in rows),
        "time_mean": mean([r["time_s"] for r in rows]),
        "time_p50": statistics.median([r["time_s"] for r in rows]),
        "walk_mean": mean([r["walk_m"] for r in rows]),
        "measures_mean": mean([r["measures"] for r in rows]),
        "outer_mean": mean([r["outer_visits"] for r in rows]),
        "inner_mean": mean([r["inner_visits"] for r in rows]),
        "stop": {k: sum(1 for r in rows if r["stop_reason"] == k) for k in
                 {r["stop_reason"] for r in rows}},
    }


def run_batch(n: int, p_dir: float, orient: bool, hide: bool, workers: int, master: int):
    payloads = [(i, p_dir, orient, hide, master) for i in range(n)]
    rows = []
    t0 = time.time()
    label = f"pdir={p_dir} orient={orient} hide={hide} n={n}"
    print(label, flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_worker, payloads, chunksize=2), 1):
            rows.append(row)
            if i % 10 == 0 or i == n:
                print(f"  {i}/{n} wall={time.time()-t0:.1f}s clear={sum(1 for r in rows if r['ok'])}", flush=True)
    return rows


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out: dict = {"workers": workers}
    # Equivalence-ish: flag off, remaining hidden, should still clear and walk outer
    off_rows = run_batch(12, 0.5, False, True, workers, 4242)
    out["off_hide_p50_n12"] = summarize(off_rows)
    # Paired 20: hide remaining, on vs off
    paired = []
    print("paired n=20 hide remaining p_dir=0.5", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = []
        for i in range(20):
            jobs.append((i, 0.5, False, True, 2027))
            jobs.append((i, 0.5, True, True, 2027))
        results = list(pool.map(_worker, jobs, chunksize=1))
    for i in range(20):
        off, on = results[2 * i], results[2 * i + 1]
        paired.append({
            "off": off, "on": on,
            "dT": on["time_s"] - off["time_s"],
            "dWalk": on["walk_m"] - off["walk_m"],
        })
    out["paired20"] = {
        "clear_off": sum(1 for p in paired if p["off"]["ok"]),
        "clear_on": sum(1 for p in paired if p["on"]["ok"]),
        "time_off": mean([p["off"]["time_s"] for p in paired]),
        "time_on": mean([p["on"]["time_s"] for p in paired]),
        "delta_time": mean([p["dT"] for p in paired]),
        "delta_walk": mean([p["dWalk"] for p in paired]),
        "outer_off": mean([p["off"]["outer_visits"] for p in paired]),
        "outer_on": mean([p["on"]["outer_visits"] for p in paired]),
        "wall_s": time.time() - t0,
    }
    print(json.dumps(out["paired20"], indent=2), flush=True)
    # Main gate
    on50 = run_batch(100, 0.5, True, True, workers, 9797)
    out["on_hide_p50_n100"] = summarize(on50)
    on100 = run_batch(50, 1.0, True, True, workers, 8888)
    out["on_hide_p100_n50"] = summarize(on100)
    dest = Path(__file__).with_name("q4_orient_cert_gate.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()}, indent=2))
    print("wrote", dest)
    fail = []
    if out["paired20"]["clear_on"] < 20:
        fail.append("paired on miss")
    if out["paired20"]["clear_off"] < 20:
        fail.append("paired off miss")
    if out["on_hide_p50_n100"]["clear"] < 100:
        fail.append("p50 n100 miss")
    if out["on_hide_p100_n50"]["clear"] < 50:
        fail.append("p100 n50 miss")
    if fail:
        raise SystemExit("GATE FAIL: " + ", ".join(fail))
    print("GATE PASS")


if __name__ == "__main__":
    main()
