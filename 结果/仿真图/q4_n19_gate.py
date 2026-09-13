#!/usr/bin/env python3
"""Local gate for fielded N19. Does not use remaining() to stop."""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "代码" / "问题4-上场"))

from problem4_belief_n19 import run_belief
from problem4_simulation_model import generate_instance

MASTER = 9797


def trial_seed(index: int, master: int = MASTER) -> int:
    return (master + 100003 * (index + 1)) % (2**31 - 1)


def run_one(index: int, p_dir: float, master: int = MASTER) -> dict:
    rng = random.Random(trial_seed(index, master))
    world = generate_instance(rng, p_dir=p_dir)
    world.remaining = lambda: None  # type: ignore[method-assign]
    res = run_belief(world, fidelity="high")
    return {
        "ok": res["cleared"] == res["n_sources"],
        "cleared": res["cleared"],
        "n_sources": res["n_sources"],
        "time_s": res["time_s"],
        "walk_m": res["walk_m"],
        "measures": res["measures"],
        "cover_visits": res["cover_visits"],
        "stop_reason": res["stop_reason"],
        "missed_dir": res.get("missed_dir", 0),
        "heard": res.get("heard"),
    }


def _worker(payload):
    return run_one(*payload)


def summarize(rows):
    return {
        "n": len(rows),
        "clear": sum(1 for r in rows if r["ok"]),
        "time_mean": statistics.fmean(r["time_s"] for r in rows),
        "time_p50": statistics.median(r["time_s"] for r in rows),
        "walk_mean": statistics.fmean(r["walk_m"] for r in rows),
        "measures_mean": statistics.fmean(r["measures"] for r in rows),
        "cover_mean": statistics.fmean(r["cover_visits"] for r in rows),
        "stop": {
            key: sum(1 for r in rows if r["stop_reason"] == key)
            for key in {r["stop_reason"] for r in rows}
        },
    }


def run_batch(n, p_dir, workers, master):
    print(f"N19 p_dir={p_dir} n={n}", flush=True)
    t0 = time.time()
    rows = []
    payloads = [(i, p_dir, master) for i in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_worker, payloads, chunksize=2), 1):
            rows.append(row)
            if i % 10 == 0 or i == n:
                print(
                    f"  {i}/{n} wall={time.time()-t0:.1f}s "
                    f"clear={sum(1 for r in rows if r['ok'])}",
                    flush=True,
                )
    return rows


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = {}
    rows50 = run_batch(40, 0.5, workers, 9797)
    out["p50_n40"] = summarize(rows50)
    rows100 = run_batch(20, 1.0, workers, 8888)
    out["p100_n20"] = summarize(rows100)
    dest = Path(__file__).with_name("q4_n19_gate.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("wrote", dest)
    fail = []
    if out["p50_n40"]["clear"] < 40:
        fail.append("p50 miss")
    if out["p100_n20"]["clear"] < 20:
        fail.append("p100 miss")
    if fail:
        raise SystemExit("GATE FAIL: " + ", ".join(fail))
    print("GATE PASS")


if __name__ == "__main__":
    main()
