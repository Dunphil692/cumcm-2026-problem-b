#!/usr/bin/env python3
"""Local Monte-Carlo for the problem-4 policy on mixed omni/directional worlds.

``--policy q4`` runs ``problem4_belief.run_belief_q4``; ``--policy q3`` runs
the frozen problem-3 policy on the *same* worlds (paper baseline: it
mistakes a directional source's silent back side for absence).
Never talks to the official simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from problem4_world import generate_mixed_instance

SEED = 2027
FIGS = Path(__file__).resolve().parents[1] / "_figs"


def _trial_seed(master_seed: int, index: int) -> int:
    return (master_seed + 100003 * (index + 1)) % (2**31 - 1)


def run_one(index: int, master_seed: int, p_dir: float, policy: str, opts: dict) -> dict:
    rng = random.Random(_trial_seed(master_seed, index))
    world = generate_mixed_instance(rng, p_dir=p_dir)
    t0 = time.perf_counter()
    if policy == "q3":
        from problem3_belief import run_belief

        stats = run_belief(world)
        stats["n_dir"] = world.n_directional()
        stats["dir_cleared"] = sum(
            1 for s in world.sources if getattr(s, "phi", None) is not None and s.cleared
        )
    else:
        from problem4_belief import run_belief_q4

        stats = run_belief_q4(
            world,
            eps_mass=opts["eps"],
            merge_stations=opts["merge"],
            cover_listen=opts["cover_listen"],
        )
        stats.pop("trace", None)
    left = [
        {
            "channel": s.channel,
            "x": round(s.x, 1),
            "y": round(s.y, 1),
            "r": round(s.r, 1),
            "phi_deg": None if getattr(s, "phi", None) is None else round(s.phi * 57.29577951, 1),
            "rho": round((s.x**2 + s.y**2) ** 0.5, 1),
        }
        for s in world.sources
        if not s.cleared
    ]
    return {
        "trial": index,
        "stats": stats,
        "left": left,
        "wall_s": time.perf_counter() - t0,
    }


def _worker(payload: tuple) -> dict:
    return run_one(*payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--p-dir", type=float, default=0.5)
    parser.add_argument("--policy", choices=("q4", "q3"), default="q4")
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--cover-listen", choices=("hear", "kill"), default="hear")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    tag = args.tag or (
        f"q4_{args.policy}_p{int(round(args.p_dir * 100)):02d}"
        + ("" if args.policy == "q3" else f"_eps{args.eps:g}_{'merge' if not args.no_merge else 'nomerge'}_{args.cover_listen}")
    )
    out = args.out_dir or (FIGS / tag)
    out.mkdir(parents=True, exist_ok=True)
    workers = args.workers
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    opts = {"eps": args.eps, "merge": not args.no_merge, "cover_listen": args.cover_listen}
    payloads = [(index, args.seed, args.p_dir, args.policy, opts) for index in range(args.n)]

    started = time.time()
    print(
        f"{tag}  n={args.n} seed={args.seed} p_dir={args.p_dir} policy={args.policy} "
        f"opts={opts} workers={workers}",
        flush=True,
    )
    rows: list[dict] = []
    mark = max(1, args.n // 10)
    pool = None
    if workers <= 1:
        iterator = (_worker(payload) for payload in payloads)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_worker, payloads, chunksize=1)
    try:
        for row in iterator:
            rows.append(row)
            if len(rows) % mark == 0 or len(rows) == args.n:
                stats = row["stats"]
                print(
                    f"{len(rows)}/{args.n} wall={time.time()-started:.0f}s "
                    f"t={stats['time_s']:.0f}s rem={stats['remaining']} "
                    f"stop={stats.get('stop_reason')}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    times = [r["stats"]["time_s"] for r in rows]
    walks = [r["stats"]["walk_m"] for r in rows]
    measures = [r["stats"]["measures"] for r in rows]
    ok = [r["stats"]["cleared"] == r["stats"]["n_sources"] for r in rows]
    ok_times = [t for t, flag in zip(times, ok) if flag]
    n_src = sum(r["stats"]["n_sources"] for r in rows)
    n_dir = sum(r["stats"].get("n_dir", 0) for r in rows)
    dir_cleared = sum(r["stats"].get("dir_cleared", 0) for r in rows)
    cleared = sum(r["stats"]["cleared"] for r in rows)
    reasons: dict[str, int] = {}
    for r in rows:
        key = r["stats"].get("stop_reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    walls = [r["wall_s"] for r in rows]
    summary = {
        "tag": tag,
        "n": args.n,
        "seed": args.seed,
        "p_dir": args.p_dir,
        "policy": args.policy,
        "opts": opts,
        "official_simulator": False,
        "stop_uses_oracle_remaining": False,
        "wall_s": time.time() - started,
        "wall_per_run_s": statistics.fmean(walls),
        "clear_rate_worlds": sum(ok) / len(rows),
        "n_not_cleared": int(sum(1 for f in ok if not f)),
        "source_clear_rate": cleared / n_src,
        "dir_source_clear_rate": (dir_cleared / n_dir) if n_dir else None,
        "omni_source_clear_rate": ((cleared - dir_cleared) / (n_src - n_dir)) if n_src > n_dir else None,
        "n_sources_total": n_src,
        "n_dir_total": n_dir,
        "stop_reason_counts": reasons,
        "time_mean": statistics.fmean(times),
        "time_p50": statistics.median(times),
        "time_p25": statistics.quantiles(times, n=4)[0] if len(times) > 3 else None,
        "time_p75": statistics.quantiles(times, n=4)[2] if len(times) > 3 else None,
        "time_min": min(times),
        "time_max": max(times),
        "time_mean_cleared": statistics.fmean(ok_times) if ok_times else None,
        "walk_mean": statistics.fmean(walks),
        "measure_mean": statistics.fmean(measures),
        "cover_mean": statistics.fmean([r["stats"].get("cover_visits") or 0 for r in rows]),
        "station_silent_mean": statistics.fmean([r["stats"].get("station_silent") or 0 for r in rows]),
        "probe_silent_mean": statistics.fmean([r["stats"].get("probe_silent") or 0 for r in rows]),
        "final_mean_mass_mean": statistics.fmean([r["stats"].get("final_mean_mass") or 0.0 for r in rows]),
    }
    fields = [
        "trial", "n_sources", "n_dir", "cleared", "dir_cleared", "remaining", "time_s",
        "walk_m", "measures", "replans", "steps", "cover_visits", "stop_reason",
        "final_mean_mass", "station_silent", "probe_silent", "wall_s", "left",
    ]
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            s = r["stats"]
            writer.writerow(
                {
                    "trial": r["trial"],
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
                    "wall_s": f"{r['wall_s']:.3f}",
                    "left": json.dumps(r["left"], ensure_ascii=False) if r["left"] else "",
                }
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\n== {tag} ==\nworlds cleared {summary['clear_rate_worlds']*100:.2f}%  "
        f"sources {summary['source_clear_rate']*100:.2f}%  dir {summary['dir_source_clear_rate']}  "
        f"time mean {summary['time_mean']:.0f} p50 {summary['time_p50']:.0f} "
        f"p25 {summary['time_p25']} p75 {summary['time_p75']} max {summary['time_max']:.0f}  "
        f"walk {summary['walk_mean']:.0f}  measures {summary['measure_mean']:.0f}  "
        f"stop {reasons}  wall {summary['wall_s']:.0f}s ({summary['wall_per_run_s']:.1f}s/run)"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
