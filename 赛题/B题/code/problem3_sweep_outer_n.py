#!/usr/bin/env python3
"""Sweep outer-ring size n = 6..100, each with 100000 paired source layouts.

Geometry matches the playground screenshot: regular n-gon, radius 1000 m,
no extra origin stop. The dog still starts at the origin (required) and walks
origin -> P1 -> ... -> Pn, then an open tour through the true source points.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ARENA = 1800.0
LISTEN = 1000.0
RHO = 1000.0
SPEED = 5.0
CHANNELS = 20
T_MEASURE = 5.0
T_SWITCH = 1.0
T_CLEAR = 5.0
T_STOP = CHANNELS * T_MEASURE + (CHANNELS - 1) * T_SWITCH
N_TRIALS = 100_000
SEED = 2026
N_MIN, N_MAX = 6, 100
N_LIST = np.arange(N_MIN, N_MAX + 1, dtype=np.int32)
N_CPU = max(1, (os.cpu_count() or 2) - 1)


def ring_points(n: int, rho: float = RHO) -> np.ndarray:
    ang = np.arange(n) * (2.0 * math.pi / n)
    return np.stack([rho * np.cos(ang), rho * np.sin(ang)], axis=1)


def covering_walk(n: int, rho: float = RHO) -> float:
    chord = 2.0 * rho * math.sin(math.pi / n)
    return rho + (n - 1) * chord


def last_point(n: int, rho: float = RHO) -> np.ndarray:
    a = 2.0 * math.pi * (n - 1) / n
    return np.array([rho * math.cos(a), rho * math.sin(a)])


def arena_samples(step: float = 8.0) -> np.ndarray:
    chunks = []
    r = 0.0
    while r <= ARENA + 1e-9:
        n_ang = 1 if r == 0 else max(16, int(round(2 * math.pi * r / step)))
        a = np.linspace(0.0, 2 * math.pi, n_ang, endpoint=False)
        chunks.append(np.stack([r * np.cos(a), r * np.sin(a)], axis=1))
        r += step
    return np.vstack(chunks)


def uncovered_count(stops: np.ndarray, samples: np.ndarray) -> tuple[int, float]:
    dmin = np.full(len(samples), np.inf)
    for sx, sy in stops:
        dmin = np.minimum(dmin, np.hypot(samples[:, 0] - sx, samples[:, 1] - sy))
    bad = dmin > LISTEN + 0.5
    if not np.any(bad):
        return 0, 0.0
    return int(bad.sum()), float(dmin[bad].max() - LISTEN)


def sample_sources(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.random(n)
    a = rng.random(n) * (2 * math.pi)
    r = ARENA * np.sqrt(u)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def open_tsp_from_start(d0: np.ndarray, dcc: np.ndarray) -> float:
    n = d0.shape[0]
    unused = np.ones(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int32)
    cur = d0
    for k in range(n):
        j = int(np.argmin(np.where(unused, cur, np.inf)))
        unused[j] = False
        order[k] = j
        cur = dcc[j]
    improved = True
    passes = 0
    while improved and passes < 20:
        improved = False
        passes += 1
        for i in range(n - 1):
            a_d = d0 if i == 0 else dcc[order[i - 1]]
            oi = order[i]
            for j in range(i + 1, n):
                oj = order[j]
                nxt = order[j + 1] if j + 1 < n else -1
                before = a_d[oi] + (0.0 if nxt < 0 else dcc[oj, nxt])
                after = a_d[oj] + (0.0 if nxt < 0 else dcc[oi, nxt])
                if after + 1e-9 < before:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    oi = order[i]
                    improved = True
    s = d0[order[0]]
    for i in range(n - 1):
        s += dcc[order[i], order[i + 1]]
    return float(s)


def trial_chunk(args: tuple[int, int]) -> dict:
    start_i, end_i = args
    m = len(N_LIST)
    starts = np.stack([last_point(int(n)) for n in N_LIST])
    walk = np.array([covering_walk(int(n)) for n in N_LIST])
    meas = N_LIST.astype(np.float64) * T_STOP
    sum_t = np.zeros(m)
    sum_tsp = np.zeros(m)
    win = np.zeros(m, dtype=np.int64)
    n_src_sum = 0
    for i in range(start_i, end_i):
        rng = np.random.default_rng((SEED, i))
        ns = int(rng.integers(10, 17))
        src = sample_sources(rng, ns)
        n_src_sum += ns
        dcc = np.sqrt(((src[:, None, :] - src[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dcc, 0.0)
        tsp = np.empty(m)
        for k in range(m):
            d0 = np.hypot(src[:, 0] - starts[k, 0], src[:, 1] - starts[k, 1])
            tsp[k] = open_tsp_from_start(d0, dcc)
        total = walk / SPEED + meas + tsp / SPEED + ns * T_CLEAR
        sum_t += total
        sum_tsp += tsp
        win[int(np.argmin(total))] += 1
    return {
        "n_trials": end_i - start_i,
        "sum_t": sum_t,
        "sum_tsp": sum_tsp,
        "win": win,
        "n_src_sum": n_src_sum,
    }


def main() -> None:
    t0 = time.perf_counter()
    out = Path(__file__).resolve().parent.parent / "_figs"
    out.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mpl"))

    samples = arena_samples(8.0)
    covers = []
    holes = []
    slack = []
    for n in N_LIST:
        bad, worst = uncovered_count(ring_points(int(n)), samples)
        covers.append(bad == 0)
        holes.append(bad)
        slack.append(worst)
    covers = np.array(covers, dtype=np.bool_)
    holes = np.array(holes)
    slack = np.array(slack)
    walk = np.array([covering_walk(int(n)) for n in N_LIST])
    meas = N_LIST.astype(np.float64) * T_STOP

    print("=== covering at rho=1000 m, no origin stop ===")
    print(f"arena samples {len(samples)}")
    for n, c, h, s, w in zip(N_LIST, covers, holes, slack, walk):
        if int(n) in {6, 7, 8, 9, 10, 12, 16, 20, 30, 50, 100} or not c:
            mark = "COVER" if c else f"HOLE {h} slack={s:.2f}m"
            print(f"  n={int(n):3d}  walk={w:7.1f} m  meas={n * T_STOP:7.0f} s  {mark}")

    chunks = []
    step = math.ceil(N_TRIALS / N_CPU)
    for a in range(0, N_TRIALS, step):
        chunks.append((a, min(N_TRIALS, a + step)))
    print(f"=== {N_TRIALS} trials x n={N_MIN}..{N_MAX}  workers={len(chunks)} ===")

    m = len(N_LIST)
    sum_t = np.zeros(m)
    sum_tsp = np.zeros(m)
    win = np.zeros(m, dtype=np.int64)
    n_src_sum = 0
    n_done = 0
    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        for part in ex.map(trial_chunk, chunks):
            sum_t += part["sum_t"]
            sum_tsp += part["sum_tsp"]
            win += part["win"]
            n_src_sum += part["n_src_sum"]
            n_done += part["n_trials"]
            print(f"  ... {n_done} trials  elapsed {time.perf_counter() - t0:.1f}s")

    mean_t = sum_t / N_TRIALS
    mean_tsp = sum_tsp / N_TRIALS
    win_pct = 100.0 * win / N_TRIALS
    mean_src = n_src_sum / N_TRIALS
    best_all = int(N_LIST[int(np.argmin(mean_t))])
    cover_idx = np.where(covers)[0]
    best_cover = int(N_LIST[cover_idx[int(np.argmin(mean_t[cover_idx]))]])

    csv_path = out / "problem3_outer_n_sweep.csv"
    with csv_path.open("w") as f:
        f.write("n,covers,hole_samples,worst_slack_m,walk_m,measure_s,mean_tsp_m,mean_total_s,win_pct\n")
        for i, n in enumerate(N_LIST):
            f.write(
                f"{int(n)},{int(covers[i])},{int(holes[i])},{slack[i]:.4f},"
                f"{walk[i]:.4f},{meas[i]:.4f},{mean_tsp[i]:.4f},{mean_t[i]:.4f},{win_pct[i]:.4f}\n"
            )

    np.savez(
        out / "problem3_outer_n_sweep.npz",
        n_list=N_LIST,
        covers=covers,
        holes=holes,
        slack=slack,
        walk=walk,
        meas=meas,
        mean_t=mean_t,
        mean_tsp=mean_tsp,
        win_pct=win_pct,
        mean_src=np.array([mean_src]),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(N_LIST[~covers], mean_t[~covers], "o", color="#b91c1c", ms=7, label="n=6, does not cover")
    ax.plot(N_LIST[covers], mean_t[covers], "-", color="#1d4ed8", lw=2.2, label="covers (n>=7)")
    ax.plot(best_cover, mean_t[best_cover - N_MIN], "o", color="#15803d", ms=9, zorder=5)
    ax.axvline(best_cover, color="#15803d", ls="--", lw=1, alpha=0.7, label=f"shortest covering n={best_cover}")
    ax.set_xlabel("outer-ring stops n")
    ax.set_ylabel("mean total virtual time (s)")
    ax.set_title("Regular n-gon at 1000 m, 100000 paired layouts")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = out / "problem3_outer_n_sweep.png"
    fig.savefig(png, dpi=140)
    plt.close()

    print("=== result ===")
    print(f"mean sources {mean_src:.3f}")
    print(f"shortest mean time among all n: n={best_all}  {mean_t.min():.2f} s")
    print(f"shortest mean time among covering n: n={best_cover}  {mean_t[best_cover - N_MIN]:.2f} s")
    print(f"n=7 (screenshot) {mean_t[7 - N_MIN]:.2f} s  walk={walk[7 - N_MIN]:.1f} m")
    print(f"saved {csv_path}")
    print(f"saved {png}")
    print(f"elapsed {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
