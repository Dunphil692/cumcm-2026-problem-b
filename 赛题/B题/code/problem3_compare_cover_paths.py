#!/usr/bin/env python3
"""Compare two covering-path schemes over 100000 random source layouts.

Scheme 1: do not measure at the origin. Walk P1..P6 (regular hexagon, 1200 m),
then one extra stop inward from P6 whose 1000 m disk just covers the centre hole.

Scheme 2: measure at the origin, then walk the outer hexagon at 1740 m (P2..P7).

Other costs are identical: 7 stops, 20 channels at each stop, then an open
shortest tour from the last covering point through the true source positions.
This isolates walking, which is what the two schemes actually change.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

ARENA = 1800.0
LISTEN = 1000.0
SPEED = 5.0
N_STOPS = 7
CHANNELS = 20
T_MEASURE = 5.0
T_SWITCH = 1.0
T_CLEAR = 5.0
T_STOP = CHANNELS * T_MEASURE + (CHANNELS - 1) * T_SWITCH  # 119 s
N_TRIALS = 100_000
SEED = 2026
RHO1 = 1200.0
RHO2 = 1740.0


def hypot(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def regular_hex(rho: float) -> np.ndarray:
    ang = np.arange(6) * (math.pi / 3.0)
    return np.stack([rho * np.cos(ang), rho * np.sin(ang)], axis=1)


def path_len(points: np.ndarray) -> float:
    d = 0.0
    for i in range(len(points) - 1):
        d += hypot(points[i + 1, 0] - points[i, 0], points[i + 1, 1] - points[i, 1])
    return d


def hole_samples(hex_pts: np.ndarray, step: float = 10.0) -> np.ndarray:
    pts = []
    r = 0.0
    while r <= ARENA + 1e-9:
        n_ang = 1 if r == 0 else max(16, int(round(2 * math.pi * r / step)))
        for i in range(n_ang):
            a = 2 * math.pi * i / n_ang
            x, y = r * math.cos(a), r * math.sin(a)
            dmin = min(hypot(x - px, y - py) for px, py in hex_pts)
            if dmin > LISTEN + 1e-6:
                pts.append((x, y))
        r += step
    return np.asarray(pts, dtype=np.float64)


def covers_hole(p: np.ndarray, hole: np.ndarray) -> bool:
    dx = hole[:, 0] - p[0]
    dy = hole[:, 1] - p[1]
    return bool(np.max(np.hypot(dx, dy)) <= LISTEN + 1e-6)


def extra_point_from_p6(hex_pts: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Walk from P6 toward the origin; stop at the farthest point that still covers the hole."""
    p6 = hex_pts[5]
    lo, hi = 0.0, 1.0
    if not covers_hole(np.zeros(2), hole):
        raise RuntimeError("origin itself does not cover the centre hole")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = p6 * mid
        if covers_hole(p, hole):
            lo = mid
        else:
            hi = mid
    p = p6 * lo
    if not covers_hole(p, hole):
        raise RuntimeError("failed to place extra covering point")
    return p


def arena_uncovered(stops: np.ndarray, step: float = 8.0) -> tuple[int, int, float]:
    total = 0
    bad = 0
    worst = 0.0
    r = 0.0
    while r <= ARENA + 1e-9:
        n_ang = 1 if r == 0 else max(16, int(round(2 * math.pi * r / step)))
        for i in range(n_ang):
            a = 2 * math.pi * i / n_ang
            x, y = r * math.cos(a), r * math.sin(a)
            dmin = min(hypot(x - px, y - py) for px, py in stops)
            total += 1
            if dmin > LISTEN + 0.5:
                bad += 1
                worst = max(worst, dmin - LISTEN)
        r += step
    return total, bad, worst


def open_tsp_len(start: np.ndarray, cities: np.ndarray) -> float:
    n = len(cities)
    if n == 0:
        return 0.0
    pts = np.empty((n + 1, 2), dtype=np.float64)
    pts[0] = start
    pts[1:] = cities
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, 0.0)

    unused = np.ones(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int32)
    cur = 0
    for k in range(n):
        dist = d[cur, 1:]
        dist = np.where(unused, dist, np.inf)
        j = int(np.argmin(dist))
        unused[j] = False
        order[k] = j + 1
        cur = j + 1

    def tour_len(ordr: np.ndarray) -> float:
        s = d[0, ordr[0]]
        for i in range(n - 1):
            s += d[ordr[i], ordr[i + 1]]
        return s

    improved = True
    passes = 0
    while improved and passes < 40:
        improved = False
        passes += 1
        for i in range(n - 1):
            a = 0 if i == 0 else int(order[i - 1])
            for j in range(i + 1, n):
                b = int(order[j])
                nxt = int(order[j + 1]) if j + 1 < n else None
                before = d[a, int(order[i])] + (0.0 if nxt is None else d[b, nxt])
                after = d[a, b] + (0.0 if nxt is None else d[int(order[i]), nxt])
                if after + 1e-9 < before:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    improved = True
    return tour_len(order)


def sample_sources(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.random(n)
    a = rng.random(n) * (2 * math.pi)
    r = ARENA * np.sqrt(u)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def main() -> None:
    t0 = time.perf_counter()
    hex1 = regular_hex(RHO1)
    hole = hole_samples(hex1, step=8.0)
    extra = extra_point_from_p6(hex1, hole)
    origin = np.zeros(2)

    stops1 = np.vstack([hex1, extra[None, :]])
    path1 = np.vstack([origin[None, :], hex1, extra[None, :]])

    hex2 = regular_hex(RHO2)
    stops2 = np.vstack([origin[None, :], hex2])
    path2 = stops2

    walk1 = path_len(path1)
    walk2 = path_len(path2)
    extra_walk = hypot(extra[0] - hex1[5, 0], extra[1] - hex1[5, 1])

    tot1, bad1, w1 = arena_uncovered(stops1)
    tot2, bad2, w2 = arena_uncovered(stops2)

    print("=== covering geometry ===")
    print(f"scheme1 extra point: ({extra[0]:.2f}, {extra[1]:.2f})")
    print(f"scheme1 extra walk from P6: {extra_walk:.2f} m")
    print(f"scheme1 covering walk (origin -> P1..P6 -> extra): {walk1:.2f} m")
    print(f"scheme2 covering walk (origin -> P2..P7): {walk2:.2f} m")
    print(f"scheme1 uncovered samples: {bad1}/{tot1} (worst slack {w1:.2f} m)")
    print(f"scheme2 uncovered samples: {bad2}/{tot2} (worst slack {w2:.2f} m)")
    print(f"centre-hole samples under hex1200: {len(hole)}")
    print(f"identical stop cost: {N_STOPS} * {T_STOP:.0f} s = {N_STOPS * T_STOP:.0f} s")

    rng = np.random.default_rng(SEED)
    t1 = np.empty(N_TRIALS)
    t2 = np.empty(N_TRIALS)
    n_src = np.empty(N_TRIALS, dtype=np.int32)
    tsp1 = np.empty(N_TRIALS)
    tsp2 = np.empty(N_TRIALS)

    meas = N_STOPS * T_STOP
    for i in range(N_TRIALS):
        n = int(rng.integers(10, 17))
        src = sample_sources(rng, n)
        n_src[i] = n
        L1 = open_tsp_len(path1[-1], src)
        L2 = open_tsp_len(path2[-1], src)
        tsp1[i] = L1
        tsp2[i] = L2
        laser = n * T_CLEAR
        t1[i] = walk1 / SPEED + meas + L1 / SPEED + laser
        t2[i] = walk2 / SPEED + meas + L2 / SPEED + laser
        if (i + 1) % 10000 == 0:
            print(f"  ... {i + 1} trials  elapsed {time.perf_counter() - t0:.1f}s")

    dt = t1 - t2
    print("=== 100000 paired trials ===")
    print(f"scheme1 time s: mean={t1.mean():.2f}  std={t1.std():.2f}  p50={np.median(t1):.2f}")
    print(f"scheme2 time s: mean={t2.mean():.2f}  std={t2.std():.2f}  p50={np.median(t2):.2f}")
    print(f"scheme1 TSP m:  mean={tsp1.mean():.1f}")
    print(f"scheme2 TSP m:  mean={tsp2.mean():.1f}")
    print(f"t1-t2 s: mean={dt.mean():.2f}  p05={np.quantile(dt, 0.05):.2f}  p95={np.quantile(dt, 0.95):.2f}")
    print(f"scheme1 shorter: {(dt < -1e-9).mean()*100:.3f}%")
    print(f"scheme2 shorter: {(dt > 1e-9).mean()*100:.3f}%")
    print(f"tie: {(np.abs(dt) <= 1e-9).mean()*100:.3f}%")
    print(f"mean sources: {n_src.mean():.3f}")
    print(f"elapsed {time.perf_counter() - t0:.1f}s")

    out = Path(__file__).resolve().parent.parent / "_figs"
    out.mkdir(exist_ok=True)
    np.savez(
        out / "problem3_cover_path_mc.npz",
        t1=t1,
        t2=t2,
        tsp1=tsp1,
        tsp2=tsp2,
        n_src=n_src,
        walk1=np.array([walk1]),
        walk2=np.array([walk2]),
        extra=extra,
    )
    print(f"saved {out / 'problem3_cover_path_mc.npz'}")


if __name__ == "__main__":
    main()
