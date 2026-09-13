#!/usr/bin/env python3
"""Sweep smaller-than-N19 census nets + a greedy set-cover net.

Never talks to the official simulator. Writes a table to stdout and
``_figs/q4_network_sweep/sweep.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from problem3_route_solver import two_opt_open
from problem3_simulation_model import ARENA, LISTEN_MAX, LISTEN_MIN
from problem4_network_design import hexagon, residual_mass, ring, tour_length
from problem4_world import census_stops_n19

FIGS = Path(__file__).resolve().parents[1] / "_figs" / "q4_network_sweep"
R_WORST = LISTEN_MIN  # 1000 m


def hull_fail_rate(
    stations: list[tuple[float, float]],
    listen: float = R_WORST,
    step: float = 25.0,
) -> dict:
    """Share of sample points that are not in conv(stations within listen)."""
    xs = np.arange(-ARENA, ARENA + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, xs)
    inside = xx * xx + yy * yy <= ARENA * ARENA
    pts = np.column_stack((xx[inside], yy[inside]))
    extra = []
    for rho in (500.0, 1000.0, 1200.0, 1500.0, 1700.0, 1790.0, 1800.0):
        for k in range(180):
            a = 2.0 * math.pi * k / 180.0
            extra.append((rho * math.cos(a), rho * math.sin(a)))
    pts = np.vstack((pts, np.asarray(extra)))
    sx = np.asarray([p[0] for p in stations], dtype=float)
    sy = np.asarray([p[1] for p in stations], dtype=float)
    n_fail = 0
    worst_gap = 0.0
    for x, y in pts:
        dx = sx - x
        dy = sy - y
        near = (dx * dx + dy * dy) <= listen * listen + 1e-9
        if not np.any(near):
            n_fail += 1
            worst_gap = 360.0
            continue
        ang = np.sort(np.arctan2(dy[near], dx[near]))
        wrap = np.empty(ang.size)
        wrap[:-1] = np.diff(ang)
        wrap[-1] = ang[0] + 2.0 * math.pi - ang[-1]
        gap = float(np.degrees(wrap.max()))
        worst_gap = max(worst_gap, gap)
        if gap > 180.0 + 1e-6:
            n_fail += 1
    return {
        "n_pts": int(len(pts)),
        "fail": n_fail,
        "fail_pct": 100.0 * n_fail / len(pts),
        "worst_gap_deg": worst_gap,
        "ok": n_fail == 0,
    }


def miss_probability(
    stations: list[tuple[float, float]],
    n: int = 200_000,
    seed: int = 1,
    radius: str = "uniform",
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = ARENA * np.sqrt(rng.random(n))
    t = 2.0 * np.pi * rng.random(n)
    gx, gy = r * np.cos(t), r * np.sin(t)
    phi = 2.0 * np.pi * rng.random(n)
    ux, uy = np.cos(phi), np.sin(phi)
    if radius == "worst":
        rad = np.full(n, R_WORST)
    else:
        rad = rng.uniform(LISTEN_MIN, LISTEN_MAX, n)
    heard = np.zeros(n, dtype=bool)
    omni = np.zeros(n, dtype=bool)
    for sx, sy in stations:
        dx, dy = sx - gx, sy - gy
        inrange = np.hypot(dx, dy) <= rad
        omni |= inrange
        heard |= inrange & (dx * ux + dy * uy >= 0.0)
    return float(1.0 - heard.mean()), float(1.0 - omni.mean())


def triangle(rho: float, rot: float = 0.0) -> list[tuple[float, float]]:
    return [
        (rho * math.cos(math.radians(rot + 120.0 * k)), rho * math.sin(math.radians(rot + 120.0 * k)))
        for k in range(3)
    ]


def square(rho: float, rot: float = 45.0) -> list[tuple[float, float]]:
    return [
        (rho * math.cos(math.radians(rot + 90.0 * k)), rho * math.sin(math.radians(rot + 90.0 * k)))
        for k in range(4)
    ]


def regular_candidates() -> list[tuple[str, list[tuple[float, float]]]]:
    origin = [(0.0, 0.0)]
    out: list[tuple[str, list[tuple[float, float]]]] = []
    out.append(("N19 current 12@1825", census_stops_n19()))
    out.append(("N19 hull 12@1864", origin + hexagon(1000.0) + ring(12, 1800.0 / math.cos(math.radians(15.0)), 15.0)))
    for n in (7, 8, 9, 10, 11, 12):
        rho_hull = 1800.0 / math.cos(math.pi / n)
        rot = 180.0 / n
        for rho, tag in (
            (1750.0, "1750"),
            (1825.0, "1825"),
            (min(rho_hull, 2100.0), f"hull{rho_hull:.0f}"),
        ):
            out.append(
                (
                    f"hex1000+{n}@{tag}",
                    origin + hexagon(1000.0) + ring(n, rho, rot),
                )
            )
    for n, rho in ((8, 1948.0), (9, 1900.0), (10, 1860.0), (7, 1998.0)):
        out.append((f"tri1000+{n}@{rho:.0f}", origin + triangle(1000.0) + ring(n, rho, 180.0 / n)))
        out.append((f"sq1000+{n}@{rho:.0f}", origin + square(1000.0) + ring(n, rho, 180.0 / n)))
    out.append(("origin+hex1000 only", origin + hexagon(1000.0)))
    out.append(("outer12@1864 only", ring(12, 1800.0 / math.cos(math.radians(15.0)), 15.0)))
    out.append(("origin+outer8@1948", origin + ring(8, 1800.0 / math.cos(math.pi / 8), 22.5)))
    return out


def _sample_sources(n: int, seed: int, radius: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    r = ARENA * np.sqrt(rng.random(n))
    t = 2.0 * np.pi * rng.random(n)
    gx, gy = r * np.cos(t), r * np.sin(t)
    phi = 2.0 * np.pi * rng.random(n)
    if radius == "worst":
        rad = np.full(n, R_WORST)
    else:
        rad = rng.uniform(LISTEN_MIN, LISTEN_MAX, n)
    return gx, gy, phi, rad


def _covers(stations, gx, gy, phi, rad) -> np.ndarray:
    heard = np.zeros(gx.shape[0], dtype=bool)
    ux, uy = np.cos(phi), np.sin(phi)
    for sx, sy in stations:
        dx, dy = sx - gx, sy - gy
        heard |= (np.hypot(dx, dy) <= rad) & (dx * ux + dy * uy >= 0.0)
    return heard


def greedy_set_cover(n_samples: int = 80_000) -> tuple[str, list[tuple[float, float]]]:
    """Cardinality-greedy: add the candidate that hears the most leftover (g,φ)."""
    pool = [(0.0, 0.0)]
    for rho in (600.0, 800.0, 1000.0, 1200.0, 1500.0, 1700.0, 1825.0, 1864.0, 1948.0, 2000.0):
        for k in range(24):
            a = 2.0 * math.pi * k / 24.0
            pool.append((rho * math.cos(a), rho * math.sin(a)))
    gx, gy, phi, rad = _sample_sources(n_samples, seed=7, radius="worst")
    alive = np.ones(n_samples, dtype=bool)
    chosen: list[tuple[float, float]] = []
    used = set()
    while alive.mean() > 0.001 and len(chosen) < 22:
        best_i = None
        best_n = -1
        idx = np.nonzero(alive)[0]
        for i, p in enumerate(pool):
            if i in used:
                continue
            dx = p[0] - gx[idx]
            dy = p[1] - gy[idx]
            hit = (np.hypot(dx, dy) <= rad[idx]) & (
                dx * np.cos(phi[idx]) + dy * np.sin(phi[idx]) >= 0.0
            )
            n_hit = int(hit.sum())
            if n_hit > best_n:
                best_n = n_hit
                best_i = i
        if best_i is None or best_n <= 0:
            break
        used.add(best_i)
        chosen.append(pool[best_i])
        alive &= ~_covers([pool[best_i]], gx, gy, phi, rad)
    return (f"greedy set-cover N{len(chosen)} leftover {100*alive.mean():.2f}%", chosen)


def path_greedy(n_samples: int = 80_000) -> tuple[str, list[tuple[float, float]]]:
    """From the origin, pick the unused candidate with best leftover-kill per metre."""
    pool = [(0.0, 0.0)]
    for rho in (800.0, 1000.0, 1500.0, 1700.0, 1825.0, 1948.0):
        n = 16 if rho >= 1700 else 12
        for k in range(n):
            a = 2.0 * math.pi * k / n
            pool.append((rho * math.cos(a), rho * math.sin(a)))
    gx, gy, phi, rad = _sample_sources(n_samples, seed=11, radius="worst")
    alive = np.ones(n_samples, dtype=bool)
    chosen = [(0.0, 0.0)]
    used = {0}
    alive &= ~_covers(chosen, gx, gy, phi, rad)
    cur = (0.0, 0.0)
    measure_m = 200.0
    while alive.mean() > 0.001 and len(chosen) < 22:
        best_i = None
        best_score = -1.0
        idx = np.nonzero(alive)[0]
        for i, p in enumerate(pool):
            if i in used:
                continue
            dx = p[0] - gx[idx]
            dy = p[1] - gy[idx]
            n_hit = int(
                (
                    (np.hypot(dx, dy) <= rad[idx])
                    & (dx * np.cos(phi[idx]) + dy * np.sin(phi[idx]) >= 0.0)
                ).sum()
            )
            if n_hit <= 0:
                continue
            travel = math.hypot(p[0] - cur[0], p[1] - cur[1]) + measure_m
            score = n_hit / travel
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        used.add(best_i)
        chosen.append(pool[best_i])
        alive &= ~_covers([pool[best_i]], gx, gy, phi, rad)
        cur = pool[best_i]
    return (f"path-greedy N{len(chosen)} leftover {100*alive.mean():.2f}%", chosen)


def prefix_curve(stations: list[tuple[float, float]]) -> list[dict]:
    """Residual / miss after 1, 2, ... stations in listed order (origin first)."""
    rows = []
    for k in range(1, len(stations) + 1):
        pref = stations[:k]
        pm, _ = miss_probability(pref, n=80_000, seed=3, radius="worst")
        rows.append({"k": k, "Pmiss1000": round(100 * pm, 3), "tour": round(tour_length(pref), 0)})
    return rows


def evaluate(name: str, stations: list[tuple[float, float]], do_residual: bool) -> dict:
    hull = hull_fail_rate(stations)
    pm_w, po_w = miss_probability(stations, radius="worst")
    pm_u, po_u = miss_probability(stations, radius="uniform")
    row = {
        "name": name,
        "N": len(stations),
        "hull_fail_pct": round(hull["fail_pct"], 3),
        "hull_ok": hull["ok"],
        "worst_gap_deg": round(hull["worst_gap_deg"], 2),
        "Pmiss1000": round(100 * pm_w, 3),
        "PmissU": round(100 * pm_u, 3),
        "PmissOmni": round(100 * po_u, 3),
        "tour_m": round(tour_length(stations), 0),
    }
    if do_residual:
        row["residual_pct"] = round(100 * residual_mass(stations), 3)
    return row


def main() -> None:
    rows: list[dict] = []
    extras = [greedy_set_cover(), path_greedy()]
    families = regular_candidates() + extras
    print(
        f"{'network':40s} {'N':>3s} {'hull%':>7s} {'gap':>7s} "
        f"{'miss1k':>8s} {'missU':>7s} {'resid':>7s} {'tour':>7s}"
    )
    for name, stations in families:
        # residual only for N<=20 and not obviously broken (or the two greedys / N19)
        do_res = len(stations) <= 20
        row = evaluate(name, stations, do_residual=do_res)
        rows.append(row)
        res = f"{row.get('residual_pct', float('nan')):6.3f}%" if "residual_pct" in row else "   n/a"
        print(
            f"{name:40s} {row['N']:3d} {row['hull_fail_pct']:6.2f}% "
            f"{row['worst_gap_deg']:6.1f} {row['Pmiss1000']:7.3f}% "
            f"{row['PmissU']:6.3f}% {res:>8s} {row['tour_m']:7.0f}"
        )

    n19 = census_stops_n19()
    # interleaved outer ring: origin, hex, then every other outer
    inter = n19[:7] + [n19[7 + k] for k in range(0, 12, 2)] + [n19[7 + k] for k in range(1, 12, 2)]
    curves = {
        "N19 sequential": prefix_curve(n19),
        "N19 interleaved outer": prefix_curve(inter),
    }
    FIGS.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "prefix_curves": curves}
    (FIGS / "sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nN19 sequential prefix P(miss|R=1000):")
    for item in curves["N19 sequential"]:
        print(f"  k={item['k']:2d}  miss={item['Pmiss1000']:6.3f}%  tour={item['tour']:.0f}")
    print(f"\nwrote {FIGS / 'sweep.json'}")


if __name__ == "__main__":
    main()
