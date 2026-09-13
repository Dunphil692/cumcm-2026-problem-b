#!/usr/bin/env python3
"""Second sweep: inner/outer radius, hole-filling, lattices, double rings."""

from __future__ import annotations

import math

import numpy as np

from problem4_network_design import hexagon, residual_mass, ring, tour_length, tri_lattice
from problem4_network_sweep import evaluate, miss_probability, _covers, _sample_sources
from problem4_world import census_stops_n19


def fill_holes(
    seed_stations: list[tuple[float, float]],
    target_miss: float = 0.0008,
    max_extra: int = 6,
) -> list[tuple[float, float]]:
    pool = []
    for rho in (1600.0, 1750.0, 1800.0, 1825.0, 1864.0, 1900.0, 1948.0):
        for k in range(36):
            a = 2.0 * math.pi * k / 36.0
            pool.append((rho * math.cos(a), rho * math.sin(a)))
    gx, gy, phi, rad = _sample_sources(120_000, seed=21, radius="worst")
    alive = ~_covers(seed_stations, gx, gy, phi, rad)
    chosen = list(seed_stations)
    used = set()
    while alive.mean() > target_miss and len(chosen) - len(seed_stations) < max_extra:
        best_i, best_n = None, -1
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
            if n_hit > best_n:
                best_n, best_i = n_hit, i
        if best_i is None or best_n <= 0:
            break
        used.add(best_i)
        chosen.append(pool[best_i])
        alive &= ~_covers([pool[best_i]], gx, gy, phi, rad)
    return chosen


def main() -> None:
    origin = [(0.0, 0.0)]
    cands: list[tuple[str, list[tuple[float, float]]]] = []
    for inner in (800.0, 900.0, 1000.0, 1100.0, 1200.0):
        for outer in (1800.0, 1825.0, 1840.0):
            cands.append(
                (
                    f"hex{inner:.0f}+12@{outer:.0f}",
                    origin + hexagon(inner) + ring(12, outer, 15.0),
                )
            )
    cands.append(("hex1000+12@1825 rot0", origin + hexagon(1000.0) + ring(12, 1825.0, 0.0)))
    cands.append(
        (
            "double 6@1600+6@1900",
            origin + hexagon(1000.0) + ring(6, 1600.0, 0.0) + ring(6, 1900.0, 30.0),
        )
    )
    cands.append(
        (
            "double 8@1600+8@1900",
            origin + hexagon(1000.0) + ring(8, 1600.0, 0.0) + ring(8, 1900.0, 22.5),
        )
    )
    cands.append(
        (
            "hex1000+6@1825+6@1600gaps",
            origin + hexagon(1000.0) + ring(6, 1825.0, 15.0) + ring(6, 1600.0, 45.0),
        )
    )
    for a, rmax in ((1200.0, 2200.0), (1400.0, 2300.0), (1500.0, 2400.0), (1000.0, 2400.0)):
        pts = tri_lattice(a, rmax)
        cands.append((f"trilattice a={a:.0f} n={len(pts)}", pts))

    seed8 = origin + hexagon(1000.0) + ring(8, 1825.0, 22.5)
    seed9 = origin + hexagon(1000.0) + ring(9, 1825.0, 20.0)
    seed10 = origin + hexagon(1000.0) + ring(10, 1825.0, 18.0)
    filled8 = fill_holes(seed8)
    filled9 = fill_holes(seed9)
    filled10 = fill_holes(seed10)
    cands.append((f"fill8→N{len(filled8)}", filled8))
    cands.append((f"fill9→N{len(filled9)}", filled9))
    cands.append((f"fill10→N{len(filled10)}", filled10))
    cands.append(("N19", census_stops_n19()))

    print(
        f"{'network':36s} {'N':>3s} {'hull%':>7s} {'miss1k':>8s} {'missU':>7s} {'resid':>7s} {'tour':>7s}"
    )
    for name, stations in cands:
        do_res = len(stations) <= 24
        row = evaluate(name, stations, do_residual=do_res)
        res = f"{row.get('residual_pct', float('nan')):6.3f}" if "residual_pct" in row else "   n/a"
        print(
            f"{name:36s} {row['N']:3d} {row['hull_fail_pct']:6.2f}% "
            f"{row['Pmiss1000']:7.3f}% {row['PmissU']:6.3f}% {res:>7s} {row['tour_m']:7.0f}"
        )


if __name__ == "__main__":
    main()
