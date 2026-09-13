#!/usr/bin/env python3
"""Problem-4 census network table (paper appendix; no official simulator).

For each candidate station set we report

* residual mass: share of (position x orientation) cells of an *empty*
  channel that survive a no_signal at every station, computed with the
  policy's own ``OrientGrid`` (10 m grid, 64 orientation bins, R = 1000 m
  worst case, grid-conservative). This is exactly the mass the policy's
  eps stop rule sees after the census.
* P(miss): Monte-Carlo probability that a directional source (uniform
  position, uniform orientation, R ~ U[1000, 1500]) is heard by none of the
  stations; and the same for an omnidirectional source.
* tour: open path from the origin through all stations (NN + 2-opt).

Theorem used in the paper: a network guarantees hearing every directional
source (worst-case orientation and R = 1000) only if its convex hull
contains the arena, so stations outside the 1800 m disk are necessary.
"""

from __future__ import annotations

import math

import numpy as np

from problem3_route_solver import two_opt_open
from problem3_simulation_model import ARENA, LISTEN_MAX, LISTEN_MIN
from problem4_orient_grid import CELLS_PER_CHANNEL, OrientGrid, popcount
from problem4_world import census_stops_n19


def hexagon(rho: float, rot: float = 0.0) -> list[tuple[float, float]]:
    return [
        (rho * math.cos(math.radians(rot + 60.0 * k)), rho * math.sin(math.radians(rot + 60.0 * k)))
        for k in range(6)
    ]


def ring(n: int, rho: float, rot: float = 0.0) -> list[tuple[float, float]]:
    return [
        (
            rho * math.cos(math.radians(rot + 360.0 * k / n)),
            rho * math.sin(math.radians(rot + 360.0 * k / n)),
        )
        for k in range(n)
    ]


def tri_lattice(a: float, rmax: float) -> list[tuple[float, float]]:
    pts = []
    h = a * math.sqrt(3.0) / 2.0
    for j in range(-6, 7):
        for i in range(-6, 7):
            x = a * i + (a / 2.0 if j % 2 else 0.0)
            y = h * j
            if math.hypot(x, y) <= rmax:
                pts.append((x, y))
    return pts


def residual_mass(stations: list[tuple[float, float]], channel: int = 1) -> float:
    grid = OrientGrid()
    for x, y in stations:
        grid.exclude_silent(channel, x, y)
    return popcount(grid.row(channel)) / CELLS_PER_CHANNEL


def miss_probability(
    stations: list[tuple[float, float]], n: int = 200_000, seed: int = 1
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = ARENA * np.sqrt(rng.random(n))
    t = 2.0 * np.pi * rng.random(n)
    gx, gy = r * np.cos(t), r * np.sin(t)
    phi = 2.0 * np.pi * rng.random(n)
    ux, uy = np.cos(phi), np.sin(phi)
    radius = rng.uniform(LISTEN_MIN, LISTEN_MAX, n)
    heard = np.zeros(n, dtype=bool)
    omni = np.zeros(n, dtype=bool)
    for sx, sy in stations:
        dx, dy = sx - gx, sy - gy
        inrange = np.hypot(dx, dy) <= radius
        omni |= inrange
        heard |= inrange & (dx * ux + dy * uy >= 0.0)
    return float(1.0 - heard.mean()), float(1.0 - omni.mean())


def tour_length(stations: list[tuple[float, float]]) -> float:
    rest = [p for p in stations if math.hypot(*p) > 1e-9]
    _, length = two_opt_open((0.0, 0.0), rest)
    return length


def candidates() -> list[tuple[str, list[tuple[float, float]]]]:
    origin = [(0.0, 0.0)]
    q3 = origin + hexagon(1200.0)
    return [
        ("Q3 seven-point (origin + hexagon 1200)", q3),
        ("Q3 seven + ring 10 @1850", q3 + ring(10, 1850.0, 18.0)),
        ("Q3 seven + ring 12 @1850", q3 + ring(12, 1850.0, 15.0)),
        ("origin + hexagon 1000 + ring 10 @1825", origin + hexagon(1000.0) + ring(10, 1825.0, 18.0)),
        ("origin + hexagon 1000 + ring 12 @1700 (rot 15)", origin + hexagon(1000.0) + ring(12, 1700.0, 15.0)),
        ("origin + hexagon 1000 + ring 12 @1750 (rot 15)", origin + hexagon(1000.0) + ring(12, 1750.0, 15.0)),
        ("N19 = origin + hexagon 1000 + ring 12 @1825 (rot 15)", census_stops_n19()),
        ("origin + hexagon 1000 + ring 12 @1864 (hull = disk)", origin + hexagon(1000.0) + ring(12, 1800.0 / math.cos(math.radians(15.0)), 15.0)),
        ("triangular lattice a=1000 (exact guarantee)", tri_lattice(1000.0, 2400.0)),
    ]


def main() -> None:
    print(f"{'network':56s} {'N':>3s} {'residual':>9s} {'P(miss dir)':>12s} {'P(miss omni)':>13s} {'tour m':>8s}")
    for name, stations in candidates():
        res = residual_mass(stations)
        pm, po = miss_probability(stations)
        length = tour_length(stations)
        print(f"{name:56s} {len(stations):3d} {100*res:8.3f}% {100*pm:11.3f}% {100*po:12.3f}% {length:8.0f}")


if __name__ == "__main__":
    main()
