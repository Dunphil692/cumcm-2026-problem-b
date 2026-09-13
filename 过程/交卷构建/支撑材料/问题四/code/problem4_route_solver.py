"""Open Held–Karp, smallest enclosing circle, and discrete TSPN entry routing."""

from __future__ import annotations

import math
import random

INF = 1e100


def smallest_enclosing_circle(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], float]:
    pts = list(points)
    if not pts:
        return (0.0, 0.0), 0.0
    rng = random.Random(0)
    rng.shuffle(pts)

    def circle_of(boundary: list[tuple[float, float]]) -> tuple[tuple[float, float], float]:
        if not boundary:
            return (0.0, 0.0), 0.0
        if len(boundary) == 1:
            return boundary[0], 0.0
        if len(boundary) == 2:
            (x1, y1), (x2, y2) = boundary
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            return center, math.dist(boundary[0], boundary[1]) / 2.0
        return _circumcircle(boundary[0], boundary[1], boundary[2])

    def welzl(index: int, boundary: list[tuple[float, float]]):
        if index == len(pts) or len(boundary) == 3:
            return circle_of(boundary)
        center, radius = welzl(index + 1, boundary)
        if math.dist(center, pts[index]) <= radius + 1e-9:
            return center, radius
        return welzl(index + 1, boundary + [pts[index]])

    return welzl(0, [])


def _circumcircle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        pairs = ((a, b), (b, c), (a, c))
        p, q = max(pairs, key=lambda pair: math.dist(pair[0], pair[1]))
        center = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
        return center, math.dist(p, q) / 2.0
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    center = (ux, uy)
    return center, math.dist(center, a)


def guaranteed_disk(
    vertices: list[tuple[float, float]],
    clear_r: float = 20.0,
) -> tuple[tuple[float, float], float] | None:
    center, radius = smallest_enclosing_circle(vertices)
    if radius > clear_r + 1e-9:
        return None
    return center, clear_r - radius


def held_karp_open(
    start: tuple[float, float],
    cities: list[tuple[float, float]],
) -> tuple[list[int], float]:
    n = len(cities)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], math.dist(start, cities[0])

    size = 1 << n
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    for j, city in enumerate(cities):
        dp[1 << j][j] = math.dist(start, city)

    for mask in range(size):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            prev_mask = mask ^ (1 << last)
            if prev_mask == 0:
                continue
            best = dp[mask][last]
            for prev in range(n):
                if not (prev_mask & (1 << prev)):
                    continue
                cand = dp[prev_mask][prev] + math.dist(cities[prev], cities[last])
                if cand < best:
                    best = cand
                    parent[mask][last] = prev
            dp[mask][last] = best

    full = size - 1
    end = min(range(n), key=lambda j: dp[full][j])
    length = dp[full][end]
    order: list[int] = []
    mask = full
    last = end
    while last != -1:
        order.append(last)
        prev = parent[mask][last]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return order, length


def two_opt_open(
    start: tuple[float, float],
    cities: list[tuple[float, float]],
    passes: int = 40,
) -> tuple[list[int], float]:
    n = len(cities)
    if n == 0:
        return [], 0.0
    unused = set(range(n))
    order: list[int] = []
    current = start
    while unused:
        nxt = min(unused, key=lambda i: math.dist(current, cities[i]))
        unused.remove(nxt)
        order.append(nxt)
        current = cities[nxt]

    def length_of(ordr: list[int]) -> float:
        total = math.dist(start, cities[ordr[0]])
        for i in range(len(ordr) - 1):
            total += math.dist(cities[ordr[i]], cities[ordr[i + 1]])
        return total

    improved = True
    rounds = 0
    while improved and rounds < passes:
        improved = False
        rounds += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                trial = order[:i] + list(reversed(order[i : j + 1])) + order[j + 1 :]
                if length_of(trial) + 1e-12 < length_of(order):
                    order = trial
                    improved = True
    return order, length_of(order)


def _project_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return start
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom
    t = min(1.0, max(0.0, t))
    return (start[0] + t * dx, start[1] + t * dy)


def best_entry(
    prev: tuple[float, float],
    center: tuple[float, float],
    radius: float,
    nxt: tuple[float, float] | None,
    samples: int = 16,
) -> tuple[float, float]:
    if radius <= 1e-9:
        return center

    def cost(point: tuple[float, float]) -> float:
        total = math.dist(prev, point)
        if nxt is not None:
            total += math.dist(point, nxt)
        return total

    best_point = center
    best = cost(center)
    for index in range(samples):
        angle = 2.0 * math.pi * index / samples
        point = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        value = cost(point)
        if value < best:
            best = value
            best_point = point
    if nxt is not None:
        projected = _project_segment(prev, nxt, center)
        if math.dist(projected, center) <= radius + 1e-9:
            value = cost(projected)
            if value < best:
                best_point = projected
    return best_point


def joint_route(
    start: tuple[float, float],
    disks: list[tuple[tuple[float, float], float]],
    rounds: int = 4,
    use_exact: bool | None = None,
) -> tuple[list[int], float, list[tuple[float, float]]]:
    n = len(disks)
    if n == 0:
        return [], 0.0, []
    if use_exact is None:
        use_exact = n <= 12
    solver = held_karp_open if use_exact else two_opt_open
    entry = [center for center, _ in disks]
    order, _ = solver(start, entry)
    for _ in range(rounds):
        new_entry = list(entry)
        for k, index in enumerate(order):
            prev = start if k == 0 else new_entry[order[k - 1]]
            nxt = new_entry[order[k + 1]] if k + 1 < n else None
            center, radius = disks[index]
            new_entry[index] = best_entry(prev, center, radius, nxt)
        entry = new_entry
        order, _ = solver(start, entry)
    path = [entry[index] for index in order]
    length = 0.0
    prev = start
    for point in path:
        length += math.dist(prev, point)
        prev = point
    return order, length, path
