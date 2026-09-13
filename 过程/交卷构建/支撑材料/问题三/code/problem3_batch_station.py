"""Batch second-station selection for unresolved problem-3 channels."""

from __future__ import annotations

import math

from problem2_model import three_circle_anchors, three_circle_feasible
from problem3_simulation_model import ARENA, LISTEN_CERT, LISTEN_MAX, LISTEN_MIN

OFFSETS = (30.0, -30.0)
SAMPLE_DISTANCES = (300.0, 600.0, 900.0, 1200.0, 1500.0)
STATION_DIST = 1200.0            # offset second station; only ±30° kept
STATION_DISTS = (1200.0,)


def clip_arena(x: float, y: float, radius: float = ARENA) -> tuple[float, float]:
    dist = math.hypot(x, y)
    if dist <= radius:
        return x, y
    scale = 0.99 * radius / dist
    return x * scale, y * scale


def bearing_point(
    x: float, y: float, deg: float, dist: float
) -> tuple[float, float]:
    rad = math.radians(deg)
    return x + dist * math.cos(rad), y + dist * math.sin(rad)


def angular_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _directions(state: dict) -> list[dict]:
    return [
        item
        for item in state.get("measurements", [])
        if item.get("status") == "direction"
    ]


def _alive(point: tuple[float, float], state: dict) -> bool:
    silences = state.get("silences") or []
    return all(math.dist(point, silence) > LISTEN_CERT for silence in silences)


def fixed_r_ok(
    point: tuple[float, float],
    state: dict,
    listen_min: float = LISTEN_MIN,
    listen_max: float = LISTEN_MAX,
) -> bool:
    """Whether one fixed R in [listen_min, listen_max] can explain P and N.

    L(g) = max(listen_min, max ||g-p|| for hearings). Need L(g) <= listen_max
    and L(g) < min ||g-n|| over silences. Does not change 990 m punches.
    """
    positives = [
        (float(item["x"]), float(item["y"]))
        for item in state.get("measurements") or []
        if item.get("status") == "direction"
    ]
    if not positives:
        return True
    lower = listen_min
    for stand in positives:
        lower = max(lower, math.dist(point, stand))
    if lower > listen_max + 1e-9:
        return False
    silences = state.get("silences") or []
    if not silences:
        return True
    nearest = min(math.dist(point, silence) for silence in silences)
    return lower < nearest - 1e-9


def possible_samples(state: dict) -> list[tuple[float, float]]:
    if state.get("vertices"):
        samples = list(state["vertices"])
    else:
        samples = [tuple(state["mec_center"])] if state.get("mec_center") else []
        for item in _directions(state):
            for dist in SAMPLE_DISTANCES:
                point = bearing_point(item["x"], item["y"], item["svd_deg"], dist)
                if math.hypot(*point) <= ARENA:
                    samples.append(point)
    return samples


def station_useful(
    station: tuple[float, float],
    state: dict,
    listen: float = LISTEN_MIN,
) -> bool:
    sx, sy = station
    directions = _directions(state)
    if directions:
        last = directions[-1]
        if math.dist(station, (last["x"], last["y"])) < 40.0:
            return False
    samples = possible_samples(state)
    if not samples:
        return False
    if not any(math.dist(station, sample) <= listen for sample in samples):
        return False
    if state.get("kind") == "too_large":
        return True
    for item in directions:
        angle = math.degrees(math.atan2(sy - item["y"], sx - item["x"])) % 360.0
        if angular_distance(angle, item["svd_deg"]) <= 1.0 + 1e-9:
            return False
    return True


def candidate_points(
    state: dict,
    current: tuple[float, float] | None = None,
    dists: tuple[float, ...] | None = None,
    offsets: tuple[float, ...] | None = None,
) -> list[tuple[float, float]]:
    del current
    dists = dists if dists is not None else STATION_DISTS
    offsets = offsets if offsets is not None else OFFSETS
    points: list[tuple[float, float]] = []
    if state.get("mec_center"):
        cx, cy = state["mec_center"]
        points.append(clip_arena(cx, cy))
        radius = float(state.get("mec_radius") or 40.0)
        for deg in (0.0, 90.0, 180.0, 270.0):
            px, py = bearing_point(cx, cy, deg, min(80.0, max(25.0, radius)))
            points.append(clip_arena(px, py))
    for item in _directions(state):
        for dist in dists:
            for offset in offsets:
                px, py = bearing_point(
                    item["x"], item["y"], item["svd_deg"] + offset, dist
                )
                points.append(clip_arena(px, py))
    return points


def nearest_in_recv(
    current: tuple[float, float],
    state: dict,
) -> tuple[float, float] | None:
    """Closest point in the three-circle sufficient set of C_recv."""
    directions = _directions(state)
    if not directions:
        return None
    last = directions[-1]
    s1 = (last["x"], last["y"])
    theta = last["svd_deg"]
    left, right = three_circle_anchors(s1, theta)
    qx, qy = current
    for _ in range(24):
        moved = False
        for cx, cy in (s1, left, right):
            dx = qx - cx
            dy = qy - cy
            dist = math.hypot(dx, dy)
            if dist > LISTEN_MIN + 1e-9:
                scale = LISTEN_MIN / dist
                qx = cx + dx * scale
                qy = cy + dy * scale
                moved = True
        if not moved:
            break
    point = clip_arena(qx, qy)
    if three_circle_feasible(s1, theta, point) and station_useful(point, state):
        return point
    return None


def propose_batch_station(
    current: tuple[float, float],
    unresolved: list[dict],
    listen: float = LISTEN_MIN,
    extra_hints: list[tuple[float, float]] | None = None,
    dists: tuple[float, ...] | None = None,
    offsets: tuple[float, ...] | None = None,
    finish_score=None,
) -> tuple[tuple[float, float], list[int]]:
    if not unresolved:
        raise ValueError("unresolved 不能为空")

    scored: list[tuple[float, tuple[float, float], list[int]]] = []
    seen: set[tuple[float, float]] = set()
    extra_points: list[tuple[float, float]] = []
    extra_points.extend(extra_hints or [])
    for state in unresolved:
        extra_points.extend(
            candidate_points(state, current, dists=dists, offsets=offsets)
        )
    for cand in extra_points:
        key = (round(cand[0], 1), round(cand[1], 1))
        if key in seen:
            continue
        seen.add(key)
        channels = [
            item["channel"]
            for item in unresolved
            if station_useful(cand, item, listen)
        ]
        if not channels:
            continue
        extra = math.dist(current, cand)
        score = len(channels) / (1.0 + extra / 1000.0)
        if item_center_bonus(cand, unresolved):
            score += 0.25
        if finish_score is not None:
            score += finish_score(cand)
        scored.append((score, cand, channels))

    if scored:
        scored.sort(key=lambda row: row[0], reverse=True)
        _, station, channels = scored[0]
        return station, channels

    first = unresolved[0]
    dirs = _directions(first)
    last = dirs[-1] if dirs else {"x": current[0], "y": current[1], "svd_deg": 0.0}
    station = clip_arena(
        *bearing_point(last["x"], last["y"], last["svd_deg"] + 40.0, STATION_DIST)
    )
    return station, [item["channel"] for item in unresolved]


def item_center_bonus(
    cand: tuple[float, float], unresolved: list[dict]
) -> bool:
    for state in unresolved:
        center = state.get("mec_center")
        if center is None:
            continue
        if math.dist(cand, tuple(center)) <= 5.0:
            return True
    return False
