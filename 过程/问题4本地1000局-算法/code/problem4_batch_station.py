"""问题四补测点选择：检测概率加权的批量站点提议。

在问题三 propose_batch_station 的基础上加入定向源意识：

- 每频道维护 n_nosig（听见后的 no_signal 次数，即"定向证据"）。
- 候选站对频道的期望可测性 w：
    * 无证据频道（大概率全向）：站距可能样本 ≤ 1000 m ⇒ 必测到，w = 1；
    * 有证据频道（疑似定向）：w = ((180 − δ)/180) ** (1 + n_nosig)，
      其中 δ 为候选站相对最近示向线的偏移角（φ 区间与扇形交叠比例，
      简化 E-IG 的检测概率权重）。δ 越小越可能仍在 180° 扇形内；
      失败的补测越多（n_nosig 越大），越偏向小偏移站（重试阶梯）。
    * MEC 圆心候选（近测）：定向源在圆心被扇形覆盖的概率约 1/2，
      取 w = 0.6 ** (1 + n_nosig)。
"""

from __future__ import annotations

import math

from problem4_simulation_model import ARENA, LISTEN_MIN

OFFSETS = (35.0, -35.0, 50.0, -50.0, 90.0, -90.0)
SAMPLE_DISTANCES = (300.0, 600.0, 900.0, 1200.0, 1500.0)


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


def possible_samples(state: dict) -> list[tuple[float, float]]:
    if state.get("vertices"):
        return list(state["vertices"])
    if state.get("mec_center"):
        samples = [tuple(state["mec_center"])]
    else:
        samples = []
    for item in _directions(state):
        for dist in SAMPLE_DISTANCES:
            point = bearing_point(item["x"], item["y"], item["svd_deg"], dist)
            if math.hypot(*point) <= ARENA:
                samples.append(point)
    return samples


def _detect_weight(candidate: tuple[float, float], state: dict) -> float:
    """候选站对该频道源的期望可测性权重 w ∈ [0, 1]。"""
    n_nosig = int(state.get("n_nosig", 0) or 0)
    directions = _directions(state)
    if not directions:
        return 1.0
    last = directions[-1]
    # 候选站相对最近示向线的偏移角 δ
    delta = angular_distance(
        math.degrees(
            math.atan2(
                candidate[1] - last["y"], candidate[0] - last["x"]
            )
        )
        % 360.0,
        last["svd_deg"],
    )
    if state.get("mec_center") and math.dist(
        candidate, tuple(state["mec_center"])
    ) <= 90.0:
        # 近圆心候选：定向扇形覆盖概率约 1/2
        return 0.6 ** (1 + n_nosig)
    if n_nosig == 0:
        return 1.0
    return ((180.0 - min(delta, 179.0)) / 180.0) ** (1 + n_nosig)


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
    # 对过大有界多边形，MEC 圆心是正确的补测点，即使它落在旧楔形内
    if state.get("kind") == "too_large":
        return True
    for item in directions:
        angle = math.degrees(math.atan2(sy - item["y"], sx - item["x"])) % 360.0
        if angular_distance(angle, item["svd_deg"]) <= 1.0 + 1e-9:
            return False
    return True


def candidate_points(state: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if state.get("mec_center"):
        cx, cy = state["mec_center"]
        points.append(clip_arena(cx, cy))
        radius = float(state.get("mec_radius") or 40.0)
        for deg in (0.0, 90.0, 180.0, 270.0):
            px, py = bearing_point(cx, cy, deg, min(80.0, max(25.0, radius)))
            points.append(clip_arena(px, py))
    for item in _directions(state):
        for offset in OFFSETS:
            px, py = bearing_point(
                item["x"], item["y"], item["svd_deg"] + offset, 900.0
            )
            points.append(clip_arena(px, py))
    return points


def propose_batch_station(
    current: tuple[float, float],
    unresolved: list[dict],
    listen: float = LISTEN_MIN,
) -> tuple[tuple[float, float], list[int]]:
    if not unresolved:
        raise ValueError("unresolved 不能为空")

    scored: list[tuple[float, tuple[float, float], list[int]]] = []
    seen: set[tuple[float, float]] = set()
    for state in unresolved:
        for cand in candidate_points(state):
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
            # 检测概率加权：Σ w(ch) / (1 + 路程/1000)
            weight = sum(
                _detect_weight(cand, item)
                for item in unresolved
                if item["channel"] in channels
            )
            score = weight / (1.0 + extra / 1000.0)
            if item_center_bonus(cand, unresolved):
                score += 0.25
            scored.append((score, cand, channels))

    if scored:
        scored.sort(key=lambda row: row[0], reverse=True)
        _, station, channels = scored[0]
        return station, channels

    first = unresolved[0]
    dirs = _directions(first)
    last = dirs[-1] if dirs else {"x": current[0], "y": current[1], "svd_deg": 0.0}
    station = clip_arena(
        *bearing_point(last["x"], last["y"], last["svd_deg"] + 40.0, 900.0)
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
