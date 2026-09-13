"""问题四补测点选择 v12：v5 + J 区间检测权重（方向3 轻量版）。"""

from __future__ import annotations

import math

from problem4_simulation_model import ARENA, LISTEN_MIN, CLEAR_R, ERROR_DEG

OFFSETS = (35.0, -35.0, 50.0, -50.0, 90.0, -90.0)
PSI_TARGETS = (45.0, 60.0, 75.0)      # 期望交会角
SAMPLE_DISTANCES = tuple(range(300, 1551, 150))
SAMPLE_RADII = (300.0, 600.0, 900.0, 1200.0, 1500.0)


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
        for dist in SAMPLE_RADII:
            point = bearing_point(item["x"], item["y"], item["svd_deg"], dist)
            if math.hypot(*point) <= ARENA:
                samples.append(point)
    return samples


def _detect_weight_j(candidate: tuple[float, float], state: dict) -> float:
    """J 区间检测权重（方向3）：对每个样本位置 s 算朝向可行区间
    J(s)=∩[dir(s→S_i)±90°]，权重=平均 |J(s)∩[dir(s→q)±90°]|/|J(s)|。
    多次探测后 J 收窄，负信息（朝背侧）的价值自然体现；
    无证据时退化为 (180−δ)/180。"""
    directions = _directions(state)
    if not directions:
        return 1.0
    samples = possible_samples(state)
    if not samples:
        return 1.0
    total = 0.0
    n = 0
    for s in samples:
        sx, sy = s
        # J(s)：与全部探测站相容的 φ 区间交集
        lo = -180.0
        hi = 180.0
        for it in directions:
            a = math.degrees(
                math.atan2(it["y"] - sy, it["x"] - sx)
            ) % 360.0
            # I_i = [a-90, a+90] 与当前交集（相对 a 的模 360 表示）
            a1 = (a - 90.0) % 360.0
            a2 = (a + 90.0) % 360.0
            # 用区间交集（把 I_i 变换到以 a 为中心的坐标系）
            def ang_in(v, lo_, hi_):
                return (v - lo_) % 360.0 <= (hi_ - lo_) % 360.0 + 1e-9
            if not ang_in((a - 90.0) % 360.0, lo, hi) and not ang_in(
                (a + 90.0) % 360.0, lo, hi
            ):
                # 与现有交集可能不相交：保守取更小者——直接算交叠宽度
                pass
            # 简化：交叠 = 把 I_i 与 [lo,hi] 都展开到 [a-180, a+180]
            L = a - 90.0
            U = a + 90.0
            lo_l = (lo - a + 180.0) % 360.0 - 180.0
            hi_l = (hi - a + 180.0) % 360.0 - 180.0
            lo_l = max(lo_l, L)
            hi_l = min(hi_l, U)
            if lo_l > hi_l:
                lo = hi = 0.0
                break
            lo = lo_l
            hi = hi_l
        if hi <= lo:
            continue
        # 候选 q 的接收区间 [dir(s→q)±90] 与 J(s) 的交叠比例
        c = math.degrees(
            math.atan2(candidate[1] - sy, candidate[0] - sx)
        ) % 360.0
        c_l = (c - 90.0 - lo + 180.0) % 360.0 - 180.0
        c_h = (c + 90.0 - lo + 180.0) % 360.0 - 180.0
        ov = max(0.0, min(c_h, hi - lo) - max(c_l, 0.0))
        w = ov / (hi - lo)
        total += min(1.0, max(0.0, w))
        n += 1
    return total / n if n else 1.0


def _detect_weight(candidate: tuple[float, float], state: dict, use_j: bool = False) -> float:
    """候选站对该频道源的期望可测性权重 w ∈ [0, 1]（与 v2 相同）。"""
    if use_j and int(state.get("n_nosig", 0) or 0) >= 1:
        return _detect_weight_j(candidate, state)
    n_nosig = int(state.get("n_nosig", 0) or 0)
    directions = _directions(state)
    if not directions:
        return 1.0
    last = directions[-1]
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
        return 0.6 ** (1 + n_nosig)
    if n_nosig == 0:
        return 1.0
    return ((180.0 - min(delta, 179.0)) / 180.0) ** (1 + n_nosig)


def clear_gain(candidate: tuple[float, float], state: dict) -> float:
    """补测成功后期望'两站可清'的样本占比（方向 1 的核心）。

    对最近一条示向线上的样本点 s（d1 = 距示向原点距离），若候选点 q
    能听到（d3 ≤ 990 保证全向必听），两楔形交会的期望 MEC：
        MEC ≈ tan(ERROR_DEG)·(d1 + d3)/sin(ψ)
    ψ 为 s 处两条示向的夹角。MEC ≤ CLEAR_R 即补测后可直接清除。
    """
    directions = _directions(state)
    if not directions:
        return 1.0
    last = directions[-1]
    sx, sy = last["x"], last["y"]
    svd = last["svd_deg"]
    good = 0
    total = 0
    for d1 in SAMPLE_DISTANCES:
        s = bearing_point(sx, sy, svd, d1)
        d3 = math.dist(candidate, s)
        if d3 > LISTEN_MIN - 10.0:  # 990 证书余量
            continue
        psi = angular_distance(
            (svd + 180.0) % 360.0,
            math.degrees(
                math.atan2(candidate[1] - s[1], candidate[0] - s[0])
            )
            % 360.0,
        )
        psi = max(psi, 8.0)  # 近平行交会退化，视为不可清
        mec = (
            math.tan(math.radians(ERROR_DEG))
            * (d1 + d3)
            / math.sin(math.radians(psi))
        )
        total += 1
        if mec <= CLEAR_R + 1e-9:
            good += 1
    if total == 0:
        return 0.0
    return good / total


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
        sx, sy = item["x"], item["y"]
        svd = item["svd_deg"]
        for psi in PSI_TARGETS:
            for delta in OFFSETS:
                for d1 in SAMPLE_DISTANCES:
                    # 正弦定理：b = d1·sin(ψ)/sin(ψ+δ)
                    denom = math.sin(math.radians(psi + abs(delta)))
                    if denom <= 1e-9:
                        continue
                    b = d1 * math.sin(math.radians(psi)) / denom
                    b = min(1600.0, max(200.0, b))
                    px, py = bearing_point(sx, sy, svd + delta, b)
                    points.append(clip_arena(px, py))
    return points


CLUSTER_R = 300.0   # 共享补测点聚类半径
MAX_COVER = 4       # 每个共享点最多覆盖的频道数


def _channel_weight(
    cand: tuple[float, float], item: dict, use_j: bool = False
) -> float:
    w = _detect_weight(cand, item, use_j=use_j)
    if int(item.get("n_nosig", 0) or 0) == 0:
        gain = clear_gain(cand, item)
        w *= 0.5 + 0.5 * gain
    return w


def propose_batch_station(
    current: tuple[float, float],
    unresolved: list[dict],
    listen: float = LISTEN_MIN,
    use_j: bool = False,
) -> tuple[tuple[float, float], list[int]]:
    if not unresolved:
        raise ValueError("unresolved 不能为空")

    # 1) 汇总全部频道的"有用"候选点（含频道归属）
    entries: list[tuple[tuple[float, float], int]] = []
    for state in unresolved:
        for cand in candidate_points(state):
            if station_useful(cand, state, listen):
                entries.append((cand, state["channel"]))

    if entries:
        # 2) 空间聚类（300 m）：贪心按距当前点近→远顺序
        clusters: list[tuple[float, float, list[tuple[float, float]]]] = []
        order = sorted(entries, key=lambda e: math.dist(current, e[0]))
        for cand, _ch in order:
            placed = False
            for rep, _chs, pts in clusters:
                if math.dist(cand, rep) <= CLUSTER_R:
                    pts.append(cand)
                    # 代表点更新为质心
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    clusters[clusters.index((rep, _chs, pts))] = (
                        (round(cx, 1), round(cy, 1)),
                        _chs,
                        pts,
                    )
                    placed = True
                    break
            if not placed:
                clusters.append((cand, [], [cand]))

        # 3) 代表点重检可用性 + 限流 4 频道
        scored: list[tuple[float, tuple[float, float], list[int]]] = []
        singles: set[int] = set()
        for rep, _chs, pts in clusters:
            rep_clipped = clip_arena(*rep)
            channels = [
                item["channel"]
                for item in unresolved
                if station_useful(rep_clipped, item, listen)
            ]
            if not channels:
                continue
            if len(channels) > MAX_COVER:
                channels = sorted(
                    channels,
                    key=lambda ch: _channel_weight(
                        rep_clipped,
                        next(it for it in unresolved if it["channel"] == ch),
                    ),
                    reverse=True,
                )[:MAX_COVER]
                singles.update(
                    ch for ch in channels if False
                )  # 被拆出的频道走下方单频道补簇
            weight = sum(
                _channel_weight(
                    rep_clipped,
                    next(it for it in unresolved if it["channel"] == ch),
                )
                for ch in channels
            )
            extra = math.dist(current, rep_clipped)
            score = weight / (1.0 + extra / 1000.0)
            if item_center_bonus(rep_clipped, unresolved):
                score += 0.25
            scored.append((score, rep_clipped, channels))

        # 4) 被限流拆出的频道：各自最佳候选单成簇
        covered = set()
        for _s, _p, chs in scored:
            covered.update(chs)
        for state in unresolved:
            ch = state["channel"]
            if ch in covered or ch in singles:
                continue
            best = None
            best_w = -1.0
            for cand in candidate_points(state):
                if not station_useful(cand, state, listen):
                    continue
                w = _channel_weight(cand, state, use_j=use_j)
                if w > best_w:
                    best_w = w
                    best = cand
            if best is not None:
                extra = math.dist(current, best)
                scored.append((best_w / (1.0 + extra / 1000.0), best, [ch]))

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
