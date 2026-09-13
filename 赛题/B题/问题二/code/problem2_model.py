"""问题 2 融合修正版核心模型（2026 国赛 B 题问题 2）。

融合与修正记录（相对「三圆叶子 + 135 m」旧稿）：
1. 源域 Ω1 = B(O,1800) ∩ W(S1,θ1,1°) ∩ {5 < d1 ≤ 1500}，逐方向目标圆裁剪；
2. 一般保证接收域 C_recv = ∩_{g∈Ω1} B(g, L(g))，L(g) = max(1000, ‖g−S1‖)。
   依据：S1 已收到 g ⟹ 接收半径 R ≥ ‖g−S1‖；要对所有相容 R 保证 S2 收到，
   必须且只需 d2 ≤ L(g)。三圆叶子 B(S1,1000)∩B(P−,1000)∩B(P+,1000) 是
   该条件的保守充分近似（近源段精确、远源段自动满足）。
3. 后验裁剪修正：定位区域 = W1 ∩ W2 ∩ Ω1 ∩ {5 < d2 ≤ 1500}。
   旧版只取 W1∩W2，把 d1>1500 m 的不可能源点算进直径，135 m 因此高估约 20%
   （修正后约 112 m）。
4. 近场 STRONG 分支：d2(q,g) ≤ 5 m 时第二次观测无示向度，后验取
   Ω1 ∩ B(q,5) 的实际交集（不虚设 0、也不虚设全圆 10）。
5. 数值口径：J(q) 是有限源网格 × 有限误差网格上的最坏裁剪后验直径估计，
   不是连续上确界的认证；优化为多级网格+局部细化，全局最优未认证。
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------- 常量
ERROR_DEG = 1.0          # 测向误差半宽（度）
R_MIN = 1000.0           # 有效接收半径下界（m）
R_MAX = 1500.0           # 有效接收半径上界（m）
TARGET_R = 1800.0        # 目标区域半径（m）
NEAR_R = 5.0             # 近场阈值（m）
ERROR_STEPS = (-ERROR_DEG, 0.0, ERROR_DEG)

Point = tuple[float, float]


# ---------------------------------------------------------------- 基础几何
def _unit(deg: float) -> Point:
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))


def _bearing(origin: Point, target: Point) -> float:
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0])) % 360.0


def _angular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _along(start: Point, deg: float, distance: float) -> Point:
    ux, uy = _unit(deg)
    return (start[0] + distance * ux, start[1] + distance * uy)


# ---------------------------------------------------------------- 源域 Ω1
def ray_exit_distance(start: Point, deg: float, radius: float = TARGET_R) -> float:
    """从 start 沿 deg 方向出发，走多远离开以原点为心、radius 为半径的圆。"""
    ux, uy = _unit(deg)
    b = start[0] * ux + start[1] * uy
    c = start[0] ** 2 + start[1] ** 2 - radius ** 2
    disc = b * b - c
    if disc < 0.0:
        return 0.0
    return max(-b + math.sqrt(disc), 0.0)


def direction_far_distance(s1: Point, deg: float) -> float:
    """该方位上楔形被目标圆与 1500 m 接收上界截断后的远端距离。"""
    return min(R_MAX, ray_exit_distance(s1, deg))


def is_in_omega1(s1: Point, theta1: float, p: Point, tol: float = 1e-9) -> bool:
    d1 = _dist(s1, p)
    if math.hypot(p[0], p[1]) > TARGET_R + tol:
        return False
    if d1 <= NEAR_R - tol or d1 > R_MAX + tol:
        return False
    return _angular_distance(_bearing(s1, p), theta1) <= ERROR_DEG + tol


def offset_from_bearing(s1: Point, theta1: float, p: Point) -> float:
    """p 相对 S1 的方位与示向线的偏角（-180..180 度）。"""
    return (_bearing(s1, p) - theta1 + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- 保证接收
def L_value(s1: Point, g: Point) -> float:
    return max(R_MIN, _dist(s1, g))


def reception_slack(s1: Point, q: Point, g: Point) -> float:
    """d2(q,g) - L(g)；≤0 ⟺ 对该源保证接收（广义口径）。"""
    return _dist(q, g) - L_value(s1, g)


def source_grid(s1: Point, theta1: float, phi_n: int = 17, r_step: float = 25.0):
    """Ω1 上的源采样（每个方向从 5 m 到该方向远端，含两端点）。yield (phi, r, g)。"""
    for i in range(phi_n):
        phi = -ERROR_DEG + i * (2.0 * ERROR_DEG / (phi_n - 1)) if phi_n > 1 else 0.0
        r_far = direction_far_distance(s1, theta1 + phi)
        if r_far <= NEAR_R:
            continue
        n_r = max(1, int(math.ceil((r_far - NEAR_R) / r_step)))
        for k in range(n_r + 1):
            r = NEAR_R + k * (r_far - NEAR_R) / n_r
            yield phi, r, _along(s1, theta1 + phi, r)


def coarse_sources(s1: Point, theta1: float, phi_n: int = 9):
    """低精度 J 用：每方向近/中/远三点。yield (phi, r, g)。"""
    for i in range(phi_n):
        phi = -ERROR_DEG + i * (2.0 * ERROR_DEG / (phi_n - 1)) if phi_n > 1 else 0.0
        r_far = direction_far_distance(s1, theta1 + phi)
        if r_far <= NEAR_R:
            continue
        for r in (NEAR_R, NEAR_R + 0.5 * (r_far - NEAR_R), r_far):
            yield phi, r, _along(s1, theta1 + phi, r)


def reception_margin(s1: Point, theta1: float, q: Point,
                     phi_n: int = 33, r_step: float = 10.0):
    """有限网格上 sup_{g∈Ω1}[d2 - L(g)] 的估计（数值口径，非区间认证）。
    返回 (max_slack, (phi, r, g))；max_slack ≤ 0 为数值接收通过。"""
    worst = -math.inf
    arg = None
    for phi, r, g in source_grid(s1, theta1, phi_n, r_step):
        s = reception_slack(s1, q, g)
        if s > worst:
            worst = s
            arg = (phi, r, g)
    return worst, arg


def three_circle_anchors(s1: Point, theta1: float) -> tuple[Point, Point]:
    """保守三圆叶子的远弧锚点：±1° 方向上 R_MIN=1000 m 处（若目标圆更早则截断）。"""
    pts = []
    for delta in (-ERROR_DEG, ERROR_DEG):
        reach = min(R_MIN, ray_exit_distance(s1, theta1 + delta))
        pts.append(_along(s1, theta1 + delta, reach))
    return pts[0], pts[1]


def three_circle_feasible(s1: Point, theta1: float, q: Point, tol: float = 1e-9) -> bool:
    """保守充分条件：S2 同时落在 S1 与远弧两端各 1000 m 盘内（近源段保证接收）。"""
    left, right = three_circle_anchors(s1, theta1)
    return (_dist(q, s1) <= R_MIN + tol
            and _dist(q, left) <= R_MIN + tol
            and _dist(q, right) <= R_MIN + tol)


# ---------------------------------------------------------------- 几何裁剪
def _disk_polygon(cx: float, cy: float, r: float, n: int = 144) -> list[Point]:
    """圆盘的多边形近似（逆时针）。"""
    return [(cx + r * math.cos(2.0 * math.pi * i / n),
             cy + r * math.sin(2.0 * math.pi * i / n)) for i in range(n)]


def _clip_halfplanes(vertices: list[Point], hps: list) -> list[Point]:
    """用半平面 (p, v)（左侧可行）逐条裁剪多边形。"""
    out = vertices
    for p, v in hps:
        cur = out
        out = []
        n = len(cur)
        if n == 0:
            return []
        for i in range(n):
            a = cur[i]
            b = cur[(i + 1) % n]
            ax, ay = a
            bx, by = b
            ca = v[0] * (ay - p[1]) - v[1] * (ax - p[0])
            cb = v[0] * (by - p[1]) - v[1] * (bx - p[0])
            ain = ca >= -1e-12
            bin_ = cb >= -1e-12
            if ain:
                out.append(a)
            if ain != bin_:
                t = ca / (ca - cb)
                out.append((ax + t * (bx - ax), ay + t * (by - ay)))
    return out


def _clip_circle(vertices: list[Point], cx: float, cy: float, r: float,
                 keep_inside: bool = True) -> list[Point]:
    """用圆裁剪多边形（keep_inside=True 保留圆内，False 保留圆外）。
    鲁棒处理「边两端都在圆外但穿过圆」的情形（此时保留两个交点即弦）。"""
    out = []
    n = len(vertices)
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        da = math.hypot(a[0] - cx, a[1] - cy)
        db = math.hypot(b[0] - cx, b[1] - cy)
        ain = (da <= r) if keep_inside else (da >= r)
        bin_ = (db <= r) if keep_inside else (db >= r)
        if ain:
            out.append(a)
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        ox, oy = ax - cx, ay - cy
        A = dx * dx + dy * dy
        B = 2.0 * (ox * dx + oy * dy)
        C = ox * ox + oy * oy - r * r
        disc = B * B - 4.0 * A * C
        if disc <= 0.0:
            continue
        sq = math.sqrt(disc)
        ts = sorted(t for t in ((-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A))
                    if 0.0 < t < 1.0)
        if not ts:
            continue
        if keep_inside:
            if ain != bin_:
                out.append((ax + ts[0] * dx, ay + ts[0] * dy))
            elif len(ts) == 2:
                # 两端都在圆外、边穿过圆：保留弦的两个端点
                out.append((ax + ts[0] * dx, ay + ts[0] * dy))
                out.append((ax + ts[1] * dx, ay + ts[1] * dy))
        else:
            if ain != bin_:
                out.append((ax + ts[0] * dx, ay + ts[0] * dy))
            elif len(ts) == 2:
                # 两端都在圆内（禁区内）、边穿过禁区：绕洞走，保留两个交点
                out.append((ax + ts[0] * dx, ay + ts[0] * dy))
                out.append((ax + ts[1] * dx, ay + ts[1] * dy))
    return out


def _dedupe(vertices: list[Point], eps: float = 1e-7) -> list[Point]:
    out = []
    for q in vertices:
        if all(_dist(q, r) > eps for r in out):
            out.append(q)
    return out


def _refine_arcs(vertices: list[Point], circles: list, max_arc_deg: float = 1.0) -> list[Point]:
    """把落在裁剪圆上的弧边按 ≤max_arc_deg 细分（弦误差 ≈ R·(1-cos(Δ/2)) ≤ 0.06 m）。"""
    n = len(vertices)
    if n == 0:
        return []
    out = []
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        out.append(a)
        for cx, cy, r in circles:
            da = abs(math.hypot(a[0] - cx, a[1] - cy) - r)
            db = abs(math.hypot(b[0] - cx, b[1] - cy) - r)
            if da < 1e-6 and db < 1e-6 and r > 0:
                chord = _dist(a, b)
                if chord > 1e-12:
                    half = min(math.pi, math.asin(min(1.0, chord / (2.0 * r))))
                    span = 2.0 * math.degrees(half)
                    k = max(1, int(math.ceil(span / max_arc_deg)))
                    am = math.atan2(a[1] - cy, a[0] - cx)
                    bm = math.atan2(b[1] - cy, b[0] - cx)
                    dm = (bm - am + math.pi) % (2.0 * math.pi) - math.pi
                    for j in range(1, k):
                        m = am + dm * j / k
                        out.append((cx + r * math.cos(m), cy + r * math.sin(m)))
                break
    return out


def _poly_diameter(vertices: list[Point]) -> float:
    m = 0.0
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(vertices[i], vertices[j])
            if d > m:
                m = d
    return m


def detection_halfplanes(S: Point, theta: float) -> list:
    """示向 θ±1° → 两条半平面（左侧可行），与 problem1_localize 同一口径。"""
    lo = theta - ERROR_DEG
    hi = theta + ERROR_DEG
    u_lo = _unit(lo)
    u_hi = _unit(hi)
    return [(S, u_lo), (S, (-u_hi[0], -u_hi[1]))]


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _line_intersection(h1, h2):
    p1, v1 = h1
    p2, v2 = h2
    d = _cross(v1, v2)
    if abs(d) < 1e-12:
        return None
    t = _cross((p2[0] - p1[0], p2[1] - p1[1]), v2) / d
    return (p1[0] + t * v1[0], p1[1] + t * v1[1])


def _on_left(h, q):
    p, v = h
    return _cross(v, (q[0] - p[0], q[1] - p[1])) >= -1e-12


def _convex_hull(points):
    pts = sorted(set((round(x, 9), round(y, 9)) for x, y in points))
    if len(pts) <= 1:
        return pts

    def ccw(o, a, b):
        return _cross((a[0] - o[0], a[1] - o[1]), (b[0] - o[0], b[1] - o[1]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and ccw(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and ccw(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def halfplane_intersection_polygon(hps):
    """半平面交（左侧可行）。返回：bounded→顶点表；unbounded→None；empty→[]。"""
    n = len(hps)
    pts = []
    nonparallel = False
    for i in range(n):
        for j in range(i + 1, n):
            q = _line_intersection(hps[i], hps[j])
            if q is None:
                continue
            nonparallel = True
            if all(_on_left(h, q) for h in hps):
                pts.append(q)
    if not nonparallel:
        return None
    if not pts:
        return []
    uniq = _dedupe(pts, 1e-7)
    # 有界性：内法向量是否包围原点
    ang = sorted(math.atan2(-v[1], v[0]) % (2.0 * math.pi) for p, v in hps)
    max_gap = max((ang[(i + 1) % n] - ang[i]) % (2.0 * math.pi) for i in range(n))
    if max_gap >= math.pi - 1e-9:
        return None
    return _convex_hull(uniq)


# ---------------------------------------------------------------- 后验与 J
def posterior_diameter(s1: Point, theta1: float, q: Point, theta2: float) -> float:
    """NORMAL 后验（W1∩W2 裁剪到 Ω1 与 d2≤1500）的直径。
    后验必含源 g（构造保证），故裁剪为空只可能是数值退化的单点情形，返回 0.0。"""
    hps = detection_halfplanes(s1, theta1) + detection_halfplanes(q, theta2)
    verts = halfplane_intersection_polygon(hps)
    if verts is None:
        # W1∩W2 无界：先与外接 B(S1,1500) 的 144 边形相交使其有界
        # （外接多边形 ⊇ 圆盘，结果对直径是保守上近似，误差 ≤0.7 m）
        circum_r = R_MAX / math.cos(math.pi / 144.0)
        disk = _disk_polygon(s1[0], s1[1], circum_r, 144)
        verts = _clip_halfplanes(disk, hps)
        if not verts:
            return 0.0
    if not verts:
        return 0.0
    circles = [(s1[0], s1[1], R_MAX), (0.0, 0.0, TARGET_R),
               (q[0], q[1], R_MAX), (s1[0], s1[1], NEAR_R), (q[0], q[1], NEAR_R)]
    verts = _clip_circle(verts, s1[0], s1[1], R_MAX, True)          # d1 ≤ 1500
    verts = _clip_circle(verts, 0.0, 0.0, TARGET_R, True)           # 目标圆
    verts = _clip_circle(verts, q[0], q[1], R_MAX, True)            # d2 ≤ 1500
    verts = _clip_circle(verts, s1[0], s1[1], NEAR_R, False)        # d1 ≥ 5
    verts = _clip_circle(verts, q[0], q[1], NEAR_R, False)          # d2 ≥ 5
    if not verts:
        return 0.0
    verts = _dedupe(verts)
    verts = _refine_arcs(verts, circles, 1.0)
    return _poly_diameter(verts)


def strong_diameter(s1: Point, theta1: float, q: Point) -> float:
    """STRONG 分支：Ω1 ∩ B(q,5) 的实际直径（B(q,5) 以 96 边形近似）。"""
    ball = _disk_polygon(q[0], q[1], NEAR_R, 96)
    verts = _clip_halfplanes(ball, detection_halfplanes(s1, theta1))
    verts = _clip_circle(verts, s1[0], s1[1], R_MAX, True)
    verts = _clip_circle(verts, 0.0, 0.0, TARGET_R, True)
    verts = _clip_circle(verts, s1[0], s1[1], NEAR_R, False)
    if not verts:
        return 0.0
    verts = _dedupe(verts)
    verts = _refine_arcs(verts, [(s1[0], s1[1], R_MAX), (0.0, 0.0, TARGET_R),
                                 (s1[0], s1[1], NEAR_R), (q[0], q[1], NEAR_R)], 2.0)
    return _poly_diameter(verts)


def worst_diameter(s1: Point, theta1: float, q: Point,
                   phi_n: int = 17, r_step: float = 10.0, err_n: int = 17,
                   coarse: bool = False):
    """J(q)：有限源网格 × 有限误差网格上最坏裁剪后验直径。
    coarse=True 时用每方向近/中/远三点源。返回 (J, 见证 (phi, r, e2, branch))。"""
    sources = coarse_sources(s1, theta1, phi_n) if coarse else source_grid(s1, theta1, phi_n, r_step)
    worst = -math.inf
    arg = None
    for phi, r, g in sources:
        d2g = _dist(q, g)
        if d2g <= NEAR_R:
            d = strong_diameter(s1, theta1, q)
            if d > worst:
                worst = d
                arg = (phi, r, 0.0, "STRONG")
            continue
        if d2g > R_MAX:
            continue  # S2 收不到该源：无第二次观测，不进入 NORMAL 口径
        t2 = _bearing(q, g)
        for i in range(err_n):
            e2 = -ERROR_DEG + i * (2.0 * ERROR_DEG / (err_n - 1)) if err_n > 1 else 0.0
            d = posterior_diameter(s1, theta1, q, t2 + e2)
            if d > worst:
                worst = d
                arg = (phi, r, e2, "NORMAL")
    if worst == -math.inf:
        return 0.0, None
    return worst, arg


# ---------------------------------------------------------------- 求解器
def _grid_points(s1: Point, half: float, step: float):
    xs = [s1[0] + i * step for i in range(-int(half // step), int(half // step) + 1)]
    ys = [s1[1] + i * step for i in range(-int(half // step), int(half // step) + 1)]
    return xs, ys


def optimize(s1: Point, theta1: float, bind_target: bool = True,
             coarse_step: float = 25.0, coarse_half: float = 1010.0,
             verbose: bool = False):
    """多级网格搜索：25 m 粗网格 → 5 m → 1 m 细化；返回 (best, records, stages)。"""
    def feasible(q, phi_n, r_step, tol):
        if abs(offset_from_bearing(s1, theta1, q)) <= ERROR_DEG + 1e-9:
            return False
        if bind_target and math.hypot(q[0], q[1]) > TARGET_R + 1e-9:
            return False
        slack, _ = reception_margin(s1, theta1, q, phi_n=phi_n, r_step=r_step)
        return slack <= tol

    def j_at(q, phi_n, r_step, err_n, coarse):
        if feasible(q, 5, 100.0, 0.5):
            return worst_diameter(s1, theta1, q, phi_n, r_step, err_n, coarse)[0]
        return math.nan

    # 阶段 1：粗网格（同时保留给图和敏感性表）
    xs, ys = _grid_points(s1, coarse_half, coarse_step)
    records = []  # (x, y, J)
    best = None
    for y in ys:
        for x in xs:
            q = (x, y)
            j = j_at(q, 9, 0.0, 3, coarse=True)  # 近/中/远三点源
            if math.isnan(j):
                continue
            records.append((x, y, j))
            if best is None or j < best[0]:
                best = (j, x, y)
    if best is None:
        raise ValueError("可行域为空")

    stages = [(coarse_step, coarse_half, best)]
    for step, half in ((5.0, 62.5), (1.0, 12.0)):
        j0, x0, y0 = best
        xs2, ys2 = _grid_points((x0, y0), half, step)
        best2 = None
        for y in ys2:
            for x in xs2:
                q = (x, y)
                j = j_at(q, 5, 50.0, 9, coarse=False)
                if math.isnan(j):
                    continue
                if best2 is None or j < best2[0]:
                    best2 = (j, x, y)
        if best2 is not None and best2[0] < best[0] - 1e-9:
            best = best2
        stages.append((step, half, best))
        if verbose:
            print(f"  refine step={step} -> J={best[0]:.3f} at ({best[1]},{best[2]})")
    return best, records, stages
