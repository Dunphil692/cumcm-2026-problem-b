#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 高教社杯全国大学生数学建模竞赛 B题 问题1
无线电干扰源的快速自动定位与清除 —— 最终融合版

融合内容：
  1. 角形区域 → 半平面（点+方向）
  2. 半平面交（边界线交点枚举 + 凸包，正确处理空/有界/无界）
  3. 有界/无界/空 状态判定（参考 geometry 版）
  4. 旋转卡壳求直径（solution 版，O(m)）
  5. Welzl 最小包围圆（solution 版，期望 O(m)）
  6. 圆覆盖判定：R_MEC ≤ D/2（solution 版，更严谨）
  7. 方法4验证：真实干扰源是否在定位区域内（solution 版）

运行：
  python problem1_final.py
"""

from __future__ import annotations

import json
import math
import random


# =====================================================================
# 基础几何
# =====================================================================

EPS = 1e-9
DEG = math.pi / 180.0
ERR_DEG = 1.0


def sgn(x: float) -> int:
    if x > EPS:
        return 1
    if x < -EPS:
        return -1
    return 0


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(a, t):
    return (a[0] * t, a[1] * t)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_of(v):
    return math.atan2(v[1], v[0])


def unit_vec(deg):
    r = deg * DEG
    return (math.cos(r), math.sin(r))


# =====================================================================
# 一、角形区域 → 半平面
# =====================================================================

def detection_to_halfplanes(S, theta_deg, err_deg=ERR_DEG):
    """检测点 S + 示向度 theta ± err → 两条半平面（点+方向，左侧可行）。"""
    lo = theta_deg - err_deg
    hi = theta_deg + err_deg
    u_lo = unit_vec(lo)
    u_hi = unit_vec(hi)
    return [(S, u_lo), (S, mul(u_hi, -1.0))]


def intersect_line(h1, h2):
    p1, v1 = h1
    p2, v2 = h2
    d = cross(v1, v2)
    if abs(d) < EPS:
        return None
    t = cross(sub(p2, p1), v2) / d
    return add(p1, mul(v1, t))


def on_left(h, q):
    p, v = h
    return sgn(cross(v, sub(q, p))) >= 0


# =====================================================================
# 二、半平面交（边界线交点枚举 + 凸包 + 衰退锥判定，正确处理空/有界/无界）
# =====================================================================

def halfplane_normal(h):
    """半平面 (p, v)（左侧可行）的内法向量 n = (-v.y, v.x)，满足 n·x >= n·p。"""
    p, v = h
    return (-v[1], v[0])


def _dedup_points(pts, eps=1e-7):
    out = []
    for q in pts:
        if all(dist(q, r) > eps for r in out):
            out.append(q)
    return out


def _normals_span_plane(hps):
    """有界性判定：内法向量若能包围原点（最大角间隔 < 180°），交集必有界。"""
    ang = sorted((angle_of(halfplane_normal(h)) % (2.0 * math.pi)) for h in hps)
    max_gap = 0.0
    for i in range(len(ang)):
        gap = (ang[(i + 1) % len(ang)] - ang[i]) % (2.0 * math.pi)
        if gap > max_gap:
            max_gap = gap
    return max_gap < math.pi - EPS


def _parallel_feasible(hps):
    """所有半平面法向量平行时的可行性判定（退化：条带/半平面/全平面/空）。"""
    n0 = halfplane_normal(hps[0])
    L = math.hypot(n0[0], n0[1])
    if L == 0.0:
        return True
    d = (n0[0] / L, n0[1] / L)
    lower = -math.inf
    upper = math.inf
    for h in hps:
        n = halfplane_normal(h)
        c = n[0] * h[0][0] + n[1] * h[0][1]
        nl = math.hypot(n[0], n[1])
        if nl == 0.0:
            continue
        dot = (n[0] * d[0] + n[1] * d[1]) / nl
        if dot > 0:
            lower = max(lower, c / nl)
        else:
            upper = min(upper, -c / nl)
    return lower <= upper + EPS


def half_plane_intersection(hps):
    """求半平面（左侧可行）交集。返回 (status, polygon)：
    status ∈ {'empty', 'bounded', 'unbounded'}；bounded 时 polygon 为逆时针顶点表。
    用边界线交点枚举求可行顶点，再由凸包得到区域；有界性由内法向量是否包围原点判定。
    """
    if not hps:
        return "unbounded", []

    pts = []
    has_nonparallel = False
    n = len(hps)
    for i in range(n):
        for j in range(i + 1, n):
            q = intersect_line(hps[i], hps[j])
            if q is None:
                continue
            has_nonparallel = True
            if all(on_left(h, q) for h in hps):
                pts.append(q)

    if not has_nonparallel:
        return ("unbounded", []) if _parallel_feasible(hps) else ("empty", [])

    if not pts:
        return "empty", []

    uniq = _dedup_points(pts)
    if _normals_span_plane(hps):
        return "bounded", convex_hull(uniq)
    return "unbounded", []


# =====================================================================
# 三、凸多边形直径（旋转卡壳，O(m)）
# =====================================================================

def polygon_diameter(poly):
    n = len(poly)
    if n == 0:
        return 0.0, None, None
    if n == 1:
        return 0.0, poly[0], poly[0]
    if n == 2:
        return dist(poly[0], poly[1]), poly[0], poly[1]

    j = 1
    best = 0.0
    A = B = poly[0]
    for i in range(n):
        ni = (i + 1) % n
        edge = sub(poly[ni], poly[i])
        while (abs(cross(edge, sub(poly[(j + 1) % n], poly[i]))) >
               abs(cross(edge, sub(poly[j], poly[i])))):
            j = (j + 1) % n
        for k in (i, ni):
            d = dist2(poly[k], poly[j])
            if d > best:
                best = d
                A, B = poly[k], poly[j]
    return math.sqrt(best), A, B


def polygon_diameter_bruteforce(poly):
    n = len(poly)
    best = 0.0
    A = B = None
    for i in range(n):
        for j in range(i + 1, n):
            d = dist2(poly[i], poly[j])
            if d > best:
                best = d
                A, B = poly[i], poly[j]
    return math.sqrt(best), A, B


# =====================================================================
# 四、最小包围圆（Welzl，期望 O(m)）
# =====================================================================

def circle_from_two(a, b):
    c = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    r = dist(a, b) / 2.0
    return c, r


def circumcircle(a, b, c):
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < EPS:
        pairs = [(a, b), (a, c), (b, c)]
        A, B = max(pairs, key=lambda p: dist2(p[0], p[1]))
        return circle_from_two(A, B)
    ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) +
          (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1]) +
          (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
    uy = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) +
          (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0]) +
          (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d
    r = dist((ux, uy), a)
    return (ux, uy), r


def min_enclosing_circle(points):
    pts = list(points)
    if not pts:
        return (0.0, 0.0), 0.0
    random.shuffle(pts)
    c = (pts[0][0], pts[0][1])
    r = 0.0
    for i in range(1, len(pts)):
        p = pts[i]
        if dist2(p, c) > r * r + EPS:
            c = (p[0], p[1])
            r = 0.0
            for j in range(i):
                if dist2(pts[j], c) > r * r + EPS:
                    c, r = circle_from_two(p, pts[j])
                    for k in range(j):
                        if dist2(pts[k], c) > r * r + EPS:
                            c, r = circumcircle(p, pts[j], pts[k])
    return c, r


def min_enclosing_circle_bruteforce(points):
    pts = list(points)
    n = len(pts)
    if n == 0:
        return (0.0, 0.0), 0.0
    if n == 1:
        return pts[0], 0.0
    if n == 2:
        return circle_from_two(pts[0], pts[1])
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            c, r = circle_from_two(pts[i], pts[j])
            if all(dist2(p, c) <= r * r + EPS for p in pts):
                if best is None or r < best[1]:
                    best = (c, r)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c, r = circumcircle(pts[i], pts[j], pts[k])
                if all(dist2(p, c) <= r * r + EPS for p in pts):
                    if best is None or r < best[1]:
                        best = (c, r)
    return best


# =====================================================================
# 五、状态判定（有界/无界/空）
# =====================================================================

def has_reflex_angle(vertices):
    if len(vertices) < 3:
        return False
    return any(
        cross(sub(vertices[(i - 1) % len(vertices)], vertices[i]),
              sub(vertices[(i + 1) % len(vertices)], vertices[i])) > EPS
        for i in range(len(vertices))
    )


# =====================================================================
# 六、圆覆盖判定 + 主流程
# =====================================================================

def circle_coverage_check(poly):
    D, A, B = polygon_diameter(poly)
    dc, dr = circle_from_two(A, B) if (A and B) else ((0, 0), 0)
    diameter_circle_covers = all(dist2(p, dc) <= dr * dr + EPS for p in poly)
    mc, R = min_enclosing_circle(poly)
    covered_by_D_circle = (R <= D / 2.0 + EPS)
    return {
        "diameter": D,
        "diameter_endpoints": (A, B),
        "diameter_circle_covers": diameter_circle_covers,
        "mec_center": mc,
        "mec_radius": R,
        "D_over_2": D / 2.0,
        "covered_by_D_circle": covered_by_D_circle,
    }


def localize(detect_points, directions_deg, err_deg=ERR_DEG):
    """主流程：返回 status（bounded/unbounded/empty）+ 直径 + 覆盖判定。
    只对原角域约束求交，不再额外加入人工方框；无界性由内法向量是否包围原点判定。
    """
    assert len(detect_points) == len(directions_deg)
    hps = []
    for S, th in zip(detect_points, directions_deg):
        hps.extend(detection_to_halfplanes(S, th, err_deg))

    status, poly = half_plane_intersection(hps)
    if status == "empty":
        return {"status": "empty", "message": "定位区域为空（数据矛盾）"}

    if status == "unbounded":
        return {
            "status": "unbounded",
            "polygon": [],
            "diameter": "Infinity",
            "message": "区域无界（定位信息不足以封闭区域）",
        }

    check = circle_coverage_check(poly)
    check["polygon"] = poly
    check["n_vertices"] = len(poly)
    check["has_reflex_angle"] = has_reflex_angle(poly)
    check["status"] = "bounded"
    check["message"] = "ok"
    return check


# =====================================================================
# 七、输出与测试用例
# =====================================================================

def _fmt(v):
    return f"{v:.6f}"


def run_test_case(name, detect_points, directions_deg):
    print("=" * 78)
    print(f"测试用例：{name}")
    print("=" * 78)
    r = localize(detect_points, directions_deg)
    if r["status"] != "bounded":
        print(f"status = {r['status']}，{r.get('message', '')}")
        return r
    poly = r["polygon"]
    D, A, B = r["diameter"], r["diameter_endpoints"][0], r["diameter_endpoints"][1]
    mc, R = r["mec_center"], r["mec_radius"]
    print(f"定位区域顶点数：{r['n_vertices']}")
    print(f"定位区域顶点（逆时针）：")
    for p in poly:
        print(f"    ({_fmt(p[0])}, {_fmt(p[1])})")
    print(f"区域直径 D          = {_fmt(D)} m")
    print(f"直径端点            = ({_fmt(A[0])},{_fmt(A[1])}) 与 ({_fmt(B[0])},{_fmt(B[1])})")
    print(f"D/2                 = {_fmt(D / 2.0)} m")
    print(f"最小包围圆圆心      = ({_fmt(mc[0])},{_fmt(mc[1])})，半径 R = {_fmt(R)} m")
    print(f"以直径线段为直径的圆覆盖区域？ {r['diameter_circle_covers']}")
    print(f"存在直径为 D 的圆覆盖区域？   {r['covered_by_D_circle']}")
    print(f"有凸角（非凸）？               {r['has_reflex_angle']}")
    return r


def test_polygon_direct(name, poly, expected_covered):
    print("=" * 78)
    print(f"测试用例（直接给定多边形）：{name}")
    print("=" * 78)
    check = circle_coverage_check(poly)
    D = check["diameter"]
    mc, R = check["mec_center"], check["mec_radius"]
    print(f"顶点：{[(round(x, 4), round(y, 4)) for x, y in poly]}")
    print(f"直径 D = {_fmt(D)} m，D/2 = {_fmt(D/2)} m")
    print(f"最小包围圆半径 R = {_fmt(R)} m")
    print(f"以直径线段为直径的圆覆盖？ {check['diameter_circle_covers']}")
    print(f"存在直径为 D 的圆覆盖？   {check['covered_by_D_circle']}")
    print(f"期望：{expected_covered}")
    assert check["covered_by_D_circle"] == expected_covered, "判定与期望不符！"
    return check


def convex_hull(points):
    pts = sorted(set((round(x, 9), round(y, 9)) for x, y in points))
    if len(pts) <= 1:
        return pts

    def ccw(o, a, b):
        return sgn(cross(sub(a, o), sub(b, o)))

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


def test_consistency():
    random.seed(0)
    for _ in range(2000):
        n = random.randint(3, 12)
        pts = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(n)]
        hull = convex_hull(pts)
        if len(hull) < 3:
            continue
        d1 = polygon_diameter(hull)[0]
        d2 = polygon_diameter_bruteforce(hull)[0]
        assert abs(d1 - d2) < 1e-6, f"直径不一致 {d1} vs {d2}"
        m1 = min_enclosing_circle(hull)
        m2 = min_enclosing_circle_bruteforce(hull)
        assert abs(m1[1] - m2[1]) < 1e-6, f"最小包围圆不一致 {m1[1]} vs {m2[1]}"
    print("交叉验证通过：旋转卡壳直径 = 暴力直径，Welzl 最小包围圆 = 暴力最小包围圆（2000 组随机凸多边形）")


# =====================================================================
# 八、方法4验证：真实干扰源是否在定位区域内
# =====================================================================

def point_in_convex_polygon(P, poly, eps=1e-9):
    """判定点 P 是否在凸多边形（逆时针）内或边上；空/单点/线段退化情形分别处理。"""
    n = len(poly)
    if n == 0:
        return False
    if n == 1:
        return dist(P, poly[0]) <= eps
    if n == 2:
        A, B = poly
        AB = sub(B, A)
        AP = sub(P, A)
        if abs(cross(AB, AP)) > eps:
            return False
        dot = AB[0] * AP[0] + AB[1] * AP[1]
        return -eps <= dot <= dist2(A, B) + eps
    for i in range(n):
        A = poly[i]
        B = poly[(i + 1) % n]
        ex, ey = B[0] - A[0], B[1] - A[1]
        px, py = P[0] - A[0], P[1] - A[1]
        if ex * py - ey * px < -eps:
            return False
    return True


def true_bearing(S, G):
    return math.degrees(math.atan2(G[1] - S[1], G[0] - S[0])) % 360.0


def single_trial_method4(k_detect=3, region_radius=1800.0,
                         min_dist=50.0, max_dist=1400.0,
                         verbose=False):
    rG = region_radius * math.sqrt(random.random())
    aG = random.uniform(0, 2 * math.pi)
    G = (rG * math.cos(aG), rG * math.sin(aG))

    detect_points, directions = [], []
    tries = 0
    while len(detect_points) < k_detect and tries < 300:
        tries += 1
        rS = region_radius * math.sqrt(random.random())
        aS = random.uniform(0, 2 * math.pi)
        S = (rS * math.cos(aS), rS * math.sin(aS))
        d = math.hypot(G[0] - S[0], G[1] - S[1])
        if d < min_dist or d > max_dist:
            continue
        theta_true = true_bearing(S, G)
        err = random.uniform(-1.0, 1.0)
        directions.append((theta_true + err) % 360.0)
        detect_points.append(S)

    if len(detect_points) < 2:
        return None

    result = localize(detect_points, directions)
    if result["status"] != "bounded":
        return None
    poly = result["polygon"]
    inside = point_in_convex_polygon(G, poly)

    if verbose:
        print(f"  G = ({G[0]:.2f}, {G[1]:.2f})")
        print(f"  检测点数: {len(detect_points)}")
        print(f"  定位区域顶点数: {len(poly)}")
        print(f"  直径 D = {result['diameter']:.4f}")
        print(f"  G 在定位区域内？ {inside}")

    return inside, G, poly, directions


def run_method4(n_trials=10000, k_detect=3, seed=42):
    random.seed(seed)
    pass_count = fail_count = skip_count = 0
    fail_cases = []
    for _ in range(n_trials):
        res = single_trial_method4(k_detect=k_detect)
        if res is None:
            skip_count += 1
            continue
        inside, G, poly, dirs = res
        if inside:
            pass_count += 1
        else:
            fail_count += 1
            if len(fail_cases) < 5:
                fail_cases.append((G, poly))
    total = pass_count + fail_count
    print("=" * 78)
    print(f"方法4验证结果（共 {n_trials} 次试验）")
    print("=" * 78)
    print(f"有效试验: {total}")
    print(f"通过（G 在区域内）: {pass_count}")
    print(f"失败（G 在区域外）: {fail_count}")
    print(f"跳过: {skip_count}")
    if total > 0:
        print(f"通过率: {pass_count / total * 100:.4f}%")
    if fail_cases:
        print("\n失败案例示例：")
        for G, poly in fail_cases:
            print(f"  G = ({G[0]:.2f}, {G[1]:.2f})，顶点数 {len(poly)}")
    else:
        print("\n✅ 所有有效试验均通过：真实干扰源始终在定位区域内")
    return pass_count, fail_count, skip_count


# =====================================================================
# 九、主程序
# =====================================================================

if __name__ == "__main__":
    print("2026 B题 问题1 求解程序（最终融合版）\n")

    run_test_case(
        "两个检测点东西对望（源在原点附近）",
        detect_points=[(-1000.0, 0.0), (1000.0, 0.0)],
        directions_deg=[0.0, 180.0],
    )

    run_test_case(
        "三个检测点 120° 均匀环绕（区域接近三角形，典型反例形态）",
        detect_points=[(1500.0, 0.0), (-750.0, 1299.04), (-750.0, -1299.04)],
        directions_deg=[180.0, 300.0, 60.0],
    )

    s = 10.0
    h = s * math.sqrt(3) / 2
    test_polygon_direct(
        "反例：等边三角形（边长为直径 D）",
        [(0.0, 0.0), (s, 0.0), (s / 2, h)],
        expected_covered=False,
    )

    test_polygon_direct(
        "正例：直角三角形（斜边为直径，直径圆可覆盖）",
        [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)],
        expected_covered=True,
    )

    print()
    test_consistency()

    print()
    print("【方法4 单次详细试验】")
    random.seed(123)
    single_trial_method4(k_detect=3, verbose=True)

    print()
    print("【方法4 批量试验】")
    run_method4(n_trials=10000, k_detect=3, seed=42)