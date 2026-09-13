#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 高教社杯全国大学生数学建模竞赛 B题 问题1
重新生成：
  ① 测试用例表 + 精简图像（3 个子图，含 120° 局部放大）
  ② 已知形状验证表 + 精简图像（3 个子图）

几何算法为修正后的版本：
  - 半平面交：边界线交点枚举 + 凸包 + 衰退锥判定（正确处理 空/有界/无界，不加入工方框）
  - 最小包围圆：Welzl 初始化已修正（不再出现负半径）
  - 凹角判定、退化点包含判定均已修正

本脚本自包含所需几何函数，不依赖外部模块文件。

运行：python 问题一_表格与图像.py
输出：测试用例表.md / 测试用例图.png
      已知形状验证表.md / 已知形状验证图.png
"""

import os
import math
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 一、基础几何（修正版，与「问题一代码final_修正版.py」一致）
# =====================================================================

EPS = 1e-9
DEG = math.pi / 180.0
ERR_DEG = 1.0


def sgn(x):
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


def half_plane_intersection(hps):
    """求半平面（左侧可行）交集。返回 (status, polygon)。"""
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


def has_reflex_angle(vertices):
    if len(vertices) < 3:
        return False
    return any(
        cross(sub(vertices[(i - 1) % len(vertices)], vertices[i]),
              sub(vertices[(i + 1) % len(vertices)], vertices[i])) > EPS
        for i in range(len(vertices))
    )


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
    """主流程：返回 status（bounded/unbounded/empty）+ 直径 + 覆盖判定。"""
    assert len(detect_points) == len(directions_deg)
    hps = []
    for S, th in zip(detect_points, directions_deg):
        hps.extend(detection_to_halfplanes(S, th, err_deg))

    status, poly = half_plane_intersection(hps)
    if status == "empty":
        return {"status": "empty", "message": "定位区域为空（数据矛盾）"}
    if status == "unbounded":
        return {"status": "unbounded", "polygon": [], "diameter": "Infinity",
                "message": "区域无界（定位信息不足以封闭区域）"}

    check = circle_coverage_check(poly)
    check["polygon"] = poly
    check["n_vertices"] = len(poly)
    check["has_reflex_angle"] = has_reflex_angle(poly)
    check["status"] = "bounded"
    check["message"] = "ok"
    return check


# =====================================================================
# 二、工具
# =====================================================================

def dir_toward(S, G):
    return math.degrees(math.atan2(G[1] - S[1], G[0] - S[0])) % 360.0


def dirs_toward(points, source):
    return [dir_toward(S, source) for S in points]


def regular_polygon(n, R, rot=0.0):
    return [(R * math.cos(rot + 2 * math.pi * k / n),
             R * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def fmt(x):
    return f"{x:.4f}"


def coord(v):
    """坐标显示：保留至 4 位小数并去掉末尾多余的 0（避免 1299.0381 被显示成 1299.04）。"""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-", "-0") else "0"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


# =====================================================================
# 三、绘图
# =====================================================================

def draw_region(ax, poly, A, B, mc, R, dc, dr, title, detect_pts=None,
                xlim=None, ylim=None):
    """绘制定位区域 + 直径 + 最小包围圆 + 直径圆。"""
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.fill(xs, ys, alpha=0.20, color="tab:blue")
    ax.plot(xs, ys, color="tab:blue", lw=1.6)
    if A is not None and B is not None:
        ax.plot([A[0], B[0]], [A[1], B[1]], "k-", lw=2.2)
    ax.add_patch(plt.Circle(mc, R, fill=False, ls="--", color="tab:orange", lw=1.8))
    ax.add_patch(plt.Circle(dc, dr, fill=False, ls=":", color="tab:green", lw=1.8))
    if detect_pts:
        for (px, py) in detect_pts:
            ax.plot(px, py, "o", color="red", ms=6, zorder=5)
    if xlim is not None and ylim is not None:
        # 手动指定范围时固定画框，再保证等比例（局部放大用）
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, lw=0.4)


def shared_legend(fig, with_detect=False):
    handles = [
        Patch(facecolor="tab:blue", alpha=0.4, edgecolor="tab:blue", label="定位区域"),
        Line2D([0], [0], color="black", lw=2.2, label="直径 D"),
        Line2D([0], [0], color="tab:orange", lw=1.8, ls="--", label="最小包围圆"),
        Line2D([0], [0], color="tab:green", lw=1.8, ls=":", label="直径圆 (半径 D/2)"),
    ]
    if with_detect:
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                              markeredgecolor="red", ms=7, lw=0, label="检测点"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False)


# =====================================================================
# 四、第一部分：测试用例表 + 精简图像（3 个子图）
# =====================================================================

def build_test_cases():
    SRC = (0.0, 0.0)
    return [
        ("两检测点：东西对望", [(-1000.0, 0.0), (1000.0, 0.0)]),
        ("两检测点：南北对望", [(0.0, -1000.0), (0.0, 1000.0)]),
        ("两检测点：直角(东+北)", [(1000.0, 0.0), (0.0, 1000.0)]),
        ("两检测点：同侧(无界)", [(500.0, 0.0), (1500.0, 0.0)]),
        ("三检测点：120° 环绕", [(1500.0, 0.0), (-750.0, 1299.0381), (-750.0, -1299.0381)]),
        ("三检测点：共线环绕", [(-1000.0, 0.0), (500.0, 0.0), (1500.0, 0.0)]),
        ("三检测点：东/北/西", [(800.0, 0.0), (0.0, 800.0), (-800.0, 0.0)]),
    ]


def test_case_table_and_image():
    SRC = (0.0, 0.0)
    random.seed(0)
    cases = build_test_cases()
    rows, results = [], []
    for i, (name, pts) in enumerate(cases, 1):
        r = localize(pts, dirs_toward(pts, SRC))
        results.append((name, pts, r))
        coords = "，".join(f"({coord(p[0])},{coord(p[1])})" for p in pts)
        if r["status"] != "bounded":
            rows.append([i, name, coords, "—", "—", "—", "—",
                         "无界（不适用）"])
            continue
        D, R = r["diameter"], r["mec_radius"]
        rows.append([i, name, coords, f"{r['n_vertices']}",
                     fmt(D), fmt(R), fmt(D / 2.0),
                     "✅ 覆盖" if r["covered_by_D_circle"] else "❌ 不覆盖"])

    headers = ["编号", "检测点配置", "检测点坐标 (m)", "顶点数",
               "直径 D (m)", "R_MEC (m)", "D/2 (m)", "直径圆是否覆盖"]
    table = "# 测试用例表（交会定位法定位区域 + 直径 + 圆覆盖判定）\n\n" \
            "示向度为各检测点指向原点（真实源）的方向；误差 ±1°。\n\n" \
            + markdown_table(headers, rows)
    with open(os.path.join(_HERE, "测试用例表.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(table)

    # 图像：东西对望（覆盖）、120° 环绕（不覆盖）+ 其局部放大
    keep = {"两检测点：东西对望", "三检测点：120° 环绕"}
    plot = {name: r for name, pts, r in results if name in keep}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # 子图1：东西对望（覆盖）
    name = "两检测点：东西对望"
    pts = [(-1000.0, 0.0), (1000.0, 0.0)]
    r = plot[name]
    A, B = r["diameter_endpoints"]
    mc, R = r["mec_center"], r["mec_radius"]
    dc, dr = circle_from_two(A, B)
    draw_region(axes[0], r["polygon"], A, B, mc, R, dc, dr,
                f"{name}\n(D={r['diameter']:.1f} m, R={R:.1f} m，可覆盖)", detect_pts=pts)

    # 子图2：120° 环绕（全局，区域在原点附近很小）
    name = "三检测点：120° 环绕"
    pts = [(1500.0, 0.0), (-750.0, 1299.0381), (-750.0, -1299.0381)]
    r = plot[name]
    A, B = r["diameter_endpoints"]
    mc, R = r["mec_center"], r["mec_radius"]
    dc, dr = circle_from_two(A, B)
    draw_region(axes[1], r["polygon"], A, B, mc, R, dc, dr,
                f"{name}（全局）\n(D={r['diameter']:.1f} m, R={R:.1f} m，不可覆盖)",
                detect_pts=pts)
    axes[1].annotate("定位区域在原点附近\n（见右图放大）", xy=(0, 0), xytext=(500, -900),
                     fontsize=9, ha="center",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))

    # 子图3：120° 环绕（局部放大）
    xs = [p[0] for p in r["polygon"]]
    ys = [p[1] for p in r["polygon"]]
    padx = (max(xs) - min(xs)) * 0.6 + 5
    pady = (max(ys) - min(ys)) * 0.6 + 5
    draw_region(axes[2], r["polygon"], A, B, mc, R, dc, dr,
                f"{name}（局部放大）\n(D={r['diameter']:.2f} m, R={R:.2f} m)",
                xlim=(min(xs) - padx, max(xs) + padx),
                ylim=(min(ys) - pady, max(ys) + pady))

    shared_legend(fig, with_detect=True)
    fig.suptitle("交会定位法：定位区域 + 直径 + 圆覆盖判定（左：可覆盖；中右：120° 环绕不可覆盖及其放大）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(os.path.join(_HERE, "测试用例图.png"), dpi=150)
    plt.close(fig)
    print("\n已保存：测试用例图.png（3 个子图：覆盖 / 全局不可覆盖 / 局部放大）")


# =====================================================================
# 五、第二部分：已知形状验证表 + 精简图像（3 个子图）
# =====================================================================

def build_known_shapes():
    h = math.sqrt(3) / 2
    return [
        ("等边三角形(边 a=10)", [(0, 0), (10, 0), (5, 10 * h)], 10.0, 10.0 / math.sqrt(3)),
        ("正方形(边 a=8)", [(0, 0), (8, 0), (8, 8), (0, 8)], 8 * math.sqrt(2), 8 * math.sqrt(2) / 2),
        ("长方形(6×4)", [(0, 0), (6, 0), (6, 4), (0, 4)], math.hypot(6, 4), math.hypot(6, 4) / 2),
        ("直角三角形(3-4-5)", [(0, 0), (3, 0), (0, 4)], 5.0, 2.5),
        ("正五边形(外接圆 R=5)", regular_polygon(5, 5, -math.pi / 2), 10 * math.sin(2 * math.pi / 5), 5.0),
        ("正六边形(外接圆 R=5)", regular_polygon(6, 5), 10.0, 5.0),
        ("正36边形(外接圆 R=5)", regular_polygon(36, 5), 10.0, 5.0),
    ]


def known_shape_table_and_image():
    random.seed(0)
    shapes = build_known_shapes()
    rows, results = [], []
    for name, poly, D_theory, R_theory in shapes:
        D_calc, A, B = polygon_diameter(poly)
        mc, R_calc = min_enclosing_circle(poly)
        results.append((name, poly, A, B, mc, R_calc, D_calc))
        errD = abs(D_calc - D_theory)
        errR = abs(R_calc - R_theory)
        covered = "✅ 覆盖" if R_calc <= D_calc / 2 + 1e-9 else "❌ 不覆盖"
        rows.append([name, f"{len(poly)}", fmt(D_theory), fmt(D_calc),
                     "✓" if errD < 1e-6 else f"{errD:.2e}",
                     fmt(R_theory), fmt(R_calc),
                     "✓" if errR < 1e-6 else f"{errR:.2e}", covered])

    headers = ["形状", "顶点数", "理论直径 D", "计算直径 D", "D 误差",
               "理论 R_MEC", "计算 R_MEC", "R 误差", "直径圆是否覆盖"]
    table = "# 已知形状验证表（验证直径与最小包围圆算法的正确性）\n\n" \
            "理论值由解析几何给出；误差列 ✓ 表示与理论值一致（<1e-6）。\n\n" \
            + markdown_table(headers, rows)
    with open(os.path.join(_HERE, "已知形状验证表.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(table)

    # 图像：只保留 3 个最有代表性的形状
    keep = {"等边三角形(边 a=10)", "直角三角形(3-4-5)", "正方形(边 a=8)"}
    plot = [r for r in results if r[0] in keep]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (name, poly, A, B, mc, R_calc, D_calc) in zip(axes, plot):
        dc, dr = circle_from_two(A, B)
        covered = "可覆盖" if R_calc <= D_calc / 2 + 1e-9 else "不可覆盖"
        draw_region(ax, poly, A, B, mc, R_calc, dc, dr,
                    f"{name}\n(D={D_calc:.2f}, R={R_calc:.2f}，{covered})")
    shared_legend(fig, with_detect=False)
    fig.suptitle("已知形状：直径 D 与最小包围圆（等边三角形不可覆盖；直角三角形、正方形可覆盖）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(os.path.join(_HERE, "已知形状验证图.png"), dpi=150)
    plt.close(fig)
    print("\n已保存：已知形状验证图.png（3 个子图）")


# =====================================================================
# 六、主程序
# =====================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("第一部分：测试用例表 + 精简图像")
    print("=" * 78)
    test_case_table_and_image()

    print()
    print("=" * 78)
    print("第二部分：已知形状验证表 + 精简图像")
    print("=" * 78)
    known_shape_table_and_image()
