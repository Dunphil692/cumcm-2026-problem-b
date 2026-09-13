"""2026 国赛 B 题问题 2：第二检测点的硬约束可行域、最坏直径与候选区域。

算法（对应进度笔记 §3 问题 2）：
1. 由 S1 和示向度 theta1 画出 G 可能所在的「缝」：2° 楔形 ∩ 目标圆 ∩ 接收半径上界。
2. 硬约束筛掉不能当 S2 的点：听得见（S1 圆 ∩ 远弧两端两圆）、不站在第一条楔形里；不出目标圆是可选假设。
3. 对每个可行的 S2，沿 ±1°/0° 各取近/中/远假想 G（远端按该方向被目标圆截断），
   用问题 1 修正版 localize 求两楔形交会直径，取最坏值 J(S2)；G 落入 S2 的 5 m 内则 J=+∞。
4. 候选区域 = 可行域中 J(S2) 不超过 factor × J_min 的部分；同时给出 J 最小的推荐点。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem1_localize import localize  # noqa: E402

ERROR_DEG = 1.0
R_MIN = 1000.0  # 有效接收半径下界（m）
R_MAX = 1500.0  # 有效接收半径上界（m）
TARGET_R = 1800.0  # 目标区域半径（m）
NEAR_R = 5.0  # 第一次已拿到示向度，说明 G 至少在 5 m 之外
ERROR_STEPS = (-ERROR_DEG, 0.0, ERROR_DEG)

Point = tuple[float, float]


def _unit(deg: float) -> Point:
    rad = math.radians(deg)
    return (math.cos(rad), math.sin(rad))


def _bearing(origin: Point, target: Point) -> float:
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0])) % 360.0


def _angular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _dist(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _along(start: Point, deg: float, distance: float) -> Point:
    ux, uy = _unit(deg)
    return (start[0] + distance * ux, start[1] + distance * uy)


def ray_exit_distance(start: Point, deg: float, radius: float = TARGET_R) -> float:
    """从 start 沿 deg 方向出发，走多远离开以原点为心、radius 为半径的圆。"""
    ux, uy = _unit(deg)
    b = start[0] * ux + start[1] * uy
    c = start[0] ** 2 + start[1] ** 2 - radius**2
    disc = b * b - c
    if disc < 0.0:
        return 0.0
    return max(-b + math.sqrt(disc), 0.0)


def slit_far_distance(s1: Point, theta1: float) -> float:
    """示向线中轴上 G 离 S1 最远可能多少。一般位置请用 direction_far_distance。"""
    return direction_far_distance(s1, theta1)


def direction_far_distance(s1: Point, deg: float) -> float:
    """该方位上楔形被目标圆和 1500 m 上界截断后的远端。"""
    return min(R_MAX, ray_exit_distance(s1, deg))


def probe_sources(s1: Point, theta1: float) -> list[Point]:
    """算 J 时抽的假想 G：每个误差边界方向各取近 / 中 / 远，远端按该方向裁剪。"""
    points: list[Point] = []
    for delta in ERROR_STEPS:
        bearing = theta1 + delta
        r_far = direction_far_distance(s1, bearing)
        if r_far <= NEAR_R:
            continue
        for r in (NEAR_R, 0.5 * r_far, r_far):
            points.append(_along(s1, bearing, min(r, r_far)))
    return points


def far_arc_anchors(s1: Point, theta1: float) -> tuple[Point, Point]:
    """R=1000 时楔形远弧的左右端点，不是示向线中轴上的单点。"""
    points: list[Point] = []
    for delta in (-ERROR_DEG, ERROR_DEG):
        reach = min(R_MIN, ray_exit_distance(s1, theta1 + delta))
        points.append(_along(s1, theta1 + delta, reach))
    return points[0], points[1]


def is_feasible(s1: Point, theta1: float, s2: Point, bind_target: bool = True) -> bool:
    """问题 2 的硬约束，不满足则 S2 直接淘汰。bind_target=False 时不要求 S2 在 1800 m 圆内。"""
    left, right = far_arc_anchors(s1, theta1)
    # 1. 不知道 G 在缝的哪一段，也不知道接收半径是 1000 还是 1500；
    #    最紧是 R=1000，且远端是 ±1° 远弧两端。S2 须同时落在 S1、左端、右端三个 1000 m 盘内。
    if _dist(s2, s1) > R_MIN or _dist(s2, left) > R_MIN or _dist(s2, right) > R_MIN:
        return False
    # 2. 不出目标区域（建模自加，题目未要求；S1 在原点时非紧）
    if bind_target and math.hypot(s2[0], s2[1]) > TARGET_R:
        return False
    # 3. 不站在第一条 2° 楔形里，否则两次测量几乎同向，交会无界
    if _angular_distance(_bearing(s1, s2), theta1) <= ERROR_DEG:
        return False
    return True


def _two_station_diameter(s1: Point, theta1: float, s2: Point, theta2: float) -> float:
    result = localize([s1, s2], [theta1, theta2])
    if result.get("status") != "bounded":
        return math.inf
    return float(result["diameter"])


def worst_diameter(s1: Point, theta1: float, s2: Point, r_far: float | None = None) -> float:
    """对缝内可能的 G（各方向近/中/远 × 第二次 ±1° 误差）取定位区域直径的最大值。

    第一次读数固定为已知的 theta1，不重新抽样。G 沿 theta1±1°/0° 放置，
    远端按该方向被目标圆截断，不再共用中轴 r_far。若某个假想 G 落在 S2 的 5 m 内，
    第二次没有示向度，记 J=+∞。
    """
    del r_far  # 保留参数以免旧调用崩溃；实际改用逐方向远端
    worst = 0.0
    for g in probe_sources(s1, theta1):
        if _dist(s2, g) <= NEAR_R:
            return math.inf
        true_theta2 = _bearing(s2, g)
        for d2 in ERROR_STEPS:
            diameter = _two_station_diameter(s1, theta1, s2, true_theta2 + d2)
            if not math.isfinite(diameter):
                return math.inf
            worst = max(worst, diameter)
    return worst


def verify_worst_case(
    s1: Point,
    theta1: float,
    s2: Point,
    d_step: float = 25.0,
    phi_step: float = 0.25,
) -> dict[str, object]:
    """对给定 S2 密扫楔形，核对三点采样是否漏掉更坏的直径。"""
    three_point = worst_diameter(s1, theta1, s2)
    worst = 0.0
    worst_at: tuple[float, float, float] | None = None
    n_phi = int(round(2.0 * ERROR_DEG / phi_step))
    for i in range(n_phi + 1):
        phi = -ERROR_DEG + i * (2.0 * ERROR_DEG / n_phi)
        r_far = direction_far_distance(s1, theta1 + phi)
        if r_far <= NEAR_R:
            continue
        n_r = max(1, int(math.floor((r_far - NEAR_R) / d_step)))
        radii = [NEAR_R + k * (r_far - NEAR_R) / n_r for k in range(n_r + 1)]
        for r in radii:
            g = _along(s1, theta1 + phi, r)
            if _dist(s2, g) <= NEAR_R:
                return {
                    "three_point_J_m": three_point,
                    "dense_J_m": math.inf,
                    "worst_distance_m": r,
                    "worst_offset_deg": phi,
                    "worst_second_error_deg": None,
                    "agrees_with_three_point": False,
                }
            true_theta2 = _bearing(s2, g)
            for d2 in ERROR_STEPS:
                diameter = _two_station_diameter(s1, theta1, s2, true_theta2 + d2)
                if not math.isfinite(diameter):
                    diameter = math.inf
                if diameter >= worst:
                    worst = diameter
                    worst_at = (r, phi, d2)
    if worst_at is None:
        raise ValueError("密扫没有合法源点")
    r, phi, d2 = worst_at
    return {
        "three_point_J_m": None if three_point == math.inf else round(three_point, 3),
        "dense_J_m": None if worst == math.inf else round(worst, 3),
        "worst_distance_m": round(r, 3),
        "worst_offset_deg": round(phi, 3),
        "worst_second_error_deg": d2,
        "agrees_with_three_point": math.isfinite(three_point)
        and math.isfinite(worst)
        and abs(worst - three_point) <= 1.0,
    }


def scan(
    s1: Point, theta1: float, step: float, bind_target: bool = True
) -> tuple[list[float], list[float], list[list[float]]]:
    """在 S1 周围 1000 m 方框内逐格计算 J；不可行格记 NaN。"""
    count = int(R_MIN // step) + 1
    xs = [s1[0] + i * step for i in range(-count, count + 1)]
    ys = [s1[1] + i * step for i in range(-count, count + 1)]
    rows: list[list[float]] = []
    for y in ys:
        row: list[float] = []
        for x in xs:
            s2 = (x, y)
            if is_feasible(s1, theta1, s2, bind_target=bind_target):
                row.append(worst_diameter(s1, theta1, s2))
            else:
                row.append(math.nan)
        rows.append(row)
    return xs, ys, rows


def _polar(s1: Point, theta1: float, point: Point) -> tuple[float, float]:
    """返回 (离 S1 的距离, 相对示向线的偏角 -180..180)。"""
    offset = (_bearing(s1, point) - theta1 + 180.0) % 360.0 - 180.0
    return _dist(point, s1), offset


def summarize(
    s1: Point, theta1: float, xs: list[float], ys: list[float], rows: list[list[float]], factor: float
) -> dict[str, object]:
    cells = [
        (value, x, y)
        for y, row in zip(ys, rows)
        for x, value in zip(xs, row)
        if not math.isnan(value) and math.isfinite(value)
    ]
    if not cells:
        raise ValueError("可行域为空，请检查 S1 是否在目标区域内")
    j_min = min(value for value, _, _ in cells)
    tied = [(x, y) for value, x, y in cells if abs(value - j_min) <= 1e-6]
    # 左右对称时取正偏角那侧，便于和正文 (800,600) 对齐；不是第二套最优。
    best_x, best_y = max(tied, key=lambda p: (_polar(s1, theta1, p)[1], p[0]))
    threshold = factor * j_min
    candidates = [(x, y, value) for value, x, y in cells if value <= threshold]

    def side_summary(sign: int) -> dict[str, object]:
        side = [(x, y, v) for x, y, v in candidates if math.copysign(1.0, _polar(s1, theta1, (x, y))[1]) == sign]
        if not side:
            return {"count": 0}
        polar = [_polar(s1, theta1, (x, y)) for x, y, _ in side]
        return {
            "count": len(side),
            "distance_m": [round(min(p[0] for p in polar)), round(max(p[0] for p in polar))],
            "offset_deg": [round(min(abs(p[1]) for p in polar), 1), round(max(abs(p[1]) for p in polar), 1)],
        }

    best_dist, best_offset = _polar(s1, theta1, (best_x, best_y))
    return {
        "s1": list(s1),
        "theta1_deg": theta1,
        "slit_far_distance_m": round(slit_far_distance(s1, theta1)),
        "grid_step_m": round(xs[1] - xs[0]),
        "feasible_cells": len(cells),
        "J_min_m": round(j_min, 1),
        "best_point": {
            "xy": [round(best_x), round(best_y)],
            "distance_from_s1_m": round(best_dist),
            "offset_from_bearing_deg": round(best_offset, 1),
        },
        "threshold_factor": factor,
        "threshold_m": round(threshold, 1),
        "candidate_cells": len(candidates),
        "candidate_left_side": side_summary(+1),
        "candidate_right_side": side_summary(-1),
    }


def _pick_cjk_font() -> str | None:
    from matplotlib import font_manager

    for name in (
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return None


def plot(
    s1: Point,
    theta1: float,
    xs: list[float],
    ys: list[float],
    rows: list[list[float]],
    summary: dict[str, object],
    output: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    font = _pick_cjk_font()
    zh = font is not None
    if zh:
        plt.rcParams["font.family"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    j_min = float(summary["J_min_m"])
    threshold = float(summary["threshold_m"])
    cap = 3.0 * j_min
    X, Y = np.meshgrid(np.array(xs), np.array(ys))
    Z = np.array(rows, dtype=float)
    Z_plot = np.ma.masked_invalid(np.where(np.isinf(Z), cap, Z))
    Z_contour = np.where(np.isnan(Z) | np.isinf(Z), cap * 10, Z)

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    mesh = ax.pcolormesh(X, Y, np.minimum(Z_plot, cap), cmap="viridis_r", shading="nearest", vmin=j_min, vmax=cap)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cbar.set_label(
        f"最坏定位区域直径 J (m)，≥{cap:.0f} m 截断显示" if zh else f"Worst-case localization diameter J (m), capped at {cap:.0f} m"
    )
    extra = [round(j_min * f, 1) for f in (1.1, 1.5) if abs(f - float(summary["threshold_factor"])) > 1e-6]
    levels = sorted(set(extra + [threshold]))
    contours = ax.contour(X, Y, Z_contour, levels=levels, colors=("#f59e0b", "#d62728", "#7c3aed")[: len(levels)], linewidths=1.6)
    ax.clabel(contours, inline=True, fmt=lambda v: f"{v:.0f} m", fontsize=8)

    r_far = slit_far_distance(s1, theta1)
    far = _along(s1, theta1, r_far)
    ax.plot([s1[0], far[0]], [s1[1], far[1]], color="#1f4fd8", lw=2.0)
    for d in (-ERROR_DEG, ERROR_DEG):
        edge = _along(s1, theta1 + d, r_far)
        ax.plot([s1[0], edge[0]], [s1[1], edge[1]], color="#1f4fd8", lw=1.0, ls="--")

    left, right = far_arc_anchors(s1, theta1)
    for center in (s1, left, right):
        ax.add_patch(plt.Circle(center, R_MIN, fill=False, ec="#555555", lw=1.0, ls=":"))
    ax.add_patch(plt.Circle((0.0, 0.0), TARGET_R, fill=False, ec="#999999", lw=1.0))

    ax.plot(*s1, "o", color="#1f4fd8", ms=9, zorder=5)
    ax.annotate("S1" if zh else "S1", s1, xytext=(8, 8), textcoords="offset points", color="#1f4fd8", fontsize=12, weight="bold")
    best = summary["best_point"]["xy"]  # type: ignore[index]
    for sign in (1, -1):
        bx, by = best
        rel = (bx - s1[0], by - s1[1])
        if sign == -1:
            # 镜像到示向线另一侧
            ux, uy = _unit(theta1)
            along = rel[0] * ux + rel[1] * uy
            perp = -rel[0] * uy + rel[1] * ux
            rel = (along * ux - (-perp) * uy, along * uy + (-perp) * ux)
        px, py = s1[0] + rel[0], s1[1] + rel[1]
        ax.plot(px, py, "*", color="#d62728", ms=16, mec="white", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(xs[0] - 50, xs[-1] + 50)
    ax.set_ylim(ys[0] - 50, ys[-1] + 50)
    if zh:
        ax.set_xlabel("x（m，正东）")
        ax.set_ylabel("y（m，正北）")
        ax.set_title(
            f"问题 2：第二检测点候选区域（S1={tuple(round(v) for v in s1)}，示向度 {theta1:g}°）\n"
            f"彩色 = 三条硬约束可行点；等值线 = 1.1/1.25/1.5×J_min；★ = J 最小 {j_min:.0f} m",
            fontsize=12,
        )
        caption = (
            f"蓝实线/虚线：第一次测得的示向线及 ±1° 边界（缝远端 {r_far:.0f} m）。点线圆：S1 与远弧两端的 1000 m 盘。\n"
            f"假设：误差 ±1°，接收半径 1000–1500 m 未知，G 可在缝内任意位置，两次误差各取 −1°/0/+1° 的最坏组合。"
        )
    else:
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
        ax.set_title(
            f"Problem 2: candidate region for S2 (S1={tuple(round(v) for v in s1)}, bearing {theta1:g} deg)\n"
            f"Colored = feasible under 3 hard constraints; inside red: J <= {threshold:.0f} m = {summary['threshold_factor']} x J_min; star = J_min {j_min:.0f} m",
            fontsize=12,
        )
        caption = (
            f"Blue solid/dashed: first bearing and +/-1 deg edges (slit far end {r_far:.0f} m). Dotted: 1000 m disks at S1 and far-arc ends.\n"
            "Assumptions: +/-1 deg error, receive radius 1000-1500 m unknown, G anywhere in slit, worst of -1/0/+1 deg errors on both readings."
        )
    fig.text(0.02, 0.01, caption, fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="B 题问题 2：第二检测点候选区域")
    parser.add_argument("--x1", type=float, default=0.0, help="S1 的 x（m）")
    parser.add_argument("--y1", type=float, default=0.0, help="S1 的 y（m）")
    parser.add_argument("--theta1", type=float, default=0.0, help="S1 处测得的示向度（度）")
    parser.add_argument("--step", type=float, default=25.0, help="网格步长（m）")
    parser.add_argument("--factor", type=float, default=1.25, help="候选阈值 = factor × J_min")
    parser.add_argument("--out", default=None, help="输出图片路径；缺省写到 ../_figs/")
    parser.add_argument("--no-plot", action="store_true", help="只打印摘要，不画图")
    parser.add_argument("--verify", action="store_true", help="对推荐点密扫，核对最坏是否在远弧两端")
    parser.add_argument("--no-target-bound", action="store_true", help="关闭「S2 必须在 1800 m 目标圆内」")
    args = parser.parse_args()

    s1 = (args.x1, args.y1)
    bind_target = not args.no_target_bound
    xs, ys, rows = scan(s1, args.theta1, args.step, bind_target=bind_target)
    summary = summarize(s1, args.theta1, xs, ys, rows, args.factor)
    summary["bind_s2_to_target_circle"] = bind_target
    if args.verify:
        best = summary["best_point"]["xy"]  # type: ignore[index]
        summary["verify_worst_case"] = verify_worst_case(s1, args.theta1, (float(best[0]), float(best[1])))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.no_plot:
        output = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "_figs", "problem2_candidate_region.png"
        )
        plot(s1, args.theta1, xs, ys, rows, summary, output)
        print(f"图已保存：{os.path.abspath(output)}")


if __name__ == "__main__":
    main()
