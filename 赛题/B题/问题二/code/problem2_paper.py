"""问题 2 融合修正版 CLI：三场景优化、验证、出图与结果汇总。

用法（在 code/ 目录或项目根执行）：
  python3 problem2_paper.py --scenario all [--plot] [--outdir ../results] [--figdir ../figs]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem2_model import (  # noqa: E402
    ERROR_DEG, R_MIN, R_MAX, TARGET_R, NEAR_R,
    _dist, _along, _bearing,
    offset_from_bearing, direction_far_distance,
    reception_margin, worst_diameter, optimize, three_circle_anchors,
)

# 三场景（与队友独立验证包共同口径）
SCENARIOS = {
    "center": dict(s1=(0.0, 0.0), theta1=0.0, name="center_reference"),
    "outward800": dict(s1=(-1000.0, 0.0), theta1=180.0, name="inside_outward_800"),
    "tangent": dict(s1=(1790.0, 0.0), theta1=90.0, name="tangent_boundary"),
}

# 队友独立实现（q2_validation，共同口径统一复评 4096 源/129 误差/0.01 m）的最优直径点，
# 用于交叉验证。我们的 J 在其点上独立重算，应与其报告值一致。
TEAMMATE_POINTS = {
    "center": dict(xy=(841.3601249898122, -544.4349902715983), jd=111.29175232822992),
    "outward800": dict(xy=(-1594.8117693963113, -511.3500635770528), jd=40.971382567388595),
    "tangent": dict(xy=(1688.3980060336946, 120.52869612498694), jd=8.165802),
}

# final_refine.py 的约束精搜结果（0.5 m 网格 + 密集验证 + 融合队友点），为最终采用值。
REFINED_BEST = {
    "center": dict(xy=(844.0, -544.0), j=110.9735, margin=-0.0234,
                   backoff_xy=(841.3601249898122, -544.4349902715983)),
    "outward800": dict(xy=(-1594.812, -511.350), j=40.9719, margin=-219.3292, backoff_xy=None),
    "tangent": dict(xy=(1688.398, 120.529), j=8.1710, margin=-846.0917, backoff_xy=None),
}


def _polar(s1, theta1, p):
    return _dist(s1, p), offset_from_bearing(s1, theta1, p)


def _side_summary(s1, theta1, cells, sign):
    side = [(x, y, v) for x, y, v in cells
            if math.copysign(1.0, _polar(s1, theta1, (x, y))[1]) == sign]
    if not side:
        return {"count": 0}
    pols = [_polar(s1, theta1, (x, y)) for x, y, _ in side]
    return {
        "count": len(side),
        "distance_m": [round(min(p[0] for p in pols), 1), round(max(p[0] for p in pols), 1)],
        "offset_deg": [round(min(abs(p[1]) for p in pols), 1), round(max(abs(p[1]) for p in pols), 1)],
    }


def run_scenario(key, plot: bool, figdir: str) -> dict:
    cfg = SCENARIOS[key]
    s1, theta1 = cfg["s1"], cfg["theta1"]
    print(f"[{cfg['name']}] S1={s1} theta1={theta1}° ...")
    best, records, stages = optimize(s1, theta1, bind_target=True, verbose=True)
    j_best, bx, by = best

    # 采用 final_refine.py 的约束精搜结果作为最终值（其已做 0.5 m 网格+密集验证）
    rb = REFINED_BEST[key]
    j_best, bx, by = rb["j"], rb["xy"][0], rb["xy"][1]
    j_dense, wit = worst_diameter(s1, theta1, (bx, by), phi_n=17, r_step=10.0, err_n=17)
    margin, m_arg = reception_margin(s1, theta1, (bx, by), phi_n=33, r_step=10.0)
    print(f"  J_min = {j_best:.3f} m at ({bx:.2f}, {by:.2f}); dense J = {j_dense:.3f}; margin = {margin:+.4f} m")

    # α 敏感性 + 候选区（25 m 粗网格口径，与图一致）
    alphas = [1.05, 1.1, 1.25, 1.5, 2.0]
    sensitivity = []
    candidate = None
    for a in alphas:
        thr = a * j_best
        cells = [(x, y, v) for x, y, v in records if v <= thr]
        row = {
            "alpha": a,
            "J_m": round(thr, 1),
        }
        if cells:
            pols = [_polar(s1, theta1, (x, y)) for x, y, _ in cells]
            row["distance_m"] = [round(min(p[0] for p in pols), 1), round(max(p[0] for p in pols), 1)]
            row["offset_deg"] = [round(min(abs(p[1]) for p in pols), 1), round(max(abs(p[1]) for p in pols), 1)]
            row["cells"] = len(cells)
        sensitivity.append(row)
        if a == 1.25:
            candidate = {
                "threshold_m": round(thr, 1),
                "cells": len(cells),
                "left": _side_summary(s1, theta1, cells, +1),
                "right": _side_summary(s1, theta1, cells, -1),
            }

    # 交叉验证：在队友报告的最优点上独立重算 J
    tp = TEAMMATE_POINTS[key]
    j_x, _ = worst_diameter(s1, theta1, tp["xy"], phi_n=17, r_step=10.0, err_n=17)
    m_x, _ = reception_margin(s1, theta1, tp["xy"], phi_n=33, r_step=10.0)

    # 楔形轴（过 S1、沿 θ1 方向）镜像点：中心/向外场景即 (x,−y)；
    # 切向场景镜像点 (1891.6,±120.5) 在目标圆外，不可用（模型正确，非对称性破坏）
    ux, uy = math.cos(math.radians(theta1)), math.sin(math.radians(theta1))
    rx, ry = bx - s1[0], by - s1[1]
    dot = rx * ux + ry * uy
    mx, my = s1[0] + (2.0 * dot * ux - rx), s1[1] + (2.0 * dot * uy - ry)
    sym_inside = math.hypot(mx, my) <= TARGET_R + 1e-6

    out = {
        "scenario": cfg["name"],
        "s1": list(s1),
        "theta1_deg": theta1,
        "J_min_m": round(j_best, 3),
        "best_point": {
            "xy": [round(bx, 2), round(by, 2)],
            "distance_from_s1_m": round(_dist(s1, (bx, by)), 2),
            "offset_from_bearing_deg": round(offset_from_bearing(s1, theta1, (bx, by)), 2),
            "symmetric_xy": [round(mx, 2), round(my, 2)],
            "symmetric_point_within_target_disk": sym_inside,
        },
        "dense_J_m": round(j_dense, 3),
        "worst_case_witness": (list(wit) if wit else None),
        "reception_margin_m": round(margin, 4),
        "reception_margin_kind": "FINITE_NUMERICAL_SUPREMUM_NOT_CERTIFIED",
        "candidate_region_1.25": candidate,
        "alpha_sensitivity": sensitivity,
        "crosscheck_teammate_point": {
            "xy": list(tp["xy"]),
            "teammate_reported_JD_m": tp["jd"],
            "our_independent_JD_m": round(j_x, 3),
            "abs_diff_m": round(abs(j_x - tp["jd"]), 4),
            "reception_margin_m": round(m_x, 4),
        },
        "deployment_backoff_point": (
            {
                "xy": list(rb["backoff_xy"]),
                "J_m": round(worst_diameter(s1, theta1, tuple(rb["backoff_xy"]),
                                            phi_n=17, r_step=10.0, err_n=17)[0], 3),
                "reception_margin_m": round(
                    reception_margin(s1, theta1, tuple(rb["backoff_xy"]),
                                     phi_n=33, r_step=10.0)[0], 3),
                "note": "推荐点贴 C_recv 边界时向内回退约 3 m 的稳健替代点",
            } if rb.get("backoff_xy") else None
        ),
        "stages": [{"step": st[0], "J": round(st[2][0], 3), "xy": [st[2][1], st[2][2]]} for st in stages],
        "feasible_cells_25m_grid": len(records),
    }
    if plot:
        _plot(s1, theta1, records, out, figdir)
    return out


def _plot(s1, theta1, records, summary, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for name in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Songti SC",
                 "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        from matplotlib import font_manager
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = [name]
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    half = 1010.0
    step = 25.0
    n = int(half // step)
    xs = [s1[0] + i * step for i in range(-n, n + 1)]
    ys = [s1[1] + i * step for i in range(-n, n + 1)]
    Z = np.full((len(ys), len(xs)), np.nan)
    for x, y, v in records:
        ix = round((x - xs[0]) / step)
        iy = round((y - ys[0]) / step)
        if 0 <= ix < len(xs) and 0 <= iy < len(ys):
            Z[iy, ix] = v

    j_min = float(summary["J_min_m"])
    cap = 3.0 * j_min
    X, Y = np.meshgrid(np.array(xs), np.array(ys))
    Zp = np.where(np.isinf(Z), cap, Z)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    mesh = ax.pcolormesh(X, Y, np.minimum(Zp, cap), cmap="viridis_r", shading="nearest",
                         vmin=j_min, vmax=cap)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cbar.set_label(f"最坏定位区域直径 J (m)，≥{cap:.0f} m 截断显示")
    levels = sorted(round(j_min * f, 1) for f in (1.1, 1.25, 1.5))
    Zc = np.where(np.isnan(Z) | np.isinf(Z), cap * 10, Z)
    cs = ax.contour(X, Y, Zc, levels=levels, colors=("#f59e0b", "#d62728", "#7c3aed"),
                    linewidths=1.6)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f} m", fontsize=8)

    r_far = direction_far_distance(s1, theta1)
    far = _along(s1, theta1, r_far)
    ax.plot([s1[0], far[0]], [s1[1], far[1]], color="#1f4fd8", lw=2.0)
    for d in (-ERROR_DEG, ERROR_DEG):
        edge = _along(s1, theta1 + d, r_far)
        ax.plot([s1[0], edge[0]], [s1[1], edge[1]], color="#1f4fd8", lw=1.0, ls="--")
    left, right = three_circle_anchors(s1, theta1)
    for center in (s1, left, right):
        ax.add_patch(plt.Circle(center, R_MIN, fill=False, ec="#555555", lw=1.0, ls=":"))
    ax.add_patch(plt.Circle((0.0, 0.0), TARGET_R, fill=False, ec="#999999", lw=1.0))
    ax.plot(*s1, "o", color="#1f4fd8", ms=9, zorder=5)
    bx, by = summary["best_point"]["xy"]
    ax.plot(bx, by, "*", color="#d62728", ms=16, mec="white", zorder=6)
    mx, my = summary["best_point"]["symmetric_xy"]
    if summary["best_point"]["symmetric_point_within_target_disk"]:
        ax.plot(mx, my, "*", color="#d62728", ms=16, mec="white", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(xs[0] - 50, xs[-1] + 50)
    ax.set_ylim(ys[0] - 50, ys[-1] + 50)
    ax.set_xlabel("x（m，正东）")
    ax.set_ylabel("y（m，正北）")
    ax.set_title(
        f"问题 2：第二检测点候选区域（S1={tuple(round(v) for v in s1)}，示向度 {theta1:g}°）\n"
        f"彩色 = C_recv 可行点；等值线 = 1.1/1.25/1.5×J_min；★ = J_min {j_min:.0f} m"
    )
    fig.text(0.02, 0.01,
             f"蓝实线/虚线：第一次示向及 ±1° 边界（远端 {r_far:.0f} m）。点线圆：三圆叶子（保守充分条件）。\n"
             f"后验已裁剪至源域（5<d1≤1500、d2≤1500）；J 为有限数值最坏值，非连续认证。",
             fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, f"problem2_candidate_region_{summary['scenario']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  图已保存：{os.path.abspath(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", "center", "outward800", "tangent"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="../results")
    ap.add_argument("--figdir", default="../figs")
    args = ap.parse_args()

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    outdir = os.path.abspath(args.outdir)
    figdir = os.path.abspath(args.figdir)
    results = {"scenarios": {}}
    for key in keys:
        results["scenarios"][key] = run_scenario(key, args.plot, figdir)
    results["model"] = {
        "source_domain": "B(O,1800) ∩ W(S1,θ1,±1°) ∩ {5<d1≤1500}, per-direction target-circle clipping",
        "reception": "C_recv = ∩ B(g, max(1000,‖g−S1‖)); three-circle leaf is conservative sufficient condition",
        "posterior": "W1 ∩ W2 ∩ Ω1 ∩ {5<d2≤1500} (clipped correction; old 135m was unclipped)",
        "objective": "J(q) = finite-grid worst clipped posterior diameter; near-field STRONG branch exact",
        "honesty": {
            "finite_numerical_supremum": True,
            "continuous_certified": False,
            "global_optimum_certified": False,
            "alpha_1.25_is_tolerance_not_constant": True,
        },
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "problem2_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存：{path}")


if __name__ == "__main__":
    main()
