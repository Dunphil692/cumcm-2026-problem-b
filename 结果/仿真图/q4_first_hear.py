#!/usr/bin/env python3
"""Teaching figure: first bearing = thin wedge; next stop is not N19."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Wedge

OUT = Path(__file__).with_name("q4_first_hear.png")
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
]
FONT_PATH = next((p for p in FONT_CANDIDATES if p.exists()), None)
if FONT_PATH is not None:
    fontManager.addfont(str(FONT_PATH))
    FONT = FontProperties(fname=str(FONT_PATH), size=11)
    FONT_SM = FontProperties(fname=str(FONT_PATH), size=9.5)
    FONT_LG = FontProperties(fname=str(FONT_PATH), size=14)
else:
    FONT = FONT_SM = FONT_LG = FontProperties()

ARENA = 1800.0
LISTEN = 1500.0
THETA = 0.0
TRUE_HALF = 1.0
TEACH_HALF = 8.0  # arena views only; otherwise the 2° slit is invisible
G = (1100.0, 20.0)
S2 = (750.0, 620.0)


def n19() -> list[tuple[float, float]]:
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = k * math.pi / 3.0
        pts.append((1000.0 * math.cos(a), 1000.0 * math.sin(a)))
    for k in range(12):
        a = math.radians(15.0 + 30.0 * k)
        pts.append((1825.0 * math.cos(a), 1825.0 * math.sin(a)))
    return pts


def wedge_poly(sx, sy, theta, half, r0, r1, n=28):
    lo, hi = math.radians(theta - half), math.radians(theta + half)
    outer = [(sx + r1 * math.cos(a), sy + r1 * math.sin(a)) for a in np.linspace(lo, hi, n)]
    inner = [(sx + r0 * math.cos(a), sy + r0 * math.sin(a)) for a in np.linspace(hi, lo, n)]
    return outer + inner


def bearing_to(src, dst):
    return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0])) % 360.0


def draw_wedge(ax, s, theta, half, r1, color, alpha, lw=1.3, z=3, r0=50.0):
    ax.add_patch(
        Polygon(
            wedge_poly(s[0], s[1], theta, half, r0, r1),
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=z,
        )
    )
    for sign in (-1.0, 1.0):
        a = math.radians(theta + sign * half)
        ax.plot(
            [s[0] + r0 * math.cos(a), s[0] + r1 * math.cos(a)],
            [s[1] + r0 * math.sin(a), s[1] + r1 * math.sin(a)],
            color=color,
            lw=lw,
            ls="--",
            zorder=z + 1,
        )
    a0 = math.radians(theta)
    ax.plot(
        [s[0], s[0] + r1 * math.cos(a0)],
        [s[1], s[1] + r1 * math.sin(a0)],
        color=color,
        lw=1.1,
        zorder=z + 1,
    )


def arena(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-2100, 2100)
    ax.set_ylim(-2100, 2100)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), ARENA, facecolor="#f6f3ec", edgecolor="#8a847c", lw=1.3, zorder=0))
    ax.add_patch(Circle((0, 0), LISTEN, facecolor="none", edgecolor="#c4beb4", lw=0.8, ls="--", zorder=1))


def ray_hit(p1, ang1, p2, ang2):
    a1, a2 = math.radians(ang1), math.radians(ang2)
    d1, d2 = (math.cos(a1), math.sin(a1)), (math.cos(a2), math.sin(a2))
    det = d1[0] * d2[1] - d1[1] * d2[0]
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def diamond(half):
    th2 = bearing_to(S2, G)
    s1 = (0.0, 0.0)
    return [
        ray_hit(s1, THETA - half, S2, th2 - half),
        ray_hit(s1, THETA - half, S2, th2 + half),
        ray_hit(s1, THETA + half, S2, th2 + half),
        ray_hit(s1, THETA + half, S2, th2 - half),
    ]


def panel_a(ax):
    arena(ax)
    ax.add_patch(Wedge((0, 0), LISTEN, -50, 50, facecolor="#d6d3d1", edgecolor="none", alpha=0.45, zorder=2))
    ax.text(0.04, 0.90, "灰块 = 常见误解\n不是 90° 大扇形", transform=ax.transAxes, fontproperties=FONT_SM, color="#78716c", va="top")
    draw_wedge(ax, (0, 0), THETA, TEACH_HALF, LISTEN, "#c2410c", 0.55, lw=1.5)
    ax.plot(0, 0, "o", color="#1d4ed8", ms=10, zorder=6)
    ax.text(80, 140, "狗 S1（原点）", fontproperties=FONT, color="#1d4ed8")
    ax.text(0.50, 0.08, "橙缝图上加宽到 8° 才能看见\n真实只有 ±1°，见右图", transform=ax.transAxes, ha="center", fontproperties=FONT_SM, color="#c2410c")
    ax.text(0.50, -0.02, "1. 听见之后：从狗出发画一条细缝", transform=ax.transAxes, ha="center", va="top", fontproperties=FONT_LG)


def panel_zoom(ax):
    ax.set_aspect("equal")
    ax.set_xlim(860, 1540)
    ax.set_ylim(-90, 90)
    draw_wedge(ax, (0, 0), THETA, TRUE_HALF, LISTEN, "#c2410c", 0.5, lw=1.5, r0=40.0)
    w = 2 * LISTEN * math.tan(math.radians(TRUE_HALF))
    ax.annotate("", xy=(LISTEN, w / 2), xytext=(LISTEN, -w / 2), arrowprops=dict(arrowstyle="<->", color="#1c1917", lw=1.0))
    ax.text(1325, 0, f"1500 m 处\n真实缝宽 {w:.0f} m", fontproperties=FONT, va="center", color="#1c1917")
    ax.set_xlabel("x / m", fontproperties=FONT_SM)
    ax.set_ylabel("y / m", fontproperties=FONT_SM)
    ax.tick_params(labelsize=8, colors="#57534e")
    for spine in ax.spines.values():
        spine.set_color("#a8a29e")
    ax.set_title("真实 ±1°：远看像一条线，不是某一个角度的圆", fontproperties=FONT, loc="left", pad=8)


def panel_b(ax):
    arena(ax)
    draw_wedge(ax, (0, 0), THETA, TEACH_HALF, LISTEN, "#c2410c", 0.40)
    ax.plot(0, 0, "o", color="#1d4ed8", ms=10, zorder=6)
    ax.text(70, 130, "S1", fontproperties=FONT, color="#1d4ed8")
    gx, gy = G
    phi = bearing_to(G, (0.0, 0.0))
    ax.add_patch(Wedge((gx, gy), 380, phi - 90, phi + 90, facecolor="#0f766e", edgecolor="#0f766e", alpha=0.30, lw=1.1, zorder=4))
    ax.add_patch(Wedge((gx, gy), 380, phi + 90, phi + 270, facecolor="#a8a29e", edgecolor="none", alpha=0.22, zorder=4))
    a = math.radians(phi)
    ax.annotate("", xy=(gx + 340 * math.cos(a), gy + 340 * math.sin(a)), xytext=(gx, gy), arrowprops=dict(arrowstyle="-|>", color="#0f766e", lw=1.7))
    ax.plot(gx, gy, "o", color="#111827", ms=8, zorder=7)
    ax.text(gx + 30, gy + 90, "G 可能在细缝里", fontproperties=FONT, color="#111827")
    ax.text(0.04, 0.92, "青绿 = 灯朝向狗的 180°\n（示意，真半径 1000–1500 m）", transform=ax.transAxes, fontproperties=FONT_SM, color="#0f766e", va="top")
    ax.text(0.96, 0.12, "灰半边 = 背面，听不到", transform=ax.transAxes, ha="right", fontproperties=FONT_SM, color="#78716c")
    ax.text(0.50, -0.02, "2. 定向额外记：手电筒必须朝向狗", transform=ax.transAxes, ha="center", va="top", fontproperties=FONT_LG)


def panel_c(ax):
    arena(ax)
    draw_wedge(ax, (0, 0), THETA, TEACH_HALF, LISTEN, "#c2410c", 0.28)
    th2 = bearing_to(S2, G)
    draw_wedge(ax, S2, th2, TEACH_HALF, 720, "#1d4ed8", 0.28)
    ax.add_patch(Polygon(diamond(TEACH_HALF), closed=True, facecolor="#b91c1c", edgecolor="#7f1d1d", alpha=0.80, zorder=5))
    for x, y in n19()[1:]:
        ax.plot(x, y, "o", color="#a8a29e", ms=5.2, zorder=4)
    ax.plot(0, 0, "o", color="#1d4ed8", ms=9, zorder=6)
    ax.plot(*S2, "s", color="#c2410c", ms=10, zorder=7)
    ax.add_patch(FancyArrowPatch((0, 0), S2, arrowstyle="-|>", mutation_scale=13, lw=1.7, color="#c2410c", zorder=6))
    ax.text(0.04, 0.93, "灰点 = N19\n给还没听见的频道", transform=ax.transAxes, fontproperties=FONT, color="#57534e", va="top")
    ax.text(S2[0] + 50, S2[1] + 40, "S2 第二站", fontproperties=FONT, color="#c2410c")
    ax.text(-80, 120, "S1", fontproperties=FONT_SM, color="#1d4ed8", ha="right")
    ax.text(0.62, 0.22, "红块 = 两站交会\n（图上加宽；真实几十米）", transform=ax.transAxes, fontproperties=FONT_SM, color="#7f1d1d")
    ax.text(0.50, -0.02, "3. 下一站去楔形旁边，不是去 N19", transform=ax.transAxes, ha="center", va="top", fontproperties=FONT_LG)


def main():
    fig = plt.figure(figsize=(13.4, 8.8), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.02, 1.0], hspace=0.28, wspace=0.10,
        left=0.03, right=0.99, top=0.88, bottom=0.06,
    )
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_zoom(fig.add_subplot(gs[0, 1]))
    panel_b(fig.add_subplot(gs[1, 0]))
    panel_c(fig.add_subplot(gs[1, 1]))
    fig.suptitle("问题 4：第一次听见之后记什么、下一步去哪", fontproperties=FONT_LG, y=0.98, color="#1c1917")
    fig.text(
        0.50, 0.925,
        "记下的不是一块扇形圆饼。位置 = 示向线两侧各 1° 的细缝（最远 1500 m）。定向再加：灯必须朝向狗。下一站是为这只源另选的第二站。",
        ha="center", fontproperties=FONT_SM, color="#57534e",
    )
    fig.savefig(OUT, dpi=165)
    print(OUT)


if __name__ == "__main__":
    main()
