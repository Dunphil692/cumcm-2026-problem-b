#!/usr/bin/env python3
"""Teaching figures: 10 m x 64-bin cell, and N19 listen disks."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

OUT_GRID = Path(__file__).with_name("q4_explain_grid64.png")
OUT_N19 = Path(__file__).with_name("q4_explain_n19_range.png")
OUT_EX = Path(__file__).with_name("q4_explain_example.png")
FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
if FONT_PATH.exists():
    fontManager.addfont(str(FONT_PATH))
    FONT = FontProperties(fname=str(FONT_PATH), size=12)
    FONT_SM = FontProperties(fname=str(FONT_PATH), size=10)
    FONT_LG = FontProperties(fname=str(FONT_PATH), size=14)
else:
    FONT = FONT_SM = FONT_LG = FontProperties()

ARENA = 1800.0
LISTEN = 1000.0


def n19():
    pts = [("圆心", 0.0, 0.0, "o")]
    for k in range(6):
        a = k * math.pi / 3.0
        pts.append((f"内{k+1}", 1000.0 * math.cos(a), 1000.0 * math.sin(a), "in"))
    for k in range(12):
        a = math.radians(15.0 + 30.0 * k)
        pts.append((f"外{k+1}", 1825.0 * math.cos(a), 1825.0 * math.sin(a), "out"))
    return pts


def fig_grid():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), facecolor="white")
    ax, az = axes

    ax.set_aspect("equal")
    ax.set_xlim(-35, 55)
    ax.set_ylim(-35, 45)
    ax.set_xlabel("x / m", fontproperties=FONT_SM)
    ax.set_ylabel("y / m", fontproperties=FONT_SM)
    for x in range(-30, 51, 10):
        ax.axvline(x, color="#e7e5e4", lw=0.8, zorder=0)
    for y in range(-30, 41, 10):
        ax.axhline(y, color="#e7e5e4", lw=0.8, zorder=0)
    ax.add_patch(Rectangle((0, 0), 10, 10, facecolor="#fed7aa", edgecolor="#c2410c", lw=1.6, zorder=2))
    ax.plot(5, 5, "o", color="#111827", ms=7, zorder=4)
    ax.annotate(
        "",
        xy=(5, 5),
        xytext=(-20, -8),
        arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=1.5),
    )
    ax.plot(-20, -8, "o", color="#1d4ed8", ms=8, zorder=4)
    ax.text(-20, -14, "狗在这边听", fontproperties=FONT, color="#1d4ed8", ha="center")
    ax.text(5, 16, "一个 10 m × 10 m 的格子\n黑点 = 这个格子的代表点", fontproperties=FONT, color="#c2410c", ha="center")
    ax.set_title("场地像方格纸，每隔 10 米一个格子", fontproperties=FONT_LG, loc="left")

    az.set_aspect("equal")
    az.set_xlim(-1.35, 1.35)
    az.set_ylim(-1.35, 1.35)
    az.axis("off")
    dog_ang = math.degrees(math.atan2(-8 - 5, -20 - 5))  # cell center -> dog
    for k in range(64):
        a0, a1 = k * 360 / 64, (k + 1) * 360 / 64
        mid = (a0 + a1) / 2
        d = abs((mid - dog_ang + 180) % 360 - 180)
        color = "#0f766e" if d <= 90 else "#d6d3d1"
        az.add_patch(Wedge((0, 0), 1.0, a0, a1, facecolor=color, edgecolor="white", lw=0.4))
    az.plot(0, 0, "o", color="#111827", ms=8)
    az.annotate(
        "",
        xy=(0.85 * math.cos(math.radians(dog_ang)), 0.85 * math.sin(math.radians(dog_ang))),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=1.8),
    )
    az.text(0, -1.22, "青绿 = 灯朝向狗，原点听得到\n灰色 = 灯朝背面，原点听不到", fontproperties=FONT, ha="center", color="#1c1917")
    az.set_title("同一个格子再切 64 块方向（每块 5.625°）", fontproperties=FONT_LG, loc="left")

    fig.suptitle("「10 米一个格子 × 64 个方向」是什么意思", fontproperties=FONT_LG, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_GRID, dpi=160, bbox_inches="tight")
    print(OUT_GRID)


def fig_n19():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), facecolor="white")
    ax, ay = axes
    pts = n19()

    for a, title in (
        (ax, "左：19 个停靠点画在哪里"),
        (ay, "右：每个点最坏能听 1000 m（圆）"),
    ):
        a.set_aspect("equal")
        a.set_xlim(-2300, 2300)
        a.set_ylim(-2300, 2300)
        a.axis("off")
        a.add_patch(Circle((0, 0), ARENA, facecolor="#f6f3ec", edgecolor="#8a847c", lw=1.4, zorder=0))
        a.set_title(title, fontproperties=FONT_LG, pad=8)

    ax.add_patch(Circle((0, 0), 1000, facecolor="none", edgecolor="#93c5fd", lw=0.8, ls="--", zorder=1))
    for name, x, y, kind in pts:
        color = {"o": "#111827", "in": "#1d4ed8", "out": "#c2410c"}[kind]
        ax.plot(x, y, "o", color=color, ms=7, zorder=3)
    ax.text(80, 80, "圆心", fontproperties=FONT_SM, color="#111827")
    ax.text(1040, 40, "里面 6 个\n半径 1000 m", fontproperties=FONT_SM, color="#1d4ed8")
    ax.text(1550, 1100, "外面 12 个\n半径 1825 m\n已经出了场地", fontproperties=FONT_SM, color="#c2410c")
    ax.text(0, -2100, "大圆 = 场地 1800 m", fontproperties=FONT_SM, color="#57534e", ha="center")

    for name, x, y, kind in pts:
        color = {"o": "#111827", "in": "#1d4ed8", "out": "#c2410c"}[kind]
        ay.add_patch(Circle((x, y), LISTEN, facecolor=color, edgecolor=color, alpha=0.10, lw=0.6, zorder=1))
        ay.plot(x, y, "o", color=color, ms=5, zorder=3)

    # one outward flashlight that only an outer stop can hear
    gx, gy = 1700.0, 455.0
    phi = math.degrees(math.atan2(gy, gx))
    ay.add_patch(Wedge((gx, gy), 1000, phi - 90, phi + 90, facecolor="#0f766e", alpha=0.22, edgecolor="#0f766e", lw=0.8, zorder=2))
    ay.plot(gx, gy, "o", color="#111827", ms=7, zorder=4)
    ay.annotate(
        "",
        xy=(gx + 380 * math.cos(math.radians(phi)), gy + 380 * math.sin(math.radians(phi))),
        xytext=(gx, gy),
        arrowprops=dict(arrowstyle="-|>", color="#0f766e", lw=1.6),
    )
    ay.text(gx - 40, gy + 80, "手电筒贴边朝外", fontproperties=FONT_SM, color="#111827", ha="right")
    ay.text(
        0,
        -2100,
        "圆 = 距离够不够。青绿半圆 = 灯朝哪。圆内蓝点在它背面，听不到。",
        fontproperties=FONT_SM,
        color="#57534e",
        ha="center",
    )

    fig.tight_layout()
    fig.savefig(OUT_N19, dpi=160, bbox_inches="tight")
    print(OUT_N19)


def fig_example():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0), facecolor="white")
    ax3, ax4 = axes
    g = (400.0, 0.0)
    for ax, title in (
        (ax3, "问题 3：圆心没声音，整圆删掉"),
        (ax4, "问题 4：圆心没声音，只划掉朝向圆心的方向"),
    ):
        ax.set_aspect("equal")
        ax.set_xlim(-900, 1100)
        ax.set_ylim(-800, 800)
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1800, facecolor="#f6f3ec", edgecolor="#d6d3d1", lw=0.8, zorder=0))
        ax.set_title(title, fontproperties=FONT_LG, pad=8)
        ax.plot(0, 0, "o", color="#1d4ed8", ms=9, zorder=5)
        ax.text(20, 50, "狗在圆心，这个频道没声音", fontproperties=FONT_SM, color="#1d4ed8")

    ax3.add_patch(Circle((0, 0), 990, facecolor="#fecaca", edgecolor="#b91c1c", alpha=0.35, lw=1.2, zorder=1))
    ax3.plot(*g, "o", color="#111827", ms=8, zorder=5)
    ax3.annotate(
        "",
        xy=(400 + 220, 0),
        xytext=g,
        arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.6),
    )
    ax3.text(420, 70, "真源在这里，灯朝东\n被问题 3 误删了", fontproperties=FONT, color="#7f1d1d")

    ax4.add_patch(Wedge(g, 280, 90, 270, facecolor="#fecaca", edgecolor="#b91c1c", alpha=0.40, lw=1.0, zorder=2))
    ax4.add_patch(Wedge(g, 280, -90, 90, facecolor="#99f6e4", edgecolor="#0f766e", alpha=0.40, lw=1.0, zorder=2))
    ax4.plot(*g, "o", color="#111827", ms=8, zorder=5)
    ax4.annotate(
        "",
        xy=(400 + 220, 0),
        xytext=g,
        arrowprops=dict(arrowstyle="-|>", color="#0f766e", lw=1.6),
    )
    ax4.text(420, 90, "朝东还留着（真方向）", fontproperties=FONT, color="#0f766e")
    ax4.text(-20, -200, "朝西划掉\n（如果灯朝狗，圆心该听得到）", fontproperties=FONT_SM, color="#b91c1c", ha="right")

    fig.tight_layout()
    fig.savefig(OUT_EX, dpi=160, bbox_inches="tight")
    print(OUT_EX)


if __name__ == "__main__":
    fig_grid()
    fig_n19()
    fig_example()
