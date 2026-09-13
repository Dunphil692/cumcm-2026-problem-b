# -*- coding: utf-8 -*-
"""图2：单频道可行域的三态演化（整圆 → 挖洞 → 交楔形 → 结案）。"""
import sys, math
sys.path.insert(0, "/Users/zhuchen/建模大赛/问题三/论文/figures_code")
from style import *  # noqa
import numpy as np
from matplotlib.patches import Circle, Wedge

R_A = 1800.0
FILL = "#CDE3F0"     # 可行域填充（低饱和蓝）
EDGE = "#333333"

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))

def arena(ax):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_A * np.cos(th), R_A * np.sin(th), color=EDGE, lw=1.4)
    ax.set_aspect("equal")
    ax.set_xlim(-2000, 2000); ax.set_ylim(-2000, 2000)
    ax.set_xticks([]); ax.set_yticks([])

# --- (a) 开局：全场都可能 ---
ax = axes[0]
arena(ax)
ax.add_patch(Circle((0, 0), R_A, color=FILL, zorder=0))
ax.set_title("(a) 开局：Ω = 全场圆", fontsize=11.5)
ax.annotate("半径 1800 m", xy=(0, 0), xytext=(300, -600), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

# --- (b) no_signal 挖洞 ---
ax = axes[1]
arena(ax)
ax.add_patch(Circle((0, 0), R_A, color=FILL, zorder=0))
S = (-900, -700)
ax.add_patch(Circle(S, 990, color="white", zorder=1))          # 挖掉的洞
ax.add_patch(Circle(S, 990, fill=False, edgecolor="#D55E00", lw=1.4, zorder=2))
ax.scatter([S[0]], [S[1]], s=45, color="#D55E00", zorder=3, edgecolor="white", lw=0.8)
ax.set_title("(b) 测到 no_signal：挖掉 B(S, 990 m)", fontsize=11.5)
ax.annotate("S（检测点）", xy=S, xytext=(S[0]-500, S[1]-900), fontsize=9,
            color="#D55E00", arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8))
ax.annotate("源必在洞外", xy=(500, 1200), xytext=(700, 1500), fontsize=9,
            color="#333333", arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

# --- (c) direction 交楔形 ---
ax = axes[2]
arena(ax)
S = (-900, -700)
angle = 32.0
w = Wedge(S, 1500, angle - 1.0, angle + 1.0, color=FILL, zorder=0)
ax.add_patch(w)
ax.add_patch(Circle(S, 1500, fill=False, edgecolor="#0072B2", lw=0.9, ls="--", zorder=1))
ax.scatter([S[0]], [S[1]], s=45, color="#D55E00", zorder=3, edgecolor="white", lw=0.8)
# 楔形两条边界射线
for da in (-1.0, 1.0):
    a = math.radians(angle + da)
    ax.plot([S[0], S[0] + 1700 * math.cos(a)], [S[1], S[1] + 1700 * math.sin(a)],
            color="#0072B2", lw=1.2, zorder=2)
ax.set_title("(c) 测到 direction：交 ±1° 楔形", fontsize=11.5)
ax.annotate("±1° 示向度", xy=(S[0] + 1500 * math.cos(math.radians(angle)),
            S[1] + 1500 * math.sin(math.radians(angle))), xytext=(400, -1400),
            fontsize=9, color="#0072B2",
            arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.8))
ax.annotate("截在 B(S,1500 m) 内", xy=(S[0] + 1500 * math.cos(math.radians(angle)),
            S[1] + 1500 * math.sin(math.radians(angle))), xytext=(600, 1500),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

plt.tight_layout()
plt.savefig("/Users/zhuchen/建模大赛/问题三/论文/figures/fig2_单频道可行域三态.png", bbox_inches="tight")
print("saved fig2")
