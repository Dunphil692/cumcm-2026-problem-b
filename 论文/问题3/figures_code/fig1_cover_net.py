# -*- coding: utf-8 -*-
"""图1：7 点覆盖网几何图（原点 + 半径 1200 m 正六边形，最坏听圈 1000 m）。"""
import sys, math
sys.path.insert(0, "/Users/zhuchen/建模大赛/问题三/论文/figures_code")
from style import *  # noqa

import numpy as np

R_ARENA = 1800.0
R_NET = 1200.0
R_LISTEN = 1000.0

fig, ax = plt.subplots(figsize=(6.4, 6.4))

# 场地圆
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(R_ARENA * np.cos(theta), R_ARENA * np.sin(theta), color="#333333", lw=1.6)
ax.fill(R_ARENA * np.cos(theta), R_ARENA * np.sin(theta), color="#F2F5F8", zorder=0)

# 7 个覆盖点
stops = [(0.0, 0.0)]
for i in range(6):
    a = i * math.pi / 3.0
    stops.append((R_NET * math.cos(a), R_NET * math.sin(a)))

# 1000 m 听圈（每个覆盖点）
for sx, sy in stops:
    ax.add_patch(plt.Circle((sx, sy), R_LISTEN, fill=False,
                            edgecolor="#0072B2", lw=0.9, ls="--", alpha=0.7, zorder=1))

# 正六边形连线（原点不算顶点，6 个外点连成环）
hex_pts = stops[1:] + [stops[1]]
hx = [p[0] for p in hex_pts]
hy = [p[1] for p in hex_pts]
ax.plot(hx, hy, color="#D55E00", lw=1.6, zorder=2)

# 覆盖点
xs = [p[0] for p in stops]
ys = [p[1] for p in stops]
ax.scatter(xs, ys, s=55, color="#D55E00", zorder=4, edgecolor="white", linewidth=0.8)

# 最坏覆盖点示意：场地边缘 30° 处（距最近覆盖点 ~969 m）
wp_ang = math.radians(30)
wp = (R_ARENA * math.cos(wp_ang), R_ARENA * math.sin(wp_ang))
ax.plot([0, wp[0]], [0, wp[1]], color="#666666", lw=1.0, ls=":")
ax.scatter([wp[0]], [wp[1]], s=60, marker="X", color="#009E73", zorder=5,
           edgecolor="white", linewidth=0.8)

# 标注
ax.annotate("原点（第 1 个覆盖点）", xy=(0, 0), xytext=(280, 220),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
ax.annotate("半径 1200 m 正六边形\n（6 个覆盖点）", xy=(R_NET * math.cos(math.radians(120)),
            R_NET * math.sin(math.radians(120))), xytext=(-1500, -1200),
            fontsize=9, color="#D55E00",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.9))
ax.annotate("听圈半径 1000 m\n（按最坏接收半径）", xy=(R_NET + R_LISTEN * 0.5, 0),
            xytext=(900, 1550), fontsize=9, color="#0072B2",
            arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9))
ax.annotate("场地边缘最坏点\n（离最近覆盖点约 969 m < 1000 m）", xy=wp, xytext=(1500, 1050),
            fontsize=9, color="#009E73",
            arrowprops=dict(arrowstyle="->", color="#009E73", lw=0.9))

ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_aspect("equal")
ax.set_xticks([-1800, -1200, -600, 0, 600, 1200, 1800])
ax.set_yticks([-1800, -1200, -600, 0, 600, 1200, 1800])
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
ax.grid(True, color="#E0E0E0", lw=0.5, ls="--")

plt.tight_layout()
plt.savefig("/Users/zhuchen/建模大赛/问题三/论文/figures/fig1_七点覆盖网.png", bbox_inches="tight")
print("saved fig1")
