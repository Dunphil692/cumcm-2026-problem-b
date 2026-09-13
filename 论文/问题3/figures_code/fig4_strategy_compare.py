# -*- coding: utf-8 -*-
"""图4：三种策略对比柱状图（均时 + 清完率）。数据取自论文定稿表（本地 1 万局）。"""
import sys
sys.path.insert(0, "/Users/zhuchen/建模大赛/问题三/论文/figures_code")
from style import *  # noqa
import numpy as np

strategies = ["静态：先 7 点普查再清", "早期动态（未加固）", "上场：可行域 + 混排"]
time_mean = [5512, 4340, 3798]
clear_rate = [99.72, 98.74, 99.90]
walk = [22536, 16617, 14065]
colors = ["#9AA5B1", "#7FB3D5", "#0072B2"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
x = np.arange(3)

# --- (a) 平均总时间 ---
bars = ax1.bar(x, time_mean, width=0.62, color=colors, edgecolor="white", lw=0.6)
for b, v, w in zip(bars, time_mean, walk):
    ax1.text(b.get_x() + b.get_width() / 2, v + 40, f"{v} s",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.text(b.get_x() + b.get_width() / 2, v / 2, f"走 {w} m",
             ha="center", va="center", fontsize=8.2, color="#333333")
ax1.set_ylim(0, 6400)
ax1.set_ylabel("平均总时间 / s")
ax1.set_xticks(x); ax1.set_xticklabels(strategies, fontsize=8.6)
ax1.set_title("(a) 平均总时间（越短越好）", fontsize=11.5)
ax1.grid(axis="y", color="#E0E0E0", lw=0.6, ls="--")

# --- (b) 清完率 ---
bars = ax2.bar(x, clear_rate, width=0.62, color=colors, edgecolor="white", lw=0.6)
for b, v in zip(bars, clear_rate):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}%",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylim(98.0, 100.15)
ax2.set_ylabel("清完率 / %")
ax2.set_xticks(x); ax2.set_xticklabels(strategies, fontsize=8.6)
ax2.set_title("(b) 清完率（纵轴自 98% 起）", fontsize=11.5)
ax2.grid(axis="y", color="#E0E0E0", lw=0.6, ls="--")
# 断轴示意
ax2.plot([-0.42, -0.42], [98.0, 98.15], color="#333333", lw=1.2, clip_on=False)
ax2.plot([-0.46, -0.38], [98.0, 98.0], color="#333333", lw=1.2, clip_on=False)
ax2.plot([-0.46, -0.38], [98.15, 98.15], color="#333333", lw=1.2, clip_on=False)

plt.tight_layout()
plt.savefig("/Users/zhuchen/建模大赛/问题三/论文/figures/fig4_策略对比.png", bbox_inches="tight")
print("saved fig4")
