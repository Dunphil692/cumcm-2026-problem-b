# -*- coding: utf-8 -*-
"""图5：三种策略总时间分布箱线图（n=10000）。
- 静态/早期动态：results/trials_10k.csv 逐局数据（whisker=1.5×IQR，异常点单独显示）
- 上场策略：fullopt_10k_summary.json 五数概括
"""
import sys, csv
sys.path.insert(0, "/Users/zhuchen/建模大赛/问题三/论文/figures_code")
from style import *  # noqa
import numpy as np

rows = {"limited": [], "belief": []}
with open("/Users/zhuchen/建模大赛/问题三/results/trials_10k.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["policy"] in rows:
            rows[r["policy"]].append(float(r["time_s"]))

def box_stats(v):
    v = np.array(v)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_w = max(v[v >= q1 - 1.5 * iqr].min(), v.min())
    hi_w = min(v[v <= q3 + 1.5 * iqr].max(), v.max())
    out = v[(v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)]
    return q1, med, q3, lo_w, hi_w, out

names = ["静态：先 7 点普查再清", "早期动态（未加固）", "上场：可行域 + 混排"]
colors = ["#9AA5B1", "#7FB3D5", "#0072B2"]
means = [np.mean(rows["limited"]), np.mean(rows["belief"]), 3797.5]

fig, ax = plt.subplots(figsize=(9.2, 5.0))

for idx, name in enumerate(names):
    pos = idx + 1
    if idx < 2:
        q1, med, q3, lo_w, hi_w, out = box_stats(rows[["limited", "belief"][idx]])
    else:
        q1, med, q3, lo_w, hi_w, out = 3506.2, 3777.1, 4058.6, 2126.4, 6864.2, np.array([])
    color = colors[idx]
    ax.add_patch(plt.Rectangle((pos - 0.26, q1), 0.52, q3 - q1,
                               facecolor=color, alpha=0.55, edgecolor=color, lw=1.0))
    ax.plot([pos - 0.26, pos + 0.26], [med, med], color="white", lw=2.4)
    ax.plot([pos, pos], [q1, lo_w], color=color, lw=1.0)
    ax.plot([pos, pos], [q3, hi_w], color=color, lw=1.0)
    ax.plot([pos - 0.1, pos + 0.1], [lo_w, lo_w], color=color, lw=1.0)
    ax.plot([pos - 0.1, pos + 0.1], [hi_w, hi_w], color=color, lw=1.0)
    if len(out):
        ax.scatter([pos] * len(out), out, s=10, color=color, alpha=0.5, zorder=4)
    ax.scatter([pos], [means[idx]], marker="D", s=40, color=color,
               edgecolor="white", linewidth=0.7, zorder=5)

ax.set_xticks([1, 2, 3]); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("总时间 / s")
ax.set_ylim(0, 9000)
ax.set_title("三种策略总时间分布（n=10000，本地仿真，非官方）", fontsize=11.5)
ax.grid(axis="y", color="#E0E0E0", lw=0.6, ls="--")
# 图例
ax.scatter([], [], marker="D", s=40, color="#333333", label="均值")
ax.plot([], [], color="#333333", lw=2.4, label="中位数")
ax.legend(loc="upper right", fontsize=8.5)
ax.text(1, 8300, "最坏 133698 s", ha="center", fontsize=8, color="#666666")
ax.text(2, 8300, "最坏 112965 s", ha="center", fontsize=8, color="#666666")
ax.text(3, 8300, "最坏 6864 s", ha="center", fontsize=8, color="#0072B2")

plt.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.16)
plt.savefig("/Users/zhuchen/建模大赛/问题三/论文/figures/fig5_时间分布箱线图.png", bbox_inches="tight")
print("saved fig5")
