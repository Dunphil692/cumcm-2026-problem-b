# -*- coding: utf-8 -*-
"""图6：覆盖点访问时序（本地 v2 策略 200 局）。
说明覆盖义务是插在路上完成的，而非先扫完 7 点再清。
"""
import sys, json, statistics
sys.path.insert(0, "/Users/zhuchen/建模大赛/问题三/论文/figures_code")
from style import *  # noqa
import numpy as np

d = json.load(open("/Users/zhuchen/建模大赛/问题三/论文/figures_code/cover_timing.json"))["rows"]
for r in d:
    # 原点覆盖点：机器狗从原点出发，t=0 即已访问
    r["first_visit"][0] = 0.0
N = len(d)
total_t = [r["time_s"] for r in d]
first_clear = [r["first_clear"] for r in d if r["first_clear"] is not None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

# ---- (a) 平均累计访问覆盖点数 vs 时间 ----
tgrid = np.linspace(0, 4000, 200)
cum = []
for t in tgrid:
    cnt = 0
    for r in d:
        cnt += sum(1 for v in r["first_visit"] if v is not None and v <= t)
    cum.append(cnt / N)
ax1.plot(tgrid, cum, color="#0072B2", lw=2.2)
ax1.set_xlabel("任务时间 / s")
ax1.set_ylabel("平均已访问覆盖点数")
ax1.set_ylim(0, 7.2)
ax1.set_title("(a) 覆盖点随任务进度被访问（插在路上做）", fontsize=11.5)
fc = statistics.fmean(first_clear)
ax1.axvline(fc, color="#D55E00", lw=1.4, ls="--")
ax1.annotate(f"首次清除 ≈ {fc:.0f} s\n此时平均仅访问 ≈2.1 点",
             xy=(fc, 2.1), xytext=(fc + 620, 1.0), fontsize=8.6, color="#D55E00",
             arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.9))
ax1.grid(True, color="#E0E0E0", lw=0.5, ls="--")

# ---- (b) 各覆盖点首次访问时刻（中位数） ----
labels = ["原点", "0°", "60°", "120°", "180°", "240°", "300°"]
med_t = []
for i in range(7):
    vals = [r["first_visit"][i] for r in d if r["first_visit"][i] is not None]
    med_t.append(statistics.median(vals) if vals else np.nan)
x = np.arange(7)
ax2.bar(x, med_t, width=0.6, color="#0072B2", edgecolor="white", lw=0.6)
for xi, v in zip(x, med_t):
    ax2.text(xi, v + 40, f"{v:.0f}", ha="center", fontsize=8.4)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("首次访问时刻中位数 / s")
ax2.set_ylim(0, max(med_t) * 1.25)
ax2.set_title("(b) 各覆盖点首次访问时刻（中位数）", fontsize=11.5)
ax2.grid(axis="y", color="#E0E0E0", lw=0.6, ls="--")

plt.tight_layout()
plt.savefig("/Users/zhuchen/建模大赛/问题三/论文/figures/fig6_覆盖点访问时序.png", bbox_inches="tight")
print("saved fig6")
print("first_clear mean", round(fc, 1), "total mean", round(statistics.fmean(total_t), 1))
print("per-cover median first visit:", [round(v) for v in med_t])
