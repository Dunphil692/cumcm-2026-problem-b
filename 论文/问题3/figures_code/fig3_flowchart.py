# -*- coding: utf-8 -*-
"""图3：三类任务混排流程图（matplotlib 获奖风格渲染；源文件见 .drawio）。"""
import sys
sys.path.insert(0, "/Users/zhuchen/建模大赛/问题三/问题三图片和论文/figures_code")
from style import *  # noqa
import matplotlib.patches as mp

fig, ax = plt.subplots(figsize=(6.8, 9.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 16)
ax.axis("off")

def box(cx, cy, w, h, text, rounded=False, fs=10.5):
    style = mp.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.06" if rounded else "square,pad=0",
                              fc="white", ec="black", lw=1.1)
    ax.add_patch(style)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

def diamond(cx, cy, w, h, text, fs=10.5):
    ax.add_patch(mp.Polygon([(cx - w/2, cy), (cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2)],
                            closed=True, fc="white", ec="black", lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))

CX = 4.0
box(CX, 15.2, 3.6, 0.8, "机器狗当前点", rounded=True, fs=11)
box(CX, 13.7, 4.6, 1.0, "更新各频道可行域\n（每频道一张可能区域）", fs=9.5)
box(CX, 11.6, 4.6, 1.5, "生成三类任务\n① 能清（MEC ≤ 20）  ② 补测（单楔形/区域太大）  ③ 听洞（未覆盖点）", fs=9)
box(CX, 9.6, 4.6, 1.3, "开路最短哈密顿路\n（Held–Karp / 2-opt）\n只走下一站", fs=9.5)
box(CX, 7.9, 3.6, 0.9, "移动到下一站\n同站测完、清完", fs=9.5)
diamond(CX, 6.0, 4.6, 1.6, "可行域空证书\n是否成立？", fs=9.5)
box(CX, 3.6, 3.0, 0.8, "结束", rounded=True, fs=11)

arrow(CX, 14.8, CX, 14.2)
arrow(CX, 13.2, CX, 12.35)
arrow(CX, 10.85, CX, 10.25)
arrow(CX, 8.95, CX, 8.35)
arrow(CX, 7.45, CX, 6.8)
arrow(CX, 5.2, CX, 4.0)
ax.text(CX + 0.15, 4.6, "是", fontsize=9)

# 否：右侧回环
ax.annotate("否", xy=(8.9, 13.9), xytext=(8.2, 6.0), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1,
                            connectionstyle="angle,angleA=0,angleB=90,rad=0"))
ax.plot([CX+2.3, 8.9], [6.0, 6.0], color="black", lw=1.1)
ax.plot([8.9, 8.9], [6.0, 13.9], color="black", lw=1.1)

plt.tight_layout()
plt.savefig("/Users/zhuchen/建模大赛/问题三/问题三图片和论文/figures/fig3_三类任务混排.png", bbox_inches="tight", dpi=330)
print("saved fig3")
