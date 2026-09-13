# -*- coding: utf-8 -*-
"""学术配图统一风格（图表美化提示词规范）：
低饱和 Nature 配色、PingFang 中文字体、≥300dpi、无多余装饰。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体回退 + 负号正常
for _name in ["PingFang SC", "Heiti SC", "Songti SC"]:
    try:
        font_manager.findfont(_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_name, "Arial Unicode MS", "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.titlesize"] = 11.5
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 330
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["grid.color"] = "#E0E0E0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False

# Nature 学术配色（通用首选）
NATURE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]
BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"
GRAY = "#666666"
