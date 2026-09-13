"""最终约束精搜：在 C_recv 边界附近做 1 m（再 0.5 m）细网格 + 密集验证，
并融合队友独立实现报告的最优点（同口径重评），输出各场景最终最优。"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem2_model import (
    TARGET_R, offset_from_bearing, reception_margin, worst_diameter,
)

SCENARIOS = {
    "center": dict(s1=(0.0, 0.0), theta1=0.0, c=(845.0, -535.0), half=30.0),
    "outward800": dict(s1=(-1000.0, 0.0), theta1=180.0, c=(-1595.0, -510.0), half=20.0),
    "tangent": dict(s1=(1790.0, 0.0), theta1=90.0, c=(1687.0, 121.0), half=15.0),
}
TEAMMATE_POINTS = {
    "center": (841.3601249898122, -544.4349902715983),
    "outward800": (-1594.8117693963113, -511.3500635770528),
    "tangent": (1688.3980060336946, 120.52869612498694),
}


def feasible(s1, theta1, q, dense=False):
    if abs(offset_from_bearing(s1, theta1, q)) <= 1.0 + 1e-9:
        return False, math.inf
    if math.hypot(q[0], q[1]) > TARGET_R + 1e-9:
        return False, math.inf
    slack, _ = reception_margin(s1, theta1, q, phi_n=33, r_step=10.0)
    return slack <= 0.0, slack


def fine_search(s1, theta1, c, half, step):
    n = int(half // step)
    cands = []
    for ix in range(-n, n + 1):
        for iy in range(-n, n + 1):
            q = (c[0] + ix * step, c[1] + iy * step)
            ok, slack = feasible(s1, theta1, q)
            if not ok:
                continue
            j = worst_diameter(s1, theta1, q, phi_n=5, r_step=50.0, err_n=9)[0]
            cands.append((j, q, slack))
    cands.sort(key=lambda t: t[0])
    return cands


def dense_verify(s1, theta1, q):
    ok, slack = feasible(s1, theta1, q)
    if not ok:
        return math.inf, slack, None
    j, wit = worst_diameter(s1, theta1, q, phi_n=17, r_step=10.0, err_n=17)
    return j, slack, wit


def main():
    for key, cfg in SCENARIOS.items():
        s1, theta1 = cfg["s1"], cfg["theta1"]
        print(f"[{key}] 1m fine search around {cfg['c']} half={cfg['half']} ...")
        cands = fine_search(s1, theta1, cfg["c"], cfg["half"], 1.0)
        top = cands[:12]
        # 融合队友点
        tp = TEAMMATE_POINTS[key]
        top.append((worst_diameter(s1, theta1, tp, phi_n=5, r_step=50.0, err_n=9)[0], tp, None))
        top.sort(key=lambda t: t[0])
        best = None
        for j, q, _ in top:
            jd, slack, wit = dense_verify(s1, theta1, q)
            if jd < math.inf and slack <= 0.0:
                if best is None or jd < best[0]:
                    best = (jd, q, slack, wit)
        print(f"  best after dense: J={best[0]:.4f} at ({best[1][0]:.3f},{best[1][1]:.3f}) margin={best[2]:+.4f}")
        # 0.5 m 精搜
        c2 = best[1]
        cands2 = fine_search(s1, theta1, c2, 3.0, 0.5)
        top2 = cands2[:6]
        top2.append((worst_diameter(s1, theta1, tp, phi_n=5, r_step=50.0, err_n=9)[0], tp, None))
        top2.sort(key=lambda t: t[0])
        best2 = None
        for j, q, _ in top2:
            jd, slack, wit = dense_verify(s1, theta1, q)
            if jd < math.inf and slack <= 0.0:
                if best2 is None or jd < best2[0]:
                    best2 = (jd, q, slack, wit)
        print(f"  final 0.5m: J={best2[0]:.4f} at ({best2[1][0]:.3f},{best2[1][1]:.3f}) margin={best2[2]:+.4f} witness={best2[3]}")
        print()


if __name__ == "__main__":
    main()
