#!/usr/bin/env python3
"""v14 K4：四点条带计划（仅当 v13 无 ≤3 点方案时尝试）。

- 构造：保守区域顶点的最小垂直半宽轴向（同 optical_plan 轴选择），
  沿轴向把投影范围均分 4 条带，每带中心 = 带中点（轴向）× 垂直中线；
- 独立条带证书（连续）：带内任意点到带中心的距离 ≤ √((w/8)² + h²)
  （带为投影切片 ∩ 保守盒，最大值在带角）；要求 √((w/8)² + h²) ≤ 20；
- 若 optical_cover_certificate 可用，额外用 P 验证器复核（COVERED 才返回）；
- 预算契约（v14 集成处执行）：仅当 optical_fails[ch]==0 且每频道每局 ≤1 个
  K4 计划；执行最坏 3 失败 + 1 成功 = 4 次调用（该频道上限 +1 仅限 K4 分支）。
  第四点仍失败 ⇒ 证据矛盾：记录快照、停止新 K4、交回 v13 异常流程。
  失败即覆盖推断：带 i ⊆ 圆(中心_i, 20)，中心 i 失败 ⇒ 源不在带 i
  ⇒（源 ∈ P 前提下）必在剩余带内，第 4 带中心必然成功。
"""
from __future__ import annotations

import math


def _axis_wh(vertices):
    best = None
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            dx = vertices[j][0] - vertices[i][0]
            dy = vertices[j][1] - vertices[i][1]
            L = math.hypot(dx, dy)
            if L < 1e-9:
                continue
            for ux, uy in ((dx / L, dy / L), (-dy / L, dx / L)):
                proj = [ux * (v[0] - vertices[i][0]) + uy * (v[1] - vertices[i][1])
                        for v in vertices]
                perp = [-uy * (v[0] - vertices[i][0]) + ux * (v[1] - vertices[i][1])
                        for v in vertices]
                w = (max(proj) - min(proj)) / 2.0
                h = (max(perp) - min(perp)) / 2.0
                if best is None or h < best[1]:
                    best = (ux, uy, w, h, min(proj), max(proj), min(perp), max(perp),
                            vertices[i])
    return best


def build_strip_plan(vertices, current_xy=None, radius: float = 20.0):
    """返回 (centers4, cert) 或 None。centers4 为 4 个 (x,y)。"""
    if not vertices or len(vertices) < 3:
        return None
    ax = _axis_wh(vertices)
    if ax is None:
        return None
    ux, uy, w, h, lo, hi, lo_q, hi_q, anchor = ax
    halfband = (hi - lo) / 8.0
    maxd2 = halfband * halfband + h * h
    if maxd2 > radius * radius + 1e-9:
        return None
    mid_q = (lo_q + hi_q) / 2.0
    centers = []
    for k in range(4):
        p = lo + (2 * k + 1) * halfband
        cx = anchor[0] + p * ux - mid_q * uy
        cy = anchor[1] + p * uy + mid_q * ux
        centers.append((cx, cy))
    # P 验证器复核（可用时）
    try:
        from optical_cover_certificate import certify
        status, cert = certify(vertices, centers, radius=radius)
        if status != "COVERED":
            return None
        strip_cert = {
            "axis": [round(ux, 9), round(uy, 9)],
            "w": w, "h": h, "halfband": halfband,
            "max_dist": math.sqrt(maxd2),
            "max_dist2": maxd2,
            "certificate": "strip_arg + certify",
            "certifier_status": status,
            "cert": cert,
        }
    except ImportError:
        strip_cert = {
            "axis": [round(ux, 9), round(uy, 9)],
            "w": w, "h": h, "halfband": halfband,
            "max_dist": math.sqrt(maxd2),
            "max_dist2": maxd2,
            "certificate": "strip_arg_only",
        }
    # 访问顺序：距当前位置最近优先（O 分支不重排 K4）
    if current_xy is not None:
        centers = sorted(centers, key=lambda c: math.dist(current_xy, c))
    return centers, strip_cert


def budget_allows(used_fails: int, cap_total: int = 4) -> bool:
    """K4 最坏 3 失败 + 1 成功 = 4 次调用；used_fails + 4 ≤ cap_total。"""
    return used_fails + 4 <= cap_total


if __name__ == "__main__":
    # 夹具自测
    P = [(0, -5), (150, -5), (150, 5), (0, 5)]
    got = build_strip_plan(P, current_xy=(0, 0))
    print("K4 夹具 P=[0,150]×[-5,5]:", "PASS" if got is not None else "FAIL")
    if got:
        centers, cert = got
        print("  中心:", [(round(c[0], 2), round(c[1], 2)) for c in centers])
        print("  最远距离:", round(cert["max_dist"], 4), "≤20:", cert["max_dist"] <= 20.0)
        print("  预算 used=0:", budget_allows(0), " used=1:", budget_allows(1))
