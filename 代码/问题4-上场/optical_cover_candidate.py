#!/usr/bin/env python3
"""v14 P 分支候选生成：在 optical_plan（保守包围盒）拒绝时，对"实际保守多边形"
直接构造 ≤3 点候选中心集，接受与否仅由连续验证器 optical_cover_certificate 决定。

- 每次决策最多评估 12 组中心（含旧构造），确定性顺序；
- 方向集合：盒长轴（垂直半宽最小轴）、最长顶点对方向、来路方向（机器人→区域质心）；
- 布局：每方向 3 点公式（a∈[w−s,2s] 三档）与 2 点公式（a=2(w−s) 夹取 ≤2s）；
  另加 1 点质心。全部经 certify 复核，COVERED 才采纳；
- 返回 (centers, w, h)（与 optical_plan 同构）或 None；
- 若验证器不可用（ImportError）→ 返回 None（不冒充证书，回退 v13）。
"""
from __future__ import annotations

import math


def _axes(vertices, cur):
    """候选方向集合：[(ux, uy, w, h, lo, hi, lo_q, hi_q, anchor), ...]"""
    out = []
    n = len(vertices)
    best = None
    longest = None
    best_L = -1.0
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
                item = (ux, uy, w, h, min(proj), max(proj), min(perp), max(perp),
                        vertices[i])
                if best is None or h < best[1]:
                    best = item
            if L > best_L:
                best_L = L
                longest = (dx / L, dy / L)
    # 去重加入：盒长轴、最长顶点对方向、来路方向
    seen = set()
    def _add(ux, uy):
        key = (round(ux, 9), round(uy, 9))
        if key in seen or (-round(ux, 9), -round(uy, 9)) in seen:
            return
        seen.add(key)
        proj = [ux * v[0] + uy * v[1] for v in vertices]
        perp = [-uy * v[0] + ux * v[1] for v in vertices]
        out.append((ux, uy, (max(proj) - min(proj)) / 2.0,
                    (max(perp) - min(perp)) / 2.0,
                    min(proj), max(proj), min(perp), max(perp), (0.0, 0.0)))
    if best is not None:
        _add(best[0], best[1])
    if longest is not None:
        _add(longest[0], longest[1])
    if cur is not None:
        cx = sum(v[0] for v in vertices) / n
        cy = sum(v[1] for v in vertices) / n
        dx = cx - cur[0]
        dy = cy - cur[1]
        L = math.hypot(dx, dy)
        if L > 1e-9:
            _add(dx / L, dy / L)
    return out


def _centers3(ax, a):
    ux, uy, w, h, lo, hi, lo_q, hi_q, _anchor = ax
    mid_p = (lo + hi) / 2.0
    mid_q = (lo_q + hi_q) / 2.0
    cx = mid_p * ux - mid_q * uy
    cy = mid_p * uy + mid_q * ux
    return [(cx - a * ux, cy - a * uy), (cx, cy), (cx + a * ux, cy + a * uy)]


def _centers2(ax, a):
    ux, uy, w, h, lo, hi, lo_q, hi_q, _anchor = ax
    mid_p = (lo + hi) / 2.0
    mid_q = (lo_q + hi_q) / 2.0
    cx = mid_p * ux - mid_q * uy
    cy = mid_p * uy + mid_q * ux
    return [(cx - a / 2 * ux, cy - a / 2 * uy), (cx + a / 2 * ux, cy + a / 2 * uy)]


def propose_polygon_plan(vertices, cur=None, max_candidates: int = 12,
                         radius: float = 20.0):
    """返回 (centers, w, h) 或 None。候选全部经连续验证器。"""
    if not vertices or len(vertices) < 3:
        return None
    try:
        from optical_cover_certificate import certify
    except ImportError:
        return None
    cands = []
    # 1 点质心
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    cands.append(([(cx, cy)], None, None))
    for ax in _axes(vertices, cur):
        _ux, _uy, w, h = ax[0], ax[1], ax[2], ax[3]
        if h <= 20.0:
            s = math.sqrt(400.0 - h * h)
            # 3 点：a 三档
            if w <= 3.0 * s + 1e-9:
                a_lo = max(0.0, w - s)
                a_hi = min(2.0 * s, w + s)
                for a in (a_lo, (a_lo + a_hi) / 2.0, a_hi):
                    cands.append((_centers3(ax, a), w, h))
            # 2 点：w ≤ 2s 时可行
            if w <= 2.0 * s + 1e-9:
                a2 = min(2.0 * s, max(0.0, 2.0 * (w - s)))
                cands.append((_centers2(ax, a2), w, h))
    cands = cands[:max_candidates]
    for centers, w, h in cands:
        status, cert = certify(vertices, centers, radius=radius)
        if status == "COVERED":
            return centers, w, h
    return None
