#!/usr/bin/env python3
"""O 分支：给定已合法生成的 ≤3 点光学计划 + 权重模型，输出确定性访问顺序。

本模块只做一次固定排序（失败后不重规划）：
  - 不生成新点、不改变预算、不改变触发条件、不重排全局任务；
  - 输入点集先去重（重合点合并），去重后 1 点直接返回该点；
  - 3 点枚举全部 6 个排列，2 点比较 2 个排列，取期望总代价最小者；
  - 并列时用 v13 基线顺序（距当前位置最近优先）作确定性 tie-break，
    同一输入永远返回同一输出。

权重模型 μ = HEURISTIC_UNIFORM：在 region（保守可行域凸多边形）上的均匀质量。
HEURISTIC 声明：这是用于排序的启发式面积权重，不是连续覆盖证书。
圆（半径 20 m）用 256 边内接多边形近似后与 region 做凸多边形裁剪，
重叠圆按“首次命中”事件计算联合质量（容斥原理，绝不独立相乘）。

计费口径（任务书 §2.3 / §4.2，与 v13 契约一致）：
  移动 5 m/s；光学失败 3 s；成功 = 3 s 光学定位 + 2 s 激光，合计 5 s。
  E[T_local] = T_start_switch + Σ_i s_i·(d_i/5 + 3) + 2
    —— Σ_i s_i·3 中已含成功那一跳的 3 s 光学定位，末尾 +2 只是激光，
       成功点总计 5 s，绝不重复计 3+5。
  T_start_switch：本环境 /clear 不切换频道 ⇒ = 0（常量显式保留，见下）。
  总排序代价 = E[T_local] + Σ_i p_i·C_resume(c_i)，
  C_resume(c_i) = dist(c_i, resume_anchor)/5，标注为
  “冻结下一合法任务锚点路程代理”，不重排全局任务；resume_anchor=None 时为 0。

weight_model 回退（fallback）：region=None、面积≤0、非有限输入、>3 点
（O 契约只覆盖 ≤3 点，K4 四点计划不套概率模型）⇒ 返回基线 nearest-first
顺序，cost_dict 标 fallback，绝不增删真实可行域。
"""
from __future__ import annotations

import math
from itertools import combinations, permutations

# ---------------------------------------------------------------------------
# 计费常量（冻结契约，任务书 §2.3；改契约只改这里）
# ---------------------------------------------------------------------------
MOVE_SPEED = 5.0        # 移动 5 m/s
SWITCH_TIME = 1.0       # 切台 1 s（本环境 /clear 不切台，未计入）
CLEAR_FAIL_TIME = 3.0   # 光学失败 3 s
CLEAR_LOCATE_TIME = 3.0 # 成功清障里的光学定位 3 s
LASER_TIME = 2.0        # 成功清障里的激光 2 s（一次成功的 +2）
CLEAR_SUCCESS_TOTAL = CLEAR_LOCATE_TIME + LASER_TIME  # 成功合计 5 s
CLEAR_RADIUS = 20.0     # 光学清除半径 20 m
# /clear 不切换频道 ⇒ 本轮环境无切台费。显式保留该项：
# 若契约未来变为先切台再 /clear，把该项改成 SWITCH_TIME 即可，
# 公式结构不变（E[T_local] = T_start_switch + ... + 2）。
T_START_SWITCH = 0.0

GON_EDGES = 256         # 圆 → 256 边内接多边形（HEURISTIC 面积权重）
_DEDUP_TOL = 1e-6       # 重合点合并容差（m）
_TIE_EPS = 1e-9         # 并列判定容差（s），用于确定性 tie-break
_CLIP_EPS = 1e-9        # 凸多边形裁剪内半平面容差
_MAX_CENTERS = 3        # O 契约：≤3 点


# ---------------------------------------------------------------------------
# 几何纯函数（无状态、无随机，输出确定）
# ---------------------------------------------------------------------------
def _signed_area(poly):
    """鞋带公式（带符号）。poly: [(x,y), ...]"""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def polygon_area(poly):
    """多边形面积（绝对值）。"""
    return abs(_signed_area(poly))


def _ensure_ccw(poly):
    """规范化方向为逆时针（CCW）；退化输入原样返回。"""
    if _signed_area(poly) >= 0.0:
        return list(poly)
    return list(reversed(poly))


def _circle_gon(cx, cy, radius, edges):
    """圆 → edges 边内接多边形（顶点在圆上、整体在圆内，CCW）。
    内接 ⇒ 面积略小于真实圆（保守方向，见报告）。"""
    pts = []
    for k in range(edges):
        ang = 2.0 * math.pi * k / edges
        pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    return pts


def _edge_inter(p, q, a, b):
    """线段 p→q 与直线 a→b 的交点（Sutherland–Hodgman 辅助）。"""
    px, py = p
    qx, qy = q
    ax, ay = a
    bx, by = b
    rx, ry = qx - px, qy - py
    ex, ey = bx - ax, by - ay
    den = rx * ey - ry * ex
    if abs(den) < 1e-14:  # 平行/共线：保守取中点
        return ((px + qx) * 0.5, (py + qy) * 0.5)
    t = ((ax - px) * ey - (ay - py) * ex) / den
    return (px + t * rx, py + t * ry)


def _clip_convex(subject, clip_poly):
    """用凸多边形 clip_poly（CCW）裁剪 subject（凸多边形）。
    Sutherland–Hodgman：内部 = 每条边（a→b）的左侧。"""
    out = [(x, y) for x, y in subject]
    if not out:
        return []
    for i in range(len(clip_poly)):
        ax, ay = clip_poly[i]
        bx, by = clip_poly[(i + 1) % len(clip_poly)]
        ex, ey = bx - ax, by - ay
        if ex == 0.0 and ey == 0.0:
            continue
        inp = out
        out = []
        if not inp:
            return []
        sx, sy = inp[-1]
        sin = ex * (sy - ay) - ey * (sx - ax) >= -_CLIP_EPS
        for j in range(len(inp)):
            tx, ty = inp[j]
            tin = ex * (ty - ay) - ey * (tx - ax) >= -_CLIP_EPS
            if tin:
                if not sin:
                    out.append(_edge_inter((sx, sy), (tx, ty), (ax, ay), (bx, by)))
                out.append((tx, ty))
            elif sin:
                out.append(_edge_inter((sx, sy), (tx, ty), (ax, ay), (bx, by)))
            sx, sy = tx, ty
            sin = tin
    return out


def _finite(p):
    return all(math.isfinite(v) for v in p)


# ---------------------------------------------------------------------------
# 首次命中概率（HEURISTIC_UNIFORM）
# ---------------------------------------------------------------------------
def first_hit_probs(centers, region, radius=CLEAR_RADIUS, gon_edges=GON_EDGES):
    """按给定访问顺序（centers 的顺序即访问顺序）计算首次命中质量。

    返回 (s, p, mass, region_area)：
      mass[i]  = μ(C_i ∩ region)；
      s[i]     = μ(X ∉ ∪_{j<i} C_j)，s[0] = 1；
      p[i]     = μ(X ∈ C_i 且 X ∉ ∪_{j<i} C_j) —— “首次命中”事件。
    联合质量用容斥原理（2 圆/3 圆交面积）计算，重叠圆不重复计质量。
    圆用 gon_edges 边内接多边形近似后与 region 做凸多边形裁剪。
    HEURISTIC 权重：不构成覆盖证书，仅用于排序。

    调用方保证 region 非空、面积 > 0 且全有限（否则应回退 fallback）。
    """
    n = len(centers)
    area = polygon_area(region)
    if not (area > 0.0) or not math.isfinite(area):
        raise ValueError("region 质量退化（面积≤0 或非有限），应先回退 fallback")
    region = _ensure_ccw(region)
    gons = [_circle_gon(cx, cy, radius, gon_edges) for cx, cy in centers]
    cache = {}

    def inter_area(idx):
        """region ∩ 各圆（gon 近似）交集的面积。"""
        key = tuple(sorted(idx))
        if key in cache:
            return cache[key]
        poly = list(region)
        for i in key:
            poly = _clip_convex(poly, gons[i])
        a = polygon_area(poly)
        cache[key] = a
        return a

    mass = [inter_area((i,)) / area for i in range(n)]

    def union_mass(idxs):
        """μ(∪_{i∈idxs} C_i)（容斥，|idxs|≤3）。"""
        idxs = tuple(idxs)
        total = 0.0
        for size in range(1, len(idxs) + 1):
            for comb in combinations(idxs, size):
                a = inter_area(comb) / area
                total += a if size % 2 == 1 else -a
        return total

    s = [1.0] + [1.0 - union_mass(range(k)) for k in range(1, n)]
    p = []
    for i in range(n):
        if i == 0:
            p.append(mass[0])
            continue
        prev = tuple(range(i))
        overlap = 0.0
        for size in range(1, i + 1):
            for comb in combinations(prev, size):
                a = inter_area((i,) + comb) / area
                overlap += a if size % 2 == 1 else -a
        p.append(mass[i] - overlap)
    # 数值保护：概率截到 [0,1]
    s = [min(1.0, max(0.0, v)) for v in s]
    p = [min(1.0, max(0.0, v)) for v in p]
    return s, p, mass, area


# ---------------------------------------------------------------------------
# 冻结接口
# ---------------------------------------------------------------------------
def rank_centers(centers, current_xy, region=None, resume_anchor=None):
    """centers: ≤3 个 (x,y)（先去重：重合点合并；去重后 1 点直接返回该点）
    region: 保守可行域凸多边形顶点（None 时无权重信息）
    resume_anchor: 冻结的下一合法任务锚点 (x,y)（None 时接续项为 0）
    返回 (order, cost_dict)
    order: 访问顺序 list[(x,y)]（与输入点集相同元素）
    cost_dict: 含 chosen_cost、per_permutation（6 排列的成本）、s_i/p_i 数组、
      weight_model（"HEURISTIC_UNIFORM"/"fallback"）、各排列的路径距离、接续代理成本等
    """
    constants = {
        "move_speed": MOVE_SPEED,
        "switch_time": SWITCH_TIME,
        "clear_fail_time": CLEAR_FAIL_TIME,
        "laser_time": LASER_TIME,
        "clear_success_total": CLEAR_SUCCESS_TOTAL,
        "T_start_switch": T_START_SWITCH,
        "clear_radius": CLEAR_RADIUS,
        "gon_edges": GON_EDGES,
    }
    centers = [tuple(map(float, c)) for c in centers]
    cur = tuple(map(float, current_xy))
    anchor = None if resume_anchor is None else tuple(map(float, resume_anchor))

    # 1) 去重：重合点合并，保留先出现者（确定性）
    dedup = []
    for c in centers:
        if not any(math.hypot(c[0] - d[0], c[1] - d[1]) <= _DEDUP_TOL
                   for d in dedup):
            dedup.append(c)
    centers = dedup
    n = len(centers)

    # 2) 基线顺序 = v13 的最近优先（与正式代码逐字一致：
    #    sorted(centers, key=lambda c: math.dist(cur, c))）
    if _finite(cur) and all(_finite(c) for c in centers):
        baseline = sorted(centers, key=lambda c: math.dist(cur, c))
    else:
        baseline = list(centers)  # 距离无定义：按输入顺序，确定性

    # 3) 权重模型可用性检查
    reason = None
    reg = None
    if n == 0:
        reason = "no_centers"
    elif n > _MAX_CENTERS:
        reason = "n_centers_gt_3"  # O 契约 ≤3；K4 四点计划不套概率模型
    elif not _finite(cur):
        reason = "current_non_finite"
    elif not all(_finite(c) for c in centers):
        reason = "center_non_finite"
    elif region is None:
        reason = "region_none"
    else:
        reg = [tuple(map(float, v)) for v in region]
        if len(reg) < 3 or not all(_finite(v) for v in reg):
            reason = "region_invalid"
        else:
            area = polygon_area(reg)
            if not (area > 0.0) or not math.isfinite(area):
                reason = "region_degenerate"

    if reason is not None:
        return baseline, _fallback_cost_dict(
            centers, cur, reg, baseline, reason, anchor, constants)

    # 4) 全排列枚举（n=1：1 个；n=2：2 个；n=3：6 个）
    baseline_indices = tuple(centers.index(pt) for pt in baseline)
    rows = []
    for perm in permutations(range(n)):
        pts = [centers[i] for i in perm]
        s, p, mass, _ = first_hit_probs(pts, reg, CLEAR_RADIUS, GON_EDGES)
        path = [cur] + pts
        dists = [
            math.hypot(path[i][0] - path[i - 1][0],
                       path[i][1] - path[i - 1][1])
            for i in range(1, n + 1)
        ]
        # E[T_local] = T_start_switch + Σ s_i·(d_i/5 + 3) + 2
        # （s_i·3 含成功点的 3 s 定位；+2 是激光；成功合计 5 s，不重复计 3+5）
        local = (
            T_START_SWITCH
            + sum(si * (di / MOVE_SPEED + CLEAR_FAIL_TIME)
                  for si, di in zip(s, dists))
            + LASER_TIME
        )
        if anchor is not None and _finite(anchor):
            resume_terms = [
                math.hypot(px - anchor[0], py - anchor[1]) / MOVE_SPEED
                for px, py in pts
            ]
        else:
            resume_terms = []
        # 接续代理：冻结下一合法任务锚点路程代理，不重排全局任务
        resume = (sum(pi * rt for pi, rt in zip(p, resume_terms))
                  if resume_terms else 0.0)
        total = local + resume
        # 剩余质量：X 不在任何已访问圆内（事件划分 {首次命中 i} ∪ {全不命中}
        # 满足 Σp + residual = 1；注意成本公式的 s_n = 1−μ(U_{n−1}) 与之不同）
        residual = max(0.0, 1.0 - sum(p))
        rows.append({
            "indices": perm,
            "order": pts,
            "is_baseline": perm == baseline_indices,
            "path_distances": dists,
            "s": s,
            "p": p,
            "mass": mass,
            "residual": residual,
            "local_cost": local,
            "resume_cost": resume,
            "total_cost": total,
            "resume_proxy_distances": resume_terms,
            "tied": False,
        })

    # 5) 选择：最小总代价；并列（差 ≤ _TIE_EPS）时优先基线顺序，
    #    否则按索引元组字典序（完全确定性）
    min_total = min(r["total_cost"] for r in rows)
    for r in rows:
        r["tied"] = abs(r["total_cost"] - min_total) <= _TIE_EPS
    tied_rows = [r for r in rows if r["tied"]]
    baseline_rows = [r for r in tied_rows if r["is_baseline"]]
    chosen = (baseline_rows[0] if baseline_rows
              else min(tied_rows, key=lambda r: r["indices"]))
    order = [centers[i] for i in chosen["indices"]]
    cost_dict = {
        "weight_model": "HEURISTIC_UNIFORM",
        "fallback": False,
        "fallback_reason": None,
        "n_points": n,
        "centers": centers,
        "current_xy": cur,
        "region": reg,
        "region_area": polygon_area(reg),
        "radius": CLEAR_RADIUS,
        "gon_edges": GON_EDGES,
        "baseline_order": baseline,
        "order": order,
        "chosen_indices": list(chosen["indices"]),
        "chosen_cost": chosen["total_cost"],
        "chosen_local_cost": chosen["local_cost"],
        "chosen_resume_cost": chosen["resume_cost"],
        "s": chosen["s"],
        "p": chosen["p"],
        "mass": chosen["mass"],
        "residual": chosen["residual"],
        "resume_anchor": anchor,
        "resume_proxy_distances": chosen["resume_proxy_distances"],
        "per_permutation": rows,
        "tie_break_used": len(tied_rows) > 1,
        "constants": constants,
        "note": ("HEURISTIC_UNIFORM 权重仅用于排序，不是连续覆盖证书；"
                 "圆用 256 边内接多边形近似（保守）。"),
    }
    return order, cost_dict


def _fallback_cost_dict(centers, cur, reg, baseline, reason, anchor, constants):
    """回退：基线 nearest-first（或输入顺序），期望成本无定义（标 fallback）。"""
    n = len(centers)
    baseline_indices = tuple(centers.index(pt) for pt in baseline)
    rows = []
    for perm in permutations(range(n)):
        pts = [centers[i] for i in perm]
        finite_ok = _finite(cur) and all(_finite(c) for c in centers)
        dists = []
        if finite_ok:
            path = [cur] + pts
            dists = [
                math.hypot(path[i][0] - path[i - 1][0],
                           path[i][1] - path[i - 1][1])
                for i in range(1, n + 1)
            ]
        resume_terms = []
        if anchor is not None and _finite(anchor) and all(_finite(c) for c in pts):
            resume_terms = [
                math.hypot(px - anchor[0], py - anchor[1]) / MOVE_SPEED
                for px, py in pts
            ]
        rows.append({
            "indices": perm,
            "order": pts,
            "is_baseline": perm == baseline_indices,
            "path_distances": dists,
            "s": None,
            "p": None,
            "mass": None,
            "residual": None,
            "local_cost": None,
            "resume_cost": None,
            "total_cost": None,
            "resume_proxy_distances": resume_terms,
            "tied": False,
        })
    base_terms = []
    if anchor is not None and _finite(anchor) and all(_finite(c) for c in baseline):
        base_terms = [
            math.hypot(px - anchor[0], py - anchor[1]) / MOVE_SPEED
            for px, py in baseline
        ]
    return {
        "weight_model": "fallback",
        "fallback": True,
        "fallback_reason": reason,
        "n_points": n,
        "centers": centers,
        "current_xy": cur,
        "region": reg,
        "region_area": None,
        "radius": CLEAR_RADIUS,
        "gon_edges": GON_EDGES,
        "baseline_order": baseline,
        "order": baseline,
        "chosen_indices": list(baseline_indices),
        "chosen_cost": None,
        "chosen_local_cost": None,
        "chosen_resume_cost": None,
        "s": None,
        "p": None,
        "mass": None,
        "residual": None,
        "resume_anchor": anchor,
        "resume_proxy_distances": base_terms,
        "per_permutation": rows,
        "tie_break_used": False,
        "constants": constants,
        "note": ("权重模型不可用（%s）⇒ 回退 v13 基线 nearest-first 顺序；"
                 "期望成本无定义，不增删真实可行域。" % reason),
    }
