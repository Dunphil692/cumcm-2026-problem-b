#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q4 v14 · P 分支：凸多边形 → 连续光学覆盖证书（纯函数、保守数值验证器）。

冻结接口（v14 主控已按此接线，不得更改签名与返回值契约）：

    certify(vertices, centers, radius=20.0, margin=0.0) -> (status, cert)

    status ∈ {"COVERED", "NOT_COVERED", "UNKNOWN"}

    vertices : 多边形顶点 list[tuple[float,float]]。可顺时针/逆时针、可含共线点
               （自行规范化）；非凸输入先取保守凸包（cert 注明 hulled=True）。
    centers  : 候选光学中心 list[tuple[float,float]]，1..4 个（可含重复/共线，
               均已处理；多于 4 个也按同一数学判据泛化处理）。
    radius   : 光学清除半径（米），必须为正有限数。
    margin   : 安全裕度（米），非负有限数；有效半径 r_eff = radius - margin。
               裕度只会使证书更严格，绝不把容差加到 radius 上放宽物理半径。

    cert（dict，至少含以下 keys）：
        status, max_dist（最坏分区顶点到其中心的距离上界）,
        partitions（list，每项 {center_index, center, vertices 保守分区顶点/端点,
                   worst_vertex, worst_dist2, empty}）,
        margin_used, source_hash（vertices+centers+radius 规范化 sha256）,
        version, error_bound

数学判据（任务书 §5.2；完整证明见 CONTINUOUS_COVERAGE_PROOF.md）：

    P_i = P ∩ {x : 2(c_j − c_i)·x ≤ ‖c_j‖² − ‖c_i‖² for all j}   （P 内离 c_i 最近的 Voronoi 分区）
    精确算术下：P 被半径 r 的圆并集覆盖 ⟺ 每个非空 P_i 的每个顶点 v 满足
    ‖v − c_i‖ ≤ r。理由：‖x − c_i‖² 是凸函数，凸函数在紧致凸多边形上的最大值
    在顶点（极值点）取得；边/点退化分区用端点或单点检查。

数值纪律（保守，详见证明文档 §数值误差处理）：

    - 半平面裁剪（Sutherland–Hodgman）使用外扩容差 CLIP_TOL_REL=1e-9（相对）：
      分区只放大不缩小，绝不向内裁剪制造缝隙。
    - 距离一律用保守上界/下界（DIST_ROUND_REL=1e-12 相对包裹浮点误差）。
    - error_bound = max(1e-6 × scale, 1e-12)，scale 为坐标/半径尺度。
    - COVERED   ：每个非空分区的每个顶点 dist² 上界 ≤ (r_eff − error_bound)²。
                  由此可严格推出 P 被半径 r_eff = radius − margin ≤ radius 的圆
                  并集覆盖；全部数值缝隙（≤1e-9 相对）被 error_bound 吸收，
                  绝不放宽 radius。
    - NOT_COVERED：存在证点（分区顶点或凸包顶点），其到所有中心的距离下界
                  > radius + 2·error_bound（用完整 radius 判定，margin 不参与，
                  绝不因 margin 放宽证否阈值）；证点必在 P 的 error_bound 邻域内，
                  故可推出 P 内存在真正未被任何半径 radius 的圆覆盖的点
                  （证据在 r+error_bound 外）。
    - UNKNOWN    ：近阈值（|dist − r| < error_bound 量级）无法可靠判定，或输入退化。
                  调用方必须回退基线策略；绝不能把 UNKNOWN 当 COVERED。
    - 网格采样只可用于自检，绝不作为判定依据（本模块完全不采样）。

状态语义提醒：

    NOT_COVERED 只否证“这组圆覆盖当前保守 P”，不代表真实可行域 Ω 无法被覆盖，
    更不代表不存在其他（≤3 圆等）覆盖方案。COVERED 只保证该光学计划在规定前提
    （P 保守包含全部可能位置、r 为物理半径）下覆盖当前可能位置，不自动证明全局
    搜索一定终止。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

VERIFIER_VERSION = "0.14.0"
ALGORITHM = "voronoi-partition + half-plane clip + vertex maximum"
CLIP_TOL_REL = 1e-9        # 裁剪外扩容差（相对）
DIST_ROUND_REL = 1e-12     # 距离上/下界包裹因子（相对）
ERROR_BOUND_REL = 1e-6     # 判定误差界（相对 scale）
MIN_ERROR_BOUND = 1e-12


class _InputError(ValueError):
    """输入格式/数值错误（统一映射为 UNKNOWN 证书）。"""


def _parse_points(raw, name):
    if not isinstance(raw, (list, tuple)):
        raise _InputError(f"{name} 必须是 (x, y) 数值对的列表")
    pts = []
    for p in raw:
        try:
            x, y = p
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            raise _InputError(f"{name} 必须由 (x, y) 数值对构成")
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise _InputError(f"{name} 含非有限数值（NaN/Inf）")
        pts.append((fx, fy))
    return pts


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _signed_area2(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total


def _convex_hull(points):
    """严格凸包：CCW、去重复、去共线（单调链，浮点精确运算）。"""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _input_flags(raw_v):
    """(normalized, hulled)：输入是否被规范化（方向/共线）、是否被凸包化。"""
    inp = []
    for p in raw_v:
        if not inp or p != inp[-1]:
            inp.append(p)
    normalized = len(inp) != len(raw_v)
    hulled = False
    if len(inp) >= 3 and inp[0] == inp[-1]:
        inp.pop()
        normalized = True
    if len(inp) >= 3:
        area2 = _signed_area2(inp)
        if area2 < 0.0:
            normalized = True
            oriented = list(reversed(inp))
        else:
            oriented = inp
        n = len(oriented)
        for k in range(n):
            cr = _cross(oriented[k - 1], oriented[k], oriented[(k + 1) % n])
            if cr < 0.0:
                hulled = True
            elif cr == 0.0:
                normalized = True
    return normalized, hulled


def _dedup_poly(poly, tol):
    if not poly:
        return []
    out = [poly[0]]
    for p in poly[1:]:
        if math.dist(p, out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and math.dist(out[0], out[-1]) <= tol:
        out.pop()
    return out


def _clip_halfplane(poly, a, b, c, tol_rel, scale):
    """poly ∩ {(x, y) : a·x + b·y ≤ c}（Sutherland–Hodgman，外扩容差）。

    (a, b) 已归一化为单位向量。顶点 p 的容差 tol_p = tol_rel·(1+|px|+|py|+|c|)：
    value ≤ +tol_p 判为在内 —— 分区只会放大（外包络），不会向内收缩。
    仅在 value 真正跨过 0 时（符号相反）求交点，交点落在分界线上。
    """
    if not poly:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        p = poly[i]
        q = poly[(i + 1) % n]
        vp = a * p[0] + b * p[1] - c
        vq = a * q[0] + b * q[1] - c
        tp = tol_rel * (1.0 + abs(p[0]) + abs(p[1]) + abs(c))
        tq = tol_rel * (1.0 + abs(q[0]) + abs(q[1]) + abs(c))
        in_p = vp <= tp
        in_q = vq <= tq
        if in_p:
            out.append(p)
        if (vp <= 0.0) != (vq <= 0.0):
            denom = vp - vq
            if denom != 0.0:
                t = vp / denom
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return _dedup_poly(out, tol_rel * max(1.0, scale) * 4.0)


def _dist2_upper(p, c, scale):
    """‖p − c‖² 的保守上界。"""
    dx = p[0] - c[0]
    dy = p[1] - c[1]
    d2 = dx * dx + dy * dy
    return d2 * (1.0 + DIST_ROUND_REL) + DIST_ROUND_REL * scale * scale


def _dist2_lower(p, c, scale):
    """‖p − c‖² 的保守下界。"""
    dx = p[0] - c[0]
    dy = p[1] - c[1]
    d2 = dx * dx + dy * dy
    val = d2 * (1.0 - DIST_ROUND_REL) - DIST_ROUND_REL * scale * scale
    return val if val > 0.0 else 0.0


def _canonical_float(x):
    return repr(float(x))


def source_hash(vertices, centers, radius):
    """vertices + centers + radius 的规范化 sha256（与输入顺序/方向/共线点无关）。

    规范化：顶点取严格凸包后按 (x, y) 排序、中心按 (x, y) 排序、浮点用 repr
    规范化，JSON（sort_keys）后 sha256。
    """
    try:
        vs = _parse_points(vertices, "vertices")
        cs = _parse_points(centers, "centers")
    except _InputError:
        vs, cs = [], []
    hull = _convex_hull(vs)
    payload = {
        "polygon": [[_canonical_float(x), _canonical_float(y)] for x, y in sorted(hull)],
        "centers": sorted(
            [[_canonical_float(x), _canonical_float(y)] for x, y in cs]
        ),
        "radius": _canonical_float(radius),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def certify(vertices, centers, radius=20.0, margin=0.0):
    """连续光学覆盖证书（冻结接口，见模块 docstring）。"""
    base = {
        "status": "UNKNOWN",
        "max_dist": None,
        "partitions": [],
        "margin_used": None,
        "source_hash": "",
        "version": VERIFIER_VERSION,
        "error_bound": None,
        "reason": None,
        "radius": None,
        "radius_effective": None,
        "n_centers": None,
        "n_partitions_nonempty": None,
        "hulled": False,
        "normalized": False,
        "hull": [],
        "witness": None,
        "algorithm": ALGORITHM,
        "eps": {
            "clip_tol_rel": CLIP_TOL_REL,
            "dist_round_rel": DIST_ROUND_REL,
            "error_bound_rel": ERROR_BOUND_REL,
            "scale": None,
        },
    }

    def finish(status, reason=None, **extra):
        base["status"] = status
        base["reason"] = reason
        base.update(extra)
        return status, base

    try:
        # ---------- 输入解析与校验（退化/非法输入绝不返回 COVERED） ----------
        try:
            radius_f = float(radius)
            margin_f = float(margin)
        except (TypeError, ValueError):
            return finish("UNKNOWN", "invalid_radius_or_margin: 必须为数值")
        if not (math.isfinite(radius_f) and radius_f > 0.0):
            return finish("UNKNOWN", "invalid_radius: 必须为正有限数")
        if not (math.isfinite(margin_f) and margin_f >= 0.0):
            return finish("UNKNOWN", "invalid_margin: 必须为非负有限数")
        base["margin_used"] = margin_f
        base["radius"] = radius_f
        try:
            raw_v = _parse_points(vertices, "vertices")
            raw_c = _parse_points(centers, "centers")
        except _InputError as exc:
            return finish("UNKNOWN", f"invalid_input: {exc}")
        base["source_hash"] = source_hash(raw_v, raw_c, radius_f)
        base["n_centers"] = len(raw_c)
        if not raw_v:
            return finish("UNKNOWN", "empty_polygon: vertices 为空")
        if len(raw_v) < 3:
            return finish(
                "UNKNOWN", "degenerate_polygon: 顶点数 < 3（线段/单点不构成二维区域）"
            )
        if not raw_c:
            return finish("UNKNOWN", "no_centers: centers 为空")

        # ---------- 规范化：保守凸包（CCW、去共线；绝不缩小可行域） ----------
        hull = _convex_hull(raw_v)
        if len(hull) < 3:
            return finish("UNKNOWN", "zero_area_polygon: 顶点共线/零面积")
        base["hull"] = [[x, y] for x, y in hull]
        normalized, hulled = _input_flags(raw_v)
        base["normalized"] = normalized
        base["hulled"] = hulled

        # ---------- 尺度与误差界 ----------
        coords = [abs(x) for v in hull for x in v] + [
            abs(x) for c in raw_c for x in c
        ]
        scale = max([1.0, radius_f, margin_f] + coords)
        error_bound = max(ERROR_BOUND_REL * scale, MIN_ERROR_BOUND)
        base["error_bound"] = error_bound
        base["eps"]["scale"] = scale

        r_eff = radius_f - margin_f
        base["radius_effective"] = r_eff
        if r_eff <= 2.0 * error_bound:
            return finish(
                "UNKNOWN", "radius_too_small: 有效半径不超过数值误差界（或 margin ≥ radius）"
            )
        cover_limit2 = (r_eff - error_bound) * (r_eff - error_bound)
        # 证否阈值用完整 radius（margin 不参与）：NOT_COVERED 是半径 r 圆组的严格否证
        refute_limit2 = (radius_f + 2.0 * error_bound) * (radius_f + 2.0 * error_bound)
        base["thresholds"] = {
            "radius_effective": r_eff,
            "error_bound": error_bound,
            "cover_limit2": cover_limit2,
            "refute_limit2": refute_limit2,
        }

        # ---------- Voronoi 分区：逐中心半平面裁剪 ----------
        partitions = []
        for i, ci in enumerate(raw_c):
            poly = list(hull)
            for j, cj in enumerate(raw_c):
                if j == i:
                    continue
                a = 2.0 * (cj[0] - ci[0])
                b = 2.0 * (cj[1] - ci[1])
                c = (cj[0] * cj[0] + cj[1] * cj[1]) - (ci[0] * ci[0] + ci[1] * ci[1])
                nrm = math.hypot(a, b)
                if nrm == 0.0 or not math.isfinite(nrm):
                    # 完全重合的中心：分界面退化为 0·x ≤ 0（恒真），跳过裁剪
                    continue
                poly = _clip_halfplane(poly, a / nrm, b / nrm, c / nrm,
                                       CLIP_TOL_REL, scale)
                if not poly:
                    break
            if poly:
                d2s = [_dist2_upper(p, ci, scale) for p in poly]
                k = max(range(len(d2s)), key=d2s.__getitem__)
                worst_v, worst_d2 = poly[k], d2s[k]
            else:
                worst_v, worst_d2 = None, None
            partitions.append({
                "center_index": i,
                "center": [ci[0], ci[1]],
                "vertices": [[x, y] for x, y in poly],
                "worst_vertex": None if worst_v is None else [worst_v[0], worst_v[1]],
                "worst_dist2": worst_d2,
                "empty": not poly,
            })
        base["partitions"] = partitions
        nonempty = [p for p in partitions if not p["empty"]]
        base["n_partitions_nonempty"] = len(nonempty)

        worst_global = max((p["worst_dist2"] for p in nonempty), default=None)
        if worst_global is not None:
            base["max_dist"] = (
                math.sqrt(worst_global) * (1.0 + DIST_ROUND_REL)
                + DIST_ROUND_REL * scale
            )
        if not nonempty:
            return finish("UNKNOWN", "empty_partitioning: 数值退化，无法建立分区")

        # ---------- 判定 1：COVERED ----------
        if all(p["worst_dist2"] <= cover_limit2 for p in nonempty):
            return finish("COVERED")

        # ---------- 判定 2：NOT_COVERED（严格证点） ----------
        candidates = []
        for p in nonempty:
            for v in p["vertices"]:
                candidates.append(
                    (tuple(v), f"partition vertex of center #{p['center_index']}")
                )
        for v in hull:
            candidates.append((v, "hull vertex of P"))
        for v, kind in candidates:
            d2_low = min(_dist2_lower(v, c, scale) for c in raw_c)
            if d2_low > refute_limit2:
                base["witness"] = {
                    "vertex": [v[0], v[1]],
                    "min_dist2_over_centers": d2_low,
                    "kind": kind,
                }
                return finish("NOT_COVERED")

        return finish("UNKNOWN", "near_threshold: 无严格在 r+error_bound 之外的证点")
    except Exception as exc:  # 验证器自身异常兜底：绝不误报 COVERED
        base["status"] = "UNKNOWN"
        base["reason"] = f"internal_error: {exc}"
        return "UNKNOWN", base


def main(argv=None):
    """独立复核命令：python3 optical_cover_certificate.py input.json

    input.json = {"vertices": [[x,y],...], "centers": [[x,y],...],
                  "radius": 20.0, "margin": 0.0}
    退出码：0=COVERED, 1=NOT_COVERED, 2=UNKNOWN。
    """
    ap = argparse.ArgumentParser(
        description="独立复核：连续光学覆盖证书（生成器与验证器分离）"
    )
    ap.add_argument("input_json", help="含 vertices/centers 的 JSON 文件")
    ap.add_argument("--radius", type=float, default=20.0, help="覆盖半径（默认 20.0）")
    ap.add_argument("--margin", type=float, default=0.0, help="安全裕度（默认 0.0）")
    args = ap.parse_args(argv)
    with open(args.input_json, encoding="utf-8") as fh:
        data = json.load(fh)
    vertices = data.get("vertices", data.get("polygon"))
    centers = data["centers"]
    status, cert = certify(
        vertices,
        centers,
        radius=float(data.get("radius", args.radius)),
        margin=float(data.get("margin", args.margin)),
    )
    print(json.dumps({"status": status, "cert": cert},
                     ensure_ascii=False, indent=2, default=str))
    return {"COVERED": 0, "NOT_COVERED": 1, "UNKNOWN": 2}[status]


if __name__ == "__main__":
    sys.exit(main())
