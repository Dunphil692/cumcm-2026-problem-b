"""问题四动态策略 v13（实验）：v12 + 四开关（默认全关 = v12 逐位复现）。

  ab_step ∈ {60,90,120}：A 态步长（Branch AB，默认 60）；
  census_variant ∈ {baseline, candidate1870, candidate1900_control}（Branch G）；
  early_cert_clear：二分夹逼中每步复用清除证书（Branch N1）；
  optical_cover：细长可行域 ≤3 点光学覆盖清除（Branch N2）。
正式版 v12 未动。"""

from __future__ import annotations

import math

import numpy as np

from problem4_batch_station_v13 import (
    clip_arena,
    possible_samples,
    propose_batch_station,
)
from problem4_census_v13 import INNER_NET, OUTER_NET, net1870, net1900
from problem4_route_solver import guaranteed_disk, joint_route, two_opt_open
from problem4_simulation_model import (
    ARENA,
    CHANNELS,
    CLEAR_R,
    LISTEN_MAX,
    LISTEN_MIN,
    NEED_MORE,
    SPEED,
    Robot,
)

BELIEF_MAX_STEPS = 240
IMMEDIATE_CLEAR_R = 500.0  # 同站即时清的圆心距离阈值（思路3）

# ---------- Branch N2：光学覆盖清除（细长可行域 ≤3 点） ----------
OPTICAL_MAX_FAIL = 3   # 每频道新增光学失败预算
OPTICAL_H_MAX = 20.0   # 可行域垂直半宽上限（盒边中点到中心距离=h）
OPTICAL_W_MAX = 115.0  # 可行域轴向半宽上限


def _farthest_pair(vertices):
    best = 0.0
    pair = (vertices[0], vertices[1])
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(vertices[i], vertices[j])
            if d > best:
                best = d
                pair = (vertices[i], vertices[j])
    return pair


def optical_plan(vertices, max_r: float = 60.0):
    """细长可行域的 ≤3 点光学覆盖计划。

    证明骨架：设可行域多边形 P ⊆ 其轴对齐包围盒 B（轴向=最长弦方向），
    B 的尺寸 2w×2h。三个光学中心取轴向 c−a·u、c、c+a·u：
    - 盒的四个角 (±w,±h) 距最近端中心 ≤20 ⟺ (w−a)²+h² ≤ 400；
    - 盒边中点 (0,±h) 距中心 ≤20 ⟺ a²+h² ≤ 400；
    两式同时满足 ⇒ B（从而 P）被 3 圆完全覆盖（盒为凸，极值在角/边中点）。
    返回 (centers, w, h) 或 None（无合法计划）。
    """
    if not vertices or len(vertices) < 3:
        return None
    # 候选主轴：所有顶点对方向 + 各自垂直方向，选最小垂直半宽 h 者
    n = len(vertices)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            dx = vertices[j][0] - vertices[i][0]
            dy = vertices[j][1] - vertices[i][1]
            L = math.hypot(dx, dy)
            if L < 1e-9:
                continue
            for ux, uy in ((dx / L, dy / L), (-dy / L, dx / L)):
                proj = [ux * (v[0] - vertices[i][0]) + uy * (v[1] - vertices[i][1]) for v in vertices]
                perp = [-uy * (v[0] - vertices[i][0]) + ux * (v[1] - vertices[i][1]) for v in vertices]
                lo_p, hi_p = min(proj), max(proj)
                lo_q, hi_q = min(perp), max(perp)
                w = (hi_p - lo_p) / 2.0
                h = (hi_q - lo_q) / 2.0
                if best is None or h < best[3]:
                    best = (ux, uy, w, h, lo_p, hi_p, lo_q, hi_q, vertices[i])
    if best is None:
        return None
    ux, uy, w, h, lo_p, hi_p, lo_q, hi_q, anchor = best
    # 盒中心（轴坐标 → 世界坐标）
    mid_p = (lo_p + hi_p) / 2.0
    mid_q = (lo_q + hi_q) / 2.0
    cx = anchor[0] + mid_p * ux - mid_q * uy
    cy = anchor[1] + mid_p * uy + mid_q * ux
    if h > OPTICAL_H_MAX or w > OPTICAL_W_MAX or w <= 20.0:
        return None
    # 覆盖条件（修正版）：
    #   角 (w,h) 距端中心 ≤20 ⟺ (w−a)²+h² ≤ 400 ⟹ a ≥ w−s；
    #   中跨段最远点 (a/2,h) 距中心 ≤20 ⟺ (a/2)²+h² ≤ 400 ⟹ a ≤ 2s；
    #   s = sqrt(400−h²)。可行 ⟺ w−s ≤ 2s ⟺ w ≤ 3s。
    s = math.sqrt(400.0 - h * h)
    if w > 3.0 * s + 1e-9:
        return None
    a = max(0.0, w - s)
    a = min(a, 2.0 * s)
    centers = [
        (cx - a * ux, cy - a * uy),
        (cx, cy),
        (cx + a * ux, cy + a * uy),
    ]
    # 数值复核（冗余保险）：盒内 2.5m 网格全部被覆盖
    for dx in range(-int(w * 4), int(w * 4) + 1):
        for dy in range(-int(h * 4), int(h * 4) + 1):
            px = cx + dx * 0.25 * ux - dy * 0.25 * uy
            py = cy + dx * 0.25 * uy + dy * 0.25 * ux
            if min(math.dist((px, py), c) for c in centers) > 20.0 + 1e-6:
                return None
    return centers, w, h



def _opt_snap(state):
    return {
        "kind": state.get("kind"),
        "mec_center": state.get("mec_center"),
        "mec_radius": state.get("mec_radius"),
        "n_measurements": len(state.get("measurements") or ()),
        "n_nosig": state.get("n_nosig"),
    }


def _optical_visit_order_v14(centers, cur, region, order_mode, diag=None,
                             resume_anchor=None):
    """光学点访问顺序。baseline = v13 的最近优先（逐字保留）。"""
    if order_mode == "baseline":
        return sorted(centers, key=lambda c: math.dist(cur, c))
    from optical_order import rank_centers
    visit, _cost = rank_centers(centers, cur, region, resume_anchor)
    return visit


def _propose_optical_tasks_v14(tasks, states, optical_fails, k4_used, robot, ctx):
    """光学任务提案（tasks_here 与 force_progress 共用）。

    ctx: dict(optical_cover, order_mode, cover_mode, enable_k4, trigger_gate, diag)
    全部新开关为 baseline/False 时，本函数与 v13 原循环逐字等价。
    """
    if not ctx["optical_cover"]:
        return
    diag = ctx.get("diag")
    for state in states.values():
        ch = state["channel"]
        if state.get("kind") != "too_large":
            continue
        verts = state.get("vertices")
        if not verts:
            continue
        if optical_fails.get(ch, 0) >= OPTICAL_MAX_FAIL:
            continue
        plan = optical_plan(verts)
        plan_src = "baseline"
        if ctx["cover_mode"] == "polygon" and plan is None:
            from optical_cover_candidate import propose_polygon_plan
            got = propose_polygon_plan(verts, (robot.x, robot.y))
            if got is not None:
                plan = got
                plan_src = "polygon"
        if (ctx["enable_k4"] and plan is None
                and optical_fails.get(ch, 0) == 0 and ch not in k4_used):
            from optical_k4 import build_strip_plan
            got = build_strip_plan(verts, (robot.x, robot.y))
            if got is not None:
                plan = (got[0], None, None)
                plan_src = "k4"
        if plan is None:
            if diag is not None:
                diag.on_optical_decision(
                    None, ch, verts, "no_plan",
                    (robot.x, robot.y), optical_fails.get(ch, 0),
                    _opt_snap(state))
            continue
        centers = plan[0]
        first = min(centers, key=lambda c: math.dist((robot.x, robot.y), c))
        if ctx["trigger_gate"] != "baseline":
            from optical_order import gate_skip
            if gate_skip(ch, verts, centers, (robot.x, robot.y),
                         optical_fails.get(ch, 0), ctx["trigger_gate"]):
                if diag is not None:
                    diag.on_optical_decision(
                        centers, ch, verts, "gated",
                        (robot.x, robot.y), optical_fails.get(ch, 0),
                        _opt_snap(state))
                continue
        if diag is not None and diag.should_skip(ch):
            diag.on_optical_decision(
                centers, ch, verts, "fork_skip",
                (robot.x, robot.y), optical_fails.get(ch, 0),
                _opt_snap(state))
            continue
        task = {
            "kind": "optical",
            "channel": ch,
            "centers": centers,
            "region": verts,
            "disk": (first, 0.0),
        }
        if plan_src != "baseline":
            task["plan_src"] = plan_src
        if plan_src == "k4":
            k4_used.add(ch)
        tasks.append(task)
        if diag is not None:
            diag.on_optical_decision(
                centers, ch, verts, plan_src,
                (robot.x, robot.y), optical_fails.get(ch, 0), dict(task),
                _opt_snap(state))

GRID_STEP = 10.0
LISTEN_CERT = 990.0
COVER_SNAP = 40.0
HERE = 1.0
PROBE_MAX_R = 200.0      # MEC 半径低于此值直接近测
UNHEARD_SPACING = 700.0  # 非普查站顺路听空频道的最小间隔
HEARD_LIMIT = 16         # 听到 16 个不同频道即停普查（题面上限）


def _arena_grid(step: float = GRID_STEP, radius: float = ARENA) -> np.ndarray:
    xs = np.arange(-radius, radius + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    inside = xx * xx + yy * yy <= radius * radius
    return np.column_stack((xx[inside], yy[inside]))


GRID_XY = _arena_grid()


class UnheardGrid:
    """逐频道未听区域（全向假设），问题三机制原样保留。"""

    def __init__(self) -> None:
        self.mask = np.ones((len(CHANNELS), GRID_XY.shape[0]), dtype=bool)

    def empty(self) -> bool:
        return not bool(self.mask.any())

    def channel_empty(self, channel: int) -> bool:
        return not bool(self.mask[channel - 1].any())

    def exclude(self, channel: int, x: float, y: float, radius: float = LISTEN_CERT) -> None:
        d2 = (GRID_XY[:, 0] - x) ** 2 + (GRID_XY[:, 1] - y) ** 2
        self.mask[channel - 1] &= d2 > radius * radius

    def forget(self, channel: int) -> None:
        self.mask[channel - 1] = False

    def intersects(self, channel: int, x: float, y: float, radius: float) -> bool:
        if self.channel_empty(channel):
            return False
        d2 = (GRID_XY[:, 0] - x) ** 2 + (GRID_XY[:, 1] - y) ** 2
        return bool(np.any(self.mask[channel - 1] & (d2 <= radius * radius)))

    def any_intersects(self, x: float, y: float, radius: float) -> bool:
        if self.empty():
            return False
        d2 = (GRID_XY[:, 0] - x) ** 2 + (GRID_XY[:, 1] - y) ** 2
        return bool(np.any(self.mask & (d2[None, :] <= radius * radius)))

    def centroid(self) -> tuple[float, float] | None:
        idx = np.flatnonzero(self.mask.any(axis=0))
        if idx.size == 0:
            return None
        return float(GRID_XY[idx, 0].mean()), float(GRID_XY[idx, 1].mean())


all_net = list(INNER_NET) + list(OUTER_NET)  # 仅 baseline 用


def _heard(robot: Robot, channel: int) -> bool:
    return any(
        item.get("status") in {"direction", "near", "cleared"}
        for item in robot.log[channel]
    )


def _measured_here(robot: Robot, channel: int, tol: float = HERE) -> bool:
    here = (robot.x, robot.y)
    return any(
        math.dist(here, (item["x"], item["y"])) < tol for item in robot.log[channel]
    )


def _cover_count(visited: list[bool]) -> int:
    return sum(1 for flag in visited if flag)


def _silent_due(
    robot: Robot, certified: dict[int, bool]
) -> list[int]:
    """仍沉默且未认证、未清除的频道。"""
    out = []
    for channel in CHANNELS:
        if certified.get(channel):
            continue
        if any(item.get("status") == "cleared" for item in robot.log[channel]):
            continue
        if _heard(robot, channel):
            continue
        out.append(channel)
    return out


def _census_useful(orient, channels: list[int], cache) -> bool:
    """This census stop would shrink a still-open silent channel (990 m kill)."""
    for channel in channels:
        if orient.channel_empty(channel):
            continue
        if orient.silence_kills_cached(channel, cache):
            return True
    return False


def run_belief(
    world,
    fidelity: str = "high",
    max_steps: int = BELIEF_MAX_STEPS,
    trace: bool = False,
    outer_partial: int = 12,
    cert_soft: int = 0,
    bracket: bool = True,
    use_j: bool = False,
    ab_step: float = 60.0,
    census_variant: str = "baseline",
    early_cert_clear: bool = False,
    optical_cover: bool = False,
    order_mode: str = "baseline",
    cover_mode: str = "baseline",
    measurement_mode: str = "baseline",
    enable_k4: bool = False,
    trigger_gate: str = "baseline",
    robot=None,
    diag=None,
    orient_cert: bool = False,
    eps_mass: float = 0.01,
) -> dict:
    from problem4_fusion_policies_v13 import (
        FINE_R,
        _clear_nears,
        _clearable,
        _fallback_home,
        _fine_homing,
        _unresolved,
        _wedge_homing_fast,
        classify_all,
    )

    fast = fidelity == "fast"
    inner_net = INNER_NET
    outer_net = OUTER_NET
    if census_variant == "candidate1870":
        inner_net = net1870()
        outer_net = []
    elif census_variant == "candidate1900_control":
        inner_net = net1900()
        outer_net = []
    if robot is None:
        robot = Robot(world)
    inner_visited = [False] * len(inner_net)
    outer_visited = [False] * len(outer_net)
    orient = None
    inner_caches: list = []
    outer_caches: list = []
    if orient_cert or eps_mass > 0.0:
        from problem4_orient_cert import OrientGrid, StopCache

        orient = OrientGrid()
        inner_caches = [StopCache(*stop) for stop in inner_net]
        outer_caches = [StopCache(*stop) for stop in outer_net]
    unheard = UnheardGrid()
    unused_listens: list[tuple[float, float]] = []
    n_nosig: dict[int, int] = {}
    certified: dict[int, bool] = {}
    outer_triggered = False
    station_ctx: set[int] = set()  # 当前站位所服务的补测频道（强证据语境）
    ring_silent: dict[int, set] = {}  # 频道在 900 环各站的沉默记录（方向3软认证）
    clear_dircount: dict[int, int] = {}  # 清除时该频道的示向条数（双站清统计）
    states = classify_all(robot, fast)
    cover_at_first_clear: int | None = None
    fine_attempts: dict[int, int] = {}

    remaining0 = [(source.x, source.y) for source in world.sources]
    _, oracle_walk = two_opt_open((0.0, 0.0), remaining0)
    oracle_time = oracle_walk / SPEED + 5.0 * len(remaining0)

    def snap_inner() -> None:
        here = (robot.x, robot.y)
        for index, stop in enumerate(inner_net):
            if math.dist(here, stop) <= COVER_SNAP:
                inner_visited[index] = True

    def snap_outer() -> None:
        here = (robot.x, robot.y)
        for index, stop in enumerate(OUTER_NET):
            if math.dist(here, stop) <= COVER_SNAP:
                outer_visited[index] = True

    net_all = list(inner_net) + list(outer_net)

    def at_net_stop() -> bool:
        here = (robot.x, robot.y)
        return any(
            math.dist(here, stop) <= COVER_SNAP for stop in net_all
        )

    def inner_done() -> bool:
        return all(inner_visited)

    def outer_done() -> bool:
        return all(outer_visited[:outer_partial])

    def cache_here():
        if orient is None:
            return None
        here = (robot.x, robot.y)
        best = None
        for index, stop in enumerate(inner_net):
            dist = math.dist(here, stop)
            if dist <= COVER_SNAP and (best is None or dist < best[0]):
                best = (dist, inner_caches[index])
        for index, stop in enumerate(outer_net):
            dist = math.dist(here, stop)
            if dist <= COVER_SNAP and (best is None or dist < best[0]):
                best = (dist, outer_caches[index])
        return None if best is None else best[1]

    def should_measure(channel: int) -> bool:
        last = robot.log[channel][-1] if robot.log[channel] else None
        if last and last.get("status") == "cleared":
            return False
        if certified.get(channel):
            return False
        state = states.get(channel)
        if state and state["kind"] in {"cleared", "near", "clearable"}:
            return False
        if _measured_here(robot, channel):
            return False
        here = (robot.x, robot.y)
        if not _heard(robot, channel):
            if orient is not None:
                if orient.channel_empty(channel):
                    return False
                cache = cache_here()
                if cache is not None:
                    return orient.silence_kills_cached(
                        channel, cache
                    ) or orient.hearable_cached(channel, cache)
                if not orient.silence_kills(channel, robot.x, robot.y):
                    return False
                return all(
                    math.dist(here, prev) >= UNHEARD_SPACING
                    for prev in unused_listens
                )
            if at_net_stop():
                # 普查义务：沉默频道在普查站一律测（含已挖空的网）
                return True
            if unheard.channel_empty(channel):
                return False
            if not unheard.intersects(channel, robot.x, robot.y, LISTEN_MIN):
                return False
            return all(
                math.dist(here, prev) >= UNHEARD_SPACING for prev in unused_listens
            )
        if state is None:
            return True
        samples = possible_samples(state)
        if not samples:
            return True
        return any(math.dist(here, sample) <= LISTEN_MAX for sample in samples)

    def _record_clear_dirs(state: dict) -> None:
        channel = state["channel"]
        n = sum(
            1
            for it in robot.log[channel]
            if it.get("status") == "direction"
        )
        if channel not in clear_dircount:
            clear_dircount[channel] = n

    def maybe_certify() -> None:
        """占空认证：全网证书 / 听到 16 只。"""
        nonlocal certified
        heard_count = sum(1 for channel in CHANNELS if _heard(robot, channel))
        if heard_count >= HEARD_LIMIT:
            for channel in CHANNELS:
                if (
                    not _heard(robot, channel)
                    and not any(
                        item.get("status") == "cleared"
                        for item in robot.log[channel]
                    )
                ):
                    certified[channel] = True
            return
        if cert_soft > 0:
            # 方向3（风险档）：900 环 ≥cert_soft 站沉默即认证为空。
            # 会破坏零漏检证书（边界外指定向源对全部内网站沉默），
            # 默认关闭；打开后清完率将随定向源占比下降。
            for channel in _silent_due(robot, certified):
                if len(ring_silent.get(channel, set())) >= cert_soft:
                    certified[channel] = True
        if orient is not None:
            silent_ch = _silent_due(robot, certified)
            if eps_mass > 0.0 and silent_ch and orient.mean_mass(silent_ch) <= eps_mass:
                for channel in silent_ch:
                    certified[channel] = True
            else:
                for channel in silent_ch:
                    if orient.channel_empty(channel):
                        certified[channel] = True
                        continue
                    still_useful = False
                    for index, cache in enumerate(inner_caches):
                        if inner_visited[index]:
                            continue
                        if _census_useful(orient, [channel], cache):
                            still_useful = True
                            break
                    if not still_useful:
                        for index, cache in enumerate(outer_caches[:outer_partial]):
                            if outer_visited[index]:
                                continue
                            if _census_useful(orient, [channel], cache):
                                still_useful = True
                                break
                    if not still_useful:
                        certified[channel] = True
        if inner_done() and outer_done():
            for channel in _silent_due(robot, certified):
                certified[channel] = True

    def work_here() -> None:
        nonlocal states, cover_at_first_clear, outer_triggered
        snap_inner()
        snap_outer()
        for channel in CHANNELS:
            if not should_measure(channel):
                continue
            was_heard = _heard(robot, channel)
            result = robot.measure(channel)
            if orient is not None:
                if result["status"] == "near":
                    orient.clear_channel(channel)
                elif result["status"] == "no_signal":
                    orient.exclude_silent(channel, robot.x, robot.y)
                elif result["status"] == "direction":
                    orient.restrict_heard(
                        channel, robot.x, robot.y, float(result["svd_deg"])
                    )
            if result["status"] == "near":
                robot.clear(channel)
            if not was_heard:
                unused_listens.append((robot.x, robot.y))
                if result["status"] == "no_signal":
                    # 全向假设的挖洞（只挖本频道；定向假设由普查证书承载）
                    unheard.exclude(channel, robot.x, robot.y, LISTEN_CERT)
                    # 900 环（INNER_NET 索引 1..6）沉默记录：方向3软认证
                    for j in range(1, 7):
                        if (
                            math.dist((robot.x, robot.y), inner_net[j])
                            <= COVER_SNAP
                        ):
                            ring_silent.setdefault(channel, set()).add(j)
                else:
                    unheard.forget(channel)
            else:
                if result["status"] == "no_signal":
                    # 强定向证据：全向源在距可能样本 ≤990 m 处必能听到；
                    # 在此听见后 no_signal ⇒ 全向假设在该样本处被证伪。
                    strong = channel in station_ctx
                    state = states.get(channel)
                    if not strong and state is not None:
                        samples = possible_samples(state)
                        strong = bool(samples) and any(
                            math.dist((robot.x, robot.y), s) <= LISTEN_CERT
                            for s in samples
                        )
                    if strong:
                        n_nosig[channel] = n_nosig.get(channel, 0) + 1
        states = classify_all(robot, fast, states)
        for state in states.values():
            state["n_nosig"] = n_nosig.get(state["channel"], 0)
        _clear_nears(robot, states)
        for state in _clearable(states):
            center = state.get("mec_center")
            radius = state.get("mec_radius")
            if center is None or radius is None:
                continue
            if math.dist((robot.x, robot.y), tuple(center)) + radius <= CLEAR_R + 1e-9:
                if robot.world.source_by_channel(state["channel"]) is not None:
                    _record_clear_dirs(state)
                    robot.clear(state["channel"])
        states = classify_all(robot, fast, states)
        # v4 思路3：同站测清合并——可清源圆心距当前站 <500 m 立即顺路清，
        # 免"回主路→重排→折返"；≥500 m 的仍作为清除任务交全局 TSPN 挂起清。
        for state in list(_clearable(states)):
            center = state.get("mec_center")
            if center is None:
                continue
            if (
                math.dist((robot.x, robot.y), tuple(center))
                <= IMMEDIATE_CLEAR_R
                and robot.world.source_by_channel(state["channel"]) is not None
            ):
                _record_clear_dirs(state)
                robot.move_to(*tuple(center))
                robot.clear(state["channel"])
        states = classify_all(robot, fast, states)
        for state in states.values():
            state["n_nosig"] = n_nosig.get(state["channel"], 0)
        for state in states.values():
            if state.get("kind") != "too_large":
                continue
            center = state.get("mec_center")
            radius = state.get("mec_radius")
            if center is None or radius is None or radius > FINE_R:
                continue
            if math.dist((robot.x, robot.y), tuple(center)) > radius + 20.0:
                continue
            channel = state["channel"]
            # 定向证据频道：只有"最近示向就在当前站位"（刚在此测到过
            # 方向，证明在扇内）才允许精测逼近；否则交给楔形追踪。
            if n_nosig.get(channel, 0) >= 1:
                dirs = [
                    it
                    for it in robot.log[channel]
                    if it.get("status") == "direction"
                ]
                fresh = dirs and math.dist(
                    (robot.x, robot.y), (dirs[-1]["x"], dirs[-1]["y"])
                ) < HERE
                if not fresh:
                    continue
            if fine_attempts.get(channel, 0) >= 3:
                continue
            fine_attempts[channel] = fine_attempts.get(channel, 0) + 1
            _fine_homing(robot, channel, fast=fast)
            states = classify_all(robot, fast, states)
            for state in states.values():
                state["n_nosig"] = n_nosig.get(state["channel"], 0)
        if cover_at_first_clear is None and robot.n_clear_ok > 0:
            cover_at_first_clear = _cover_count(inner_visited)
        if orient is not None:
            for channel in CHANNELS:
                if any(
                    item.get("status") == "cleared" for item in robot.log[channel]
                ):
                    orient.clear_channel(channel)
        maybe_certify()

    probed_key: tuple | None = None
    fallback_calls: dict[int, int] = {}
    homing_proposed: set[int] = set()
    probe_attempts: dict[int, int] = {}
    optical_fails: dict[int, int] = {}
    k4_used: set[int] = set()  # v14 K4：每频道/每局至多一个四点计划
    mstats: list = []  # v14 M：重排统计（基线模式恒为空）
    _opt_ctx = {
        "optical_cover": optical_cover,
        "order_mode": order_mode,
        "cover_mode": cover_mode,
        "enable_k4": enable_k4,
        "trigger_gate": trigger_gate,
        "diag": diag,
    }
    walk_kind: dict[str, float] = {}  # 各任务类型的移动路程统计

    def tasks_here() -> list[dict]:
        nonlocal probed_key, outer_triggered
        tasks: list[dict] = []
        for state in _clearable(states):
            vertices = state.get("vertices")
            if not vertices:
                continue
            disk = guaranteed_disk(vertices)
            if disk is None:
                continue
            tasks.append(
                {"kind": "clear", "channel": state["channel"], "disk": disk}
            )
        for state in states.values():
            if state.get("kind") != "too_large":
                continue
            center = state.get("mec_center")
            radius = state.get("mec_radius")
            if center is None or radius is None or radius > PROBE_MAX_R:
                continue
            channel = state["channel"]
            # probe 上限：3 次探圆心仍无进展 ⇒ 两个频道的 probe 会
            # 跨频道乒乓（圆心随每次测量漂移，dedup 键失效），
            # 直接升级楔形追踪。
            if probe_attempts.get(channel, 0) >= 3:
                continue
            key = (state["channel"], round(center[0], 1), round(center[1], 1))
            if key == probed_key:
                continue
            tasks.append(
                {"kind": "probe", "channel": state["channel"], "disk": (tuple(center), 0.0)}
            )
        _propose_optical_tasks_v14(
            tasks, states, optical_fails, k4_used, robot, _opt_ctx
        )
        unresolved = _unresolved(states)
        # 定向证据（补测站测到 no_signal）驱动的楔形追踪触发：
        #  - wedge/unbounded（单楔形或近平行楔形）：第 1 次证据即追踪——
        #    源沿射线距离未知，换站要么 50% 听不到、要么交会角差，
        #    沿最近原点射线 + 折返捕获 near 是确定性方案；
        #  - bounded（有界多边形）：第 2 次证据再追踪（先让 probe 试一次）。
        station_unresolved = []
        for state in unresolved:
            channel = state["channel"]
            ev = n_nosig.get(channel, 0)
            trigger = ev >= 1 and state.get("kind") in {
                "wedge",
                "unbounded",
                "empty_intersection",
            }
            trigger = trigger or ev >= 2 or probe_attempts.get(channel, 0) >= 3
            if trigger:
                if channel not in homing_proposed:
                    # 从完整日志选距当前位置最近的示向原点
                    origins = [
                        it
                        for it in robot.log[channel]
                        if it.get("status") == "direction"
                    ]
                    if origins:
                        best = min(
                            origins,
                            key=lambda it: math.dist(
                                (robot.x, robot.y), (it["x"], it["y"])
                            ),
                        )
                        tasks.append(
                            {
                                "kind": "homing",
                                "channel": channel,
                                "disk": ((best["x"], best["y"]), 0.0),
                            }
                        )
                continue
            station_unresolved.append(state)
        if station_unresolved:
            station, channels = propose_batch_station(
                (robot.x, robot.y), station_unresolved, use_j=use_j
            )
            tasks.append(
                {
                    "kind": "station",
                    "channels": channels,
                    "disk": (station, 0.0),
                }
            )
        silent = _silent_due(robot, certified)
        if silent:
            for index, stop in enumerate(inner_net):
                if inner_visited[index]:
                    continue
                if orient is not None and not _census_useful(
                    orient, silent, inner_caches[index]
                ):
                    continue
                tasks.append({"kind": "cover", "disk": (stop, 0.0)})
            allow_outer = inner_done()
            if allow_outer:
                added_outer = False
                skip_outer = bool(
                    eps_mass > 0.0
                    and orient is not None
                    and orient.mean_mass(silent) <= eps_mass
                )
                if not skip_outer:
                    for index, stop in enumerate(outer_net[:outer_partial]):
                        if outer_visited[index]:
                            continue
                        if orient is not None and not _census_useful(
                            orient, silent, outer_caches[index]
                        ):
                            continue
                        tasks.append({"kind": "outer", "disk": (stop, 0.0)})
                        added_outer = True
                if added_outer or inner_done():
                    outer_triggered = True
        if measurement_mode != "baseline":
            from measurement_completion_value import rerank_station
            from state_snapshot import capture_m
            tasks, m_stat = rerank_station(tasks, capture_m(
                states=states, robot=robot, optical_fails=optical_fails,
                k4_used=k4_used, n_nosig=n_nosig, certified=certified,
                probe_attempts=probe_attempts, fallback_calls=fallback_calls,
                inner_visited=inner_visited, outer_visited=outer_visited,
                unheard=unheard, station_ctx=station_ctx,
            ))
            mstats.append(m_stat)
        return tasks

    def force_progress() -> bool:
        nonlocal station_ctx
        station_ctx = set()
        if trace:
            rem = [
                source.channel
                for source in robot.world.sources
                if not source.cleared
            ]
            print(
                f"FORCE rem={rem} inner={sum(inner_visited)}/13 "
                f"silent={len(_silent_due(robot, certified))} "
                f"unres={[s['channel'] for s in _unresolved(states)]}",
                flush=True,
            )
        _propose_optical_tasks_v14(
            tasks, states, optical_fails, k4_used, robot, _opt_ctx
        )
        unresolved = _unresolved(states)
        silent = _silent_due(robot, certified)
        if orient is not None and silent:
            needed_inner = [
                stop
                for index, stop in enumerate(inner_net)
                if not inner_visited[index]
                and _census_useful(orient, silent, inner_caches[index])
            ]
            if needed_inner:
                robot.move_to(*needed_inner[0])
                robot.n_replan += 1
                work_here()
                return True
            needed_outer = [
                stop
                for index, stop in enumerate(outer_net[:outer_partial])
                if not outer_visited[index]
                and _census_useful(orient, silent, outer_caches[index])
            ]
            if needed_outer:
                robot.move_to(*needed_outer[0])
                robot.n_replan += 1
                work_here()
                return True
        pending = [stop for flag, stop in zip(inner_visited, inner_net) if not flag]
        if pending:
            robot.move_to(*pending[0])
            robot.n_replan += 1
            work_here()
            return True
        if silent:
            pending_o = [
                stop for flag, stop in zip(outer_visited, outer_net) if not flag
            ]
            if pending_o:
                robot.move_to(*pending_o[0])
                robot.n_replan += 1
                work_here()
                return True
        if unresolved:
            state = unresolved[0]
            channel = state["channel"]
            calls = fallback_calls.get(channel, 0) + 1
            fallback_calls[channel] = calls
            # 前两次兜底走问题三原逻辑（有界 3000 m）；3 次仍无进展 ⇒
            # 无条件升级快速楔形追踪：对两类源都确定性可达（全向 near
            # 只按距离；定向靠四态状态机），不依赖证据计数（兜底乱走时
            # 的 no_signal 距样本远，未必算强证据，曾造成零证据定向源
            # 无限乱走）。
            _fallback_home(
                robot,
                state,
                fast=fast,
                directional=(calls >= 3),
            )
            robot.n_replan += 1
            work_here()
            return True
        target = unheard.centroid()
        if target is None:
            return False
        dest = clip_arena(*target)
        if math.dist((robot.x, robot.y), dest) < HERE:
            return False
        robot.move_to(*dest)
        robot.n_replan += 1
        work_here()
        return True

    def mission_complete() -> bool:
        """Stop without the local oracle remaining(). Official-safe."""
        maybe_certify()
        blocking = {"near", "clearable"} | set(NEED_MORE)
        if any(state["kind"] in blocking for state in states.values()):
            return False
        return not _silent_due(robot, certified)

    work_here()
    steps = 0
    stuck = 0
    last_xy = (robot.x, robot.y)
    last_station_key: tuple | None = None
    stop_reason = "max_steps"

    while steps < max_steps:
        rem = robot.world.remaining()
        if rem is not None:
            if rem <= 0:
                stop_reason = "cleared"
                break
        elif mission_complete():
            stop_reason = "certificate"
            break
        steps += 1
        tasks = tasks_here()
        if not tasks:
            if not force_progress():
                stop_reason = "stuck"
                break
            continue

        disks = [task["disk"] for task in tasks]
        order, _, path = joint_route((robot.x, robot.y), disks)
        pick = 0
        dest = path[0]
        while pick < len(path) and math.dist((robot.x, robot.y), dest) < HERE:
            pick += 1
            if pick < len(path):
                dest = path[pick]
        if pick >= len(path):
            final_task = tasks[order[-1]] if order else None
            if final_task is not None and final_task["kind"] in {"clear", "probe"}:
                if final_task["kind"] == "probe":
                    # 与正常 pick 分支一致：执行过的 probe 打标记，
                    # 否则同一 probe 会无限重提（同点重复测量被跳过，
                    # 证据不增长，死循环直到步数上限）。
                    center = final_task["disk"][0]
                    probed_key = (
                        final_task["channel"],
                        round(center[0], 1),
                        round(center[1], 1),
                    )
                    probe_attempts[final_task["channel"]] = (
                        probe_attempts.get(final_task["channel"], 0) + 1
                    )
                station_ctx = set()
                if diag is not None:
                    diag.on_non_optical_action()
                robot.move_to(*final_task["disk"][0])
                robot.n_replan += 1
                work_here()
                continue
            if not force_progress():
                stop_reason = "stuck"
                break
            last_station_key = None
            continue

        task = tasks[order[pick]]
        if diag is not None and task["kind"] != "optical":
            diag.on_non_optical_action()
        station_ctx = (
            set(task.get("channels") or ()) if task["kind"] == "station" else set()
        )
        if trace:
            kinds = [(t["kind"], t.get("channel", 0)) for t in tasks]
            print(
                f"step{steps:3d} pos=({robot.x:6.0f},{robot.y:6.0f}) "
                f"ntasks={len(tasks)} pick={task['kind']}:{task.get('channel',0)} "
                f"dest=({dest[0]:6.0f},{dest[1]:6.0f}) "
                f"nosig={dict(n_nosig)} rem={robot.world.remaining()}",
                flush=True,
            )
        if task["kind"] == "probe":
            center = task["disk"][0]
            probed_key = (task["channel"], round(center[0], 1), round(center[1], 1))
            probe_attempts[task["channel"]] = (
                probe_attempts.get(task["channel"], 0) + 1
            )
        if task["kind"] == "station":
            key = (
                round(dest[0], 0),
                round(dest[1], 0),
                tuple(task.get("channels") or ()),
            )
            if key == last_station_key:
                if not force_progress():
                    stop_reason = "stuck"
                    break
                last_station_key = None
                continue
            last_station_key = key
        if task["kind"] == "optical":
            station_ctx = set()
            centers = task.get("centers") or []
            nxt = tasks[order[pick + 1]] if pick + 1 < len(order) else None
            anchor = nxt["disk"][0] if nxt is not None else None
            order = _optical_visit_order_v14(
                centers, (robot.x, robot.y), task.get("region"),
                "baseline" if task.get("plan_src") == "k4" else order_mode,
                diag, anchor
            )
            cleared = False
            fails_before = optical_fails.get(task["channel"], 0)
            success_index = None
            for idx, c in enumerate(order):
                robot.move_to(*c)
                robot.n_replan += 1
                if robot.clear(task["channel"]):
                    cleared = True
                    success_index = idx
                    break
                optical_fails[task["channel"]] = (
                    optical_fails.get(task["channel"], 0) + 1
                )
            if (task.get("plan_src") == "k4" and not cleared
                    and diag is not None):
                diag.on_k4_contradiction(
                    task["channel"], order, success_index,
                    optical_fails.get(task["channel"], 0) - fails_before)
            if diag is not None:
                diag.on_optical_exec(
                    task["channel"], order, cleared, success_index,
                    (robot.x, robot.y),
                    optical_fails.get(task["channel"], 0) - fails_before,
                    robot.time)
            work_here()
            last_station_key = None
            continue
        if task["kind"] == "homing":
            # 楔形追踪任务：先到原点再沿射线追踪，一步到位
            station_ctx = set()
            homing_proposed.add(task["channel"])
            w0 = robot.walk_m
            robot.move_to(*dest)
            robot.n_replan += 1
            _wedge_homing_fast(
                robot,
                task["channel"],
                fast=fast,
                from_xy=dest,
                bracket=bracket,
                ab_step=ab_step,
                eps_B=1.0,
                early_cert_clear=early_cert_clear,
            )
            walk_kind["homing"] = walk_kind.get("homing", 0.0) + (
                robot.walk_m - w0
            )
            work_here()
            last_station_key = None
            if robot.world.source_by_channel(task["channel"]) is not None:
                # 追踪后源仍在（异常）：允许下次重新提议
                homing_proposed.discard(task["channel"])
            continue
        w0 = robot.walk_m
        robot.move_to(*dest)
        walk_kind[task["kind"]] = walk_kind.get(task["kind"], 0.0) + (
            robot.walk_m - w0
        )
        robot.n_replan += 1
        work_here()
        # 方向4：补测站行程后链式探圆心——站上刚测完，若目标频道仍
        # too_large 且 MEC ≤ PROBE_MAX_R，直接顺路去圆心补一测再重排，
        # 省一趟"回主路→再折返"的重复路程。
        if task["kind"] == "station":
            for ch in task.get("channels") or ():
                st = states.get(ch)
                if (
                    st is not None
                    and st.get("kind") == "too_large"
                    and st.get("mec_center") is not None
                    and float(st.get("mec_radius") or 999.0) <= PROBE_MAX_R
                    and probe_attempts.get(ch, 0) < 3
                ):
                    center = tuple(st["mec_center"])
                    probed_key = (ch, round(center[0], 1), round(center[1], 1))
                    probe_attempts[ch] = probe_attempts.get(ch, 0) + 1
                    station_ctx = set()
                    w1 = robot.walk_m
                    robot.move_to(*center)
                    walk_kind["probe"] = walk_kind.get("probe", 0.0) + (
                        robot.walk_m - w1
                    )
                    robot.n_replan += 1
                    work_here()
                    break
        here = (robot.x, robot.y)
        stuck = stuck + 1 if math.dist(here, last_xy) < HERE else 0
        last_xy = here
        if stuck >= 3:
            if not force_progress():
                stop_reason = "stuck"
                break
            stuck = 0
            last_station_key = None

    heard_channels = {
        channel for channel in CHANNELS if _heard(robot, channel)
    }
    missed_dir = sum(
        1
        for source in world.sources
        if source.directional and source.channel not in heard_channels
    )
    return {
        "policy": "belief4v13",
        "outer_partial": outer_partial,
        "cert_soft": cert_soft,
        "fidelity": "fast" if fast else "high",
        "cleared": sum(1 for source in robot.world.sources if source.cleared),
        "n_sources": len(robot.world.sources),
        "n_dir": world.n_directional(),
        "n_dir_cleared": world.n_directional_cleared(),
        "missed_dir": missed_dir,
        "time_s": robot.time,
        "walk_m": robot.walk_m,
        "measures": robot.n_measure,
        "replans": robot.n_replan,
        "clear_fail": robot.n_clear_fail,
        "steps": steps,
        "census_time_s": 0.0,
        "oracle_time_s": oracle_time,
        "regret_s": robot.time - oracle_time,
        "remaining": robot.world.remaining(),
        "stop_reason": stop_reason,
        "inner_visits": _cover_count(inner_visited),
        "outer_visits": _cover_count(outer_visited),
        "outer_triggered": outer_triggered,
        "n_nosig_channels": sum(
            1 for channel in CHANNELS if n_nosig.get(channel, 0) > 0
        ),
        "walk_kind": dict(walk_kind),
        "bracket": bracket,
        "use_j": use_j,
        "ab_step": ab_step,
        "census_variant": census_variant,
        "early_cert_clear": early_cert_clear,
        "optical_cover": optical_cover,
        "orient_cert": orient_cert,
        "order_mode": order_mode,
        "cover_mode": cover_mode,
        "measurement_mode": measurement_mode,
        "enable_k4": enable_k4,
        "trigger_gate": trigger_gate,
        "optical_fails_total": sum(optical_fails.values()),
        "m_rerank_calls": len(mstats),
        "m_action_changes": sum(1 for s in mstats if s.get("changed")),
        "n_clear_2station": sum(
            1 for n in clear_dircount.values() if n == 2
        ),
        "n_clear_total": len(clear_dircount),
        "cover_visits_at_first_clear": (
            cover_at_first_clear
            if cover_at_first_clear is not None
            else _cover_count(inner_visited)
        ),
    }
