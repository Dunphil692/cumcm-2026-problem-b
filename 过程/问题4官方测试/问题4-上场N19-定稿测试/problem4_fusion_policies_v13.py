"""问题四共享策略件 v13（实验）：v12 + A 步长参数 + B 预算公式。

默认 ab_step=60、eps_B=1 时行为与 v12 逐位一致（v12 硬编码 6 次二分，
60/2^6=0.9375<1，与公式 ceil(log2(w/1)) 一致）。"""

from __future__ import annotations

import math

from problem4_batch_station import bearing_point, clip_arena
from problem4_route_solver import guaranteed_disk
from problem4_simulation_model import CHANNELS, NEED_MORE, Robot, classify_channel

FALLBACK_MAX_WALK = 3000.0   # 单次兜底最多多走这些米
FALLBACK_MAX_JUMP = 400.0    # 单次追 MEC 圆心的步长上限
FINE_R = 40.0                # 低于该 MEC 半径走精测逼近
FINE_STEP = 6.0              # 精测步长（≤10 m，不会跨过 5 m 近场区）
FINE_STEPS = 16              # 精测最大步数
WEDGE_STEPS = 260            # 沿楔形追源最大步数
FAST_STEP = 60.0             # 快速楔形追踪步长
FAST_STEPS = 30              # 快速楔形追踪最大步数（1500/60 + 余量）
FINE2_STEP = 4.0             # 折返精步（≤5 m 近场区直径，两步内必入）


def _wedge_homing_fast(
    robot: Robot,
    channel: int,
    fast: bool = False,
    from_xy: tuple | None = None,
    bracket: bool = True,
    ab_step: float = FAST_STEP,
    eps_B: float = 1.0,
    early_cert_clear: bool = False,
) -> None:
    """快速楔形追踪（问题四定向源兜底），四态状态机。

    核心几何：沿"示向原点"射线朝源走，dir(g→dog) 恒定（近侧在扇内），
    越过源后翻转 180° 落出 φ±90° 扇区 ⇒ 远侧 no_signal。但还有第二种
    no_signal：原点方向擦着扇区边界时（边距 < ~10°），横向漂移会让犬
    在离源几十到几百米处提前出扇——折返若沿原射线走，方向从 g 看几乎
    不变，永远在边界上振荡。状态机区分两种情况：

    A 前进：60 m 步沿最新示向逼近（每步重瞄，漂移 ≤ ~2 m）。
       no_signal → B。
    B 折返：4 m 步沿反方向回退。
       near → 清（跨过源的情形，近侧 ≤5 m 必触发）；
       测到 direction（回到扇内但 >5 m）⇒ 侧向出扇，→ C；
       no_signal 持续 → 继续 B。
    C 垂直切入：沿最近行进方向的 +90° 方向 6 m 步横移找扇区内部；
       连续 2 步 no_signal ⇒ 换 -90° 侧；测到 direction 后再横移
       约 60 m 建立边距（边距 ≈ atan(横移量/距离)，对付漂移绰绰有余）。
       → D。
    D 内部逼近：4 m 步沿新鲜示向逼近（扇内边距大，dir(g→dog) 恒定
       直到跨过源），near 必在 ≤5 m 近侧触发。
    """
    dirs = [it for it in robot.log[channel] if it.get("status") == "direction"]
    if not dirs:
        return
    if from_xy is None:
        first = dirs[-1]
    else:
        first = min(dirs, key=lambda it: math.dist(from_xy, (it["x"], it["y"])))
    robot.move_to(first["x"], first["y"])
    state = "A"
    last_dir_xy = (robot.x, robot.y)  # 最近一次 direction 的测点（二分下界）
    sweep_side = 90.0   # C 态横移方向（相对行进方向）
    sweep_steps = 0     # C 态已横移步数
    side_miss = 0       # C 态连续 no_signal 计数
    no_sig_run = 0      # B 态连续 no_signal 计数
    for _ in range(FAST_STEPS + 12 * FINE_STEPS + 40):
        if robot.world.source_by_channel(channel) is None:
            return
        result = robot.measure(channel)
        if result["status"] == "near":
            robot.clear(channel)
            return
        label = classify_channel(robot.log[channel], fast=fast)
        if label["kind"] == "clearable" and label.get("vertices"):
            disk = guaranteed_disk(label["vertices"])
            if disk is not None:
                robot.move_to(*disk[0])
                if robot.clear(channel):
                    return
        if result["status"] == "direction":
            no_sig_run = 0
            if state == "A":
                step = ab_step
                svd = result["svd_deg"]
                last_dir_xy = (robot.x, robot.y)
            elif state == "B":
                # 回到扇内但未触发 near ⇒ 侧向出扇，转垂直切入
                state = "C"
                sweep_steps = 0
                side_miss = 0
                step = 6.0
                svd = (result["svd_deg"] + sweep_side) % 360.0
            elif state == "C":
                sweep_steps += 1
                side_miss = 0
                if sweep_steps >= 12:
                    state = "D"  # 已建立约 60-72 m 边距，转内部逼近
                    step = FINE2_STEP
                    svd = result["svd_deg"]
                else:
                    step = 6.0
                    svd = (result["svd_deg"] + sweep_side) % 360.0
            else:  # D
                step = FINE2_STEP
                svd = result["svd_deg"]
        else:  # no_signal
            if state == "A":
                if bracket and last_dir_xy is not None:
                    # 有限二分：lo=方向点, hi=无信号点，夹逼覆盖边界。
                    # 预算按实际端点距离与目标宽度：n_B=ceil(log2(w/eps_B))
                    lo = last_dir_xy
                    hi = (robot.x, robot.y)
                    w = math.dist(lo, hi)
                    n_b = 0
                    if w > eps_B:
                        n_b = int(math.ceil(math.log2(w / eps_B))) + 0
                    n_b = max(0, min(n_b, 24))
                    for _ in range(n_b):
                        mx = (lo[0] + hi[0]) / 2.0
                        my = (lo[1] + hi[1]) / 2.0
                        if math.dist((robot.x, robot.y), (mx, my)) < 0.25:
                            break
                        robot.move_to(mx, my)
                        r = robot.measure(channel)
                        if r["status"] == "near":
                            robot.clear(channel)
                            return
                        if early_cert_clear and r["status"] == "direction":
                            # N1：每次新观测后复用既有清除证书
                            lab = classify_channel(
                                robot.log[channel], fast=fast
                            )
                            if (
                                lab["kind"] == "clearable"
                                and lab.get("vertices")
                            ):
                                disk = guaranteed_disk(lab["vertices"])
                                if disk is not None:
                                    robot.move_to(*disk[0])
                                    if robot.clear(channel):
                                        return
                        if r["status"] == "direction":
                            lo = (mx, my)
                        else:
                            hi = (mx, my)
                    # 二分结束仍未清：侧向出扇，转入 C 态垂直切入
                    state = "C"
                    sweep_steps = 0
                    side_miss = 0
                    last_dir = next(
                        (
                            it
                            for it in reversed(robot.log[channel])
                            if it.get("status") == "direction"
                        ),
                        None,
                    )
                    if last_dir is None:
                        return
                    svd = (last_dir["svd_deg"] + sweep_side) % 360.0
                    step = 6.0
                    continue
                state = "B"
                step = FINE2_STEP
                last_dir = next(
                    (
                        it
                        for it in reversed(robot.log[channel])
                        if it.get("status") == "direction"
                    ),
                    None,
                )
                if last_dir is None:
                    return
                svd = (last_dir["svd_deg"] + 180.0) % 360.0
            elif state == "B":
                no_sig_run += 1
                if no_sig_run > 18:
                    # 回退 72 m 仍全是 no_signal（跨过源的近侧早该触发
                    # near 了）：远距出扇的兜底，转垂直切入
                    state = "C"
                    sweep_steps = 0
                    side_miss = 0
                    last_dir = next(
                        (
                            it
                            for it in reversed(robot.log[channel])
                            if it.get("status") == "direction"
                        ),
                        None,
                    )
                    if last_dir is None:
                        return
                    svd = (last_dir["svd_deg"] + sweep_side) % 360.0
                    step = 6.0
                else:
                    last_dir = next(
                        (
                            it
                            for it in reversed(robot.log[channel])
                            if it.get("status") == "direction"
                        ),
                        None,
                    )
                    if last_dir is None:
                        return
                    svd = (last_dir["svd_deg"] + 180.0) % 360.0
                    step = FINE2_STEP
            elif state == "C":
                side_miss += 1
                if side_miss >= 2:
                    sweep_side = -sweep_side
                    side_miss = 0
                    sweep_steps = 0
                last_dir = next(
                    (
                        it
                        for it in reversed(robot.log[channel])
                        if it.get("status") == "direction"
                    ),
                    None,
                )
                if last_dir is None:
                    return
                svd = (last_dir["svd_deg"] + sweep_side) % 360.0
                step = 6.0
            else:  # D：内部逼近途中出扇（漂移），折返小步
                last_dir = next(
                    (
                        it
                        for it in reversed(robot.log[channel])
                        if it.get("status") == "direction"
                    ),
                    None,
                )
                if last_dir is None:
                    return
                svd = (last_dir["svd_deg"] + 180.0) % 360.0
                step = FINE2_STEP
        target = bearing_point(robot.x, robot.y, svd, step)
        if math.dist(target, (robot.x, robot.y)) < 0.5:
            return
        robot.move_to(*target)



def _wedge_homing(
    robot: Robot, channel: int, fast: bool = False, from_xy: tuple | None = None
) -> None:
    """单示向楔形：回到示向原点，沿示向 6 m 小步逐点测。

    定向源下同样安全：沿射线接近时 dir(g→dog) 恒定，狗始终在 φ±90°
    扇形内，5 m near 区不会因出扇而漏。
    from_xy 指定后从最近的原点出发（省掉长距离折返）。
    """
    dirs = [it for it in robot.log[channel] if it.get("status") == "direction"]
    if not dirs:
        return
    if from_xy is None:
        first = dirs[-1]
    else:
        first = min(dirs, key=lambda it: math.dist(from_xy, (it["x"], it["y"])))
    robot.move_to(first["x"], first["y"])
    for _ in range(WEDGE_STEPS):
        if robot.world.source_by_channel(channel) is None:
            return
        label = classify_channel(robot.log[channel], fast=fast)
        if label["kind"] == "near":
            robot.clear(channel)
            return
        if label["kind"] == "clearable" and label.get("vertices"):
            disk = guaranteed_disk(label["vertices"])
            if disk is not None:
                robot.move_to(*disk[0])
                robot.clear(channel)
                return
        dirs = [it for it in robot.log[channel] if it.get("status") == "direction"]
        if not dirs:
            return
        last = dirs[-1]
        result = robot.measure(channel)
        if result["status"] == "near":
            robot.clear(channel)
            return
        target = clip_arena(
            *bearing_point(robot.x, robot.y, last["svd_deg"], FINE_STEP)
        )
        if math.dist(target, (robot.x, robot.y)) < 0.5:
            return
        robot.move_to(*target)


def _fine_homing(robot: Robot, channel: int, fast: bool = False) -> None:
    """近源精测逼近：沿最近示向 6 m 步逐点测。"""
    for _ in range(FINE_STEPS):
        if robot.world.source_by_channel(channel) is None:
            return
        label = classify_channel(robot.log[channel], fast=fast)
        if label["kind"] == "near":
            robot.clear(channel)
            return
        if label["kind"] == "clearable" and label.get("vertices"):
            disk = guaranteed_disk(label["vertices"])
            if disk is not None:
                robot.move_to(*disk[0])
                robot.clear(channel)
                return
        center = label.get("mec_center")
        if center is not None and math.dist((robot.x, robot.y), tuple(center)) <= 20.0:
            if robot.clear(channel):
                return
        directions = [
            item
            for item in robot.log[channel]
            if item.get("status") == "direction"
        ]
        if not directions:
            return
        last = directions[-1]
        result = robot.measure(channel)
        if result["status"] == "near":
            robot.clear(channel)
            return
        target = clip_arena(
            *bearing_point(robot.x, robot.y, last["svd_deg"], FINE_STEP)
        )
        if math.dist(target, (robot.x, robot.y)) < 0.5:
            return
        robot.move_to(*target)


def classify_all(
    robot: Robot, fast: bool, previous: dict[int, dict] | None = None
) -> dict[int, dict]:
    states = {}
    for channel in CHANNELS:
        log = robot.log[channel]
        if (
            previous
            and channel in previous
            and previous[channel].get("_nlog") == len(log)
        ):
            states[channel] = previous[channel]
            continue
        label = classify_channel(log, fast=fast)
        label["channel"] = channel
        label["_nlog"] = len(log)
        states[channel] = label
    return states


def _unresolved(states: dict[int, dict]) -> list[dict]:
    return [state for state in states.values() if state["kind"] in NEED_MORE]


def _clearable(states: dict[int, dict]) -> list[dict]:
    return [state for state in states.values() if state["kind"] == "clearable"]


def _clear_nears(robot: Robot, states: dict[int, dict]) -> None:
    for state in states.values():
        if state["kind"] != "near":
            continue
        if robot.world.source_by_channel(state["channel"]) is None:
            continue
        robot.clear(state["channel"])


def _fallback_home(
    robot: Robot,
    state: dict,
    fast: bool = False,
    directional: bool = False,
    ab_step: float = FAST_STEP,
    eps_B: float = 1.0,
) -> None:
    """兜底逼近。

    问题四新增：directional=True（该频道有"听见后 no_signal"的定向证据，
    且已被兜底过 2 次仍无进展）时，多边形追心 / 沿过期示向跳步都可能
    走出 φ±90° 扇形导致无限循环，直接升级为从最近示向原点的快速楔形
    追踪——沿射线接近恒在扇内，确定性可达且最多 ~30 次测量。
    """
    channel = state["channel"]
    if directional:
        _wedge_homing_fast(
            robot,
            channel,
            fast=fast,
            from_xy=(robot.x, robot.y),
            bracket=True,
            ab_step=ab_step,
            eps_B=eps_B,
        )
        return
    # 有定向证据的频道缩短兜底步行预算（站梯多半失灵，别乱走）
    budget = 800.0 if int(state.get("n_nosig", 0) or 0) >= 1 else FALLBACK_MAX_WALK
    jumped = False
    walk0 = robot.walk_m
    for index in range(24 if fast else 72):
        if robot.world.source_by_channel(channel) is None:
            return
        if robot.walk_m - walk0 > budget:
            return
        label = classify_channel(robot.log[channel], fast=fast)
        label["channel"] = channel
        if label["kind"] == "near":
            robot.clear(channel)
            return
        if label["kind"] == "wedge":
            _wedge_homing(robot, channel, fast=fast)
            return
        if label["kind"] == "clearable" and label.get("vertices"):
            disk = guaranteed_disk(label["vertices"])
            if disk is not None:
                robot.move_to(*disk[0])
                if robot.clear(channel):
                    return
        if (
            label.get("kind") == "too_large"
            and label.get("mec_center") is not None
        ):
            center = tuple(label["mec_center"])
            d = math.dist((robot.x, robot.y), center)
            radius = float(label.get("mec_radius") or 0.0)
            if radius <= FINE_R and d <= radius + 20.0:
                _fine_homing(robot, channel, fast=fast)
                return
            if d > 12.0:
                if d > FALLBACK_MAX_JUMP:
                    directions = [
                        item
                        for item in robot.log[channel]
                        if item.get("status") == "direction"
                    ]
                    if not directions:
                        return
                    last = directions[-1]
                    target = clip_arena(
                        *bearing_point(
                            robot.x, robot.y, last["svd_deg"], FALLBACK_MAX_JUMP
                        )
                    )
                    if math.dist(target, (robot.x, robot.y)) < 0.5:
                        return
                    robot.move_to(*target)
                    continue
                robot.move_to(*center)
                if robot.clear(channel):
                    return
                result = robot.measure(channel)
                if result["status"] == "near":
                    robot.clear(channel)
                continue
        directions = [
            item
            for item in robot.log[channel]
            if item.get("status") == "direction"
        ]
        if not directions:
            return
        last = directions[-1]
        step = 18.0
        if not jumped and label.get("kind") == "unbounded":
            step = 800.0
            jumped = True
        target = clip_arena(
            *bearing_point(robot.x, robot.y, last["svd_deg"], step)
        )
        if math.dist(target, (robot.x, robot.y)) < 0.5:
            return
        robot.move_to(*target)
        if robot.clear(channel):
            return
        if index % 4 != 3:
            continue
        result = robot.measure(channel)
        if result["status"] == "near":
            robot.clear(channel)
            return
        if result["status"] == "no_signal":
            if label.get("kind") == "unbounded":
                continue
            back = clip_arena(
                *bearing_point(robot.x, robot.y, last["svd_deg"] + 180.0, 25.0)
            )
            robot.move_to(*back)
            if robot.clear(channel):
                return
            return
