"""Local problem-3 policies: strict, full, limited, and dynamic belief."""

from __future__ import annotations

import math
from collections import Counter

from problem3_batch_station import bearing_point, clip_arena, propose_batch_station
from problem3_route_solver import guaranteed_disk, joint_route, two_opt_open
from problem3_simulation_model import (
    CHANNELS,
    NEED_MORE,
    SPEED,
    Robot,
    World,
    census_stops,
    classify_channel,
)

MAX_STEPS = 80
FALLBACK_MAX_WALK = 3000.0   # give up after this many extra metres in one fallback
FALLBACK_MAX_JUMP = 400.0    # cap a single chase step toward a far MEC centre
FINE_R = 40.0                # MEC radius below which we run the final approach
FINE_STEP = 6.0              # step length of the final approach (<= 10 m so the
                             # 5 m "near" zone cannot be stepped over)
FINE_STEPS = 16              # max steps of the final approach
WEDGE_STEPS = 260            # max steps of the origin-anchored wedge homing


def _wedge_homing(robot: Robot, channel: int, fast: bool = False) -> None:
    """Resolve a single-bearing wedge by walking from the bearing origin.

    The source sits within its listen radius (<= 1500 m) of the bearing
    origin, so walk from that origin along the (re-aimed) bearing in small
    steps and measure every step. Re-aiming every step keeps the lateral
    error ~0.1 m, so the 5 m "near" zone cannot be missed. This only runs
    as a fallback when the usual second-station proposal overshot a source
    that happened to be very close to the origin.
    """
    dirs = [it for it in robot.log[channel] if it.get("status") == "direction"]
    if not dirs:
        return
    first = dirs[-1]
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
    """Final approach for a nearly-localised source.

    Walks small steps along the latest bearing and measures every step, so
    the 5 m "near" detection zone cannot be stepped over (18 m homing steps
    with sparse measurements could skip right past the source).
    """
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


def run_census(robot: Robot) -> None:
    for stop in census_stops():
        robot.move_to(*stop)
        for channel in CHANNELS:
            result = robot.measure(channel)
            if result["status"] == "near":
                robot.clear(channel)


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


def _fallback_home(robot: Robot, state: dict, fast: bool = False) -> None:
    channel = state["channel"]
    jumped = False
    walk0 = robot.walk_m
    for index in range(24 if fast else 72):
        if robot.world.source_by_channel(channel) is None:
            return
        # v2: walk budget — stop chasing a drifting polygon instead of
        # wandering for kilometres (worst observed: one call walked 88 km).
        if robot.walk_m - walk0 > FALLBACK_MAX_WALK:
            return
        label = classify_channel(robot.log[channel], fast=fast)
        label["channel"] = channel
        if label["kind"] == "near":
            robot.clear(channel)
            return
        if label["kind"] == "wedge":
            # v2: single bearing — walk from the bearing origin along the ray.
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
            # v2: nearly localised but still above the clear threshold —
            # switch to the fine homing approach instead of chasing bearings.
            if radius <= FINE_R and d <= radius + 20.0:
                _fine_homing(robot, channel, fast=fast)
                return
            if d > 12.0:
                # v2: only chase the MEC centre if it is close; a far centre
                # that keeps jumping around (inconsistent bearings) must not
                # be chased.
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
            # v2: an unbounded intersection (nearly parallel bearings) means
            # the source sits far along the common direction; jump towards it
            # before homing, otherwise the first no_signal aborts the walk.
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
                # still far along the common direction: keep walking
                continue
            back = clip_arena(
                *bearing_point(robot.x, robot.y, last["svd_deg"] + 180.0, 25.0)
            )
            robot.move_to(*back)
            if robot.clear(channel):
                return
            return


def _tasks(robot: Robot, unresolved: list[dict], clearable: list[dict]) -> list[dict]:
    tasks: list[dict] = []
    for state in clearable:
        vertices = state.get("vertices")
        if not vertices:
            continue
        disk = guaranteed_disk(vertices)
        if disk is None:
            continue
        tasks.append(
            {"kind": "clear", "channel": state["channel"], "disk": disk}
        )
    if unresolved:
        station, channels = propose_batch_station(
            (robot.x, robot.y), unresolved
        )
        tasks.append(
            {
                "kind": "station",
                "channels": channels,
                "disk": (station, 0.0),
            }
        )
    return tasks


def _measure_channels(robot: Robot, channels: list[int]) -> None:
    for channel in channels:
        source = robot.world.source_by_channel(channel)
        last = robot.log[channel][-1] if robot.log[channel] else None
        if source is None and last and last.get("status") == "cleared":
            continue
        last_dir = next(
            (
                item
                for item in reversed(robot.log[channel])
                if item.get("status") == "direction"
            ),
            None,
        )
        if last_dir is not None and math.dist(
            (robot.x, robot.y), (last_dir["x"], last_dir["y"])
        ) < 40.0:
            continue
        result = robot.measure(channel)
        if result["status"] == "near":
            robot.clear(channel)


def _execute(
    robot: Robot,
    tasks: list[dict],
    replan_on_clear: bool,
    stop_after_station: bool,
) -> bool:
    if not tasks:
        return False
    disks = [task["disk"] for task in tasks]
    order, _, path = joint_route((robot.x, robot.y), disks)
    for step, index in enumerate(order):
        robot.move_to(*path[step])
        task = tasks[index]
        if task["kind"] == "station":
            _measure_channels(robot, task["channels"])
            robot.n_replan += 1
            if stop_after_station:
                return True
            continue
        ok = robot.clear(task["channel"])
        if not ok:
            result = robot.measure(task["channel"])
            if result["status"] == "near":
                robot.clear(task["channel"])
            robot.n_replan += 1
            return True
        if replan_on_clear:
            robot.n_replan += 1
            return True
    return True


def _strict_step(
    robot: Robot,
    unresolved: list[dict],
    clearable: list[dict],
    fast: bool,
) -> bool:
    if unresolved:
        station, channels = propose_batch_station((robot.x, robot.y), unresolved)
        key = (round(station[0], 0), round(station[1], 0), tuple(channels))
        if getattr(robot, "_station_key", None) == key:
            _fallback_home(robot, unresolved[0], fast=fast)
            robot.n_replan += 1
            return True
        robot._station_key = key
        robot.move_to(*station)
        _measure_channels(robot, channels)
        robot.n_replan += 1
        return True
    if clearable:
        tasks = _tasks(robot, [], clearable)
        return _execute(
            robot, tasks, replan_on_clear=False, stop_after_station=False
        )
    return False


def copy_robot(robot: Robot) -> Robot:
    copy = Robot(robot.world.clone(keep_cleared=True))
    copy.x = robot.x
    copy.y = robot.y
    copy.channel = robot.channel
    copy.time = robot.time
    copy.walk_m = robot.walk_m
    copy.n_measure = robot.n_measure
    copy.n_clear_ok = robot.n_clear_ok
    copy.n_clear_fail = robot.n_clear_fail
    copy.n_replan = robot.n_replan
    copy.log = {channel: list(items) for channel, items in robot.log.items()}
    return copy


def _census_meta(robot: Robot, fast: bool) -> dict:
    census_states = classify_all(robot, fast)
    remaining = [
        (source.x, source.y)
        for source in robot.world.sources
        if not source.cleared
    ]
    _, oracle_walk = two_opt_open((robot.x, robot.y), remaining)
    return {
        "census_time": robot.time,
        "census_kinds": Counter(
            state["kind"]
            for state in census_states.values()
            if state["kind"] != "empty"
        ),
        "oracle_time": robot.time + oracle_walk / SPEED + 5.0 * len(remaining),
        "states": census_states,
    }


def _finish_policy(
    robot: Robot,
    name: str,
    fast: bool,
    meta: dict,
    max_steps: int,
) -> dict:
    steps = 0
    prev_states: dict[int, dict] | None = meta.get("states")
    idle = 0
    last_remaining = robot.world.remaining()
    while robot.world.remaining() > 0 and steps < max_steps:
        steps += 1
        states = classify_all(robot, fast, prev_states)
        _clear_nears(robot, states)
        if robot.world.remaining() == 0:
            break
        states = classify_all(robot, fast, states)
        prev_states = states
        unresolved = _unresolved(states)
        clearable = _clearable(states)
        if idle >= 1 and unresolved:
            index = getattr(robot, "_home_index", 0) % len(unresolved)
            robot._home_index = index + 1
            _fallback_home(robot, unresolved[index], fast=fast)
            robot.n_replan += 1
            idle = 0
            last_remaining = robot.world.remaining()
            continue
        if name == "strict":
            progressed = _strict_step(robot, unresolved, clearable, fast)
        elif name == "full":
            progressed = _execute(
                robot,
                _tasks(robot, unresolved, clearable),
                replan_on_clear=True,
                stop_after_station=True,
            )
        else:
            progressed = _execute(
                robot,
                _tasks(robot, unresolved, clearable),
                replan_on_clear=False,
                stop_after_station=True,
            )
        if not progressed:
            if unresolved:
                _fallback_home(robot, unresolved[0], fast=fast)
                robot.n_replan += 1
            else:
                break
        remaining = robot.world.remaining()
        idle = idle + 1 if remaining >= last_remaining else 0
        last_remaining = remaining

    return {
        "policy": name,
        "fidelity": "fast" if fast else "high",
        "cleared": sum(1 for source in robot.world.sources if source.cleared),
        "n_sources": len(robot.world.sources),
        "time_s": robot.time,
        "walk_m": robot.walk_m,
        "measures": robot.n_measure,
        "replans": robot.n_replan,
        "clear_fail": robot.n_clear_fail,
        "steps": steps,
        "census_time_s": meta["census_time"],
        "oracle_time_s": meta["oracle_time"],
        "regret_s": robot.time - meta["oracle_time"],
        "census_kinds": dict(meta["census_kinds"]),
        "remaining": robot.world.remaining(),
    }


def run_policy(
    world: World,
    name: str,
    fidelity: str = "high",
    max_steps: int = MAX_STEPS,
) -> dict:
    if name == "belief_ctrl":
        from problem3_belief import BELIEF_MAX_STEPS, run_belief

        return run_belief(
            world,
            fidelity=fidelity,
            max_steps=max(max_steps, BELIEF_MAX_STEPS),
        )
    if name in {"belief", "belief_struct", "belief_punch", "belief_delay"}:
        from problem3_belief_struct import BELIEF_MAX_STEPS, run_belief_struct

        mode = {
            "belief": "e15",
            "belief_struct": "e15",
            "belief_punch": "punch",
            "belief_delay": "delay",
        }[name]
        return run_belief_struct(
            world,
            fidelity=fidelity,
            max_steps=max(max_steps, BELIEF_MAX_STEPS),
            mode=mode,
        )
    if name not in {"strict", "full", "limited"}:
        raise ValueError(f"unknown policy {name}")
    fast = fidelity == "fast"
    robot = Robot(world)
    run_census(robot)
    meta = _census_meta(robot, fast)
    return _finish_policy(robot, name, fast, meta, max_steps)


def run_policy_set(
    world: World,
    fidelity: str = "high",
    max_steps: int = MAX_STEPS,
) -> dict[str, dict]:
    fast = fidelity == "fast"
    robot = Robot(world)
    run_census(robot)
    meta = _census_meta(robot, fast)
    return {
        name: _finish_policy(copy_robot(robot), name, fast, meta, max_steps)
        for name in ("strict", "full", "limited")
    }
