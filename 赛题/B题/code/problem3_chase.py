"""Cautious-hunt policy for problem 3.

New root. Not a patch of fullopt / speed6 / Held–Karp.

Web sources this file actually uses:

1. ARDF / fox hunting (IARU, Fox Hunting 101):
   do not run down a single bearing; step sideways (~35–60°) and
   triangulate; scan every still-open transmitter at each stop;
   home in small steps once close.
2. Vander Hook, Tokekar, Isler, J. Field Robotics 2014,
   “Cautious Greedy Strategy for Bearing-only Active Localization”:
   the next station must stay off the current bearing line, because a
   near-collinear second ray does not shrink the wedge; travel time
   and measure time both count.
3. Two-station DF geometry (最优交会角 / GDOP):
   cut angle in about 40–120°, baseline around 1000–1200 m, so the
   ±1° diamond can fall inside the 20 m clear disk.
4. Disk covering: worst-case listen radius 1000 m on an 1800 m arena
   needs origin + a 1200 m hexagon (7 points). Those vertices are the
   second stations, not a second net.

Loop: listen → if a channel is clearable, go clear it → else take a
cautious second station (prefer a still-needed hex vertex) → else
walk leftover cover vertices in angular order → stop on the Ω
certificate (no unresolved heard channel, no leftover cover hole).
Never uses world.remaining() to stop.
"""

from __future__ import annotations

import math

from problem3_batch_station import angular_distance, bearing_point, clip_arena
from problem3_belief import UnheardGrid, _heard, _measured_here
from problem3_fusion_policies import (
    FINE_R,
    _clear_nears,
    _fallback_home,
    _fine_homing,
    classify_all,
)
from problem3_route_solver import guaranteed_disk
from problem3_simulation_model import (
    ARENA,
    CHANNELS,
    CLEAR_R,
    LISTEN_CERT,
    LISTEN_MAX,
    LISTEN_MIN,
    NEED_MORE,
    Robot,
    census_stops,
    classify_channel,
)

MAX_STEPS = 160
MAX_HEARD = 16
HERE = 1.0
COVER_SNAP = 40.0
MIN_BASELINE = 180.0
CUT_LO = 30.0
CUT_HI = 150.0
OFFSET_ANGLES = (35.0, 45.0, 60.0, 90.0)
OFFSET_DISTS = (400.0, 600.0, 800.0, 1000.0, 1200.0)
GUESS_DISTS = (300.0, 500.0, 800.0, 1100.0, 1400.0)
BLOCKING = frozenset(
    {
        "near",
        "clearable",
        "wedge",
        "unbounded",
        "too_large",
        "empty_intersection",
    }
)


def _dirs(state: dict) -> list[dict]:
    return [
        item
        for item in state.get("measurements", [])
        if item.get("status") == "direction"
    ]


def _last_dir(state: dict) -> dict | None:
    dirs = _dirs(state)
    return dirs[-1] if dirs else None


def _alive(point: tuple[float, float], state: dict) -> bool:
    for silence in state.get("silences") or []:
        if math.dist(point, (float(silence[0]), float(silence[1]))) <= LISTEN_CERT:
            return False
    return True


def _guess_point(last: dict, dist: float) -> tuple[float, float] | None:
    point = bearing_point(last["x"], last["y"], last["svd_deg"], dist)
    if math.hypot(*point) > ARENA + 1e-6:
        return None
    return point


def _cut_at_guess(
    station: tuple[float, float],
    last: dict,
    guess_dist: float,
) -> float:
    guess = _guess_point(last, guess_dist)
    if guess is None:
        return 0.0
    gx, gy = guess
    a1 = math.degrees(math.atan2(last["y"] - gy, last["x"] - gx)) % 360.0
    a2 = math.degrees(math.atan2(station[1] - gy, station[0] - gx)) % 360.0
    return angular_distance(a1, a2)


def _tried(state: dict) -> set[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    for item in state.get("measurements", []):
        seen.add((round(float(item["x"]), 0), round(float(item["y"]), 0)))
    for silence in state.get("silences") or []:
        seen.add((round(float(silence[0]), 0), round(float(silence[1]), 0)))
    return seen


def station_is_cautious(station: tuple[float, float], state: dict) -> bool:
    """Off the ray, far enough, and some still-alive guess still hears it."""
    last = _last_dir(state)
    if last is None:
        return False
    if math.dist(station, (last["x"], last["y"])) < MIN_BASELINE:
        return False
    heading = (
        math.degrees(math.atan2(station[1] - last["y"], station[0] - last["x"]))
        % 360.0
    )
    if angular_distance(heading, last["svd_deg"]) < 20.0:
        return False
    for dist in GUESS_DISTS:
        guess = _guess_point(last, dist)
        if guess is None or not _alive(guess, state):
            continue
        if math.dist(station, guess) <= LISTEN_MAX:
            cut = _cut_at_guess(station, last, dist)
            if CUT_LO <= cut <= CUT_HI:
                return True
    return False


def _offset_candidates(last: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for alpha in OFFSET_ANGLES:
        for sign in (1.0, -1.0):
            for dist in OFFSET_DISTS:
                points.append(
                    clip_arena(
                        *bearing_point(
                            last["x"],
                            last["y"],
                            last["svd_deg"] + sign * alpha,
                            dist,
                        )
                    )
                )
    return points


def score_station(
    here: tuple[float, float],
    station: tuple[float, float],
    state: dict,
    dual_use: bool,
) -> float:
    last = _last_dir(state)
    if last is None or not station_is_cautious(station, state):
        return 1e18
    travel = math.dist(here, station)
    cuts = [_cut_at_guess(station, last, dist) for dist in GUESS_DISTS]
    good = [cut for cut in cuts if CUT_LO <= cut <= CUT_HI]
    if not good:
        return 1e18
    cut_pen = min(abs(cut - 80.0) for cut in good)
    bonus = 150.0 if dual_use else 0.0
    return travel + 12.0 * cut_pen - bonus


def pick_second_station(
    here: tuple[float, float],
    state: dict,
    cover_stops: list[tuple[float, float]],
    needed_cover: list[tuple[float, float]],
) -> tuple[float, float] | None:
    last = _last_dir(state)
    if last is None:
        return None
    tried = _tried(state)
    tried.add((round(here[0], 0), round(here[1], 0)))
    needed_set = {(round(x, 1), round(y, 1)) for x, y in needed_cover}
    candidates: list[tuple[tuple[float, float], bool]] = []
    for stop in cover_stops[1:]:
        candidates.append((stop, (round(stop[0], 1), round(stop[1], 1)) in needed_set))
    for point in _offset_candidates(last):
        candidates.append((point, False))
    ranked: list[tuple[float, tuple[float, float]]] = []
    for point, dual in candidates:
        if (round(point[0], 0), round(point[1], 0)) in tried:
            continue
        ranked.append((score_station(here, point, state, dual), point))
    ranked.sort(key=lambda item: item[0])
    if ranked and ranked[0][0] < 1e17:
        return ranked[0][1]
    for sign in (1.0, -1.0):
        for dist in (400.0, 700.0, 1000.0):
            point = clip_arena(
                *bearing_point(
                    last["x"], last["y"], last["svd_deg"] + sign * 45.0, dist
                )
            )
            if (round(point[0], 0), round(point[1], 0)) in tried:
                continue
            if math.dist(here, point) >= HERE:
                return point
    return None


def _clock_next(
    here: tuple[float, float],
    stops: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if not stops:
        return None
    heading = math.atan2(here[1], here[0])

    def delta(stop: tuple[float, float]) -> float:
        angle = math.atan2(stop[1], stop[0])
        return (angle - heading) % (2.0 * math.pi)

    return min(stops, key=delta)


def run_chase(world, max_steps: int = MAX_STEPS, debug: bool = False) -> dict:
    robot = Robot(world)
    cover_stops = census_stops()
    visited = [False] * len(cover_stops)
    unheard = UnheardGrid()
    states = classify_all(robot, False)
    cover_at_first_clear: int | None = None
    fine_used: set[int] = set()

    def n_heard() -> int:
        return sum(1 for channel in CHANNELS if _heard(robot, channel))

    def close_census() -> None:
        if n_heard() < MAX_HEARD:
            return
        for channel in CHANNELS:
            if not _heard(robot, channel):
                unheard.forget(channel)
        for index in range(len(visited)):
            visited[index] = True

    def needed_covers() -> list[tuple[float, float]]:
        close_census()
        needed: list[tuple[float, float]] = []
        for index, stop in enumerate(cover_stops):
            if visited[index]:
                continue
            if unheard.cover_still_needed(stop):
                needed.append(stop)
            else:
                visited[index] = True
        return needed

    def snap_cover() -> None:
        here = (robot.x, robot.y)
        for index, stop in enumerate(cover_stops):
            if math.dist(here, stop) <= COVER_SNAP:
                visited[index] = True

    def at_cover() -> bool:
        here = (robot.x, robot.y)
        return any(math.dist(here, stop) <= COVER_SNAP for stop in cover_stops)

    def should_measure(channel: int) -> bool:
        last = robot.log[channel][-1] if robot.log[channel] else None
        if last and last.get("status") == "cleared":
            return False
        state = states.get(channel)
        if state and state["kind"] in {"cleared", "near", "clearable"}:
            return False
        if _measured_here(robot, channel):
            return False
        here = (robot.x, robot.y)
        if not _heard(robot, channel):
            if unheard.channel_empty(channel):
                return False
            if at_cover():
                return unheard.intersects(channel, robot.x, robot.y, LISTEN_MAX)
            return False
        if state is None:
            return True
        if state["kind"] in NEED_MORE:
            if state["kind"] == "wedge":
                return station_is_cautious(here, state) or at_cover()
            return True
        return False

    def work_here() -> None:
        nonlocal states, cover_at_first_clear
        snap_cover()
        for channel in CHANNELS:
            if not should_measure(channel):
                continue
            was_heard = _heard(robot, channel)
            result = robot.measure(channel)
            if result["status"] == "near":
                robot.clear(channel)
            if not was_heard:
                if result["status"] == "no_signal":
                    unheard.exclude(channel, robot.x, robot.y, LISTEN_CERT)
                    unheard.record_listen(channel, robot.x, robot.y)
                else:
                    unheard.forget(channel)
        states = classify_all(robot, False, states)
        _clear_nears(robot, states)
        for state in states.values():
            if state["kind"] != "clearable":
                continue
            center = state.get("mec_center")
            radius = state.get("mec_radius")
            if center is None or radius is None:
                continue
            if math.dist((robot.x, robot.y), tuple(center)) + radius <= CLEAR_R:
                if robot.world.source_by_channel(state["channel"]) is not None:
                    robot.clear(state["channel"])
        states = classify_all(robot, False, states)
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
            if channel in fine_used:
                continue
            fine_used.add(channel)
            _fine_homing(robot, channel, fast=False)
            states = classify_all(robot, False, states)
        close_census()
        needed_covers()
        if cover_at_first_clear is None and robot.n_clear_ok > 0:
            cover_at_first_clear = sum(visited)

    def mission_complete() -> bool:
        close_census()
        if any(state["kind"] in BLOCKING for state in states.values()):
            return False
        return not needed_covers()

    def next_dest() -> tuple[str, tuple[float, float], int | None]:
        here = (robot.x, robot.y)
        clears: list[tuple[float, tuple[float, float], int]] = []
        for state in states.values():
            if state["kind"] == "near":
                return "clear", here, state["channel"]
            if state["kind"] != "clearable":
                continue
            vertices = state.get("vertices") or []
            disk = guaranteed_disk(vertices) if vertices else None
            if disk is None:
                center = state.get("mec_center")
                if center is None:
                    continue
                dest = tuple(center)
            else:
                dest = disk[0]
            clears.append((math.dist(here, dest), dest, state["channel"]))
        if clears:
            clears.sort()
            return "clear", clears[0][1], clears[0][2]

        probes: list[tuple[float, tuple[float, float], int]] = []
        for state in states.values():
            if state["kind"] != "too_large":
                continue
            center = state.get("mec_center")
            if center is None:
                continue
            dest = tuple(center)
            probes.append((math.dist(here, dest), dest, state["channel"]))
        if probes:
            probes.sort()
            return "probe", probes[0][1], probes[0][2]

        needed = needed_covers()
        wedges = [
            state
            for state in states.values()
            if state["kind"] in NEED_MORE and state["kind"] != "too_large"
        ]
        if wedges:
            options: list[tuple[float, tuple[float, float], int]] = []
            for state in wedges:
                dest = pick_second_station(here, state, cover_stops, needed)
                if dest is None:
                    continue
                options.append(
                    (
                        score_station(
                            here,
                            dest,
                            state,
                            (round(dest[0], 1), round(dest[1], 1))
                            in {(round(x, 1), round(y, 1)) for x, y in needed},
                        ),
                        dest,
                        state["channel"],
                    )
                )
            if options:
                options.sort()
                return "station", options[0][1], options[0][2]

        cover = _clock_next(here, needed)
        if cover is not None:
            return "cover", cover, None
        hole = unheard.centroid()
        if hole is not None:
            return "hole", clip_arena(*hole), None
        return "idle", here, None

    work_here()
    steps = 0
    stuck = 0
    last_xy = (robot.x, robot.y)
    last_key: tuple | None = None
    recent_dests: list[tuple[float, float]] = []
    station_hits: dict[int, int] = {}
    stop_reason = "max_steps"

    while steps < max_steps:
        if mission_complete():
            stop_reason = "certificate"
            break
        steps += 1
        kind, dest, channel = next_dest()
        if debug:
            kinds = {ch: states[ch]["kind"] for ch in CHANNELS}
            print(
                f"step={steps} here=({robot.x:.0f},{robot.y:.0f}) "
                f"-> {kind} {dest} ch={channel} "
                f"cleared={robot.n_clear_ok} rem={robot.world.remaining()} "
                f"block={[c for c,k in kinds.items() if k in BLOCKING]}",
                flush=True,
            )
        if kind == "idle":
            stop_reason = "stuck"
            break
        key = (kind, round(dest[0], 1), round(dest[1], 1), channel)
        if math.dist((robot.x, robot.y), dest) < HERE:
            if kind == "clear" and channel is not None:
                robot.clear(channel)
                states = classify_all(robot, False, states)
                continue
            if kind == "probe" and channel is not None:
                if not _measured_here(robot, channel):
                    result = robot.measure(channel)
                    if result["status"] == "near":
                        robot.clear(channel)
                    elif result["status"] == "no_signal":
                        unheard.exclude(channel, robot.x, robot.y, LISTEN_CERT)
                    states = classify_all(robot, False, states)
                    continue
                if channel not in fine_used:
                    fine_used.add(channel)
                    _fine_homing(robot, channel, fast=False)
                    states = classify_all(robot, False, states)
                    continue
            cover = _clock_next((robot.x, robot.y), needed_covers())
            if cover is not None and math.dist((robot.x, robot.y), cover) >= HERE:
                dest = cover
                kind = "cover"
                channel = None
            elif key == last_key:
                stop_reason = "stuck"
                break
        last_key = key
        if (
            kind == "station"
            and channel is not None
            and station_hits.get(channel, 0) >= 3
        ):
            _fallback_home(robot, states[channel], fast=False)
            states = classify_all(robot, False, states)
            continue
        if recent_dests and math.dist(dest, recent_dests[-1]) < HERE:
            if channel is not None:
                _fallback_home(robot, states[channel], fast=False)
                states = classify_all(robot, False, states)
            else:
                work_here()
            continue
        if (
            len(recent_dests) >= 2
            and math.dist(dest, recent_dests[-2]) < 5.0
        ):
            if channel is not None:
                _fallback_home(robot, states[channel], fast=False)
                states = classify_all(robot, False, states)
                continue
        recent_dests.append(dest)
        if kind == "station" and channel is not None:
            station_hits[channel] = station_hits.get(channel, 0) + 1
        robot.move_to(*dest)
        robot.n_replan += 1
        if kind == "clear" and channel is not None:
            if robot.world.source_by_channel(channel) is not None:
                if not robot.clear(channel):
                    work_here()
                    label = classify_channel(robot.log[channel])
                    if label.get("kind") == "too_large" and channel not in fine_used:
                        fine_used.add(channel)
                        _fine_homing(robot, channel, fast=False)
            states = classify_all(robot, False, states)
        else:
            work_here()
        here = (robot.x, robot.y)
        stuck = stuck + 1 if math.dist(here, last_xy) < HERE else 0
        last_xy = here
        if stuck >= 4:
            stop_reason = "stuck"
            break

    if mission_complete():
        stop_reason = "certificate"

    return {
        "policy": "chase",
        "fidelity": "high",
        "cleared": sum(1 for source in robot.world.sources if source.cleared),
        "n_sources": len(robot.world.sources),
        "time_s": robot.time,
        "walk_m": robot.walk_m,
        "measures": robot.n_measure,
        "replans": robot.n_replan,
        "clear_fail": robot.n_clear_fail,
        "steps": steps,
        "census_time_s": 0.0,
        "oracle_time_s": 0.0,
        "regret_s": 0.0,
        "census_kinds": {},
        "remaining": robot.world.remaining(),
        "stop_reason": stop_reason,
        "cover_visits": sum(visited),
        "cover_visits_at_first_clear": (
            cover_at_first_clear
            if cover_at_first_clear is not None
            else sum(visited)
        ),
    }
