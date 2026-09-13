"""Dynamic belief policy for problem 4 (mixed omni / directional sources).

Same skeleton as the frozen problem-3 policy (``problem3_belief.run_belief``:
per-channel set-membership state, three task families, open Held–Karp +
TSPN, replan after every stop) with four problem-4 changes:

1. State: ``OrientGrid`` (position x orientation bit rows) instead of the
   position-only ``UnheardGrid``. no_signal no longer punches a disk.
2. Census network: N19 (origin + hexagon 1000 m + 12 stops on a ring of
   1825 m *outside* the arena). A directional source on the rim facing
   outward can only be heard from outside, so the station hull must
   contain the arena (B题-进度 §3 问题 4).
3. Localisation: second stations / probes are chosen by the share of
   surviving (g, phi) cells that are certified audible ("approach from
   the side we already heard it from"); a silent second station flips
   that share to the mirror side (mirror-station lemma). Sudden silence in
   the fine homing walk means we crossed the source's half-plane boundary,
   which passes through the source: clear here and a few metres back.
4. Stop rule: heard 16, or every unheard channel's mask is empty
   (certificate), or mean surviving mass <= EPS_MASS (uniform prior:
   posterior probability of a hidden source <= EPS_MASS). Never uses the
   local oracle ``remaining()`` for the decision.

Never talks to the official simulator.
"""

from __future__ import annotations

import math

from problem3_batch_station import (
    STATION_DIST,
    bearing_point,
    candidate_points,
    clip_arena,
    item_center_bonus,
    possible_samples,
    station_useful,
)
from problem3_route_solver import guaranteed_disk, joint_route, two_opt_open
from problem3_simulation_model import (
    CHANNELS,
    CLEAR_R,
    LISTEN_MAX,
    LISTEN_MIN,
    SPEED,
    Robot,
    classify_channel,
)
from problem4_orient_grid import OrientGrid, StopCache
from problem4_world import census_stops_n19

BELIEF_MAX_STEPS = 260
COVER_SNAP = 40.0
HERE = 1.0
PROBE_MAX_R = 200.0      # MEC radius below which we probe the source directly
PROBE_MAX_TRIES = 3      # probes per channel before leaving it to stations
PROBE_AUDIBLE_OK = 0.9   # probe at the MEC centre itself if this audible
UNHEARD_SPACING = 700.0  # min spacing between opportunistic unheard listens
MAX_HEARD = 16
EPS_MASS = 0.01          # mean surviving mass below which the census is closed
MERGE_STATIONS = True    # let a pending census stop serve as the second station
MERGE_AUDIBLE = 0.5      # ... if at least this share of surviving cells is audible there
COVER_LISTEN = "hear"    # "hear": listen if any feasible cell is audible within 1500 m
                         # "kill": listen only if silence would certify something (990 m)
FINE_R = 40.0
FINE_STEP = 6.0
FINE_STEPS = 16
BACK_STEPS = (12.0, 12.0)  # retreat steps after silence in the fine homing
STAND_MAX_RHO = 1990.0     # farthest we ever stand from the origin
# (along-track, lateral) second-station offsets w.r.t. a bearing, metres
LATERAL_STATIONS = ((750.0, 600.0), (450.0, 350.0), (250.0, 200.0))
HOME_HOP = 150.0           # bearing-following hop of the fallback homing
HOME_MIN_HOP = 25.0
HOME_MAX_HOPS = 40
HOME_MAX_WALK = 3000.0
HOME_MAX_TRIES = 2         # fallbacks per channel before it is parked


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


def fine_homing_q4(robot: Robot, channel: int, fast: bool = False) -> None:
    """Final approach with the problem-4 silence rule.

    Walk 6 m steps along the latest bearing measuring every step. If a step
    returns no_signal right after bearings, we just crossed the source's
    half-plane boundary; that line passes through the source, so the source
    is within a few steps behind us: try /clear here and while backing up.
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
            item for item in robot.log[channel] if item.get("status") == "direction"
        ]
        if not directions:
            return
        last = directions[-1]
        result = robot.measure(channel)
        if result["status"] == "near":
            robot.clear(channel)
            return
        if result["status"] == "no_signal":
            if robot.clear(channel):
                return
            for back in BACK_STEPS:
                target = bearing_point(robot.x, robot.y, last["svd_deg"] + 180.0, back)
                robot.move_to(*target)
                if robot.clear(channel):
                    return
            return
        target = bearing_point(robot.x, robot.y, last["svd_deg"], FINE_STEP)
        if math.dist(target, (robot.x, robot.y)) < 0.5:
            return
        robot.move_to(*target)


def clip_q4(x: float, y: float, radius: float = STAND_MAX_RHO) -> tuple[float, float]:
    """Stations may stand outside the arena (the N19 ring does); cap the radius."""
    return clip_arena(x, y, radius)


def _directions(state: dict) -> list[dict]:
    return [
        item
        for item in state.get("measurements", [])
        if item.get("status") == "direction"
    ]


def candidate_points_q4(state: dict) -> list[tuple[float, float]]:
    """Second-station candidates for one unresolved channel.

    * (a, +-h) lateral stations w.r.t. each bearing: a = 750 m along the
      bearing, h = 600 m sideways is within 1000 m of every point of a
      0–1500 m wedge and crosses it at >= 38.7 deg (problem-2 guaranteed
      reception, C_recv); the closer pairs serve short wedges cheaply.
    * the problem-3 set (1200 m at +-35/50/90 deg) for compatibility;
    * the MEC centre only when it is meaningful (radius <= 600 m): three
      nearly collinear bearings give a 30 km sliver whose centre is junk.
    """
    points: list[tuple[float, float]] = []
    center = state.get("mec_center")
    radius = float(state.get("mec_radius") or 0.0)
    if center is not None and radius <= 600.0:
        cx, cy = center
        points.append(clip_q4(cx, cy))
        for deg in (0.0, 90.0, 180.0, 270.0):
            px, py = bearing_point(cx, cy, deg, min(80.0, max(25.0, radius)))
            points.append(clip_q4(px, py))
    for item in _directions(state):
        theta = item["svd_deg"]
        for along, side in LATERAL_STATIONS:
            ax, ay = bearing_point(item["x"], item["y"], theta, along)
            for sign in (1.0, -1.0):
                px, py = bearing_point(ax, ay, theta + 90.0 * sign, side)
                points.append(clip_q4(px, py))
        for offset in (35.0, -35.0, 50.0, -50.0, 90.0, -90.0):
            px, py = bearing_point(item["x"], item["y"], theta + offset, STATION_DIST)
            points.append(clip_q4(px, py))
    return points


def propose_station_q4(
    current: tuple[float, float],
    unresolved: list[dict],
    orient: OrientGrid,
    measured: dict[int, list[tuple[float, float]]] | None = None,
    listen: float = LISTEN_MIN,
) -> tuple[tuple[float, float], list[int], float]:
    """Candidate stations scored by geometry x certified audibility.

    Returns (station, channels served, mean audible share). A candidate
    within 40 m of any earlier measurement of a channel cannot serve it
    (repeating a measurement at the same point gives nothing new).
    """
    scored: list[tuple[float, tuple[float, float], list[int], float]] = []
    seen: set[tuple[float, float]] = set()
    for state in unresolved:
        for cand in candidate_points_q4(state):
            key = (round(cand[0], 1), round(cand[1], 1))
            if key in seen:
                continue
            seen.add(key)
            channels = [
                item["channel"]
                for item in unresolved
                if station_useful(cand, item, listen)
                and not (
                    measured
                    and any(
                        math.dist(cand, point) < 40.0
                        for point in measured.get(item["channel"], ())
                    )
                )
            ]
            if not channels:
                continue
            audible = sum(
                orient.audible_fraction(channel, cand[0], cand[1]) for channel in channels
            ) / len(channels)
            weight = max(audible, 0.05)
            extra = math.dist(current, cand)
            score = len(channels) * weight / (1.0 + extra / 1000.0)
            if item_center_bonus(cand, unresolved):
                score += 0.25 * weight
            scored.append((score, cand, channels, audible))
    if scored:
        scored.sort(key=lambda row: row[0], reverse=True)
        _, station, channels, audible = scored[0]
        return station, channels, audible

    first = unresolved[0]
    dirs = [
        item
        for item in first.get("measurements", [])
        if item.get("status") == "direction"
    ]
    last = dirs[-1] if dirs else {"x": current[0], "y": current[1], "svd_deg": 0.0}
    station = clip_q4(
        *bearing_point(last["x"], last["y"], last["svd_deg"] + 40.0, STATION_DIST)
    )
    return station, [item["channel"] for item in unresolved], 1.0


def fallback_home_q4(
    robot: Robot,
    state: dict,
    fast: bool = False,
    hop: float = HOME_HOP,
    max_walk: float = HOME_MAX_WALK,
) -> None:
    """Bearing-following homing for a channel the station logic cannot fix.

    Start at the hearing station nearest to the robot and hop along the
    latest bearing, re-measuring after every hop. The segment from a
    hearing station to the source lies inside the source's (convex)
    coverage half-plane, so the walk stays audible for a directional
    source. A bearing flip means the source is behind us (omni source
    passed): shrink the hop. Silence means we crossed the boundary that
    passes through the source: return to the last audible point, shrink
    the hop, and finish with /clear attempts once the hop is small.
    """
    channel = state["channel"]
    dirs = _directions(state)
    if not dirs:
        return
    walk0 = robot.walk_m
    base = min(dirs, key=lambda it: math.dist((robot.x, robot.y), (it["x"], it["y"])))
    robot.move_to(base["x"], base["y"])
    bearing = float(base["svd_deg"])
    last_audible = (robot.x, robot.y)
    for _ in range(HOME_MAX_HOPS):
        if robot.world.source_by_channel(channel) is None:
            return
        if robot.walk_m - walk0 > max_walk:
            return
        label = classify_channel(robot.log[channel], fast=fast)
        if label["kind"] == "near":
            robot.clear(channel)
            return
        if label["kind"] == "clearable" and label.get("vertices"):
            disk = guaranteed_disk(label["vertices"])
            if disk is not None:
                robot.move_to(*disk[0])
                if robot.clear(channel):
                    return
        center = label.get("mec_center")
        radius = float(label.get("mec_radius") or 0.0)
        if (
            label.get("kind") == "too_large"
            and center is not None
            and radius <= FINE_R
            and math.dist((robot.x, robot.y), tuple(center)) <= radius + 20.0
        ):
            fine_homing_q4(robot, channel, fast=fast)
            return
        target = clip_q4(*bearing_point(robot.x, robot.y, bearing, hop))
        if math.dist(target, (robot.x, robot.y)) < 0.5:
            return
        robot.move_to(*target)
        result = robot.measure(channel)
        status = result["status"]
        if status == "near":
            robot.clear(channel)
            return
        if status == "direction":
            new = float(result["svd_deg"])
            flipped = abs((new - bearing + 180.0) % 360.0 - 180.0) > 90.0
            bearing = new
            last_audible = (robot.x, robot.y)
            if flipped:
                hop = max(HOME_MIN_HOP, hop / 3.0)
            continue
        # silence
        if hop <= HOME_MIN_HOP + 1e-9:
            if robot.clear(channel):
                return
            for back in BACK_STEPS:
                robot.move_to(*bearing_point(robot.x, robot.y, bearing + 180.0, back))
                if robot.clear(channel):
                    return
            return
        robot.move_to(*last_audible)
        hop = max(HOME_MIN_HOP, hop / 3.0)


def probe_point_q4(
    state: dict,
    orient: OrientGrid,
    here: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """Where to take the close-range bearing of a nearly-localised source.

    Problem 3 probes the MEC centre. For a possibly directional source the
    centre may lie on the silent back side, so if the centre is not
    certified audible enough, stand just outside the polygon on the most
    audible side instead.
    """
    center = tuple(state["mec_center"])
    radius = float(state.get("mec_radius") or 0.0)
    channel = state["channel"]
    audible_center = orient.audible_fraction(channel, center[0], center[1])
    if audible_center >= PROBE_AUDIBLE_OK:
        return center, audible_center
    rho = max(radius, 30.0)
    best_point, best_key, best_aud = center, (-audible_center, 0.0), audible_center
    for k in range(8):
        point = bearing_point(center[0], center[1], 45.0 * k, rho)
        audible = orient.audible_fraction(channel, point[0], point[1])
        key = (-round(audible, 2), math.dist(here, point))
        if key < best_key:
            best_key, best_point, best_aud = key, point, audible
    return best_point, best_aud


def run_belief_q4(
    world,
    fidelity: str = "high",
    max_steps: int = BELIEF_MAX_STEPS,
    eps_mass: float = EPS_MASS,
    cover_stops: list[tuple[float, float]] | None = None,
    merge_stations: bool = MERGE_STATIONS,
    cover_listen: str = COVER_LISTEN,
) -> dict:
    from problem3_fusion_policies import (
        _clear_nears,
        _clearable,
        _unresolved,
        classify_all,
    )

    fast = fidelity == "fast"
    robot = Robot(world)
    cover_stops = list(cover_stops) if cover_stops is not None else census_stops_n19()
    stop_caches = [StopCache(*stop) for stop in cover_stops]
    visited = [False] * len(cover_stops)
    orient = OrientGrid()
    unused_listens: list[tuple[float, float]] = []
    states = classify_all(robot, fast)
    cover_at_first_clear: int | None = None
    fine_attempts: dict[int, int] = {}
    probe_tries: dict[int, int] = {}
    counters = {"station_silent": 0, "probe_silent": 0, "eps_closed": 0}
    cover_cache: dict[int, tuple[int, bool]] = {}
    physically_visited = [False] * len(cover_stops)
    eps_closed_at: int | None = None
    trace: list[tuple[str, float, float, float]] = []
    home_tries: dict[int, int] = {}
    parked: set[int] = set()  # heard channels we gave up on (bounded loss)

    def measured_points() -> dict[int, list[tuple[float, float]]]:
        return {
            channel: [(item["x"], item["y"]) for item in robot.log[channel]]
            for channel in CHANNELS
        }

    def live_unresolved() -> list[dict]:
        return [state for state in _unresolved(states) if state["channel"] not in parked]

    def home(state: dict) -> None:
        channel = state["channel"]
        home_tries[channel] = home_tries.get(channel, 0) + 1
        fallback_home_q4(robot, state, fast=fast)
        if (
            home_tries[channel] >= HOME_MAX_TRIES
            and robot.world.source_by_channel(channel) is not None
        ):
            parked.add(channel)

    remaining0 = [(source.x, source.y) for source in world.sources]
    _, oracle_walk = two_opt_open((0.0, 0.0), remaining0)
    oracle_time = oracle_walk / SPEED + 5.0 * len(remaining0)

    def unheard_channels() -> list[int]:
        return [
            channel
            for channel in CHANNELS
            if not _heard(robot, channel) and not orient.channel_empty(channel)
        ]

    def n_heard() -> int:
        return sum(1 for channel in CHANNELS if _heard(robot, channel))

    def close_census() -> None:
        if n_heard() >= MAX_HEARD:
            orient.forget_unheard([c for c in CHANNELS if not _heard(robot, c)])
            for index in range(len(visited)):
                visited[index] = True

    def needed_covers() -> list[tuple[int, tuple[float, float]]]:
        close_census()
        channels = unheard_channels()
        if not channels:
            for index in range(len(visited)):
                visited[index] = True
            return []
        if eps_mass > 0.0 and orient.mean_mass(channels) <= eps_mass:
            nonlocal eps_closed_at
            if not counters["eps_closed"]:
                counters["eps_closed"] = 1
                eps_closed_at = _cover_count(physically_visited)
            for index in range(len(visited)):
                visited[index] = True
            return []
        needed: list[tuple[int, tuple[float, float]]] = []
        for index, stop in enumerate(cover_stops):
            if visited[index]:
                continue
            cached = cover_cache.get(index)
            if cached is not None and cached[0] == orient.version:
                still = cached[1]
            else:
                still = any(
                    orient.silence_kills_cached(channel, stop_caches[index])
                    for channel in channels
                )
                cover_cache[index] = (orient.version, still)
            if still:
                needed.append((index, stop))
            else:
                visited[index] = True
        return needed

    def snap_cover() -> None:
        here = (robot.x, robot.y)
        for index, stop in enumerate(cover_stops):
            if math.dist(here, stop) <= COVER_SNAP:
                visited[index] = True
                physically_visited[index] = True

    def cache_here() -> StopCache | None:
        here = (robot.x, robot.y)
        best = None
        for index, stop in enumerate(cover_stops):
            d = math.dist(here, stop)
            if d <= COVER_SNAP and (best is None or d < best[0]):
                best = (d, stop_caches[index])
        return None if best is None else best[1]

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
            if orient.channel_empty(channel):
                return False
            cache = cache_here()
            if cache is not None:
                if cover_listen == "kill":
                    return orient.silence_kills_cached(channel, cache)
                return orient.hearable_cached(channel, cache)
            if not orient.silence_kills(channel, robot.x, robot.y):
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

    def sync_cleared() -> None:
        for channel in CHANNELS:
            log = robot.log[channel]
            if log and log[-1].get("status") == "cleared":
                orient.clear_channel(channel)

    def work_here(expect: dict | None = None) -> None:
        nonlocal states, cover_at_first_clear
        snap_cover()
        here = (robot.x, robot.y)
        for channel in CHANNELS:
            if not should_measure(channel):
                continue
            was_heard = _heard(robot, channel)
            result = robot.measure(channel)
            status = result["status"]
            if status == "near":
                robot.clear(channel)
                orient.clear_channel(channel)
            elif status == "no_signal":
                orient.exclude_silent(channel, robot.x, robot.y)
                if expect and channel in expect.get("channels", ()):
                    counters[expect["kind"]] += 1
            else:
                orient.restrict_heard(channel, robot.x, robot.y, result["svd_deg"])
            if not was_heard:
                unused_listens.append(here)
        states = classify_all(robot, fast, states)
        _clear_nears(robot, states)
        for state in _clearable(states):
            center = state.get("mec_center")
            radius = state.get("mec_radius")
            if center is None or radius is None:
                continue
            if math.dist((robot.x, robot.y), tuple(center)) + radius <= CLEAR_R + 1e-9:
                if robot.world.source_by_channel(state["channel"]) is not None:
                    robot.clear(state["channel"])
        states = classify_all(robot, fast, states)
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
            if fine_attempts.get(channel, 0) >= 3:
                continue
            fine_attempts[channel] = fine_attempts.get(channel, 0) + 1
            fine_homing_q4(robot, channel, fast=fast)
            states = classify_all(robot, fast, states)
        sync_cleared()
        close_census()
        needed_covers()
        if cover_at_first_clear is None and robot.n_clear_ok > 0:
            cover_at_first_clear = _cover_count(visited)

    def mission_complete() -> bool:
        close_census()
        blocking = {
            "near",
            "clearable",
            "wedge",
            "unbounded",
            "too_large",
            "empty_intersection",
        }
        if any(
            state["kind"] in blocking and state["channel"] not in parked
            for state in states.values()
        ):
            return False
        return not needed_covers()

    probed_keys: set[tuple] = set()

    def tasks_here() -> list[dict]:
        tasks: list[dict] = []
        here = (robot.x, robot.y)
        for state in _clearable(states):
            vertices = state.get("vertices")
            if not vertices:
                continue
            disk = guaranteed_disk(vertices)
            if disk is None:
                continue
            tasks.append({"kind": "clear", "channel": state["channel"], "disk": disk})
        for state in states.values():
            if state.get("kind") != "too_large":
                continue
            center = state.get("mec_center")
            radius = state.get("mec_radius")
            if center is None or radius is None or radius > PROBE_MAX_R:
                continue
            channel = state["channel"]
            if channel in parked or probe_tries.get(channel, 0) >= PROBE_MAX_TRIES:
                continue
            point, audible = probe_point_q4(state, orient, here)
            key = (channel, round(point[0], 0), round(point[1], 0))
            if key in probed_keys:
                continue
            tasks.append(
                {
                    "kind": "probe",
                    "channel": channel,
                    "disk": (point, 0.0),
                    "key": key,
                    "audible": audible,
                }
            )
        unresolved = live_unresolved()
        if unresolved and merge_stations:
            # A census stop we must visit anyway that is a useful, mostly
            # audible second station for this channel: let it serve instead
            # of inserting a dedicated 1200 m station off the tour.
            pending = [stop for _, stop in needed_covers()]

            def served(state: dict) -> bool:
                return any(
                    station_useful(stop, state)
                    and orient.audible_fraction(state["channel"], stop[0], stop[1])
                    >= MERGE_AUDIBLE
                    for stop in pending
                )

            unresolved = [state for state in unresolved if not served(state)]
        if unresolved:
            station, channels, audible = propose_station_q4(
                here, unresolved, orient, measured_points()
            )
            tasks.append(
                {
                    "kind": "station",
                    "channels": channels,
                    "disk": (station, 0.0),
                    "audible": audible,
                }
            )
        for _, stop in needed_covers():
            tasks.append({"kind": "cover", "disk": (stop, 0.0)})
        return tasks

    def force_progress() -> bool:
        unresolved = live_unresolved()
        pending = [stop for _, stop in needed_covers()]
        if pending:
            robot.move_to(*pending[0])
            robot.n_replan += 1
            work_here()
            return True
        if unresolved:
            home(unresolved[0])
            robot.n_replan += 1
            work_here()
            return True
        target = orient.centroid(unheard_channels())
        if target is None:
            return False
        dest = clip_q4(*target)
        if math.dist((robot.x, robot.y), dest) < HERE:
            return False
        robot.move_to(*dest)
        robot.n_replan += 1
        work_here()
        return True

    work_here()
    steps = 0
    stuck = 0
    last_xy = (robot.x, robot.y)
    last_station_key: tuple | None = None
    stop_reason = "max_steps"

    while steps < max_steps:
        if mission_complete():
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
        expect: dict | None = None
        if task["kind"] == "probe":
            probed_keys.add(task["key"])
            probe_tries[task["channel"]] = probe_tries.get(task["channel"], 0) + 1
            expect = {"kind": "probe_silent", "channels": (task["channel"],)}
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
            expect = {"kind": "station_silent", "channels": tuple(task["channels"])}
        robot.move_to(*dest)
        robot.n_replan += 1
        trace.append((task["kind"], robot.x, robot.y, robot.time))
        work_here(expect)
        here = (robot.x, robot.y)
        stuck = stuck + 1 if math.dist(here, last_xy) < HERE else 0
        last_xy = here
        if stuck >= 3:
            if not force_progress():
                stop_reason = "stuck"
                break
            stuck = 0
            last_station_key = None

    if stop_reason == "max_steps" and mission_complete():
        stop_reason = "certificate"
    if stop_reason == "certificate" and counters["eps_closed"]:
        stop_reason = "epsilon"

    n_dir = sum(1 for s in world.sources if getattr(s, "phi", None) is not None)
    dir_cleared = sum(
        1 for s in world.sources if getattr(s, "phi", None) is not None and s.cleared
    )
    return {
        "policy": "belief_q4",
        "fidelity": "fast" if fast else "high",
        "cleared": sum(1 for source in robot.world.sources if source.cleared),
        "n_sources": len(robot.world.sources),
        "n_dir": n_dir,
        "dir_cleared": dir_cleared,
        "time_s": robot.time,
        "walk_m": robot.walk_m,
        "measures": robot.n_measure,
        "replans": robot.n_replan,
        "clear_fail": robot.n_clear_fail,
        "steps": steps,
        "census_time_s": 0.0,
        "oracle_time_s": oracle_time,
        "regret_s": robot.time - oracle_time,
        "census_kinds": {},
        "remaining": robot.world.remaining(),
        "cover_visits": _cover_count(physically_visited),
        "cover_total": len(cover_stops),
        "eps_closed_at_cover": eps_closed_at,
        "trace": trace,
        "cover_visits_at_first_clear": (
            cover_at_first_clear
            if cover_at_first_clear is not None
            else _cover_count(visited)
        ),
        "stop_reason": stop_reason,
        "final_mean_mass": orient.mean_mass(
            [c for c in CHANNELS if not _heard(robot, c)]
        ),
        "station_silent": counters["station_silent"],
        "probe_silent": counters["probe_silent"],
        "heard": n_heard(),
    }
