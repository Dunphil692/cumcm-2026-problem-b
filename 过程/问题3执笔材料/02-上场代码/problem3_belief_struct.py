"""Q3 experiment policy. Control is problem3_belief.py (do not edit).

E15: keep E14b + probe. Do not add a dedicated second station for a
channel if a clear, probe, or leftover cover we already plan to
visit is already a useful second stand. E16 structural extras on
cover/clear/probe did not move the 1000-game mean; numbers stay frozen.

Modes: e15 (default), punch (planned stops punch leftover at 1000 m;
listen empty there only to drop a cover vertex), delay (no dedicated
second station while another task still exists).

Each channel keeps a set-membership feasible region. Next stop is chosen
from three neighbourhoods (clear / second station / coverage hole) by
open Held–Karp + TSPN. Never talks to the official simulator.

v2 changes vs v1 (see 优化说明.md):
  * Per-channel unheard grid: a no_signal punch only removes area from the
    channel that was actually measured. The v1 shared grid could erase the
    true location of another still-undetected source (main cause of the
    ~1.3% "one source left" failures).
  * Clear-disk skip fix: when the route entry point of a clear disk lies
    within 1 m of the robot, step to the disk centre instead of dropping
    the task (v1 left clearable sources uncleared).
  * Probe task: a too_large source with MEC radius <= PROBE_MAX_R is
    visited at its MEC centre (a close-range bearing there usually shrinks
    the polygon below the clear threshold, replacing a far second station).
  * Fine homing: for a nearly localised source, walk 6 m steps along the
    latest bearing measuring every step, so the 5 m "near" zone cannot be
    stepped over.
  * Fallback hardening: walk budget + unbounded-jump + fine homing so the
    fallback can neither wander for kilometres nor skip past the source.
  * Wedge second-station distance STATION_DIST = 1200 m at ±30°.
  * Census hexagon radius is CENSUS_RHO (1150 m); origin is still required.
  * fullopt (only the pieces that keep 100%): fake-cover ring
    via per-channel 1000 m listen disks; stop census after 16 hearings.
  * Official-safe stop: Ω empty certificate (cover holes gone + no
    unresolved heard channel). Never uses world.remaining() to stop.
  * Heard-channel no_signal is recorded but does not filter samples.
    Unheard channels off a cover stop use the 700 m spacing.
  * Second station: 1200 m at ±30° only (50°/90° offsets dropped).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from problem3_batch_station import (
    _alive,
    clip_arena,
    fixed_r_ok,
    possible_samples,
    propose_batch_station,
    station_useful,
)
from problem3_route_solver import (
    guaranteed_disk,
    joint_route,
    smallest_enclosing_circle,
    two_opt_open,
)
from problem3_simulation_model import (
    ARENA,
    CHANNELS,
    CLEAR_R,
    LISTEN_CERT,
    LISTEN_MAX,
    LISTEN_MIN,
    SPEED,
    Robot,
    census_stops,
)

BELIEF_MAX_STEPS = 160
GRID_STEP = 10.0
COVER_SNAP = 40.0
HERE = 1.0
PROBE_MAX_R = 200.0      # MEC radius below which we probe the source directly
UNHEARD_SPACING = 700.0  # kept for control-compat comments; E14 does not use it
MAX_HEARD = 16
PUNCH_HELP_MIN_CELLS = 50
PUNCH_HELP_MIN_FRAC = 0.70  # only listen if this stand can nearly finish exclusive leftover
PUNCH_HELP_MIN_R = 750.0


def _arena_grid(step: float = GRID_STEP, radius: float = ARENA) -> np.ndarray:
    xs = np.arange(-radius, radius + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    inside = xx * xx + yy * yy <= radius * radius
    return np.column_stack((xx[inside], yy[inside]))


GRID_XY = _arena_grid()


class UnheardGrid:
    """Per-channel remaining set for channels that have never returned a bearing.

    One boolean row per channel (CHANNELS = 1..20). A no_signal result at a
    stop removes the 990 m certified disk from *that channel's own* row only,
    so the feasible region of one channel can never swallow another
    undetected source.
    """

    def __init__(self) -> None:
        n_ch, n_pts = len(CHANNELS), GRID_XY.shape[0]
        self.mask = np.ones((n_ch, n_pts), dtype=bool)
        self.heard_cover = np.zeros((n_ch, n_pts), dtype=bool)

    def empty(self) -> bool:
        return not bool(self.mask.any())

    def channel_empty(self, channel: int) -> bool:
        return not bool(self.mask[channel - 1].any())

    def exclude(self, channel: int, x: float, y: float, radius: float = LISTEN_CERT) -> None:
        d2 = (GRID_XY[:, 0] - x) ** 2 + (GRID_XY[:, 1] - y) ** 2
        self.mask[channel - 1] &= d2 > radius * radius

    def record_listen(
        self,
        channel: int,
        x: float,
        y: float,
        radius: float = LISTEN_MIN,
    ) -> None:
        d2 = (GRID_XY[:, 0] - x) ** 2 + (GRID_XY[:, 1] - y) ** 2
        self.heard_cover[channel - 1] |= d2 <= radius * radius

    def forget(self, channel: int) -> None:
        """Channel just returned a bearing: it is no longer unheard."""
        self.mask[channel - 1] = False
        self.heard_cover[channel - 1] = False

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

    def cover_still_needed(
        self,
        stop: tuple[float, float],
        cover_r: float = LISTEN_MIN,
    ) -> bool:
        if self.empty():
            return False
        d2 = (GRID_XY[:, 0] - stop[0]) ** 2 + (GRID_XY[:, 1] - stop[1]) ** 2
        leftover = (
            self.mask
            & ~self.heard_cover
            & (d2[None, :] <= cover_r * cover_r)
        )
        return bool(leftover.any())

    def leftover_intersects(
        self,
        channel: int,
        x: float,
        y: float,
        radius: float,
    ) -> bool:
        if self.channel_empty(channel):
            return False
        d2 = (GRID_XY[:, 0] - x) ** 2 + (GRID_XY[:, 1] - y) ** 2
        return bool(
            np.any(
                self.mask[channel - 1]
                & ~self.heard_cover[channel - 1]
                & (d2 <= radius * radius)
            )
        )

    def cover_slack(
        self,
        stop: tuple[float, float],
        cover_r: float = LISTEN_MIN,
    ) -> float:
        d2 = (GRID_XY[:, 0] - stop[0]) ** 2 + (GRID_XY[:, 1] - stop[1]) ** 2
        leftover = (
            self.mask
            & ~self.heard_cover
            & (d2[None, :] <= cover_r * cover_r)
        )
        idx = leftover.any(axis=0)
        if not bool(idx.any()):
            return cover_r
        dist = np.sqrt(
            (GRID_XY[idx, 0] - stop[0]) ** 2 + (GRID_XY[idx, 1] - stop[1]) ** 2
        )
        return float(cover_r - dist.max())

    def centroid(self) -> tuple[float, float] | None:
        idx = np.flatnonzero(self.mask.any(axis=0))
        if idx.size == 0:
            return None
        return float(GRID_XY[idx, 0].mean()), float(GRID_XY[idx, 1].mean())

    def punch_help_channels(
        self,
        point: tuple[float, float],
        vertices: list[tuple[float, float]],
        cover_r: float = LISTEN_MIN,
        punch_r: float = LISTEN_MIN,
    ) -> set[int]:
        """Unheard channels that eat leftover only this vertex still covers.

        Overlap with a neighbour is left to that neighbour. Listening those
        cells does not drop a vertex and only costs 5 s.
        """
        if not vertices or self.empty():
            return set()
        d2p = (GRID_XY[:, 0] - point[0]) ** 2 + (GRID_XY[:, 1] - point[1]) ** 2
        in_p = d2p <= punch_r * punch_r
        leftover = self.mask & ~self.heard_cover
        leftover_cells = leftover.any(axis=0)
        disks = []
        for vx, vy in vertices:
            d2v = (GRID_XY[:, 0] - vx) ** 2 + (GRID_XY[:, 1] - vy) ** 2
            disks.append(d2v <= cover_r * cover_r)
        best_hit = 0
        best_channels: set[int] = set()
        for index, in_v in enumerate(disks):
            others = np.zeros(leftover_cells.shape, dtype=bool)
            for other, in_w in enumerate(disks):
                if other != index:
                    others |= in_w
            exclusive = leftover_cells & in_v & ~others
            n_ex = int(exclusive.sum())
            n_hit = int((exclusive & in_p).sum())
            if (
                n_ex == 0
                or n_hit < PUNCH_HELP_MIN_CELLS
                or n_hit / n_ex < PUNCH_HELP_MIN_FRAC
                or n_hit < best_hit
            ):
                continue
            chs: set[int] = set()
            hit = exclusive & in_p
            for channel in range(leftover.shape[0]):
                if bool(np.any(leftover[channel] & hit)):
                    chs.add(channel + 1)
            best_hit = n_hit
            best_channels = chs
        return best_channels

    def greedy_needed(
        self,
        pending: list[tuple[int, tuple[float, float]]],
        cover_r: float = LISTEN_MIN,
        helpers: list[tuple[float, float]] | tuple[()] = (),
    ) -> list[tuple[int, tuple[float, float]]]:
        """Smallest greedy subset of unvisited vertices that covers leftover.

        A vertex is not needed just because leftover still sits in its
        1000 m disk: overlap with a neighbour can be left to that neighbour.
        Keep every pending vertex only if leftover sits outside all of them.
        Optional helpers are already-planned stops treated as 1000 m punches
        for routing only; the certificate path must pass helpers=().
        """
        leftover = (self.mask & ~self.heard_cover).any(axis=0)
        if helpers:
            pre = np.zeros(leftover.shape, dtype=bool)
            for hx, hy in helpers:
                d2 = (GRID_XY[:, 0] - hx) ** 2 + (GRID_XY[:, 1] - hy) ** 2
                pre |= d2 <= cover_r * cover_r
            leftover = leftover & ~pre
        if not pending or self.empty() or not bool(leftover.any()):
            return []
        disks = []
        for _, stop in pending:
            d2 = (GRID_XY[:, 0] - stop[0]) ** 2 + (GRID_XY[:, 1] - stop[1]) ** 2
            disks.append(d2 <= cover_r * cover_r)
        coverable = np.zeros(leftover.shape, dtype=bool)
        for disk in disks:
            coverable |= disk
        if bool((leftover & ~coverable).any()):
            return list(pending)
        remaining = leftover & coverable
        chosen: list[tuple[int, tuple[float, float]]] = []
        used = [False] * len(pending)
        while bool(remaining.any()):
            best_i = -1
            best_n = 0
            for index, disk in enumerate(disks):
                if used[index]:
                    continue
                n_hit = int((remaining & disk).sum())
                if n_hit > best_n:
                    best_i = index
                    best_n = n_hit
            if best_n == 0:
                return list(pending)
            used[best_i] = True
            chosen.append(pending[best_i])
            remaining = remaining & ~disks[best_i]
        return chosen


def _dedup_points(
    points: list[tuple[float, float]],
    tol: float = COVER_SNAP,
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for point in points:
        if all(math.dist(point, prev) > tol for prev in kept):
            kept.append(point)
    return kept


def shadow_joint_cover(
    unheard: UnheardGrid,
    pending: list[tuple[int, tuple[float, float]]],
    helpers: list[tuple[float, float]],
    cover_r: float = LISTEN_MIN,
    max_helpers: int = 6,
) -> dict:
    """How many greedy cover vertices 1 / 2 / 3 existing stops could drop.

    Decision-free: leftover is only punched on paper. helpers are stands
    already visited or already in this replan (clear / probe / station).
    """
    leftover = (unheard.mask & ~unheard.heard_cover).any(axis=0)
    empty = {
        "n_needed": 0,
        "n_helpers": 0,
        "drop1": 0,
        "drop2": 0,
        "drop3": 0,
        "drop_all": 0,
        "joint23_extra": 0,
        "in_region": 0,
        "slack_mean": 0.0,
    }
    if not pending or not bool(leftover.any()):
        return empty
    helpers = _dedup_points(helpers)[:max_helpers]
    vdisks = []
    slacks: list[float] = []
    for _, stop in pending:
        d2 = (GRID_XY[:, 0] - stop[0]) ** 2 + (GRID_XY[:, 1] - stop[1]) ** 2
        disk = d2 <= cover_r * cover_r
        vdisks.append(disk)
        cells = leftover & disk
        if bool(cells.any()):
            dist = np.sqrt(
                (GRID_XY[cells, 0] - stop[0]) ** 2
                + (GRID_XY[cells, 1] - stop[1]) ** 2
            )
            slacks.append(float(cover_r - dist.max()))
    hdisks = []
    for hx, hy in helpers:
        d2 = (GRID_XY[:, 0] - hx) ** 2 + (GRID_XY[:, 1] - hy) ** 2
        hdisks.append(d2 <= cover_r * cover_r)

    def needed_count(pre: np.ndarray) -> int:
        remaining = leftover & ~pre
        if not bool(remaining.any()):
            return 0
        coverable = np.zeros(leftover.shape, dtype=bool)
        for disk in vdisks:
            coverable |= disk
        if bool((remaining & ~coverable).any()):
            return len(pending)
        remaining = remaining & coverable
        used = [False] * len(pending)
        chosen = 0
        while bool(remaining.any()):
            best_i = -1
            best_n = 0
            for index, disk in enumerate(vdisks):
                if used[index]:
                    continue
                n_hit = int((remaining & disk).sum())
                if n_hit > best_n:
                    best_i = index
                    best_n = n_hit
            if best_n == 0:
                return len(pending)
            used[best_i] = True
            chosen += 1
            remaining = remaining & ~vdisks[best_i]
        return chosen

    zeros = np.zeros(leftover.shape, dtype=bool)
    n_plain = needed_count(zeros)
    if n_plain == 0:
        empty["n_helpers"] = len(helpers)
        return empty

    def drop_with(idxs: tuple[int, ...] | range) -> int:
        if not idxs:
            return 0
        pre = np.zeros(leftover.shape, dtype=bool)
        for index in idxs:
            pre |= hdisks[index]
        return max(0, n_plain - needed_count(pre))

    n_h = len(hdisks)
    best1 = max((drop_with((i,)) for i in range(n_h)), default=0)
    best2 = best1
    if n_h >= 2:
        best2 = max(
            best1,
            max(
                drop_with(pair)
                for pair in itertools.combinations(range(n_h), 2)
            ),
        )
    best3 = best2
    if n_h >= 3:
        best3 = max(
            best2,
            max(
                drop_with(triple)
                for triple in itertools.combinations(range(n_h), 3)
            ),
        )
    drop_all = drop_with(range(n_h)) if n_h else 0
    in_region = 0
    for disk in vdisks:
        cells = leftover & disk
        if not bool(cells.any()):
            continue
        if any(not bool((cells & ~hd).any()) for hd in hdisks):
            in_region += 1
    return {
        "n_needed": n_plain,
        "n_helpers": n_h,
        "drop1": best1,
        "drop2": best2,
        "drop3": best3,
        "drop_all": drop_all,
        "joint23_extra": max(0, best3 - best1),
        "in_region": in_region,
        "slack_mean": float(sum(slacks) / len(slacks)) if slacks else 0.0,
    }


def shadow_layer1_heard(states: dict) -> dict:
    """If heard silences punched 990 m disks, would tasks move?

    Decision-free. Does not apply L(g) cross-station linkage.
    """
    n_heard_with_sil = 0
    vertex_killed = 0
    become_clear = 0
    enter_probe = 0
    samples_all_dead = 0
    mec_shrink: list[float] = []
    for state in states.values():
        if state.get("kind") in {"cleared", "near", "empty", "empty_intersection"}:
            continue
        if not (state.get("silences") or []):
            continue
        if not any(
            item.get("status") == "direction"
            for item in state.get("measurements") or []
        ):
            continue
        n_heard_with_sil += 1
        verts = [tuple(point) for point in (state.get("vertices") or [])]
        if verts:
            alive = [point for point in verts if _alive(point, state)]
            killed = len(verts) - len(alive)
            vertex_killed += killed
            if killed and alive:
                _, radius_new = smallest_enclosing_circle(alive)
                radius_old = float(state.get("mec_radius") or 0.0)
                mec_shrink.append(max(0.0, radius_old - radius_new))
                if state.get("kind") != "clearable" and radius_new <= CLEAR_R + 1e-9:
                    become_clear += 1
                if (
                    state.get("kind") == "too_large"
                    and radius_old > PROBE_MAX_R
                    and radius_new <= PROBE_MAX_R
                ):
                    enter_probe += 1
            elif killed and not alive:
                samples_all_dead += 1
            continue
        samples = possible_samples(state)
        alive_s = [point for point in samples if _alive(point, state)]
        if samples and not alive_s:
            samples_all_dead += 1
    return {
        "n_heard_with_sil": n_heard_with_sil,
        "vertex_killed": vertex_killed,
        "become_clear": become_clear,
        "enter_probe": enter_probe,
        "samples_all_dead": samples_all_dead,
        "mec_shrink_max": max(mec_shrink) if mec_shrink else 0.0,
        "mec_shrink_mean": float(sum(mec_shrink) / len(mec_shrink)) if mec_shrink else 0.0,
    }


def shadow_layer2_rlink(states: dict) -> dict:
    """If candidates need one common R, would clear / probe / S2 move?"""
    n_heard_with_sil = 0
    vertex_killed = 0
    become_clear = 0
    enter_probe = 0
    leave_unresolved = 0
    samples_all_dead = 0
    mec_shrink: list[float] = []
    for state in states.values():
        kind = state.get("kind")
        if kind in {"cleared", "near", "empty", "empty_intersection"}:
            continue
        if not any(
            item.get("status") == "direction"
            for item in state.get("measurements") or []
        ):
            continue
        if not (state.get("silences") or []):
            continue
        n_heard_with_sil += 1
        unresolved_now = kind in {
            "wedge",
            "unbounded",
            "too_large",
            "empty_intersection",
        }
        verts = [tuple(point) for point in (state.get("vertices") or [])]
        if verts:
            alive = [point for point in verts if fixed_r_ok(point, state)]
            killed = len(verts) - len(alive)
            vertex_killed += killed
            if killed and alive:
                _, radius_new = smallest_enclosing_circle(alive)
                radius_old = float(state.get("mec_radius") or 0.0)
                mec_shrink.append(max(0.0, radius_old - radius_new))
                now_clear = radius_new <= CLEAR_R + 1e-9
                if kind != "clearable" and now_clear:
                    become_clear += 1
                    if unresolved_now:
                        leave_unresolved += 1
                if (
                    kind == "too_large"
                    and radius_old > PROBE_MAX_R
                    and radius_new <= PROBE_MAX_R
                ):
                    enter_probe += 1
            elif killed and not alive:
                samples_all_dead += 1
            continue
        samples = possible_samples(state)
        alive_s = [point for point in samples if fixed_r_ok(point, state)]
        if samples and not alive_s:
            samples_all_dead += 1
    return {
        "n_heard_with_sil": n_heard_with_sil,
        "vertex_killed": vertex_killed,
        "become_clear": become_clear,
        "enter_probe": enter_probe,
        "leave_unresolved": leave_unresolved,
        "samples_all_dead": samples_all_dead,
        "mec_shrink_max": max(mec_shrink) if mec_shrink else 0.0,
        "mec_shrink_mean": float(sum(mec_shrink) / len(mec_shrink)) if mec_shrink else 0.0,
    }


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


def run_belief_struct(
    world,
    fidelity: str = "high",
    max_steps: int = BELIEF_MAX_STEPS,
    mode: str = "e15",
    shadow: list | None = None,
    silence_shadow: list | None = None,
    rlink_shadow: list | None = None,
    robot=None,
) -> dict:
    from problem3_fusion_policies import (
        FINE_R,
        _clear_nears,
        _clearable,
        _fallback_home,
        _fine_homing,
        _unresolved,
        classify_all,
    )

    fast = fidelity == "fast"
    if robot is None:
        robot = Robot(world)
    cover_stops = census_stops()
    visited = [False] * len(cover_stops)
    snapped = [False] * len(cover_stops)
    unheard = UnheardGrid()
    unused_listens: list[tuple[float, float]] = []
    states = classify_all(robot, fast)
    cover_at_first_clear: int | None = None
    fine_attempts: dict[int, int] = {}
    punch_helpers: list[tuple[float, float]] = []
    stood_at: list[tuple[float, float]] = []
    n_heard_nosignal = 0

    remaining0 = [(source.x, source.y) for source in world.sources]
    _, oracle_walk = two_opt_open((0.0, 0.0), remaining0)
    oracle_time = oracle_walk / SPEED + 5.0 * len(remaining0)

    def n_heard() -> int:
        return sum(1 for channel in CHANNELS if _heard(robot, channel))

    def close_census() -> None:
        if n_heard() >= MAX_HEARD:
            for channel in CHANNELS:
                if not _heard(robot, channel):
                    unheard.forget(channel)
            for index in range(len(visited)):
                visited[index] = True

    def needed_covers(
        use_helpers: bool = False,
    ) -> list[tuple[int, tuple[float, float]]]:
        close_census()
        pending = [
            (index, stop)
            for index, stop in enumerate(cover_stops)
            if not snapped[index]
        ]
        helpers = punch_helpers if (mode == "punch" and use_helpers) else ()
        return unheard.greedy_needed(pending, helpers=helpers)

    def _punch_buys_skip(here: tuple[float, float]) -> bool:
        pending = [
            (index, stop)
            for index, stop in enumerate(cover_stops)
            if not snapped[index]
        ]
        if not pending:
            return False
        plain = unheard.greedy_needed(pending)
        helped = unheard.greedy_needed(pending, helpers=[here])
        return len(helped) < len(plain)

    def snap_cover() -> None:
        here = (robot.x, robot.y)
        for index, stop in enumerate(cover_stops):
            if math.dist(here, stop) <= COVER_SNAP:
                visited[index] = True
                snapped[index] = True

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
            at_cover = any(
                math.dist(here, stop) <= COVER_SNAP for stop in cover_stops
            )
            if at_cover:
                return unheard.intersects(channel, robot.x, robot.y, LISTEN_MIN)
            if mode == "punch" and _punch_buys_skip(here):
                return unheard.leftover_intersects(
                    channel, robot.x, robot.y, LISTEN_MIN
                )
            if not unheard.intersects(channel, robot.x, robot.y, LISTEN_MIN):
                return False
            if any(math.dist(here, prev) < HERE for prev in unused_listens):
                return False
            if not unused_listens:
                return True
            return all(
                math.dist(here, prev) >= UNHEARD_SPACING for prev in unused_listens
            )
        if state is None:
            return True
        samples = possible_samples(state)
        if not samples:
            return True
        if not any(math.dist(here, sample) <= LISTEN_MAX for sample in samples):
            return False
        return station_useful(here, state)

    def work_here() -> None:
        nonlocal states, cover_at_first_clear, n_heard_nosignal
        snap_cover()
        stood_at.append((robot.x, robot.y))
        for channel in CHANNELS:
            if not should_measure(channel):
                continue
            was_heard = _heard(robot, channel)
            result = robot.measure(channel)
            if result["status"] == "near":
                robot.clear(channel)
            if was_heard and result["status"] == "no_signal":
                n_heard_nosignal += 1
            if not was_heard:
                unused_listens.append((robot.x, robot.y))
                if result["status"] == "no_signal":
                    # punch only this channel's own feasible set
                    unheard.exclude(channel, robot.x, robot.y, LISTEN_CERT)
                    unheard.record_listen(channel, robot.x, robot.y)
                else:
                    unheard.forget(channel)
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
        # v2: if we stand at the MEC centre of a nearly-localised source,
        # finish it right here with the fine homing walk (the 18 m homing
        # steps with sparse measurements can skip over the 5 m near zone).
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
            _fine_homing(robot, channel, fast=fast)
            states = classify_all(robot, fast, states)
        close_census()
        needed_covers()
        if cover_at_first_clear is None and robot.n_clear_ok > 0:
            cover_at_first_clear = _cover_count(visited)

    def mission_complete() -> bool:
        """Stop without the local oracle remaining(). Official-safe.

        Done when every heard channel is cleared / not chasing, and the
        7-point cover (or heard-16 census close) has no leftover hole.
        Fake 990–1000 m rings are not holes: cover_still_needed skips them.
        """
        close_census()
        blocking = {
            "near",
            "clearable",
            "wedge",
            "unbounded",
            "too_large",
            "empty_intersection",
        }
        if any(state["kind"] in blocking for state in states.values()):
            return False
        return not needed_covers()

    probed_key: tuple | None = None

    def tasks_here() -> list[dict]:
        nonlocal probed_key
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
        # Probe nearly-localised sources at their MEC centre: a close-range
        # bearing there usually shrinks the polygon below the clear
        # threshold, avoiding an extra far second station.
        probe_channels: set[int] = set()
        here = (robot.x, robot.y)
        for state in states.values():
            if state.get("kind") != "too_large":
                continue
            center = state.get("mec_center")
            radius = state.get("mec_radius")
            if center is None or radius is None or radius > PROBE_MAX_R:
                continue
            if math.dist(here, tuple(center)) > 900.0:
                continue
            key = (state["channel"], round(center[0], 1), round(center[1], 1))
            if key == probed_key:
                continue
            probe_channels.add(state["channel"])
            tasks.append(
                {"kind": "probe", "channel": state["channel"], "disk": (tuple(center), 0.0)}
            )
        punch_helpers.clear()
        if mode == "punch":
            for task in tasks:
                center = task["disk"][0]
                if _punch_buys_skip(center):
                    punch_helpers.append(center)
        needed = needed_covers(use_helpers=True)
        planned = [stop for _, stop in needed]
        for task in tasks:
            planned.append(task["disk"][0])
        unresolved = []
        for item in _unresolved(states):
            if item["channel"] in probe_channels:
                continue
            if any(station_useful(point, item) for point in planned):
                continue
            unresolved.append(item)
        if unresolved:
            has_other = bool(tasks) or bool(needed)
            if not (mode == "delay" and has_other):
                station, channels = propose_batch_station(
                    here, unresolved, dists=(1000.0,)
                )
                tasks.append(
                    {
                        "kind": "station",
                        "channels": channels,
                        "disk": (station, 0.0),
                    }
                )
                if mode == "punch" and _punch_buys_skip(station):
                    punch_helpers.append(station)
                    needed = needed_covers(use_helpers=True)
        for _, stop in needed:
            tasks.append({"kind": "cover", "disk": (stop, 35.0)})
        if shadow is not None:
            pending = [
                (index, stop)
                for index, stop in enumerate(cover_stops)
                if not snapped[index]
            ]
            helpers = [(robot.x, robot.y)]
            helpers.extend(stood_at)
            helpers.extend(
                task["disk"][0] for task in tasks if task["kind"] != "cover"
            )
            shadow.append(
                shadow_joint_cover(
                    unheard, unheard.greedy_needed(pending), helpers
                )
            )
        if silence_shadow is not None:
            row = shadow_layer1_heard(states)
            row["n_heard_nosignal"] = n_heard_nosignal
            row["n_station_tasks"] = sum(1 for task in tasks if task["kind"] == "station")
            row["n_probe_tasks"] = sum(1 for task in tasks if task["kind"] == "probe")
            row["n_clear_tasks"] = sum(1 for task in tasks if task["kind"] == "clear")
            silence_shadow.append(row)
        if rlink_shadow is not None:
            row = shadow_layer2_rlink(states)
            row["n_station_tasks"] = sum(1 for task in tasks if task["kind"] == "station")
            rlink_shadow.append(row)
        return tasks

    def force_progress() -> bool:
        unresolved = _unresolved(states)
        pending = [stop for _, stop in needed_covers()]
        if pending:
            dest = pending[0]
            robot.move_to(*dest)
            robot.n_replan += 1
            work_here()
            return True
        if unresolved:
            _fallback_home(robot, unresolved[0], fast=fast)
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

    work_here()
    steps = 0
    stuck = 0
    last_xy = (robot.x, robot.y)
    last_station_key: tuple | None = None
    stop_reason = "max_steps"
    kind_walk = {"station": 0.0, "cover": 0.0, "clear": 0.0, "probe": 0.0, "other": 0.0}
    kind_n = {"station": 0, "cover": 0, "clear": 0, "probe": 0, "other": 0}

    while steps < max_steps:
        if mission_complete():
            stop_reason = "certificate"
            break
        steps += 1
        unresolved = _unresolved(states)
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
            # All route destinations are within HERE of the robot. For a
            # clear disk the exact stand point matters: the robot can sit
            # just outside the guaranteed disk (skipping would drop the
            # task and leave a clearable source uncleared), so step to the
            # disk centre and clear from there.
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
        if task["kind"] == "probe":
            center = task["disk"][0]
            probed_key = (task["channel"], round(center[0], 1), round(center[1], 1))
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
        kind = task.get("kind") or "other"
        if kind not in kind_walk:
            kind = "other"
        kind_walk[kind] += math.dist((robot.x, robot.y), dest)
        kind_n[kind] += 1
        robot.move_to(*dest)
        robot.n_replan += 1
        work_here()
        here = (robot.x, robot.y)
        stuck = stuck + 1 if math.dist(here, last_xy) < HERE else 0
        last_xy = here
        if stuck >= 3:
            if not force_progress():
                stop_reason = "stuck"
                break
            stuck = 0
            last_station_key = None

    if steps >= max_steps and stop_reason == "max_steps":
        stop_reason = "max_steps"
    elif mission_complete():
        stop_reason = "certificate"

    return {
        "policy": {
            "e15": "belief_struct",
            "punch": "belief_punch",
            "delay": "belief_delay",
        }.get(mode, f"belief_struct_{mode}"),
        "fidelity": "fast" if fast else "high",
        "cleared": sum(1 for source in robot.world.sources if source.cleared),
        "n_sources": len(robot.world.sources),
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
        "stop_reason": stop_reason,
        "cover_visits": _cover_count(visited),
        "cover_snaps": _cover_count(snapped),
        "cover_visits_at_first_clear": (
            cover_at_first_clear
            if cover_at_first_clear is not None
            else _cover_count(visited)
        ),
        "walk_station": kind_walk["station"],
        "walk_cover": kind_walk["cover"],
        "walk_clear": kind_walk["clear"],
        "walk_probe": kind_walk["probe"],
        "n_station_moves": kind_n["station"],
        "n_cover_moves": kind_n["cover"],
        "n_clear_moves": kind_n["clear"],
        "n_probe_moves": kind_n["probe"],
    }
