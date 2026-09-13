"""Position x orientation set-membership grid for problem 4.

Problem 3 kept, per channel, the set of positions where a still-unheard
source could be (10 m grid). A directional source breaks the problem-3
update "no_signal at S => no source within 990 m of S": the source may sit
inside that disk with S on its silent back side. So problem 4 tracks, per
channel and per grid node g, the set of *orientations* phi still consistent
with every observation, as a 64-bit mask (64 bins of 5.625 deg). An
omnidirectional source is the special case "every phi works".

Update rules (all conservative w.r.t. the 10 m grid: the true source can be
up to 7.07 m from its nearest node, which perturbs the g -> S direction by
at most delta = asin(7.07 / |g - S|)):

* no_signal at S: for nodes with |g - S| <= r_cert (990 m for an unheard
  channel; for a heard channel also |g - S_h| - 15 m for every hearing
  station S_h, because R >= |g* - S_h|), kill the orientation bins that lie
  completely inside the half-circle facing S, shrunk by delta.
* direction at S with bearing theta: keep only nodes inside the +-1 deg
  wedge (widened by delta) within 1500 m, and only the bins that touch the
  half-circle facing S (widened by delta).
* near / cleared: the channel is done, mask := 0.

The certificate "channel k has no undetected source" is mask[k] == 0.
Copied into the v14 field pack so official Windows runs stay self-contained.
"""

from __future__ import annotations

import math

import numpy as np

from problem4_simulation_model import ARENA, CHANNELS, LISTEN_MAX

GRID_STEP = 10.0
GRID_SLACK = GRID_STEP * math.sqrt(0.5)  # 7.07 m: node -> true point
LISTEN_CERT = 990.0                       # 990 + 7.07 < 1000 <= R
HEARD_RANGE_SLACK = 15.0                  # |g-S_h| - 15 is a certified in-range radius
NBITS = 64
BIN_DEG = 360.0 / NBITS
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
ZERO = np.uint64(0)
ONE = np.uint64(1)
U64_NBITS = np.uint64(NBITS)


def _arena_grid(step: float = GRID_STEP, radius: float = ARENA) -> np.ndarray:
    xs = np.arange(-radius, radius + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    inside = xx * xx + yy * yy <= radius * radius
    return np.column_stack((xx[inside], yy[inside]))


GRID_XY = _arena_grid()
GX = np.ascontiguousarray(GRID_XY[:, 0])
GY = np.ascontiguousarray(GRID_XY[:, 1])
N_NODES = GRID_XY.shape[0]
CELLS_PER_CHANNEL = float(N_NODES * NBITS)


def range_mask(lo: np.ndarray, count: np.ndarray) -> np.ndarray:
    """uint64 masks with bits lo, lo+1, ..., lo+count-1 (mod 64) set."""
    count = np.clip(np.asarray(count, dtype=np.int64), 0, NBITS)
    lo = np.mod(np.asarray(lo, dtype=np.int64), NBITS).astype(np.uint64)
    cnt_safe = np.minimum(count, NBITS - 1).astype(np.uint64)
    base = (ONE << cnt_safe) - ONE
    base = np.where(count >= NBITS, FULL, base).astype(np.uint64)
    base = np.where(count <= 0, ZERO, base).astype(np.uint64)
    lo_safe = np.maximum(lo, ONE)
    rotated = (base << lo) | (base >> (U64_NBITS - lo_safe))
    return np.where(lo == 0, base, rotated).astype(np.uint64)


def bin_of(phi_deg: float) -> int:
    return int(math.floor((phi_deg % 360.0) / BIN_DEG)) % NBITS


def popcount(values: np.ndarray) -> int:
    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(values).sum())
    return int(np.unpackbits(np.ascontiguousarray(values).view(np.uint8)).sum())


def _delta_deg(dist: np.ndarray) -> np.ndarray:
    ratio = np.minimum(1.0, GRID_SLACK / np.maximum(dist, 1e-9))
    return np.degrees(np.arcsin(ratio))


def kill_masks(x: float, y: float, idx: np.ndarray) -> np.ndarray:
    """Bins fully inside the half-circle facing station (x, y), shrunk by delta.

    These are the orientations a source at node idx cannot have if the
    station heard nothing (and the station is certified in range).
    """
    dx = x - GX[idx]
    dy = y - GY[idx]
    dist = np.hypot(dx, dy)
    a = np.degrees(np.arctan2(dy, dx))
    w = 90.0 - _delta_deg(dist)
    lo = np.ceil((a - w) / BIN_DEG)
    hi = np.floor((a + w) / BIN_DEG) - 1.0
    count = (hi - lo + 1.0).astype(np.int64)
    return range_mask(lo.astype(np.int64), count)


def front_masks(x: float, y: float, idx: np.ndarray) -> np.ndarray:
    """Bins touching the half-circle facing station (x, y), widened by delta.

    These are the orientations under which a source at node idx could be
    heard from the station (superset, grid-conservative).
    """
    dx = x - GX[idx]
    dy = y - GY[idx]
    dist = np.hypot(dx, dy)
    a = np.degrees(np.arctan2(dy, dx))
    w = 90.0 + _delta_deg(dist)
    lo = np.floor((a - w) / BIN_DEG)
    hi = np.floor((a + w) / BIN_DEG)
    count = (hi - lo + 1.0).astype(np.int64)
    return range_mask(lo.astype(np.int64), count)


def _angular_distance(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


class StopCache:
    """Static geometry of a fixed listening stop (cover station)."""

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)
        d2 = (GX - x) ** 2 + (GY - y) ** 2
        self.idx_cert = np.flatnonzero(d2 <= LISTEN_CERT * LISTEN_CERT)
        self.kill_cert = kill_masks(x, y, self.idx_cert)
        self.idx_hear = np.flatnonzero(d2 <= (LISTEN_MAX + GRID_SLACK) ** 2)
        self.front_hear = front_masks(x, y, self.idx_hear)


class OrientGrid:
    """Per-channel (position x orientation) feasible set as uint64 bit rows."""

    def __init__(self) -> None:
        self.mask = np.full((len(CHANNELS), N_NODES), FULL, dtype=np.uint64)
        self.version = 0
        self.heard_stations: dict[int, list[tuple[float, float]]] = {
            channel: [] for channel in CHANNELS
        }
        self._mass_cache: dict[int, tuple[int, float]] = {}

    # ---- queries -----------------------------------------------------
    def row(self, channel: int) -> np.ndarray:
        return self.mask[channel - 1]

    def channel_empty(self, channel: int) -> bool:
        return not bool(self.row(channel).any())

    def channel_mass(self, channel: int) -> float:
        cached = self._mass_cache.get(channel)
        if cached is not None and cached[0] == self.version:
            return cached[1]
        value = popcount(self.row(channel)) / CELLS_PER_CHANNEL
        self._mass_cache[channel] = (self.version, value)
        return value

    def mean_mass(self, channels: list[int]) -> float:
        if not channels:
            return 0.0
        return sum(self.channel_mass(channel) for channel in channels) / len(channels)

    def omni_possible(self, channel: int, silent_stations: list[tuple[float, float]]) -> bool:
        """Some surviving node has no certified silent station within 990 m."""
        row = self.row(channel)
        alive = row != 0
        if not alive.any():
            return False
        if not silent_stations:
            return True
        covered = np.zeros(N_NODES, dtype=bool)
        for sx, sy in silent_stations:
            covered |= (GX - sx) ** 2 + (GY - sy) ** 2 <= LISTEN_CERT * LISTEN_CERT
        return bool((alive & ~covered).any())

    def centroid(self, channels: list[int]) -> tuple[float, float] | None:
        alive = np.zeros(N_NODES, dtype=bool)
        for channel in channels:
            alive |= self.row(channel) != 0
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            return None
        return float(GX[idx].mean()), float(GY[idx].mean())

    def _in_range_radius(self, channel: int) -> np.ndarray | float:
        """Certified in-range radius per node for a silence update."""
        stations = self.heard_stations[channel]
        if not stations:
            return LISTEN_CERT
        rad = np.full(N_NODES, LISTEN_CERT)
        for sx, sy in stations:
            rad = np.maximum(rad, np.hypot(GX - sx, GY - sy) - HEARD_RANGE_SLACK)
        return rad

    def silence_kills(self, channel: int, x: float, y: float) -> bool:
        """Would a no_signal at (x, y) remove any surviving cell of the channel?"""
        row = self.row(channel)
        if not row.any():
            return False
        d2 = (GX - x) ** 2 + (GY - y) ** 2
        rad = self._in_range_radius(channel)
        sel = (row != 0) & (d2 <= rad * rad)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            return False
        return bool(np.any(row[idx] & kill_masks(x, y, idx)))

    def silence_kills_cached(self, channel: int, cache: StopCache) -> bool:
        row = self.row(channel)
        sub = row[cache.idx_cert]
        if not sub.any():
            return False
        return bool(np.any(sub & cache.kill_cert))

    def hearable(self, channel: int, x: float, y: float, radius: float = LISTEN_MAX) -> bool:
        """Could a still-feasible source of this channel be heard from (x, y)?"""
        row = self.row(channel)
        if not row.any():
            return False
        d2 = (GX - x) ** 2 + (GY - y) ** 2
        sel = (row != 0) & (d2 <= (radius + GRID_SLACK) ** 2)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            return False
        return bool(np.any(row[idx] & front_masks(x, y, idx)))

    def hearable_cached(self, channel: int, cache: StopCache) -> bool:
        row = self.row(channel)
        sub = row[cache.idx_hear]
        if not sub.any():
            return False
        return bool(np.any(sub & cache.front_hear))

    def audible_fraction(self, channel: int, x: float, y: float) -> float:
        """Share of surviving (g, phi) cells certified audible from (x, y).

        Cells outside the certified in-range radius count as not audible.
        Returns 1.0 when the channel row is empty (no information).
        """
        row = self.row(channel)
        idx_alive = np.flatnonzero(row != 0)
        if idx_alive.size == 0:
            return 1.0
        total = popcount(row[idx_alive])
        d2 = (GX[idx_alive] - x) ** 2 + (GY[idx_alive] - y) ** 2
        rad = self._in_range_radius(channel)
        rad_sel = rad[idx_alive] if isinstance(rad, np.ndarray) else rad
        inrange = d2 <= rad_sel * rad_sel
        idx = idx_alive[inrange]
        if idx.size == 0:
            return 0.0
        # audible bins: fully inside the facing half-circle (kill-style,
        # i.e. certified), not the widened front superset
        audible = popcount(row[idx] & kill_masks(x, y, idx))
        return audible / max(total, 1)

    # ---- updates -----------------------------------------------------
    def clear_channel(self, channel: int) -> None:
        if self.row(channel).any():
            self.mask[channel - 1] = ZERO
            self.version += 1

    def exclude_silent(self, channel: int, x: float, y: float) -> int:
        """no_signal at (x, y). Returns the number of nodes touched."""
        row = self.row(channel)
        if not row.any():
            return 0
        d2 = (GX - x) ** 2 + (GY - y) ** 2
        rad = self._in_range_radius(channel)
        sel = (row != 0) & (d2 <= rad * rad)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            return 0
        row[idx] &= ~kill_masks(x, y, idx)
        self.version += 1
        return int(idx.size)

    def restrict_heard(self, channel: int, x: float, y: float, svd_deg: float) -> None:
        """direction at (x, y) with bearing svd_deg."""
        row = self.row(channel)
        dx = GX - x
        dy = GY - y
        dist = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx)) % 360.0  # station -> node
        delta = _delta_deg(dist)
        inside = (
            (_angular_distance(ang, svd_deg) <= 1.0 + delta + 1e-9)
            & (dist <= LISTEN_MAX + GRID_SLACK)
        )
        row[~inside] = ZERO
        idx = np.flatnonzero(inside & (row != 0))
        if idx.size:
            row[idx] &= front_masks(x, y, idx)
        self.heard_stations[channel].append((float(x), float(y)))
        self.version += 1

    def forget_unheard(self, channels: list[int]) -> None:
        """Census closed (16 heard): the remaining channels are empty."""
        for channel in channels:
            self.clear_channel(channel)

    # ---- diagnostics ---------------------------------------------------
    def true_cell_alive(self, channel: int, x: float, y: float, phi: float | None) -> bool:
        """For tests: is the cell of the true source still feasible?"""
        node = int(np.argmin((GX - x) ** 2 + (GY - y) ** 2))
        value = int(self.row(channel)[node])
        if phi is None:
            return value == int(FULL)
        return bool(value >> bin_of(math.degrees(phi)) & 1)
