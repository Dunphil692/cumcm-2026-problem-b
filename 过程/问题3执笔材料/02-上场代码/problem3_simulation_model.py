"""Local omnidirectional world for problem-3 policy experiments.

Does not talk to the official simulator. One measurement error is stored per
(channel, rounded station) pair, matching the statement that repeating a
measurement at the same point does not resample the ±1° error.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from problem1_localize import localize

ARENA = 1800.0
LISTEN_MIN = 1000.0
LISTEN_MAX = 1500.0
LISTEN_CERT = 990.0
SPEED = 5.0
CLEAR_R = 20.0
NEAR_R = 5.0
ERROR_DEG = 1.0
CENSUS_RHO = 1150.0
T_MEASURE = 5.0
T_SWITCH = 1.0
T_CLEAR_OK = 5.0
T_CLEAR_FAIL = 3.0
CHANNELS = tuple(range(1, 21))

NEED_MORE = frozenset({"wedge", "unbounded", "too_large", "empty_intersection"})


def analyze_localization(payload: list[dict]) -> dict:
    """Q3 几何入口：调用问题 1 修正版 localize，映射成 classify 用的字段。"""
    points = [(float(item["x"]), float(item["y"])) for item in payload]
    angles = [float(item["angle_deg"]) for item in payload]
    result = localize(points, angles)
    status = result["status"]
    if status == "empty":
        return {"status": "empty", "vertices": [], "diameter": None}
    if status == "unbounded":
        return {
            "status": "unbounded",
            "vertices": [],
            "diameter": "Infinity",
        }
    polygon = result.get("polygon") or []
    vertices = [(float(point[0]), float(point[1])) for point in polygon]
    return {
        "status": "bounded",
        "vertices": vertices,
        "diameter": result["diameter"],
        "mec_center": result.get("mec_center"),
        "mec_radius": result.get("mec_radius"),
    }


@dataclass
class Source:
    channel: int
    x: float
    y: float
    r: float
    cleared: bool = False


class World:
    def __init__(
        self,
        sources: list[Source],
        error_rng: random.Random,
        error_seed: int = 0,
    ) -> None:
        self.sources = sources
        self.error_rng = error_rng
        self.error_seed = error_seed
        self._errors: dict[tuple[int, tuple[float, float]], float] = {}

    @classmethod
    def from_sources(
        cls,
        triples: list[tuple[int, float, float, float]],
        error_seed: int = 0,
    ) -> "World":
        sources = [
            Source(int(channel), float(x), float(y), float(radius))
            for channel, x, y, radius in triples
        ]
        return cls(sources, random.Random(error_seed), error_seed)

    def clone(self, keep_cleared: bool = False) -> "World":
        triples = [
            (source.channel, source.x, source.y, source.r)
            for source in self.sources
        ]
        copy = World.from_sources(triples, error_seed=self.error_seed)
        copy._errors = dict(self._errors)
        copy.error_rng.setstate(self.error_rng.getstate())
        if keep_cleared:
            for src, dst in zip(self.sources, copy.sources):
                dst.cleared = src.cleared
        return copy

    def source_by_channel(self, channel: int) -> Source | None:
        for source in self.sources:
            if source.channel == channel and not source.cleared:
                return source
        return None

    def _error(self, channel: int, x: float, y: float) -> float:
        key = (channel, (round(x, 2), round(y, 2)))
        if key not in self._errors:
            self._errors[key] = self.error_rng.uniform(-ERROR_DEG, ERROR_DEG)
        return self._errors[key]

    def measure(self, x: float, y: float, channel: int) -> dict:
        source = self.source_by_channel(channel)
        if source is None:
            return {"status": "no_signal"}
        dist = math.hypot(source.x - x, source.y - y)
        if dist > source.r + 1e-9:
            return {"status": "no_signal"}
        if dist <= NEAR_R + 1e-9:
            return {"status": "near"}
        bearing = math.degrees(math.atan2(source.y - y, source.x - x)) % 360.0
        svd = (bearing + self._error(channel, x, y)) % 360.0
        return {"status": "direction", "svd_deg": svd}

    def clear(self, x: float, y: float, channel: int) -> bool:
        source = self.source_by_channel(channel)
        if source is None:
            return False
        if math.hypot(source.x - x, source.y - y) > CLEAR_R + 1e-9:
            return False
        source.cleared = True
        return True

    def remaining(self) -> int:
        return sum(0 if source.cleared else 1 for source in self.sources)


def census_stops(rho: float = CENSUS_RHO) -> list[tuple[float, float]]:
    stops = [(0.0, 0.0)]
    for index in range(6):
        angle = index * math.pi / 3.0
        stops.append((rho * math.cos(angle), rho * math.sin(angle)))
    return stops


def generate_instance(
    rng: random.Random,
    n_sources: int | None = None,
) -> World:
    if n_sources is None:
        n_sources = rng.randint(10, 16)
    channels = rng.sample(list(CHANNELS), n_sources)
    triples: list[tuple[int, float, float, float]] = []
    for channel in channels:
        radius = ARENA * math.sqrt(rng.random())
        theta = 2.0 * math.pi * rng.random()
        listen = rng.uniform(LISTEN_MIN, LISTEN_MAX)
        triples.append(
            (
                channel,
                radius * math.cos(theta),
                radius * math.sin(theta),
                listen,
            )
        )
    return World.from_sources(triples, error_seed=rng.randint(0, 2**31 - 1))


def classify_channel(
    measurements: list[dict],
    clear_r: float = CLEAR_R,
    fast: bool = False,
) -> dict:
    silences = [
        (float(item["x"]), float(item["y"]))
        for item in measurements
        if item.get("status") == "no_signal"
    ]

    def finish(label: dict) -> dict:
        label["silences"] = silences
        return label

    if any(item.get("status") == "cleared" for item in measurements):
        return finish({"kind": "cleared", "measurements": measurements})
    if any(item.get("status") == "near" for item in measurements):
        return finish({"kind": "near", "measurements": measurements})

    directions = [
        item for item in measurements if item.get("status") == "direction"
    ]
    if not directions:
        return finish({"kind": "empty", "measurements": measurements})
    if len(directions) == 1:
        return finish({"kind": "wedge", "measurements": directions})

    directions = _select_bearings(directions, limit=4)
    if fast:
        return finish(_classify_fast(directions, clear_r))

    payload = [
        {"x": item["x"], "y": item["y"], "angle_deg": item["svd_deg"]}
        for item in directions
    ]
    result = analyze_localization(payload)
    if result["status"] == "empty":
        return finish(
            {
                "kind": "empty_intersection",
                "measurements": directions,
            }
        )
    if result["status"] == "unbounded":
        return finish(
            {
                "kind": "unbounded",
                "measurements": directions,
                "vertices": list(result.get("vertices") or []),
            }
        )

    vertices = list(result.get("vertices") or [])
    center = result.get("mec_center")
    radius = result.get("mec_radius")
    if center is None or radius is None:
        from problem3_route_solver import smallest_enclosing_circle

        center, radius = smallest_enclosing_circle(vertices)
    kind = "clearable" if radius <= clear_r + 1e-9 else "too_large"
    return finish(
        {
            "kind": kind,
            "measurements": directions,
            "vertices": vertices,
            "mec_center": center,
            "mec_radius": radius,
            "diameter": result["diameter"],
        }
    )


def _classify_fast(directions: list[dict], clear_r: float) -> dict:
    first, second = directions[0], directions[-1]
    point = _ray_intersection(
        first["x"],
        first["y"],
        first["svd_deg"],
        second["x"],
        second["y"],
        second["svd_deg"],
    )
    if point is None:
        return {"kind": "unbounded", "measurements": directions}
    r1 = math.hypot(point[0] - first["x"], point[1] - first["y"])
    r2 = math.hypot(point[0] - second["x"], point[1] - second["y"])
    psi = abs(
        ((first["svd_deg"] - second["svd_deg"] + 180.0) % 360.0) - 180.0
    )
    psi = max(psi, 1.0)
    diameter = (
        2.0
        * math.tan(math.radians(ERROR_DEG))
        / math.sin(math.radians(psi))
        * math.sqrt(r1 * r1 + r2 * r2 + 2.0 * r1 * r2 * abs(math.cos(math.radians(psi))))
    )
    mec_radius = 0.5 * diameter
    kind = "clearable" if mec_radius <= clear_r + 1e-9 else "too_large"
    vertices = [
        (point[0] + mec_radius, point[1]),
        (point[0] - mec_radius, point[1]),
        (point[0], point[1] + mec_radius),
        (point[0], point[1] - mec_radius),
    ]
    return {
        "kind": kind,
        "measurements": directions,
        "vertices": vertices,
        "mec_center": point,
        "mec_radius": mec_radius,
        "diameter": diameter,
    }


def _select_bearings(directions: list[dict], limit: int = 4) -> list[dict]:
    if len(directions) <= limit:
        return directions
    selected = [directions[-1]]
    remaining = list(directions[:-1])
    while len(selected) < limit and remaining:
        def score(item: dict) -> float:
            return min(
                math.dist((item["x"], item["y"]), (other["x"], other["y"]))
                for other in selected
            )

        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best)
    return selected


def _ray_intersection(
    x1: float, y1: float, a1: float, x2: float, y2: float, a2: float
) -> tuple[float, float] | None:
    dx1, dy1 = math.cos(math.radians(a1)), math.sin(math.radians(a1))
    dx2, dy2 = math.cos(math.radians(a2)), math.sin(math.radians(a2))
    det = dx1 * (-dy2) - dy1 * (-dx2)
    if abs(det) < 1e-9:
        return None
    t = ((x2 - x1) * (-dy2) - (y2 - y1) * (-dx2)) / det
    s = ((x2 - x1) * (-dy1) - (y2 - y1) * (-dx1)) / det
    if t < -1e-6 or s < -1e-6:
        return None
    return (x1 + t * dx1, y1 + t * dy1)


@dataclass
class Robot:
    world: World
    x: float = 0.0
    y: float = 0.0
    channel: int = 1
    time: float = 0.0
    walk_m: float = 0.0
    n_measure: int = 0
    n_clear_ok: int = 0
    n_clear_fail: int = 0
    n_replan: int = 0
    log: dict[int, list[dict]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.log:
            self.log = {channel: [] for channel in CHANNELS}

    def move_to(self, x: float, y: float) -> None:
        dist = math.hypot(x - self.x, y - self.y)
        self.walk_m += dist
        self.time += dist / SPEED
        self.x, self.y = x, y

    def measure(self, channel: int) -> dict:
        if channel != self.channel:
            self.time += T_SWITCH
            self.channel = channel
        self.time += T_MEASURE
        self.n_measure += 1
        result = self.world.measure(self.x, self.y, channel)
        record = {"x": self.x, "y": self.y, **result}
        self.log[channel].append(record)
        return result

    def clear(self, channel: int) -> bool:
        ok = self.world.clear(self.x, self.y, channel)
        if ok:
            self.time += T_CLEAR_OK
            self.n_clear_ok += 1
            self.log[channel].append(
                {"x": self.x, "y": self.y, "status": "cleared"}
            )
        else:
            self.time += T_CLEAR_FAIL
            self.n_clear_fail += 1
        return ok
