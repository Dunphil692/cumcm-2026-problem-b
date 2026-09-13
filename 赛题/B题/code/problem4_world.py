"""Local mixed omni/directional world for problem-4 policy experiments.

Kept separate from ``problem3_simulation_model`` so the frozen problem-3
code is untouched. A source with ``phi is None`` is omnidirectional; a
directional source radiates only into the closed half-plane facing ``phi``
(180 deg coverage, boundary included), exactly as 附件 2 §2.2 states.
Never talks to the official simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from problem3_simulation_model import (
    ARENA,
    CHANNELS,
    LISTEN_MAX,
    LISTEN_MIN,
    NEAR_R,
    Source,
    World,
)

# N19 census network (see B题-进度 §3 问题 4): origin + hexagon(1000) +
# 12-ring outside the arena. Worst-case residual (position x orientation)
# mass 0.06 %, open tour from the origin about 16.5 km.
INNER_RHO = 1000.0
OUTER_RHO = 1825.0
OUTER_N = 12
OUTER_ROT_DEG = 15.0


@dataclass
class DirSource(Source):
    phi: float | None = None  # radians; None = omnidirectional

    def covers(self, x: float, y: float) -> bool:
        """Station (x, y) lies inside this source's coverage angle."""
        if self.phi is None:
            return True
        dot = (x - self.x) * math.cos(self.phi) + (y - self.y) * math.sin(self.phi)
        return dot >= -1e-9


class DirWorld(World):
    """World whose sources may be directional."""

    def measure(self, x: float, y: float, channel: int) -> dict:
        source = self.source_by_channel(channel)
        if source is None:
            return {"status": "no_signal"}
        dist = math.hypot(source.x - x, source.y - y)
        if dist > source.r + 1e-9:
            return {"status": "no_signal"}
        if isinstance(source, DirSource) and not source.covers(x, y):
            return {"status": "no_signal"}
        if dist <= NEAR_R + 1e-9:
            return {"status": "near"}
        bearing = math.degrees(math.atan2(source.y - y, source.x - x)) % 360.0
        svd = (bearing + self._error(channel, x, y)) % 360.0
        return {"status": "direction", "svd_deg": svd}

    def clone(self, keep_cleared: bool = False) -> "DirWorld":
        sources = [
            DirSource(
                s.channel,
                s.x,
                s.y,
                s.r,
                s.cleared if keep_cleared else False,
                getattr(s, "phi", None),
            )
            for s in self.sources
        ]
        copy = DirWorld(sources, random.Random(self.error_seed), self.error_seed)
        copy._errors = dict(self._errors)
        copy.error_rng.setstate(self.error_rng.getstate())
        return copy

    def n_directional(self) -> int:
        return sum(1 for s in self.sources if getattr(s, "phi", None) is not None)


def census_stops_n19(
    inner_rho: float = INNER_RHO,
    outer_rho: float = OUTER_RHO,
    outer_n: int = OUTER_N,
    outer_rot_deg: float = OUTER_ROT_DEG,
) -> list[tuple[float, float]]:
    stops = [(0.0, 0.0)]
    for index in range(6):
        angle = index * math.pi / 3.0
        stops.append((inner_rho * math.cos(angle), inner_rho * math.sin(angle)))
    for index in range(outer_n):
        angle = math.radians(outer_rot_deg + 360.0 * index / outer_n)
        stops.append((outer_rho * math.cos(angle), outer_rho * math.sin(angle)))
    return stops


def generate_mixed_instance(
    rng: random.Random,
    p_dir: float = 0.5,
    n_sources: int | None = None,
) -> DirWorld:
    """Uniform positions in the disk, uniform orientations, R ~ U[1000, 1500].

    Each source is directional with probability ``p_dir`` (the official
    mix is unknown; sweep 0.3 / 0.5 / 0.7 locally).
    """
    if n_sources is None:
        n_sources = rng.randint(10, 16)
    channels = rng.sample(list(CHANNELS), n_sources)
    sources: list[DirSource] = []
    for channel in channels:
        radius = ARENA * math.sqrt(rng.random())
        theta = 2.0 * math.pi * rng.random()
        listen = rng.uniform(LISTEN_MIN, LISTEN_MAX)
        directional = rng.random() < p_dir
        phi = 2.0 * math.pi * rng.random() if directional else None
        sources.append(
            DirSource(
                channel,
                radius * math.cos(theta),
                radius * math.sin(theta),
                listen,
                False,
                phi,
            )
        )
    seed = rng.randint(0, 2**31 - 1)
    return DirWorld(sources, random.Random(seed), seed)
