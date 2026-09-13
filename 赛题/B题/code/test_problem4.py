#!/usr/bin/env python3
"""Problem-4 local checks (no official simulator)."""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

from problem4_orient_grid import (
    BIN_DEG,
    FULL,
    NBITS,
    OrientGrid,
    bin_of,
    front_masks,
    kill_masks,
    range_mask,
)
from problem4_world import DirSource, DirWorld, census_stops_n19, generate_mixed_instance


class RangeMaskTests(unittest.TestCase):
    def test_full_and_empty(self):
        lo = np.array([0, 10], dtype=np.int64)
        self.assertEqual(int(range_mask(lo, np.array([0, 0]))[0]), 0)
        self.assertEqual(int(range_mask(lo, np.array([64, 64]))[0]), int(FULL))

    def test_wraps_around_zero(self):
        mask = int(range_mask(np.array([62]), np.array([4]))[0])
        bits = [i for i in range(64) if mask >> i & 1]
        self.assertEqual(bits, [0, 1, 62, 63])


class DirWorldTests(unittest.TestCase):
    def test_back_side_is_silent_front_is_heard(self):
        # Source at origin facing +x. Station at (-100, 0) is behind.
        src = DirSource(1, 0.0, 0.0, 1200.0, False, 0.0)
        world = DirWorld([src], random.Random(0), 0)
        self.assertEqual(world.measure(-100.0, 0.0, 1)["status"], "no_signal")
        self.assertEqual(world.measure(100.0, 0.0, 1)["status"], "direction")

    def test_clear_ignores_orientation(self):
        src = DirSource(1, 0.0, 0.0, 1200.0, False, 0.0)
        world = DirWorld([src], random.Random(0), 0)
        self.assertTrue(world.clear(-10.0, 0.0, 1))

    def test_boundary_of_half_plane_is_included(self):
        src = DirSource(1, 0.0, 0.0, 1200.0, False, 0.0)
        world = DirWorld([src], random.Random(0), 0)
        self.assertEqual(world.measure(0.0, 100.0, 1)["status"], "direction")


class OrientGridTests(unittest.TestCase):
    def test_silence_does_not_erase_the_disk(self):
        grid = OrientGrid()
        grid.exclude_silent(1, 0.0, 0.0)
        self.assertFalse(grid.channel_empty(1))
        self.assertGreater(grid.channel_mass(1), 0.2)
        self.assertLess(grid.channel_mass(1), 0.9)

    def test_true_cell_survives_back_side_silence(self):
        # Source at (800, 0) facing +x. Origin is behind it.
        grid = OrientGrid()
        grid.exclude_silent(1, 0.0, 0.0)
        self.assertTrue(grid.true_cell_alive(1, 800.0, 0.0, 0.0))

    def test_true_cell_dies_on_front_side_silence(self):
        grid = OrientGrid()
        grid.exclude_silent(1, 1500.0, 0.0)
        self.assertFalse(grid.true_cell_alive(1, 800.0, 0.0, 0.0))

    def test_hearing_keeps_wedge_and_front_half(self):
        # Heard at the origin with bearing 0: the source is east of us and
        # facing us, so phi ≈ 180 deg, not 0 (which would point away).
        grid = OrientGrid()
        grid.restrict_heard(1, 0.0, 0.0, 0.0)
        self.assertTrue(grid.true_cell_alive(1, 800.0, 0.0, math.pi))
        self.assertFalse(grid.true_cell_alive(1, 800.0, 0.0, 0.0))

    def test_n19_has_nineteen_stops(self):
        stops = census_stops_n19()
        self.assertEqual(len(stops), 19)
        self.assertEqual(stops[0], (0.0, 0.0))
        self.assertGreater(math.hypot(*stops[-1]), 1800.0)


class PolicySmokeTests(unittest.TestCase):
    def test_one_mixed_world_finishes(self):
        from problem4_belief import run_belief_q4

        rng = random.Random(2027)
        world = generate_mixed_instance(rng, p_dir=0.5, n_sources=10)
        stats = run_belief_q4(world)
        self.assertIn(stats["stop_reason"], {"certificate", "epsilon", "max_steps", "stuck"})
        self.assertGreaterEqual(stats["cleared"], 0)
        self.assertLessEqual(stats["cleared"], stats["n_sources"])
        self.assertNotEqual(stats["stop_reason"], "stuck")


if __name__ == "__main__":
    unittest.main()
