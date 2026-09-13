"""Unit tests for along-cover opportunistic waypoints (not N19 census)."""

from __future__ import annotations

import math
import unittest

from problem4_belief_n19 import along_cover_segment


class AlongCoverSegmentTests(unittest.TestCase):
    def test_zero_waypoints(self) -> None:
        self.assertEqual(along_cover_segment((0.0, 0.0), (1000.0, 0.0), 0), [])

    def test_one_midpoint(self) -> None:
        points = along_cover_segment((0.0, 0.0), (1000.0, 0.0), 1)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0][0], 500.0)
        self.assertAlmostEqual(points[0][1], 0.0)

    def test_two_thirds(self) -> None:
        points = along_cover_segment((0.0, 0.0), (1200.0, 0.0), 2)
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 400.0)
        self.assertAlmostEqual(points[1][0], 800.0)

    def test_short_hop_skipped(self) -> None:
        self.assertEqual(along_cover_segment((0.0, 0.0), (100.0, 0.0), 2), [])

    def test_does_not_include_endpoints(self) -> None:
        here, dest = (10.0, 20.0), (1010.0, 20.0)
        points = along_cover_segment(here, dest, 3)
        self.assertEqual(len(points), 3)
        for point in points:
            self.assertGreater(math.dist(here, point), 1.0)
            self.assertGreater(math.dist(dest, point), 1.0)


if __name__ == "__main__":
    unittest.main()
