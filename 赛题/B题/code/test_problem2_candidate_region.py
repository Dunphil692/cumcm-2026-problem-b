"""问题 2 硬伤回归：逐方向裁剪、5 m near、楔形禁区、推荐点口径。"""

from __future__ import annotations

import math
import unittest

from problem2_candidate_region import (
    NEAR_R,
    TARGET_R,
    is_feasible,
    probe_sources,
    ray_exit_distance,
    verify_worst_case,
    worst_diameter,
)


class ProbeSourcesStayPhysical(unittest.TestCase):
    def test_origin_far_end_is_1500(self):
        points = probe_sources((0.0, 0.0), 0.0)
        radii = [math.hypot(*p) for p in points]
        self.assertAlmostEqual(max(radii), 1500.0, places=6)
        self.assertTrue(all(r <= TARGET_R + 1e-8 for r in radii))

    def test_edge_s1_does_not_place_g_outside_target_circle(self):
        s1 = (1790.0, 0.0)
        theta1 = 90.0
        centerline = min(1500.0, ray_exit_distance(s1, theta1))
        self.assertGreater(centerline, 180.0)
        points = probe_sources(s1, theta1)
        self.assertTrue(points)
        for g in points:
            self.assertLessEqual(math.hypot(*g), TARGET_R + 1e-6, msg=g)
            bearing = math.degrees(math.atan2(g[1] - s1[1], g[0] - s1[0]))
            offset = abs((bearing - theta1 + 180.0) % 360.0 - 180.0)
            self.assertLessEqual(offset, 1.0 + 1e-8)


class FeasibilityAndNear(unittest.TestCase):
    def test_point_on_bearing_is_infeasible(self):
        self.assertFalse(is_feasible((0.0, 0.0), 0.0, (500.0, 0.0)))

    def test_recommended_point_is_feasible(self):
        self.assertTrue(is_feasible((0.0, 0.0), 0.0, (800.0, 600.0)))

    def test_s2_within_five_metres_of_a_probe_source_is_infinite(self):
        s1 = (0.0, 0.0)
        s2 = (
            6.0 * math.cos(math.radians(2.0)),
            6.0 * math.sin(math.radians(2.0)),
        )
        self.assertTrue(is_feasible(s1, 0.0, s2))
        self.assertTrue(any(math.hypot(s2[0] - g[0], s2[1] - g[1]) <= NEAR_R for g in probe_sources(s1, 0.0)))
        self.assertEqual(worst_diameter(s1, 0.0, s2), math.inf)

    def test_target_bound_can_be_turned_off(self):
        s1 = (1700.0, 0.0)
        s2 = (1700.0, 800.0)
        self.assertGreater(math.hypot(*s2), TARGET_R)
        self.assertFalse(is_feasible(s1, 0.0, s2, bind_target=True))
        self.assertTrue(is_feasible(s1, 0.0, s2, bind_target=False))
        self.assertTrue(is_feasible((0.0, 0.0), 0.0, (800.0, 600.0), bind_target=True))
        self.assertTrue(is_feasible((0.0, 0.0), 0.0, (800.0, 600.0), bind_target=False))
        self.assertFalse(is_feasible((0.0, 0.0), 0.0, (100.0, 0.0), bind_target=False))


class RecommendedPoint(unittest.TestCase):
    def test_j_at_800_600_is_135(self):
        j = worst_diameter((0.0, 0.0), 0.0, (800.0, 600.0))
        self.assertTrue(math.isfinite(j))
        self.assertAlmostEqual(j, 135.0, delta=1.0)
        self.assertAlmostEqual(j, worst_diameter((0.0, 0.0), 0.0, (800.0, -600.0)), places=6)

    def test_verify_worst_case_agrees_at_recommended_point(self):
        report = verify_worst_case((0.0, 0.0), 0.0, (800.0, 600.0), d_step=50.0, phi_step=0.5)
        self.assertAlmostEqual(report["three_point_J_m"], 135.0, delta=1.0)
        self.assertAlmostEqual(report["dense_J_m"], report["three_point_J_m"], delta=1.0)
        self.assertGreaterEqual(report["worst_distance_m"], 1400.0)
        self.assertGreaterEqual(abs(report["worst_offset_deg"]), 0.5)


if __name__ == "__main__":
    unittest.main()
