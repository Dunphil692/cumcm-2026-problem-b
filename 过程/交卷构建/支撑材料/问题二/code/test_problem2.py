"""问题 2 融合修正版回归测试：逐方向裁剪、广义接收、后验裁剪修正、三场景交叉验证。"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problem2_model import (
    NEAR_R, R_MAX, TARGET_R, ERROR_DEG,
    ray_exit_distance, direction_far_distance, is_in_omega1,
    reception_margin, worst_diameter, posterior_diameter, strong_diameter,
    three_circle_feasible, three_circle_anchors, offset_from_bearing, _dist,
)
from problem1_localize import localize


class BoundaryClipping(unittest.TestCase):
    def test_tangent_exit_distances_per_direction(self):
        s1 = (1790.0, 0.0)
        got = [ray_exit_distance(s1, a) for a in (89.0, 90.0, 91.0)]
        want = [160.791246148427, 189.472953214964, 223.270861193902]
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=6)

    def test_outward_800_midline_exit(self):
        self.assertAlmostEqual(ray_exit_distance((-1000.0, 0.0), 180.0), 800.0, places=8)

    def test_far_source_never_escapes_target_circle(self):
        # S1 贴圆边时，逐方向远端不得越过 1800 m 圆
        s1 = (1790.0, 0.0)
        for phi in (-1.0, 0.0, 1.0):
            r = direction_far_distance(s1, 90.0 + phi)
            g = (s1[0] + r * math.cos(math.radians(90.0 + phi)),
                 s1[1] + r * math.sin(math.radians(90.0 + phi)))
            self.assertLessEqual(math.hypot(*g), TARGET_R + 1e-8)

    def test_omega1_membership(self):
        self.assertTrue(is_in_omega1((0.0, 0.0), 0.0, (1500.0, 0.0)))
        self.assertFalse(is_in_omega1((0.0, 0.0), 0.0, (1500.0, 40.0)))   # 越出楔形
        self.assertFalse(is_in_omega1((0.0, 0.0), 0.0, (1501.0, 0.0)))    # 超出接收上界
        self.assertFalse(is_in_omega1((0.0, 0.0), 0.0, (3.0, 0.0)))       # 近场


class ReceptionCriterion(unittest.TestCase):
    def test_recommended_region_margin(self):
        # 队友独立实现的最优直径点：广义口径下保证接收（数值）
        q = (841.3601249898122, -544.4349902715983)
        slack, _ = reception_margin((0.0, 0.0), 0.0, q, phi_n=33, r_step=10.0)
        self.assertLessEqual(slack, 0.5)
        self.assertGreaterEqual(slack, -15.0)

    def test_point_beyond_1000_is_rejected_when_needed(self):
        # (900,-500)：|q|=1029.6>1000，近源 5 m 处 d2≈1025>1000 → 不可保证
        slack, _ = reception_margin((0.0, 0.0), 0.0, (900.0, -500.0), phi_n=33, r_step=10.0)
        self.assertGreater(slack, 0.0)

    def test_three_circle_is_conservative_subset(self):
        # (600,800) 在三圆叶子内；队友最优点在叶子外但仍在 C_recv 内
        self.assertTrue(three_circle_feasible((0.0, 0.0), 0.0, (600.0, 800.0)))
        self.assertFalse(three_circle_feasible((0.0, 0.0), 0.0, (841.3601249898122, -544.4349902715983)))
        slack, _ = reception_margin((0.0, 0.0), 0.0, (841.3601249898122, -544.4349902715983),
                                    phi_n=33, r_step=10.0)
        self.assertLessEqual(slack, 0.0)

    def test_three_circle_anchors_on_1000m_arc(self):
        pneg, ppos = three_circle_anchors((0.0, 0.0), 0.0)
        self.assertAlmostEqual(_dist((0.0, 0.0), pneg), 1000.0, places=6)
        self.assertAlmostEqual(_dist((0.0, 0.0), ppos), 1000.0, places=6)
        self.assertAlmostEqual(offset_from_bearing((0.0, 0.0), 0.0, ppos), 1.0, places=4)


class PosteriorClipping(unittest.TestCase):
    def test_clip_fix_reduces_diameter(self):
        # 旧版 135 m 的最坏情形：4 个顶点中 3 个在 d1>1500 m（不可能位置）
        s1 = (0.0, 0.0)
        q = (800.0, 600.0)
        theta2 = 320.648 + 1.0  # 与旧稿 --verify 最坏情形一致（θ2=321.648）
        unclipped = localize([s1, q], [0.0, theta2])["diameter"]
        clipped = posterior_diameter(s1, 0.0, q, theta2)
        self.assertAlmostEqual(unclipped, 134.976, delta=0.01)
        self.assertLess(clipped, 60.0)  # 裁剪后只留下真源附近的小角

    def test_j_at_800_600_is_about_112(self):
        j, arg = worst_diameter((0.0, 0.0), 0.0, (800.0, 600.0),
                                phi_n=17, r_step=10.0, err_n=17)
        self.assertAlmostEqual(j, 112.13, delta=0.5)
        self.assertIsNotNone(arg)

    def test_strong_branch_finite(self):
        # S2 贴第一楔形近端：对落在 5 m 内的源，近场分支给出有限小直径（旧稿记 +∞）
        s2 = (6.0 * math.cos(math.radians(2.0)), 6.0 * math.sin(math.radians(2.0)))
        self.assertLessEqual(strong_diameter((0.0, 0.0), 0.0, s2), 10.0 + 1e-6)
        j, arg = worst_diameter((0.0, 0.0), 0.0, s2, phi_n=17, r_step=10.0, err_n=17)
        # 该退化点整体 J 有限（其他远源仍给出大但有限的近平行后验），不再虚设 +∞
        self.assertTrue(math.isfinite(j))
        self.assertLess(j, 3000.0)


class ScenarioCrosscheck(unittest.TestCase):
    """在队友独立实现报告的最优直径点上重算 J，应与其数值一致（两套实现互证）。"""

    def test_center_reference(self):
        q = (841.3601249898122, -544.4349902715983)
        j, _ = worst_diameter((0.0, 0.0), 0.0, q, phi_n=17, r_step=10.0, err_n=17)
        self.assertAlmostEqual(j, 111.2918, delta=0.5)

    def test_inside_outward_800(self):
        q = (-1594.8117693963113, -511.3500635770528)
        j, _ = worst_diameter((-1000.0, 0.0), 180.0, q, phi_n=17, r_step=10.0, err_n=17)
        self.assertAlmostEqual(j, 40.9714, delta=0.5)

    def test_tangent_boundary(self):
        q = (1688.3980060336946, 120.52869612498694)
        j, _ = worst_diameter((1790.0, 0.0), 90.0, q, phi_n=17, r_step=10.0, err_n=17)
        self.assertAlmostEqual(j, 8.1658, delta=0.5)


if __name__ == "__main__":
    unittest.main()
