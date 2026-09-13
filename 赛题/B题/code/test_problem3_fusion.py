#!/usr/bin/env python3
"""Unit tests for problem-3 local fusion routing (no official simulator)."""

from __future__ import annotations

import math
import random
import unittest

from problem3_batch_station import STATION_DIST, propose_batch_station, station_useful, possible_samples
from problem3_belief import GRID_XY, LISTEN_CERT, UnheardGrid
from problem3_simulation_model import CHANNELS
from problem3_route_solver import (
    guaranteed_disk,
    held_karp_open,
    joint_route,
    smallest_enclosing_circle,
)
from problem3_simulation_model import (
    ARENA,
    CENSUS_RHO,
    CLEAR_R,
    NEAR_R,
    World,
    classify_channel,
    census_stops,
    generate_instance,
)


class SimulationModelTests(unittest.TestCase):
    def test_measure_returns_direction_when_inside_listen_radius(self):
        world = World.from_sources(
            [(1, 400.0, 0.0, 1000.0)],
            error_seed=1,
        )
        result = world.measure(0.0, 0.0, 1)
        self.assertEqual(result["status"], "direction")
        wrapped = abs((result["svd_deg"] % 360.0 + 180.0) % 360.0 - 180.0)
        self.assertLessEqual(wrapped, 1.0)

    def test_measure_returns_no_signal_when_farther_than_r(self):
        world = World.from_sources(
            [(1, 1500.0, 0.0, 1000.0)],
            error_seed=1,
        )
        result = world.measure(0.0, 0.0, 1)
        self.assertEqual(result["status"], "no_signal")

    def test_measure_returns_near_within_five_metres(self):
        world = World.from_sources(
            [(1, 3.0, 0.0, 1000.0)],
            error_seed=1,
        )
        result = world.measure(0.0, 0.0, 1)
        self.assertEqual(result["status"], "near")

    def test_clear_succeeds_only_inside_twenty_metres(self):
        world = World.from_sources(
            [(1, 50.0, 0.0, 1000.0)],
            error_seed=1,
        )
        self.assertFalse(world.clear(0.0, 0.0, 1))
        self.assertTrue(world.clear(40.0, 0.0, 1))
        self.assertIsNone(world.source_by_channel(1))

    def test_unused_channel_is_no_signal(self):
        world = World.from_sources(
            [(1, 100.0, 0.0, 1000.0)],
            error_seed=1,
        )
        self.assertEqual(world.measure(0.0, 0.0, 2)["status"], "no_signal")


class ClassifyTests(unittest.TestCase):
    def test_all_no_signal_is_empty(self):
        label = classify_channel(
            [
                {"x": 0.0, "y": 0.0, "status": "no_signal"},
                {"x": 1200.0, "y": 0.0, "status": "no_signal"},
            ]
        )
        self.assertEqual(label["kind"], "empty")

    def test_single_direction_is_wedge(self):
        label = classify_channel(
            [
                {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 20.0},
                {"x": 1200.0, "y": 0.0, "status": "no_signal"},
            ]
        )
        self.assertEqual(label["kind"], "wedge")

    def test_near_is_immediate(self):
        label = classify_channel(
            [{"x": 0.0, "y": 0.0, "status": "near"}]
        )
        self.assertEqual(label["kind"], "near")

    def test_two_crossing_bearings_can_be_clearable_or_too_large(self):
        label = classify_channel(
            [
                {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 0.0},
                {"x": 0.0, "y": 400.0, "status": "direction", "svd_deg": 270.0},
            ]
        )
        self.assertIn(label["kind"], {"clearable", "too_large", "unbounded"})
        if label["kind"] == "clearable":
            self.assertLessEqual(label["mec_radius"], CLEAR_R)

    def test_classify_uses_problem1_localize_mec(self):
        label = classify_channel(
            [
                {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 0.0},
                {"x": 0.0, "y": 200.0, "status": "direction", "svd_deg": 270.0},
                {"x": 200.0, "y": 0.0, "status": "direction", "svd_deg": 180.0},
            ]
        )
        self.assertEqual(label["kind"], "clearable")
        self.assertIsNotNone(label.get("mec_center"))
        self.assertLessEqual(label["mec_radius"], CLEAR_R)
        self.assertGreaterEqual(len(label.get("vertices") or []), 3)


class RouteSolverTests(unittest.TestCase):
    def test_held_karp_matches_brute_force_on_five_cities(self):
        rng = random.Random(2026)
        start = (0.0, 0.0)
        cities = [
            (rng.uniform(-200, 200), rng.uniform(-200, 200)) for _ in range(5)
        ]
        order, length = held_karp_open(start, cities)

        def path_len(perm):
            prev = start
            total = 0.0
            for i in perm:
                total += math.dist(prev, cities[i])
                prev = cities[i]
            return total

        best = min(
            path_len(perm)
            for perm in _permutations(range(5))
        )
        self.assertAlmostEqual(length, best, places=6)
        self.assertAlmostEqual(path_len(order), length, places=6)

    def test_smallest_enclosing_circle_covers_triangle(self):
        pts = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
        center, radius = smallest_enclosing_circle(pts)
        for p in pts:
            self.assertLessEqual(math.dist(center, p), radius + 1e-8)

    def test_guaranteed_disk_keeps_true_source_inside_clearance(self):
        vertices = [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)]
        disk = guaranteed_disk(vertices, clear_r=20.0)
        self.assertIsNotNone(disk)
        center, radius = disk
        for g in vertices:
            self.assertLessEqual(math.dist(center, g), 20.0 - radius + 1e-8)
            self.assertLessEqual(math.dist(center, g) + radius, 20.0 + 1e-8)

    def test_joint_route_is_not_longer_than_center_tour(self):
        start = (0.0, 0.0)
        disks = [
            ((100.0, 0.0), 20.0),
            ((100.0, 80.0), 20.0),
            ((0.0, 80.0), 20.0),
        ]
        centers = [c for c, _ in disks]
        _, center_len = held_karp_open(start, centers)
        _, joint_len, _ = joint_route(start, disks)
        self.assertLessEqual(joint_len + 1e-6, center_len)


class UnheardCoverTests(unittest.TestCase):
    def _only_channel(self, channel: int) -> UnheardGrid:
        grid = UnheardGrid()
        for other in CHANNELS:
            if other != channel:
                grid.forget(other)
        return grid

    def test_punch_only_clears_that_channel(self):
        grid = UnheardGrid()
        grid.exclude(1, 1200.0, 0.0, LISTEN_CERT)
        self.assertFalse(grid.intersects(1, 1200.0, 0.0, LISTEN_CERT))
        self.assertTrue(grid.intersects(2, 1200.0, 0.0, LISTEN_CERT))

    def test_outer_hex_still_needed_after_origin_punch(self):
        grid = self._only_channel(1)
        grid.exclude(1, 0.0, 0.0, LISTEN_CERT)
        grid.record_listen(1, 0.0, 0.0)
        self.assertTrue(grid.cover_still_needed((1200.0, 0.0)))

    def test_fake_ring_is_not_a_new_cover_stop(self):
        grid = self._only_channel(1)
        grid.exclude(1, 1200.0, 0.0, LISTEN_CERT)
        self.assertTrue(grid.any_intersects(1200.0, 0.0, 1000.0))
        self.assertTrue(grid.cover_still_needed((1200.0, 0.0)))
        grid.record_listen(1, 1200.0, 0.0)
        self.assertFalse(grid.cover_still_needed((1200.0, 0.0)))
        self.assertFalse(grid.leftover_intersects(1, 1200.0, 0.0, 1000.0))

    def test_cover_slack_is_full_when_the_disk_has_no_leftover(self):
        grid = self._only_channel(1)
        grid.record_listen(1, 1200.0, 0.0)
        self.assertFalse(grid.cover_still_needed((1200.0, 0.0)))
        self.assertGreaterEqual(grid.cover_slack((1200.0, 0.0)), 1000.0 - 1e-6)

    def test_finish_vertex_only_when_leftover_is_inside_the_punch(self):
        grid = self._only_channel(1)
        d2 = (GRID_XY[:, 0] - 500.0) ** 2 + (GRID_XY[:, 1] - 0.0) ** 2
        grid.mask[0] = d2 <= 80.0 ** 2
        self.assertIn(1, grid.channels_that_finish_a_vertex(500.0, 0.0, [(1200.0, 0.0)]))
        self.assertNotIn(
            1, grid.channels_that_finish_a_vertex(1800.0, 0.0, [(1200.0, 0.0)])
        )


class StructHelperCoverTests(unittest.TestCase):
    def test_planned_stop_can_drop_a_hex_vertex(self):
        from problem3_belief_struct import UnheardGrid as StructGrid

        grid = StructGrid()
        for channel in CHANNELS:
            grid.exclude(channel, 0.0, 0.0, LISTEN_CERT)
            grid.record_listen(channel, 0.0, 0.0)
        hex_stops = census_stops()[1:]
        pending = list(enumerate(hex_stops))
        plain = grid.greedy_needed(pending)
        helped = grid.greedy_needed(pending, helpers=[hex_stops[0]])
        self.assertTrue(plain)
        self.assertLess(len(helped), len(plain))
        self.assertFalse(any(stop == hex_stops[0] for _, stop in helped))

    def test_shadow_joint_two_stops_can_beat_one(self):
        from problem3_belief_struct import UnheardGrid as StructGrid
        from problem3_belief_struct import shadow_joint_cover

        grid = StructGrid()
        for channel in CHANNELS:
            grid.exclude(channel, 0.0, 0.0, LISTEN_CERT)
            grid.record_listen(channel, 0.0, 0.0)
        hex_stops = census_stops()[1:]
        pending = list(enumerate(hex_stops))
        needed = grid.greedy_needed(pending)
        one = shadow_joint_cover(grid, needed, [hex_stops[0]])
        two = shadow_joint_cover(grid, needed, [hex_stops[0], hex_stops[1]])
        self.assertGreaterEqual(one["drop1"], 1)
        self.assertGreaterEqual(two["drop2"], two["drop1"])
        self.assertGreaterEqual(two["drop2"], one["drop1"])


class PolicySmokeTests(unittest.TestCase):
    def test_limited_policy_clears_a_tiny_instance(self):
        from problem3_fusion_policies import run_policy

        rng = random.Random(7)
        world = generate_instance(rng, n_sources=3)
        stats = run_policy(world, "limited")
        self.assertEqual(stats["cleared"], 3)
        self.assertGreater(stats["time_s"], 0.0)
        self.assertIn(stats["replans"], range(0, 40))

    def test_belief_policy_clears_a_tiny_instance(self):
        from problem3_fusion_policies import run_policy

        rng = random.Random(7)
        world = generate_instance(rng, n_sources=3)
        stats = run_policy(world, "belief")
        self.assertEqual(stats["cleared"], 3)
        self.assertEqual(stats["remaining"], 0)
        self.assertGreater(stats["time_s"], 0.0)
        self.assertEqual(stats["stop_reason"], "certificate")

    def test_belief_is_e15_and_ctrl_still_clears(self):
        from problem3_fusion_policies import run_policy

        rng = random.Random(7)
        world = generate_instance(rng, n_sources=3)
        e15 = run_policy(world.clone(), "belief_struct")
        field = run_policy(world.clone(), "belief")
        ctrl = run_policy(world.clone(), "belief_ctrl")
        self.assertEqual(field["remaining"], 0)
        self.assertEqual(e15["remaining"], 0)
        self.assertEqual(ctrl["remaining"], 0)
        self.assertAlmostEqual(field["time_s"], e15["time_s"], places=6)
        self.assertEqual(field["policy"], e15["policy"])

    def test_belief_can_clear_before_seven_census_stops(self):
        from problem3_fusion_policies import run_policy

        world = World.from_sources(
            [(3, 220.0, 40.0, 1200.0), (8, 480.0, 80.0, 1300.0)],
            error_seed=11,
        )
        stats = run_policy(world, "belief")
        self.assertEqual(stats["cleared"], 2)
        self.assertLess(stats["cover_visits_at_first_clear"], 7)

    def test_belief_clears_the_eight_seed_2027_instances(self):
        from problem3_fusion_policies import run_policy

        rng = random.Random(2027)
        leftover = []
        for index in range(8):
            world = generate_instance(rng)
            stats = run_policy(world, "belief")
            if stats["remaining"] != 0:
                leftover.append((index, stats["remaining"], stats["steps"], stats.get("stop_reason")))
        self.assertEqual(leftover, [])

    def test_belief_punch_clears_a_tiny_instance(self):
        from problem3_fusion_policies import run_policy

        rng = random.Random(7)
        world = generate_instance(rng, n_sources=3)
        stats = run_policy(world, "belief_punch")
        self.assertEqual(stats["cleared"], 3)
        self.assertEqual(stats["remaining"], 0)
        self.assertEqual(stats["stop_reason"], "certificate")

    def test_belief_delay_clears_a_tiny_instance(self):
        from problem3_fusion_policies import run_policy

        rng = random.Random(7)
        world = generate_instance(rng, n_sources=3)
        stats = run_policy(world, "belief_delay")
        self.assertEqual(stats["cleared"], 3)
        self.assertEqual(stats["remaining"], 0)
        self.assertEqual(stats["stop_reason"], "certificate")


class BatchStationTests(unittest.TestCase):
    def test_one_station_can_serve_two_nearby_wedges(self):
        unresolved = [
            {
                "channel": 1,
                "kind": "wedge",
                "measurements": [
                    {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 10.0}
                ],
            },
            {
                "channel": 2,
                "kind": "wedge",
                "measurements": [
                    {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 25.0}
                ],
            },
        ]
        station, channels = propose_batch_station((0.0, 0.0), unresolved)
        self.assertEqual(len(station), 2)
        self.assertLessEqual(math.hypot(*station), ARENA + 1e-6)
        self.assertGreaterEqual(len(channels), 2)

    def test_wedge_station_uses_tuned_second_distance(self):
        unresolved = [
            {
                "channel": 1,
                "kind": "wedge",
                "measurements": [
                    {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 0.0}
                ],
            }
        ]
        station, channels = propose_batch_station((0.0, 0.0), unresolved)
        self.assertEqual(channels, [1])
        self.assertAlmostEqual(math.hypot(*station), STATION_DIST, delta=80.0)
        self.assertFalse(station_useful((0.0, 0.0), unresolved[0]))

    def test_fixed_r_keeps_near_sample_and_drops_far_conflict(self):
        from problem3_batch_station import fixed_r_ok

        state = {
            "kind": "wedge",
            "measurements": [
                {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 0.0}
            ],
            "silences": [(1400.0, 0.0)],
        }
        self.assertTrue(fixed_r_ok((800.0, 0.0), state))
        self.assertFalse(fixed_r_ok((1450.0, 0.0), state))

    def test_heard_no_signal_still_keeps_bearing_samples(self):
        state = {
            "kind": "wedge",
            "measurements": [
                {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 0.0}
            ],
            "silences": [(1400.0, 0.0)],
        }
        samples = possible_samples(state)
        self.assertTrue(samples)
        self.assertTrue(any(abs(sample[0] - 300.0) < 1e-6 for sample in samples))

    def test_classify_keeps_silences_on_a_wedge(self):
        label = classify_channel(
            [
                {"x": 0.0, "y": 0.0, "status": "direction", "svd_deg": 0.0},
                {"x": 600.0, "y": 0.0, "status": "no_signal"},
            ]
        )
        self.assertEqual(label["kind"], "wedge")
        self.assertIn((600.0, 0.0), label["silences"])


def _permutations(items):
    items = list(items)
    if len(items) <= 1:
        yield items
        return
    for i, value in enumerate(items):
        rest = items[:i] + items[i + 1 :]
        for perm in _permutations(rest):
            yield [value] + perm


class OfficialRobotMappingTests(unittest.TestCase):
    def test_map_measure_keeps_direction_and_near(self):
        from problem3_official_robot import map_measure

        self.assertEqual(
            map_measure({"measure_result": "direction", "svd_deg": 12.34}),
            {"status": "direction", "svd_deg": 12.34},
        )
        self.assertEqual(map_measure({"measure_result": "near"}), {"status": "near"})
        self.assertEqual(
            map_measure({"measure_result": "no_signal"}),
            {"status": "no_signal"},
        )

    def test_official_world_dummy_until_cleared(self):
        from problem3_official_robot import OfficialWorld

        world = OfficialWorld()
        self.assertIsNotNone(world.source_by_channel(7))
        world.mark_cleared(7)
        self.assertIsNone(world.source_by_channel(7))
        self.assertEqual(world.remaining(), None)


class CensusGeometryTests(unittest.TestCase):
    def test_seven_census_stops_include_origin_and_hexagon(self):
        stops = census_stops()
        self.assertEqual(len(stops), 7)
        self.assertEqual(stops[0], (0.0, 0.0))
        radii = [math.hypot(x, y) for x, y in stops[1:]]
        for r in radii:
            self.assertAlmostEqual(r, CENSUS_RHO, places=6)


if __name__ == "__main__":
    unittest.main()
