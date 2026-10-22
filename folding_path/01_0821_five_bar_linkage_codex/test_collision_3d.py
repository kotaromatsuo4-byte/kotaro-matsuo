"""Focused semantics and regression tests for :mod:`collision_3d`."""

import unittest

import numpy as np

from collision_3d import (
    COLLISION_BACKEND_NAME,
    COLLISION_BACKEND_VERSION,
    ContinuousBarCollisionChecker,
    has_proper_self_crossing,
    make_legacy_signature_checker,
)


TWO_NONADJACENT_BARS = np.asarray([[0, 1], [2, 3]], dtype=int)


class ContinuousBarCollisionTests(unittest.TestCase):
    def checker(self, member=TWO_NONADJACENT_BARS):
        return ContinuousBarCollisionChecker(
            member,
            distance_tolerance=1.0e-9,
            time_tolerance=1.0e-9,
        )

    def test_detects_non_coplanar_midstep_pass_through(self):
        # Endpoint configurations are separated by distance one.  The moving
        # y-directed bar crosses the stationary x-directed bar only at t=0.5.
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        after = before.copy()
        after[2:, 2] = -1.0
        result = self.checker().check(before, after)
        self.assertTrue(result.collision)
        self.assertIsNotNone(result.event)
        self.assertAlmostEqual(result.event.time, 0.5, places=10)
        self.assertAlmostEqual(result.event.first_parameter, 0.5, places=9)
        self.assertAlmostEqual(result.event.second_parameter, 0.5, places=9)
        self.assertFalse(result.event.coplanar_transition)

    def test_detects_coplanar_midstep_pass_through(self):
        # Both bars remain in z=0; the vertical bar sweeps from x=-2 to x=2.
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-2.0, -1.0, 0.0],
            [-2.0, 1.0, 0.0],
        ])
        after = before.copy()
        after[2:, 0] = 2.0
        result = self.checker().check(before, after)
        self.assertTrue(result.collision)
        self.assertTrue(result.event.coplanar_transition)
        self.assertGreater(result.event.time, 0.0)
        self.assertLess(result.event.time, 1.0)

    def test_terminal_proper_contact_is_allowed(self):
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        after = before.copy()
        after[2:, 2] = 0.0
        self.assertFalse(self.checker()(before, after))

    def test_initial_proper_contact_while_separating_is_allowed(self):
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        after = before.copy()
        after[2:, 2] = 1.0
        self.assertFalse(self.checker()(before, after))

    def test_parallel_collinear_terminal_overlap_is_allowed(self):
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.0, 1.0],
            [0.5, 0.0, 1.0],
        ])
        after = before.copy()
        after[2:, 2] = 0.0
        self.assertFalse(self.checker()(before, after))

    def test_persistent_collinear_overlap_is_allowed(self):
        nodes = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        self.assertFalse(self.checker()(nodes, nodes))

    def test_persistent_transversal_intersection_is_allowed(self):
        nodes = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self.assertFalse(self.checker()(nodes, nodes))

    def test_waterbomb_spread_3_flat_solution_is_allowed(self):
        member = np.asarray([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3],
            [2, 4], [2, 5], [2, 6], [3, 5], [4, 6], [5, 6],
        ], dtype=int)
        flat = np.asarray([
            [6.481601867972252e-09, -1.500000335710634, -0.4090362661012954],
            [-6.502876870599493e-09, 1.499999664287157, -0.4090355946777588],
            [1.7669e-11, 0.0, -1.909035799808172],
            [-1.0635e-11, -9.133271676330971e-07, -0.409036],
            [-1.064e-11, 2.419037251634442e-07, -0.409036],
            [6.481601868321374e-09, -1.500000335712764, -0.4090362661353328],
            [-6.502876870947344e-09, 1.499999664289296, -0.4090355947118035],
        ])
        self.assertFalse(self.checker(member)(flat, flat))

    def test_waterbomb_spread_3sqrt2_flat_solution_is_allowed(self):
        member = np.asarray([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3],
            [2, 4], [2, 5], [2, 6], [3, 5], [4, 6], [5, 6],
        ], dtype=int)
        flat = np.asarray([
            [-3.894400716381219e-11, -8.168861220313866e-07, -1.469696267170883],
            [1.766398312759474e-11, 2.121320251223860, 0.6516233518399089],
            [1.7669e-11, 0.0, 0.6516239840494429],
            [-1.0635e-11, -1.060660076396409, -0.409036],
            [-1.064e-11, 1.060660076396409, -0.409036],
            [1.767398312975619e-11, -2.121320251201299, 0.6516233519035186],
            [-3.894400677714368e-11, 8.168572215480553e-07, -1.469696267155264],
        ])
        self.assertFalse(self.checker(member)(flat, flat))

    def test_midstep_endpoint_only_touch_is_allowed(self):
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        after = before.copy()
        after[2:, 2] = -1.0
        self.assertFalse(self.checker()(before, after))

    def test_members_sharing_a_node_are_excluded(self):
        member = np.asarray([[0, 1], [1, 2]], dtype=int)
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        after = before.copy()
        after[2] = [-1.0, 0.0, 0.0]
        result = self.checker(member).check(before, after)
        self.assertFalse(result.collision)
        self.assertEqual(result.total_nonadjacent_pairs, 0)

    def test_swept_aabb_rejects_separated_pairs(self):
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, -1.0, 0.0],
            [10.0, 1.0, 0.0],
        ])
        result = self.checker().check(before, before)
        self.assertFalse(result.collision)
        self.assertEqual(result.total_nonadjacent_pairs, 1)
        self.assertEqual(result.broad_phase_candidates, 0)
        self.assertEqual(result.narrow_phase_checks, 0)

    def test_legacy_signatures_and_metadata(self):
        before = np.asarray([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        after = before.copy()
        after[2:, 2] = -1.0
        pairs = [(0, 1)]
        self.assertTrue(has_proper_self_crossing(
            before, after, TWO_NONADJACENT_BARS, pairs, 1.0e-9
        ))
        callback = make_legacy_signature_checker(
            TWO_NONADJACENT_BARS,
            pair_indices=pairs,
            distance_tolerance=1.0e-9,
        )
        self.assertTrue(callback(
            before, after, TWO_NONADJACENT_BARS, pairs, 1.0e-9
        ))
        self.assertEqual(COLLISION_BACKEND_NAME, "continuous_proper_crossing_3d")
        self.assertRegex(COLLISION_BACKEND_VERSION, r"^\d+\.\d+\.\d+$")

    def test_rejects_non_3d_input(self):
        checker = self.checker()
        with self.assertRaises(ValueError):
            checker(np.zeros((4, 2)), np.zeros((4, 2)))


if __name__ == "__main__":
    unittest.main()
