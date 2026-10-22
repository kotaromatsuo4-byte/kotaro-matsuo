"""Mechanical and learning checks; no experiment artifacts are written."""
import unittest

import numpy as np
import torch

import mechanics
from supervised_folding import (BASE, Dataset, DirectionNet, Geometry, GOALS,
                                generate_starts, load_targets, rollout,
                                student_direction, teacher_advance, teacher_direction, train)


class SupervisedFoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.geometry = Geometry(BASE / "model/five_bar_linkage")
        cls.targets = load_targets(cls.geometry)

    def test_teacher_is_projected_negative_gradient(self):
        geometry = self.geometry
        node = geometry.initial
        modes = geometry.modes(node)
        jacobian = mechanics.dedx(node, geometry.member, geometry.free)
        for target in self.targets.values():
            direction, norm = teacher_direction(node, target, modes)
            self.assertGreater(norm, 1e-8)
            self.assertAlmostEqual(np.linalg.norm(direction), 1.0)
            np.testing.assert_allclose(jacobian @ direction[geometry.free], 0, atol=1e-10)
            self.assertLess(float((node - target).reshape(-1) @ direction), 0)
            candidate, status = teacher_advance(geometry, node, target, direction)
            self.assertEqual(status, "accepted")
            self.assertLess(np.linalg.norm(candidate - target), np.linalg.norm(node - target))
            self.assertLessEqual(mechanics.current_strain(candidate, geometry.member, geometry.lengths), 1e-8)
            fixed = (geometry.fix[:, 0], geometry.fix[:, 1])
            np.testing.assert_array_equal(candidate[fixed], node[fixed])

    def test_zero_gradient_is_excluded(self):
        node = self.geometry.initial
        data = Dataset(self.geometry, {2.0: node, 3.0: self.targets[3.0]})
        data.add_path([node], 0, 0)
        inputs, _, _, goals = data.tensors()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(goals.tolist(), [1])
        self.assertEqual(data.skipped_zero, 1)

    def test_distinct_feasible_reproducible_starts(self):
        g = self.geometry
        train_starts = generate_starts(g, 3, np.random.default_rng(42), 5, 15)
        replay = generate_starts(g, 3, np.random.default_rng(42), 5, 15)
        np.testing.assert_array_equal(train_starts, replay)
        test_starts = generate_starts(g, 2, np.random.default_rng(43), 5, 15, train_starts)
        for node in [*train_starts, *test_starts]:
            self.assertLessEqual(mechanics.current_strain(node, g.member, g.lengths), 1e-8)
        self.assertTrue(all(np.linalg.norm(a - b) / g.scale >= .03 for a in train_starts for b in test_starts))

    def test_aggregation_keeps_both_goals_and_old_data(self):
        g = self.geometry
        data = Dataset(g, self.targets)
        data.add_path([g.initial], 0, 0)
        torch.manual_seed(3)
        model = DirectionNet(24)
        path, _ = rollout(g, g.initial, 2.0, 3, model=model)
        data.add_path(path, 1, 1)
        arrays = data.arrays()
        self.assertEqual(set(arrays["source_round"]), {0, 1})
        inputs, _, _, _ = data.tensors()
        for row in range(0, len(inputs), 2):
            np.testing.assert_array_equal(inputs[row, :-1], inputs[row + 1, :-1])
            self.assertEqual(inputs[row:row + 2, -1].tolist(), list(GOALS))

    def test_direction_training_improves_and_student_is_feasible(self):
        g = self.geometry
        torch.manual_seed(11)
        data = Dataset(g, self.targets)
        for goal in GOALS:
            path, _ = rollout(g, g.initial, goal, 10, target=self.targets[goal])
            data.add_path(path, 0, int(goal))
        model = DirectionNet(32)
        inputs, projectors, labels, _ = data.tensors()
        with torch.no_grad():
            before = ((model(inputs, projectors) - labels)**2).sum(-1).mean().item()
        train(model, torch.optim.Adam(model.parameters(), lr=.003), data, 80, 32,
              torch.Generator().manual_seed(9))
        with torch.no_grad():
            after = ((model(inputs, projectors) - labels)**2).sum(-1).mean().item()
        self.assertLess(after, before * .5)
        direction = student_direction(model, g, g.initial, 2.5)
        jacobian = mechanics.dedx(g.initial, g.member, g.free)
        np.testing.assert_allclose(jacobian @ direction[g.free], 0, atol=1e-6)
        # Unknown D uses the model only; no target coordinates are supplied.
        path, _ = rollout(g, g.initial, 2.5, 5, model=model)
        for node in path:
            self.assertLessEqual(mechanics.current_strain(node, g.member, g.lengths), 1e-8)

    def test_seven_bar_dimensions_and_d4_without_teacher(self):
        g = Geometry(BASE / "model/seven_bar_linkage")
        targets = load_targets(g)
        self.assertEqual(set(targets), {2.0, 3.0})
        data = Dataset(g, targets)
        data.add_path([g.initial], 0, 0)
        inputs, projectors, labels, goals = data.tensors()
        self.assertEqual(tuple(inputs.shape), (2, 22))
        self.assertEqual(tuple(labels.shape), (2, 21))
        model = DirectionNet(32, g.coordinate_size)
        prediction = model(inputs, projectors)
        self.assertEqual(tuple(prediction.shape), (2, 21))
        loss = ((prediction - labels) ** 2).sum(-1).mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        path, _ = rollout(g, g.initial, 4.0, 5, model=model)
        self.assertEqual(path.shape[1:], (7, 3))
        for node in path:
            self.assertLessEqual(mechanics.current_strain(node, g.member, g.lengths), 1e-8)


if __name__ == "__main__":
    unittest.main()
