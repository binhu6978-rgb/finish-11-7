import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from Algorithm.FedEcho import (FedEchoController, Persistence, make_basis,
                               project_update, solve_dispatch)
from experiments.fedecho_reproduction import aggregate_add
from models.Fed import Aggregation


class FedEchoTests(unittest.TestCase):
    def test_root_conv_uses_spatial_component(self):
        model = torch.nn.Conv2d(1, 1, 3, bias=False).double()
        with torch.no_grad():
            model.weight.copy_(torch.arange(9).reshape(1, 1, 3, 3))
        old = {"weight": torch.zeros_like(model.weight)}
        basis, _, _, components = make_basis(model, model.state_dict(), old)
        self.assertEqual(components, [0, 1])
        torch.testing.assert_close(basis[1]["weight"], model.weight - 4)
        torch.testing.assert_close(basis[1]["weight"].mean((-2, -1)), torch.zeros(1, 1).double())

    def test_nonfinite_geometry_and_weights_are_rejected(self):
        for p, norms, radius in [([np.nan, .5], [1.], 1.),
                                  ([.5, .5], [np.inf], 1.),
                                  ([.5, .5], [1.], np.nan)]:
            with self.assertRaises(ValueError):
                solve_dispatch([[1.], [2.]], p, norms, radius)
        model = torch.nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            make_basis(model, model.state_dict(), None, epsilon=np.nan)

    def test_projection_subtracts_actual_dispatch_not_global(self):
        # A large nonzero dispatch must not be counted as local training.
        basis = [{"w": torch.tensor([1., 0.]).double()},
                 {"w": torch.tensor([0., 2.]).double()}]
        start = {"w": torch.tensor([100., -200.]).double()}
        returned = {"w": start["w"] + torch.tensor([3., 8.]).double()}
        np.testing.assert_allclose(project_update(returned, start, basis, [1., 2.]), [3., 4.])

    def test_prediction_is_past_only_and_uses_round_gap(self):
        ctrl = FedEchoController()
        ctrl.persistence.add(1., .8, 1)
        ctrl.histories[4] = {0: (3., 2), 1: (-2., 4)}
        _, beta, rho, _ = ctrl.dispatch([4, 7], [.3, .7], 9, [.6, .8], 1., [0, 1])
        np.testing.assert_allclose(beta[0], [3 * rho**7, -2 * rho**5])
        np.testing.assert_array_equal(beta[1], 0)
        self.assertEqual(ctrl.persistence.count, 1)
        self.assertEqual(ctrl.histories[4][0], (3., 2))

    def test_joint_solver_matches_independent_constrained_oracle(self):
        beta = np.array([[9., -3.], [-4., 7.], [2., 8.], [-8., -1.]])
        p = np.array([.1, .2, .3, .4])
        norms, radius = np.array([.6, .8]), 1.
        alpha = solve_dispatch(beta, p, norms, radius)
        z = alpha * norms
        c = -.5 * beta * norms
        oracle, ca, cb = c.copy(), np.zeros_like(c), np.zeros_like(c)
        for _ in range(30000):
            v = oracle+ca
            center = v-p @ v
            ca = v-center
            v = center+cb
            updated = v/np.maximum(1., np.linalg.norm(v, axis=1))[:, None]
            cb = v-updated
            difference = np.max(abs(updated-oracle))
            oracle = updated
            if difference < 1e-13 and np.linalg.norm(p @ oracle) < 1e-12:
                break
        self.assertAlmostEqual(float(np.sum(p[:, None]*(z-c)**2)),
                               float(np.sum(p[:, None]*(oracle-c)**2)), places=8)
        np.testing.assert_allclose(p @ z, 0, atol=1e-10)
        self.assertLessEqual(np.linalg.norm(z, axis=1).max(), 1+1e-12)

    def test_unconstrained_half_and_weighted_centering(self):
        beta = np.array([[.1, .3], [-.2, .2], [.5, -.4]])
        p = np.array([.2, .3, .5])
        actual = solve_dispatch(beta, p, [.6, .8], 1.)
        np.testing.assert_allclose(actual, -.5*(beta-p @ beta), atol=1e-15)

    def test_rank_one_saturation_and_unequal_weights(self):
        p = np.array([.15, .35, .5])
        alpha = solve_dispatch([[100], [-12], [4]], p, [2.], 2.)
        self.assertLessEqual(abs(alpha).max(), 1)
        np.testing.assert_allclose(p @ alpha, 0, atol=1e-12)

    def test_joint_solver_stress_feasibility(self):
        rng = np.random.default_rng(17)
        for scale in [0, .2, 2, 20, 200]:
            p = rng.uniform(.1, 1, 10)
            p /= p.sum()
            alpha = solve_dispatch(rng.normal(size=(10, 2))*scale, p, [.1, .99], 1.)
            z = alpha * [.1, .99]
            np.testing.assert_allclose(p @ z, 0, atol=1e-10)
            self.assertLessEqual(np.linalg.norm(z, axis=1).max(), 1+1e-12)

    def test_pooled_decay_matches_direct_prediction_loss(self):
        pool = Persistence()
        for x, gap in [(1., 1), (-2., 2), (3., 5), (.3, 10)]:
            pool.add(x, .8**gap * x, gap)
        self.assertAlmostEqual(pool.estimate(), .8, places=7)
        self.assertEqual(pool.count, 4)

    def test_empty_and_negative_correlation_decay(self):
        pool = Persistence()
        self.assertEqual(pool.estimate(), 0)
        pool.add(1, -1, 1)
        self.assertEqual(pool.estimate(), 0)

    def test_decay_minimum_inside_first_grid_interval(self):
        pool = Persistence()
        pool.add(1., .00001, 1)
        self.assertAlmostEqual(pool.estimate(), .00001, places=10)

    def test_on_policy_relative_update_and_previous_pair(self):
        ctrl = FedEchoController()
        relative = ctrl.observe([4, 7], [.25, .75], 2, [[3, 1], [1, 5]], [0, 1])
        np.testing.assert_allclose(relative, [[1.5, -3], [-.5, 1]])
        ctrl.observe([4, 7], [.25, .75], 5, [[2, 2], [1, 3]], [0, 1])
        self.assertEqual(ctrl.persistence.count, 4)
        self.assertIn(3, ctrl.persistence.by_gap)
        self.assertEqual(ctrl.histories[4][0][1], 5)

    def test_no_history_can_receive_centered_offset(self):
        ctrl = FedEchoController()
        ctrl.histories[4] = {0: (2., 1)}
        ctrl.persistence.add(1, 1, 1)
        alpha, beta, rho, count = ctrl.dispatch([4, 7], [.5, .5], 2, [1.], 1., [0])
        self.assertEqual(beta[1, 0], 0)
        self.assertGreater(alpha[1, 0], 0)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(rho, 1.)

    def test_invalid_component_preserves_history(self):
        ctrl = FedEchoController()
        ctrl.histories[1] = {0: (3., 1), 1: (2., 2)}
        ctrl.observe([1], [1.], 5, [[8.]], [1])
        self.assertEqual(ctrl.histories[1][0], (3., 1))
        self.assertEqual(ctrl.histories[1][1], (0., 5))
        self.assertEqual(ctrl.persistence.count, 1)

    def test_future_history_rejected(self):
        ctrl = FedEchoController()
        ctrl.histories[1] = {0: (3., 2)}
        with self.assertRaises(RuntimeError):
            ctrl.dispatch([1], [1.], 2, [1.], 1., [0])

    def test_structured_basis_reconstruction_orthogonality_and_projection(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 3), torch.nn.Flatten(), torch.nn.Linear(3, 2)).double()
        old = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        current = model.state_dict()
        basis, norms, radius, components = make_basis(model, current, old)
        self.assertEqual(components, [0, 1])
        dot = sum((basis[0][n]*basis[1][n]).sum().item() for n in old)
        self.assertAlmostEqual(dot, 0., places=12)
        for name in old:
            torch.testing.assert_close(basis[0][name]+basis[1][name], current[name])
        self.assertTrue(torch.count_nonzero(basis[1]['2.weight']) == 0)
        returned = {n: current[n]+2*basis[0][n]-3*basis[1][n] for n in old}
        np.testing.assert_allclose(project_update(returned, current, basis, norms), [2, -3], atol=1e-12)
        self.assertAlmostEqual(radius**2, float(np.sum(norms**2)))

    def test_first_round_and_small_trajectory_have_no_control(self):
        model = torch.nn.Linear(2, 2)
        self.assertEqual(make_basis(model, model.state_dict(), None)[3], [])
        same = copy.deepcopy(model.state_dict())
        self.assertEqual(make_basis(model, model.state_dict(), same)[3], [])

    def test_streamed_aggregation_matches_original_including_integer_buffers(self):
        states = [{"w": torch.tensor([i*.2, i*.7]), "count": torch.tensor(i, dtype=torch.int64)}
                  for i in range(1, 5)]
        sizes = [11, 3, 8, 7]
        acc = None
        for state, size in zip(states, sizes):
            acc = aggregate_add(acc, state, size)
        acc = {k: torch.div(v, sum(sizes)) for k, v in acc.items()}
        expected = Aggregation(states, sizes)
        for name in expected:
            self.assertTrue(torch.equal(expected[name], acc[name]))

    def test_controller_checkpoint_roundtrip_continues_identically(self):
        import pickle
        a = FedEchoController()
        a.observe([1, 2], [.5, .5], 1, [[2., 1.], [1., 3.]], [0, 1])
        a.observe([1, 2], [.5, .5], 3, [[1., 1.], [.5, 2.]], [0, 1])
        b = pickle.loads(pickle.dumps(a))
        x = a.dispatch([1, 2], [.5, .5], 6, [.6, .8], 1., [0, 1])
        y = b.dispatch([1, 2], [.5, .5], 6, [.6, .8], 1., [0, 1])
        np.testing.assert_array_equal(x[0], y[0])
        self.assertEqual(x[2], y[2])


if __name__ == '__main__':
    unittest.main()
