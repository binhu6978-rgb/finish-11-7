import copy
import pickle
import unittest

import numpy as np
import torch

from Algorithm.FedEcho import FedEchoController, make_basis, project_update, solve_dispatch
from Algorithm.FedEchoModelView import train_in_model_view
from Algorithm.SymmetryView import ViewScheduler, apply_horizontal_view_


def toy_model():
    return torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.Tanh(),
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                               torch.nn.Linear(2, 2)).double()


class ToyLocal:
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def train(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.5)
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(self.images), self.labels)
            loss.backward()
            optimizer.step()
        return model.state_dict()


class ModelViewTests(unittest.TestCase):
    def test_identity_view_is_exact_original_local_training(self):
        model = toy_model()
        images = torch.arange(4*1*5*5, dtype=torch.float64).reshape(4, 1, 5, 5)/100
        local = ToyLocal(images, torch.tensor([0, 1, 0, 1]))
        original = local.train(copy.deepcopy(model))
        actual = train_in_model_view(copy.deepcopy(model), local, "I")
        for name in original:
            self.assertTrue(torch.equal(original[name], actual[name]))

    def test_mirror_training_and_response_are_in_canonical_coordinates(self):
        model = toy_model()
        current = copy.deepcopy(model.state_dict())
        previous = {n: torch.zeros_like(v) for n, v in current.items()}
        basis, norms, radius, _ = make_basis(model, current, previous)
        alpha = np.array([.04, -.03])
        with torch.no_grad():
            for name, p in model.named_parameters():
                for a in range(2):
                    p.add_(basis[a][name], alpha=float(alpha[a]))
        start = copy.deepcopy(model.state_dict())
        images = torch.arange(4*1*5*5, dtype=torch.float64).reshape(4, 1, 5, 5)/100
        labels = torch.tensor([0, 1, 0, 1])
        actual = train_in_model_view(copy.deepcopy(model), ToyLocal(images, labels), "F")
        expected = ToyLocal(torch.flip(images, [-1]), labels).train(copy.deepcopy(model))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-12, rtol=1e-12)
        response = project_update(actual, start, basis, norms)
        expected_response = project_update(expected, start, basis, norms)
        np.testing.assert_allclose(response, expected_response, atol=1e-12)
        # Measuring from w instead of x would erroneously include alpha.
        wrong = project_update(actual, current, basis, norms)
        np.testing.assert_allclose(wrong-response, alpha, atol=1e-12)

    def test_zero_sum_start_survives_view_and_inverse(self):
        model = toy_model()
        current = copy.deepcopy(model.state_dict())
        previous = {n: torch.zeros_like(v) for n, v in current.items()}
        basis, norms, radius, _ = make_basis(model, current, previous)
        p = np.array([.2, .3, .5])
        alpha = solve_dispatch([[4., -2.], [-3., 1.], [1., 6.]], p, norms, radius)
        starts = []
        for j in range(3):
            local = copy.deepcopy(model)
            with torch.no_grad():
                for name, parameter in local.named_parameters():
                    for a in range(2):
                        parameter.add_(basis[a][name], alpha=float(alpha[j, a]))
            if j != 1:
                apply_horizontal_view_(local)
                apply_horizontal_view_(local)
            starts.append(local.state_dict())
        for name in current:
            mean = sum(float(p[j])*starts[j][name] for j in range(3))
            torch.testing.assert_close(mean, current[name], atol=1e-12, rtol=1e-12)

    def test_scheduler_checkpoint_and_global_rng(self):
        controller = FedEchoController()
        controller.view_scheduler = ViewScheduler("alt", 1)
        rng = torch.get_rng_state().clone()
        first = controller.view_scheduler.choose(7, 0)[0]
        restored = pickle.loads(pickle.dumps(controller))
        next_a = controller.view_scheduler.choose(7, 9)[0]
        next_b = restored.view_scheduler.choose(7, 9)[0]
        self.assertNotEqual(first, next_a)
        self.assertEqual(next_a, next_b)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(restored.view_scheduler.participation_count[7], 2)

    def test_view_does_not_change_core_dispatch_or_decay(self):
        plain = FedEchoController()
        viewed = FedEchoController()
        viewed.view_scheduler = ViewScheduler("alt", 1)
        for controller in [plain, viewed]:
            controller.observe([1, 2], [.3, .7], 1, [[3, 1], [1, 4]], [0, 1])
            controller.observe([1, 2], [.3, .7], 3, [[2, 1], [1, 3]], [0, 1])
        viewed.view_scheduler.choose(1, 0)
        a = plain.dispatch([1, 2], [.3, .7], 6, [.6, .8], 1., [0, 1])
        b = viewed.dispatch([1, 2], [.3, .7], 6, [.6, .8], 1., [0, 1])
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2:], b[2:])


if __name__ == '__main__':
    unittest.main()
