import pickle
import random
import unittest

import numpy as np
import torch
from Algorithm.SymmetryView import ViewScheduler
from experiments.fedecho_random_view import prepare_args


class RandomModelViewTests(unittest.TestCase):
    def test_data_augmentation_stays_disabled(self):
        args = prepare_args(["--output_dir", "results/unused_random_view_test"])
        self.assertEqual(args.input_hflip, 0)
        self.assertTrue(args.data_augmentation_forbidden)
        self.assertEqual(args.view_mode, "rand")
        self.assertEqual(args.algorithm, "FedEchoView")
        self.assertEqual((args.seed, args.local_ep, args.local_bs), (1, 5, 50))
        self.assertFalse(args.disable_control)

    def test_identity_gate_has_same_data_pipeline(self):
        args = prepare_args(["--output_dir", "results/unused_random_view_test", "--view_policy", "none"])
        self.assertEqual(args.view_mode, "none")
        self.assertEqual(args.input_hflip, 0)

    def test_private_random_stream_resume_and_global_rng(self):
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        schedule = ViewScheduler("rand", 1)
        first = [schedule.choose(7, t)[0] for t in range(8)]
        restored = pickle.loads(pickle.dumps(schedule))
        a = [schedule.choose(7, t)[0] for t in range(8, 20)]
        b = [restored.choose(7, t)[0] for t in range(8, 20)]
        self.assertEqual(a, b)
        self.assertEqual(schedule.participation_count[7], 20)
        self.assertTrue(any(x == y for x, y in zip(first+a, (first+a)[1:])))
        self.assertEqual(before_python, random.getstate())
        now_numpy = np.random.get_state()
        np.testing.assert_array_equal(before_numpy[1], now_numpy[1])
        self.assertEqual(before_numpy[2:], now_numpy[2:])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_conflicting_view_override_rejected(self):
        with self.assertRaises(ValueError):
            prepare_args(["--output_dir", "results/unused_random_view_test", "--view_mode", "alt"])


if __name__ == '__main__':
    unittest.main()
