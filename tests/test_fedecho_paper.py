"""Protocol/reporting checks that prevent a misleading comparison."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.fedecho_reproduction import output_lock
from scripts.run_fedecho_paper import exact, metrics, report


class PaperProtocolTests(unittest.TestCase):
    def test_late_peak_cannot_contaminate_first500_metrics(self):
        rows = [{"test_accuracy": 50 if i < 500 else 80} for i in range(1000)]
        result = metrics(rows)
        self.assertEqual(result["mean_1_500"], 50)
        self.assertEqual(result["mean_501_1000"], 80)
        self.assertEqual(result["mean_901_1000"], 80)
        self.assertEqual(result["peak_round"], 501)
        self.assertEqual(result["best_rolling10"], 80)

    def test_output_lock_rejects_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with output_lock(path):
                with self.assertRaises(RuntimeError):
                    with output_lock(path):
                        pass
            with output_lock(path):
                pass

    def test_exact_gate_checks_rng_not_only_accuracy(self):
        with self.assertRaises(RuntimeError):
            exact([{"accuracy": "1", "rng": "a"}], [{"accuracy": "1", "rng": "b"}], ["accuracy", "rng"], 1)

    def test_report_uses_only_requested_horizon_and_needs_no_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "FedEcho").mkdir()
            baseline = out / "baseline.log"
            baseline.write_text("\n".join(
                f"ROUND_ACCURACY method=FedPhoenix round={i} accuracy={50 if i <= 1000 else 99}"
                for i in range(1, 1201)), encoding="utf-8")
            rows = [{"round": i, "test_accuracy": 51., "rho": 0., "max_offset_ratio": 0.,
                     "consensus_residual": 0., "response_center_residual": 0., "response_pairs": i,
                     "gpu_peak_bytes": 100, "round_seconds": 1.} for i in range(1, 1001)]
            with (out / "FedEcho/round_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            value = report(out, baseline)
            self.assertEqual(value["FedPhoenix_historical"]["peak"], 50.)
            self.assertEqual(value["delta_pp"]["mean_1_500"], 1.)
            self.assertTrue((out / "accuracy_curve.svg").exists())
            self.assertTrue((out / "RESULTS_SUMMARY.md").exists())


if __name__ == "__main__":
    unittest.main()
