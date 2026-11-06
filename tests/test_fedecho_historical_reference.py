import tempfile
import unittest
from pathlib import Path

from scripts.run_fedecho_reproduction import historical_summary, read_baseline_log


class HistoricalReferenceTests(unittest.TestCase):
    def test_excludes_later_rounds_and_peak_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"baseline.log"
            path.write_text("\n".join(
                [f"ROUND_ACCURACY method=FedPhoenix round={i} accuracy={i/10}" for i in range(1, 601)]
                + ["PEAK_ACCURACY method=FedPhoenix round=600 accuracy=99.0"]), encoding="utf-8")
            rows = read_baseline_log(path)
            summary = historical_summary(rows, [])
            self.assertEqual(len(rows), 500)
            self.assertEqual(summary["peak_round"], 500)
            self.assertEqual(summary["peak_accuracy"], 50.)
            self.assertAlmostEqual(summary["mean_rounds_1_500"], 25.05)

    def test_missing_round_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"baseline.log"
            path.write_text("ROUND_ACCURACY method=FedPhoenix round=1 accuracy=10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing rounds"):
                read_baseline_log(path, 2)

    def test_duplicate_round_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"baseline.log"
            path.write_text("ROUND_ACCURACY method=FedPhoenix round=1 accuracy=10\n"*2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_baseline_log(path, 1)


if __name__ == '__main__':
    unittest.main()
