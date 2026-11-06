"""Run random model-side FedEcho views with all input augmentation OFF."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from experiments.fedecho_reproduction import atomic_json, sha_file
from scripts.run_fedecho_reproduction import read_baseline_log, historical_summary, read_csv


def compare_prefix(a, b, fields, count):
    if len(a) < count or len(b) < count:
        raise RuntimeError("Missing gate records")
    for i in range(count):
        for field in fields:
            if a[i][field] != b[i][field]:
                raise RuntimeError(f"Parity mismatch record {i+1}: {field}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=Path, default=ROOT/"results/fedecho_random_view_seed1")
    p.add_argument("--skip_gates", action="store_true")
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output/"logs"
    logs.mkdir(exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for item in old.get("active", []):
            if psutil.pid_exists(item["pid"]):
                raise RuntimeError("Existing worker is alive; do not start a duplicate")
        if old.get("status") == "completed":
            return
        old_pid = old.get("pid")
        if old_pid and psutil.pid_exists(old_pid):
            command = psutil.Process(old_pid).cmdline()
            if str(Path(__file__).resolve()) in command:
                raise RuntimeError("Existing launcher is alive")
    baseline_path = ROOT/"result_other_method/cifar10/0.3/vgg_FedPhoenix.log"
    baseline = read_baseline_log(baseline_path)
    baseline_hash = sha_file(baseline_path)
    pure = ROOT/"results/fedecho_reproduction_seed1/FedEcho"
    sources = ["Algorithm/FedEcho.py", "Algorithm/FedEchoModelView.py", "Algorithm/SymmetryView.py",
               "experiments/fedecho_random_view.py", "experiments/fedecho_model_view.py",
               "experiments/fedecho_reproduction.py", "utils/get_dataset.py"]
    fingerprints = {f: sha_file(ROOT/f) for f in sources}
    manifest = {"status": "gates", "pid": os.getpid(), "active": [], "attempts": [],
                "definition": "FedEcho unchanged core + RANDOM client model view; input augmentation OFF; no reset",
                "seed": 1, "epochs": 500, "max_additional_training_workers": 1,
                "baseline": {"path": str(baseline_path), "sha256": baseline_hash,
                             "type": "historical log, first 500 rounds", "paired_sampling_verified": False}}
    atomic_json(manifest_path, manifest)
    current = None
    handles = []

    def run_job(dest, rounds, tag, extra=()):
        nonlocal current
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT/"experiments/fedecho_random_view.py"),
               "--gpu", str(args.gpu), "--epochs", str(rounds), "--output_dir", str(dest), *extra]
        stdout = (logs/f"{tag}.stdout.log").open("a", encoding="utf-8")
        stderr = (logs/f"{tag}.stderr.log").open("a", encoding="utf-8")
        handles.extend([stdout, stderr])
        current = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr)
        manifest["active"] = [{"pid": current.pid, "tag": tag, "command": cmd}]
        atomic_json(manifest_path, manifest)
        print(f"LAUNCHED {tag} pid={current.pid}", flush=True)
        code = current.wait()
        manifest["active"] = []
        manifest["attempts"].append({"tag": tag, "return_code": code})
        atomic_json(manifest_path, manifest)
        current = None
        if code:
            raise RuntimeError(f"{tag} failed with code {code}; inspect stderr")

    try:
        gate_path = output/"gate_report.json"
        gates = output/"gates"
        if not args.skip_gates:
            with (logs/"regression.log").open("w", encoding="utf-8") as f:
                completed = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", str(ROOT/"tests")],
                    cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), stdout=f, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError("Regression tests failed")
            run_job(gates/"identity", 3, "identity3", ["--view_policy", "none", "--check_aggregation"])
            compare_prefix(read_csv(pure/"round_metrics.csv"), read_csv(gates/"identity/round_metrics.csv"),
                           ["global_state_sha256", "selected_clients", "test_accuracy", "rho"], 3)
            compare_prefix(read_csv(pure/"client_events.csv"), read_csv(gates/"identity/client_events.csv"),
                           ["sgd_rng_sha256", "prediction", "observed_response"], 30)
            run_job(gates/"resumed", 4, "random4")
            run_job(gates/"resumed", 8, "random_resume8", ["--resume"])
            run_job(gates/"continuous", 8, "random_continuous8", ["--check_aggregation"])
            compare_prefix(read_csv(gates/"resumed/round_metrics.csv"), read_csv(gates/"continuous/round_metrics.csv"),
                           ["global_state_sha256", "selected_clients", "rho", "views", "response_pairs"], 8)
            compare_prefix(read_csv(gates/"resumed/client_events.csv"), read_csv(gates/"continuous/client_events.csv"),
                           ["prediction", "observed_response", "current_view", "previous_view", "visit_index"], 80)
            active = json.loads((gates/"continuous/summary.json").read_text())
            metrics = read_csv(gates/"continuous/round_metrics.csv")
            if active["alt_violations"] or not 0 < active["mean_flip_fraction"] < 1:
                raise RuntimeError("Random model view scheduling failed")
            if int(metrics[-1]["response_pairs"]) < 1:
                raise RuntimeError("Response history not exercised")
            config = json.loads((gates/"continuous/config.json").read_text())
            pure_config = json.loads((pure/"config.json").read_text())
            for key in ["initial_state_sha256", "partition_sha256", "train_transform"]:
                if config[key] != pure_config[key]:
                    raise RuntimeError(f"Pure/variant protocol differs: {key}")
            for key in ["models/Nets.py", "models/Update.py", "models/Fed.py", "Algorithm/FedEcho.py"]:
                if config["source_sha256"][key] != pure_config["source_sha256"][key]:
                    raise RuntimeError(f"Core source differs: {key}")
            if config["input_hflip"] != 0 or "Random" in config["train_transform"]:
                raise RuntimeError("Input augmentation must remain OFF")
            if config["view_mode"] != "rand":
                raise RuntimeError("Wrong model-side policy")
            alt = ROOT/"results/fedecho_model_view_seed1/FedEchoView"
            compare_prefix(read_csv(alt/"client_events.csv"), read_csv(gates/"continuous/client_events.csv"),
                           ["sgd_rng_sha256"], 80)
            alt_config = json.loads((alt/"config.json").read_text())
            if config["source_sha256"]["experiments/fedecho_model_view.py"] != alt_config["source_sha256"]["experiments/fedecho_model_view.py"]:
                raise RuntimeError("Training engine differs from alternating-view arm")
            atomic_json(gate_path, {"passed": True, "source_sha256": fingerprints,
                        "input_augmentation_off": True, "same_preprocessing_as_pure": True,
                        "same_training_engine_as_alt": True, "sgd_rng_matches_alt": True,
                        "identity_vs_pure_exact_rounds": 3, "resume_exact_rounds": 8,
                        "original_aggregation_exact": True, "same_paper_core_sources": True,
                        "active_view_smoke": active, "view_equivariance_error": config["view_equivariance_error"],
                        "control_exercised": active["controlled_rounds"] > 0})
        else:
            gate = json.loads(gate_path.read_text())
            if not gate.get("passed") or gate["source_sha256"] != fingerprints:
                raise RuntimeError("Code changed or gates absent")
        manifest["status"] = "formal_running"
        atomic_json(manifest_path, manifest)
        destination = output/"FedEchoRandomView"
        for attempt in range(3):
            status_path = destination/"status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("status") == "completed" and status.get("completed_rounds") == 500:
                break
            try:
                extra = ["--resume"] if (destination/"config.json").exists() else []
                run_job(destination, 500, f"formal.attempt{attempt+1}", extra)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                manifest.setdefault("retries", []).append({"attempt": attempt+1, "error": traceback.format_exc()})
                atomic_json(manifest_path, manifest)
                time.sleep(5)
        echo = json.loads((destination/"summary.json").read_text())
        if echo["completed_rounds"] != 500 or sha_file(baseline_path) != baseline_hash:
            raise RuntimeError("Incomplete output or baseline log changed")
        all_rows = read_csv(destination/"round_metrics.csv")
        summaries = {"FedPhoenix_historical": historical_summary(baseline, echo.keys()), "FedEcho_random_model_view": echo}
        trajectories = {"FedPhoenix historical": [float(r["test_accuracy"]) for r in baseline],
                        "FedEcho + random model view": [float(r["test_accuracy"]) for r in all_rows]}
        if (pure/"summary.json").exists():
            pure_rows = read_csv(pure/"round_metrics.csv")
            if len(pure_rows) >= 500:
                base = historical_summary(pure_rows[:500], echo.keys())
                summaries["FedEcho_pure"] = base
                trajectories["FedEcho pure"] = [float(r["test_accuracy"]) for r in pure_rows[:500]]
                compare_prefix(pure_rows, all_rows, ["selected_clients"], 500)
        alt_path = ROOT/"results/fedecho_model_view_seed1/FedEchoView"
        if (alt_path/"summary.json").exists():
            alt_summary = json.loads((alt_path/"summary.json").read_text())
            if alt_summary["completed_rounds"] == 500:
                alt_rows = read_csv(alt_path/"round_metrics.csv")
                summaries["FedEcho_alternating_model_view"] = alt_summary
                trajectories["FedEcho + alternating model view"] = [float(r["test_accuracy"]) for r in alt_rows]
                compare_prefix(alt_rows, all_rows, ["selected_clients"], 500)
        gain = echo["mean_rounds_1_500"]-summaries["FedPhoenix_historical"]["mean_rounds_1_500"]
        report = {"seed": 1, "rounds": 500, "summaries": summaries,
                  "gain_vs_historical_FedPhoenix_pp": gain,
                  "minimum_target_met": gain >= 1, "desired_target_met": gain >= 2,
                  "baseline": manifest["baseline"],
                  "interpretation": "Augmented variant, not unmodified FedEcho. Report history-control activity separately."}
        atomic_json(output/"comparison.json", report)
        with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["method", *echo.keys()])
            writer.writeheader()
            for name, value in summaries.items():
                writer.writerow({"method": name, **value})
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, accuracy in trajectories.items():
            line, = ax.plot(range(1, 501), accuracy, alpha=.15, linewidth=.7)
            smooth = np.convolve(accuracy, np.ones(20)/20, mode="valid")
            ax.plot(range(20, 501), smooth, label=name, color=line.get_color(), linewidth=1.5)
        ax.set(xlabel="Communication round", ylabel="Test accuracy (%)", title="Seed 1; 20-round moving mean")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/"accuracy_curve.png", dpi=180)
        plt.close(fig)
        (output/"RESULTS_SUMMARY.md").write_text(
            "# FedEcho core + random MODEL view, seed 1; input augmentation OFF\n\n"
            f"First-500 mean gain vs historical FedPhoenix: {gain:+.6f} pp. "
            f"Minimum +1 pp criterion {'MET' if gain >= 1 else 'NOT MET'}.\n\n"
            f"Nonzero control in {echo['controlled_rounds']}/500 rounds; "
            f"mean model-F fraction {echo['mean_flip_fraction']:.6f}; schedule errors {echo['alt_violations']}.\n\n"
            "All input preprocessing is ToTensor/Normalize only (input_hflip=0). "
            "The same paper core and model-view training engine are reused, changing only model-view scheduling. "
            "Accuracy gains alone do not establish that historical prediction adds value over augmentation. "
            "The historical FedPhoenix log lacks saved sampling/config metadata. "
            "No further experiment is automatically scheduled. See comparison.json/csv and curve.\n",
            encoding="utf-8")
        manifest["status"] = "completed"
        atomic_json(manifest_path, manifest)
    except BaseException:
        if current is not None and current.poll() is None:
            current.terminate()
            current.wait()
        manifest["active"] = []
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        atomic_json(manifest_path, manifest)
        raise
    finally:
        for f in handles:
            f.close()


if __name__ == "__main__":
    main()
