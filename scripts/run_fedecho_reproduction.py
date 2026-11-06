"""FedEcho-only reproduction against an existing FedPhoenix log. No tuning."""
import argparse
import csv
import hashlib
import json
import os
import re
from statistics import mean
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fedecho_reproduction import atomic_json, sha_file


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_baseline_log(path, horizon=500):
    """Use exactly rounds 1..horizon; never use a later logged peak."""
    observed = {}
    pattern = re.compile(r"^ROUND_ACCURACY method=FedPhoenix round=(\d+) accuracy=([\d.eE+-]+)$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            number, value = int(match[1]), float(match[2])
            if number in observed:
                raise ValueError(f"Duplicate baseline round {number}")
            if not 0 <= value <= 100:
                raise ValueError(f"Invalid baseline accuracy at round {number}")
            observed[number] = value
    missing = [i for i in range(1, horizon+1) if i not in observed]
    if missing:
        raise ValueError(f"Baseline is missing rounds: {missing[:10]}")
    return [{"round": i, "test_accuracy": observed[i]} for i in range(1, horizon+1)]


def historical_summary(rows, keys):
    accuracy = [float(row["test_accuracy"]) for row in rows]
    result = dict.fromkeys(keys)
    result.update(completed_rounds=len(rows), peak_accuracy=max(accuracy),
                  peak_round=accuracy.index(max(accuracy))+1,
                  last_observed_accuracy=accuracy[-1], mean_rounds_1_500=mean(accuracy),
                  mean_rounds_401_500=mean(accuracy[400:500]))
    for n in [5, 10, 20, 50, 100]:
        result[f"last{n}_mean"] = mean(accuracy[-n:])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, default=ROOT/"results/fedecho_reproduction_seed1")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip_gates", action="store_true")
    p.add_argument("--baseline_log", type=Path,
                   default=ROOT/"result_other_method/cifar10/0.3/vgg_FedPhoenix.log")
    p.add_argument("--adopt_pid", type=int,
                   help="Attach to the verified existing FedEcho worker without interrupting it")
    args = p.parse_args()
    if args.adopt_pid and not args.skip_gates:
        p.error("Adopting a running worker requires already-passed gates and --skip_gates")
    baseline_path = args.baseline_log.resolve()
    baseline_rows = read_baseline_log(baseline_path)
    baseline_hash = sha_file(baseline_path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output/"logs"
    logs.mkdir(exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in old.get("active", []):
            # Never duplicate a live launcher/training job, even on manual resume.
            import psutil
            if psutil.pid_exists(item["pid"]) and item["pid"] != args.adopt_pid:
                raise RuntimeError(f"Inspect live process before resume: {item}")
        if old.get("status") == "completed":
            return
    manifest = {"status": "gates", "pid": os.getpid(), "active": [], "attempts": [],
                "protocol": "FedEcho only: VGG/CIFAR10/alpha0.3/seed1/500 rounds; rank2; no reset",
                "baseline": {"path": str(baseline_path), "sha256": baseline_hash,
                    "source": "existing historical log, rounds 1-500; no baseline training",
                    "paired_client_sampling_verified": False}}
    handles, live = [], []
    adopted = None
    if args.adopt_pid:
        import psutil
        adopted = psutil.Process(args.adopt_pid)
        command = adopted.cmdline()
        if (str(ROOT/"experiments/fedecho_reproduction.py") not in command
                or "--algorithm" not in command
                or command[command.index("--algorithm")+1] != "FedEcho"
                or str(output/"FedEcho") not in command):
            raise RuntimeError("Adopted PID is not this output directory's FedEcho worker")
        manifest["active"] = [{"pid": adopted.pid, "tag": "adopted_FedEcho", "command": command}]

    def launch(algorithm, destination, epochs, tag, extra=()):
        command = [sys.executable, "-u", "-X", "utf8", str(ROOT/"experiments/fedecho_reproduction.py"),
                   "--algorithm", algorithm, "--epochs", str(epochs), "--gpu", str(args.gpu),
                   "--output_dir", str(destination), *extra]
        stdout = (logs/f"{tag}.stdout.log").open("a", encoding="utf-8")
        stderr = (logs/f"{tag}.stderr.log").open("a", encoding="utf-8")
        handles.extend([stdout, stderr])
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        item = {"process": child, "pid": child.pid, "tag": tag, "command": command}
        live.append(item)
        manifest["active"] = [{k: v for k, v in x.items() if k != "process"} for x in live]
        atomic_json(manifest_path, manifest)
        print(f"LAUNCHED {tag} pid={child.pid}", flush=True)
        return item

    def wait(items):
        while any(x["process"].poll() is None for x in items):
            # An early failure stops its paired gate immediately.
            for x in items:
                if x["process"].poll() not in (None, 0):
                    for other in items:
                        if other["process"].poll() is None:
                            other["process"].terminate()
                    for other in items:
                        other["process"].wait()
                    raise RuntimeError(f"Run failed: {x['tag']}; inspect corresponding stderr")
            time.sleep(2)
        for x in items:
            code = x["process"].returncode
            manifest["attempts"].append({"tag": x["tag"], "return_code": code})
            live.remove(x)
            if code:
                raise RuntimeError(f"Run failed: {x['tag']} return code={code}")
        manifest["active"] = [{k: v for k, v in x.items() if k != "process"} for x in live]
        atomic_json(manifest_path, manifest)

    def check_equal(left, right, fields, rounds):
        if len(left) != rounds or len(right) != rounds:
            raise RuntimeError("Incorrect gate horizon")
        for i in range(rounds):
            for field in fields:
                if left[i][field] != right[i][field]:
                    raise RuntimeError(f"Gate mismatch round {i+1}: {field}")

    try:
        code_hashes = {f: sha_file(ROOT/f) for f in ["Algorithm/FedEcho.py", "experiments/fedecho_reproduction.py"]}
        gate_path = output/"gate_report.json"
        if not args.skip_gates:
            with (logs/"regression.log").open("w", encoding="utf-8") as f:
                env = dict(os.environ, PYTHONPATH=str(ROOT))
                result = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", str(ROOT/"tests")],
                                        cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError("Regression gate failed")
            gates = output/"gates"
            avg = launch("FedAvg", gates/"FedAvg", 3, "parity_avg", ["--check_aggregation"])
            disabled = launch("FedEcho", gates/"FedEcho_disabled", 3, "parity_disabled",
                              ["--disable_control", "--check_aggregation"])
            wait([avg, disabled])
            check_equal(read_csv(gates/"FedAvg/round_metrics.csv"),
                        read_csv(gates/"FedEcho_disabled/round_metrics.csv"),
                        ["global_state_sha256", "selected_clients", "test_accuracy"], 3)
            check_equal(read_csv(gates/"FedAvg/client_events.csv"),
                        read_csv(gates/"FedEcho_disabled/client_events.csv"), ["sgd_rng_sha256"], 30)
            full = launch("FedEcho", gates/"FedEcho_active", 20, "active20", ["--check_aggregation"])
            short = launch("FedEcho", gates/"FedEcho_resume", 7, "resume7")
            wait([short])
            resumed = launch("FedEcho", gates/"FedEcho_resume", 20, "resume20", ["--resume"])
            wait([full, resumed])
            check_equal(read_csv(gates/"FedEcho_active/round_metrics.csv"),
                        read_csv(gates/"FedEcho_resume/round_metrics.csv"),
                        ["global_state_sha256", "selected_clients", "test_accuracy", "rho", "response_pairs"], 20)
            check_equal(read_csv(gates/"FedEcho_active/client_events.csv"),
                        read_csv(gates/"FedEcho_resume/client_events.csv"),
                        ["sgd_rng_sha256", "prediction", "offset_coefficients", "observed_response"], 200)
            active_summary = json.loads((gates/"FedEcho_active/summary.json").read_text())
            if active_summary["controlled_rounds"] < 1:
                raise RuntimeError("20-round smoke did not exercise active dispatch; inspect before formal training")
            report = {"passed": True, "source_sha256": code_hashes, "regression": "passed",
                      "disabled_vs_FedAvg_exact_rounds": 3, "active_smoke_rounds": 20,
                      "active_smoke_summary": active_summary, "resume_exact_rounds": 20,
                      "original_aggregation_exact": True,
                      "baseline_policy": "read existing log only"}
            atomic_json(gate_path, report)
        else:
            report = json.loads(gate_path.read_text())
            if not report.get("passed") or report["source_sha256"] != code_hashes:
                raise RuntimeError("Gates absent or code changed; skipping gates is not safe")
        manifest["status"] = "formal_running"
        atomic_json(manifest_path, manifest)
        if adopted is not None:
            print(f"ADOPTED FedEcho pid={adopted.pid}; baseline is read-only", flush=True)
            while adopted.is_running():
                status_file = output/"FedEcho/status.json"
                status = json.loads(status_file.read_text()) if status_file.exists() else {}
                if status.get("status") == "completed" and status.get("completed_rounds") == 500:
                    break
                time.sleep(2)
            manifest["active"] = []
            atomic_json(manifest_path, manifest)
        # Only FedEcho is launched or resumed. FedPhoenix is never started.
        serial = False
        for attempt in range(3):
            formal = []
            try:
                for name in ["FedEcho"]:
                    dest = output/name
                    if (dest/"status.json").exists():
                        status = json.loads((dest/"status.json").read_text())
                        if status["status"] == "completed" and status["completed_rounds"] == 500:
                            continue
                    extra = ["--resume"] if (dest/"config.json").exists() else []
                    item = launch(name, dest, 500, f"formal_{name}.attempt{attempt+1}", extra)
                    formal.append(item)
                    if serial:
                        wait([item])
                if not serial:
                    wait(formal)
                break
            except RuntimeError:
                for item in list(live):
                    if item["process"].poll() is None:
                        item["process"].terminate()
                        item["process"].wait()
                    live.remove(item)
                manifest["active"] = []
                errors = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                                   for path in logs.glob(f"formal_*.attempt{attempt+1}.stderr.log"))
                if "out of memory" in errors.lower():
                    serial = True
                manifest.setdefault("retries", []).append({"attempt": attempt+1,
                    "error": traceback.format_exc(), "next_execution": "serial" if serial else "parallel"})
                atomic_json(manifest_path, manifest)
                if attempt == 2:
                    raise
        echo_rows = read_csv(output/"FedEcho/round_metrics.csv")
        if len(echo_rows) != 500 or sha_file(baseline_path) != baseline_hash:
            raise RuntimeError("Incomplete FedEcho output or historical baseline log changed")
        echo_summary = json.loads((output/"FedEcho/summary.json").read_text())
        summaries = {"FedEcho": echo_summary,
                     "FedPhoenix": historical_summary(baseline_rows, echo_summary.keys())}
        with (output/"historical_FedPhoenix_first500.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "test_accuracy"])
            writer.writeheader()
            writer.writerows(baseline_rows)
        gain = summaries["FedEcho"]["mean_rounds_1_500"] - summaries["FedPhoenix"]["mean_rounds_1_500"]
        comparison = {"status": "completed", "seed": 1, "rounds": 500, "summaries": summaries,
                      "gain_first500_pp": gain, "minimum_target_met": gain >= 1., "desired_target_met": gain >= 2.,
                      "baseline": manifest["baseline"]}
        atomic_json(output/"comparison.json", comparison)
        with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["algorithm", *summaries["FedEcho"].keys()])
            writer.writeheader()
            for name, summary in summaries.items():
                writer.writerow({"algorithm": name, **summary})
        (output/"RESULTS_SUMMARY.md").write_text(
            f"# FedEcho reproduction, seed 1, 500 rounds\n\n"
            f"Mean accuracy 1--500: FedPhoenix {summaries['FedPhoenix']['mean_rounds_1_500']:.6f}%, "
            f"FedEcho {summaries['FedEcho']['mean_rounds_1_500']:.6f}%. Gain {gain:+.6f} pp.\n\n"
            f"FedPhoenix source: existing `{baseline_path}` (first 500 rounds), SHA256 `{baseline_hash}`. "
            "Only this historical log is used for the comparison; the partial fresh baseline was stopped "
            "at the user's request and excluded. The historical log does not contain "
            "client selections or full configuration, so paired RNG/partition identity is not verified.\n\n"
            f"Minimum +1 pp criterion {'MET' if gain >= 1 else 'NOT MET'}. "
            "No additional training or search is automatically scheduled. Full metrics are in comparison.json/csv.\n",
            encoding="utf-8")
        manifest["status"] = "completed"
        atomic_json(manifest_path, manifest)
    except BaseException:
        for item in live:
            if item["process"].poll() is None:
                item["process"].terminate()
                item["process"].wait()
        manifest["active"] = ([{"pid": adopted.pid, "tag": "adopted_FedEcho"}]
                              if adopted is not None and adopted.is_running() else [])
        manifest["status"] = "failed"
        manifest["traceback"] = traceback.format_exc()
        atomic_json(manifest_path, manifest)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
