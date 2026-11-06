"""Audited paper-only FedEcho, one fixed seed/setting and 1000 rounds.

At most two GPU workers are used for correctness gates; only one formal worker
is launched. FedPhoenix is read from its existing log and is never trained.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fedecho_reproduction import atomic_json, output_lock, sha_file
from scripts.run_fedecho_reproduction import read_baseline_log


SOURCES = ["Algorithm/FedEcho.py", "experiments/fedecho_reproduction.py", "main_fedecho.py",
           "scripts/run_fedecho_paper.py", "models/Nets.py", "models/Update.py", "models/Fed.py",
           "models/test.py", "utils/get_dataset.py", "utils/set_seed.py", "utils/dataset_utils.py",
           "utils/sampling.py", "tests/test_fedecho.py", "tests/test_fedecho_paper.py"]


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def exact(left, right, fields, count):
    if len(left) != count or len(right) != count:
        raise RuntimeError(f"Incorrect parity row count, expected {count}")
    for i, (a, b) in enumerate(zip(left, right), 1):
        for field in fields:
            if a[field] != b[field]:
                raise RuntimeError(f"Exact parity mismatch: row {i}, {field}")


def metrics(rows):
    accuracy = [float(r["test_accuracy"]) for r in rows]
    count = len(accuracy)
    if not count:
        raise ValueError("No observations")
    def window(lo, hi):
        return sum(accuracy[lo-1:hi]) / (hi-lo+1) if count >= hi else None
    result = {"rounds": count, "peak": max(accuracy), "peak_round": accuracy.index(max(accuracy))+1,
              "final": accuracy[-1], "mean_1_500": window(1, 500), "mean_501_1000": window(501, 1000),
              "mean_151_770": window(151, 770), "mean_901_1000": window(901, 1000)}
    for n in (20, 50, 100):
        result[f"last{n}"] = sum(accuracy[-n:]) / n if count >= n else None
    for n in (5, 10):
        result[f"best_rolling{n}"] = max(sum(accuracy[i-n:i])/n for i in range(n, count+1)) if count >= n else None
    for lo in range(1, 1001, 100):
        result[f"mean_{lo}_{lo+99}"] = window(lo, lo+99)
    return result


def report(output, baseline_path):
    """Reporting uses the fixed horizon, never a later historical peak."""
    rows = read_rows(output / "FedEcho/round_metrics.csv")
    if len(rows) != 1000 or [int(r["round"]) for r in rows] != list(range(1, 1001)):
        raise RuntimeError("Incomplete/duplicate formal horizon")
    baseline = read_baseline_log(baseline_path, 1000)
    comparison = {"FedEcho": metrics(rows), "FedPhoenix_historical": metrics(baseline)}
    comparison["delta_pp"] = {key: comparison["FedEcho"][key] - comparison["FedPhoenix_historical"][key]
                              for key in comparison["FedEcho"] if key not in ("rounds", "peak_round")}
    rho = [float(r["rho"]) for r in rows]
    nonzero = [int(r["round"]) for r in rows if float(r["max_offset_ratio"]) > 0]
    comparison["mechanism"] = {
        "rank": 2, "nonzero_dispatch_rounds": len(nonzero),
        "last_nonzero_dispatch_round": nonzero[-1] if nonzero else None,
        "rho_positive_rounds": sum(x > 0 for x in rho), "rho_mean": sum(rho)/len(rho),
        "max_offset_over_trajectory_norm": max(float(r["max_offset_ratio"]) for r in rows),
        "max_weighted_consensus_residual": max(float(r["consensus_residual"]) for r in rows),
        "max_response_center_residual": max(float(r["response_center_residual"]) for r in rows),
        "response_pairs": int(rows[-1]["response_pairs"]),
        "max_gpu_allocated_bytes": max(int(r["gpu_peak_bytes"]) for r in rows),
        "runtime_seconds_sum_rounds": sum(float(r["round_seconds"]) for r in rows)}
    comparison["scope"] = "One seed; historical comparator has no verified paired task/RNG traces. No significance claim."
    comparison["baseline_sha256"] = sha_file(baseline_path)
    comparison["performance_target_first500_plus1pp_met"] = comparison["delta_pp"]["mean_1_500"] >= 1
    atomic_json(output / "comparison.json", comparison)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *comparison["FedEcho"]])
        writer.writeheader()
        writer.writerows({"method": name, **comparison[name]} for name in ("FedEcho", "FedPhoenix_historical"))
    # A standalone scientific line chart without a plotting/GUI dependency.
    points = lambda rs: " ".join(f"{60+(int(r['round'])-1)*.78:.2f},{430-float(r['test_accuracy'])*4:.2f}" for r in rs)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">',
           '<rect width="900" height="500" fill="white"/>',
           '<text x="60" y="25" font-size="17">FedEcho / VGG / CIFAR-10 / alpha=0.3 / seed=1</text>']
    for value in range(0, 101, 20):
        y = 430-value*4
        svg += [f'<line x1="60" y1="{y}" x2="840" y2="{y}" stroke="#ddd"/>',
                f'<text x="25" y="{y+5}" font-size="12">{value}</text>']
    for value in (1, 200, 400, 600, 800, 1000):
        x = 60+(value-1)*.78
        svg.append(f'<text x="{x-10}" y="452" font-size="12">{value}</text>')
    for rs, color in ((baseline, "#e28b22"), (rows, "#1769aa")):
        svg.append(f'<polyline points="{points(rs)}" fill="none" stroke="{color}" stroke-width="1"/>')
    svg += ['<text x="60" y="477" fill="#1769aa">FedEcho (paper only)</text>',
            '<text x="310" y="477" fill="#b36b16">FedPhoenix (historical, rounds 1-1000)</text>', '</svg>']
    (output / "accuracy_curve.svg").write_text("\n".join(svg), encoding="utf-8")
    lines = ["# FedEcho paper reproduction", "", comparison["scope"], "",
             "Fixed: seed=1, VGG, CIFAR-10, existing alpha=0.3 partition, 1000 rounds; no augmentation or reset.", "",
             "| Metric | FedEcho | Historical FedPhoenix | Difference (pp) |", "|---|---:|---:|---:|"]
    for key in comparison["FedEcho"]:
        lines.append(f"| {key} | {comparison['FedEcho'][key]:.6f} | {comparison['FedPhoenix_historical'][key]:.6f} | {comparison['delta_pp'].get(key, 0):.6f} |")
    lines += ["", "Mechanism diagnostics:", "", "```json", json.dumps(comparison["mechanism"], indent=2), "```", "",
              "The primary practical comparison is mean accuracy in rounds 1-500, with late-window and full-horizon metrics reported separately.",
              "A fitted rho of zero is an allowed paper outcome. It is not replaced by a floor or a selected recent-history window.",
              f"First-500 mean gain >=1pp: {comparison['performance_target_first500_plus1pp_met']}.",
              "This run tests a faithful algorithm implementation; it does not guarantee reproduction of the paper's reported accuracy."]
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/fedecho_paper_seed1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--gates_only", action="store_true")
    args = parser.parse_args()
    if args.max_retries < 0 or args.gpu < 0:
        parser.error("A GPU and nonnegative retry count are required")
    output = args.output_dir.resolve()
    with output_lock(output, "launcher.lock"):
        orchestrate(args, output)


def orchestrate(args, output):
    baseline_path = ROOT / "result_other_method/cifar10/0.3/vgg_FedPhoenix.log"
    read_baseline_log(baseline_path, 1000)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    hashes = {name: sha_file(ROOT/name) for name in SOURCES}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        if original["source_sha256"] != hashes:
            raise RuntimeError("Source changed: use a new output directory; do not mix trajectories")
        # A worker can outlive a launcher crash. Do not restart it.
        import psutil
        for item in original.get("active", []):
            if psutil.pid_exists(item["pid"]):
                process = psutil.Process(item["pid"])
                if str(output) in " ".join(process.cmdline()):
                    raise RuntimeError(f"Existing worker {item['pid']} still alive; do not duplicate")
        manifest = original
    else:
        manifest = {"protocol": "paper FedEcho; VGG/CIFAR10/alpha0.3/seed1/1000 rounds; no augmentation/reset/views",
                    "source_sha256": hashes, "baseline_sha256": sha_file(baseline_path),
                    "active": [], "attempts": [], "status": "gates"}
    manifest["launcher_pid"] = os.getpid()
    atomic_json(manifest_path, manifest)
    live = []

    def start(destination, epochs, tag, algorithm="FedEcho", extra=(), paper=False):
        script = "main_fedecho.py" if paper else "experiments/fedecho_reproduction.py"
        command = [sys.executable, "-u", "-X", "utf8", str(ROOT/script), "--epochs", str(epochs),
                   "--gpu", str(args.gpu), "--output_dir", str(destination)]
        if not paper:
            command += ["--algorithm", algorithm]
        command += list(extra)
        stdout = (logs/f"{tag}.stdout.log").open("a", encoding="utf-8")
        stderr = (logs/f"{tag}.stderr.log").open("a", encoding="utf-8")
        try:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        finally:
            stdout.close()
            stderr.close()
        item = {"process": child, "pid": child.pid, "tag": tag, "command": command}
        live.append(item)
        manifest["active"] = [{k: v for k, v in x.items() if k != "process"} for x in live]
        atomic_json(manifest_path, manifest)
        print(f"START {tag} pid={child.pid}", flush=True)
        return item

    def wait(items):
        while any(x["process"].poll() is None for x in items):
            if any(x["process"].poll() not in (None, 0) for x in items):
                for x in items:
                    if x["process"].poll() is None:
                        x["process"].terminate()
            time.sleep(1)
        failed = []
        for item in items:
            code = item["process"].returncode
            manifest["attempts"].append({"tag": item["tag"], "return_code": code})
            live.remove(item)
            if code:
                failed.append(item["tag"])
        manifest["active"] = [{k: v for k, v in x.items() if k != "process"} for x in live]
        atomic_json(manifest_path, manifest)
        if failed:
            raise RuntimeError(f"Worker failed: {failed}; inspect matching stderr")

    def extend(path, horizon, tag, **kwargs):
        status = json.loads((path/"status.json").read_text()) if (path/"status.json").exists() else {}
        if status.get("status") == "completed" and status.get("completed_rounds", 0) >= horizon:
            return None
        extra = list(kwargs.pop("extra", ()))
        if (path/"config.json").exists():
            if not (path/"checkpoint.pt").exists():
                raise RuntimeError(f"No recoverable checkpoint in {path}; preserve and inspect")
            extra.append("--resume")
        return start(path, horizon, tag, extra=extra, **kwargs)

    try:
        gate_report = output / "gate_report.json"
        if not gate_report.exists():
            manifest["status"] = "gates"
            atomic_json(manifest_path, manifest)
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            with (logs/"regression.log").open("w", encoding="utf-8") as handle:
                result = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", str(ROOT/"tests")],
                                        cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("Regression failed; inspect regression.log")
            regression_text = (logs/"regression.log").read_text(encoding="utf-8")
            count = int(re.search(r"Ran (\d+) tests?", regression_text).group(1))
            gates = output / "gates"
            jobs = [extend(gates/"FedAvg", 3, "avg3", algorithm="FedAvg", extra=["--check_aggregation"]),
                    extend(gates/"disabled", 3, "disabled3", extra=["--disable_control", "--check_aggregation"])]
            wait([x for x in jobs if x])
            exact(read_rows(gates/"FedAvg/round_metrics.csv"), read_rows(gates/"disabled/round_metrics.csv"),
                  ["selected_clients", "global_state_sha256", "test_accuracy"], 3)
            exact(read_rows(gates/"FedAvg/client_events.csv"), read_rows(gates/"disabled/client_events.csv"),
                  ["sgd_rng_sha256", "client_id", "aggregation_weight"], 30)
            full = extend(gates/"active10", 10, "active10", paper=True, extra=["--check_aggregation"])
            short = extend(gates/"resume", 5, "smoke5", paper=True, extra=["--check_aggregation"])
            wait([short] if short else [])
            resumed = extend(gates/"resume", 10, "resume10", paper=True)
            wait([x for x in [full, resumed] if x])
            exact(read_rows(gates/"active10/round_metrics.csv"), read_rows(gates/"resume/round_metrics.csv"),
                  ["selected_clients", "global_state_sha256", "test_accuracy", "rho", "response_pairs"], 10)
            exact(read_rows(gates/"active10/client_events.csv"), read_rows(gates/"resume/client_events.csv"),
                  ["sgd_rng_sha256", "prediction", "offset_coefficients", "observed_response", "raw_update_projection"], 100)
            active = read_rows(gates/"active10/round_metrics.csv")
            if not any(float(r["max_offset_ratio"]) > 0 for r in active):
                raise RuntimeError("Active smoke did not exercise dispatch")
            config = json.loads((gates/"active10/config.json").read_text())
            if "Random" in config["train_transform"] or config["input_hflip"] != 0:
                raise RuntimeError("Unexpected augmentation")
            archive = ROOT/"results/fedecho_reproduction_seed1/FedEcho/round_metrics.csv"
            legacy_match = None
            if archive.exists():
                exact(read_rows(archive)[:10], active, ["global_state_sha256", "selected_clients", "rho"], 10)
                legacy_match = True
            atomic_json(gate_report, {"passed": True, "source_sha256": hashes, "regression_tests": count,
                        "disabled_FedAvg_exact_rounds": 3, "smoke_rounds": 5, "active_smoke_rounds": 10,
                        "checkpoint_resume_exact_rounds": 10, "original_aggregation_exact": True,
                        "input_augmentation": False, "legacy_VGG_prefix_exact": legacy_match,
                        "formal_baseline_retraining": False})
        else:
            gate = json.loads(gate_report.read_text())
            if not gate["passed"] or gate["source_sha256"] != hashes:
                raise RuntimeError("Stale gates")
        if args.gates_only:
            manifest["status"] = "gates_passed"
            atomic_json(manifest_path, manifest)
            return
        manifest["status"] = "formal_running"
        atomic_json(manifest_path, manifest)
        for attempt in range(args.max_retries+1):
            try:
                job = extend(output/"FedEcho", 1000, f"formal_attempt{attempt+1}", paper=True)
                wait([job] if job else [])
                break
            except RuntimeError:
                if attempt == args.max_retries:
                    raise
                print("RETRY from latest exact checkpoint; protocol unchanged", flush=True)
        manifest["status"] = "reporting"
        atomic_json(manifest_path, manifest)
        report(output, baseline_path)
        manifest["status"] = "completed"
        atomic_json(manifest_path, manifest)
        print("COMPLETED paper FedEcho and historical comparison", flush=True)
    except BaseException:
        manifest["status"] = "failed"
        manifest["traceback"] = traceback.format_exc()
        atomic_json(manifest_path, manifest)
        raise
    finally:
        for item in live:
            if item["process"].poll() is None:
                item["process"].terminate()
                item["process"].wait()


if __name__ == "__main__":
    main()
