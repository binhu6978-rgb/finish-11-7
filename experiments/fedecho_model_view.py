"""FedEcho core with client-local alternating model-view augmentation, seed 1.

Run from any directory. The original loaders/model/local SGD are imported, not
rewritten. Streaming aggregation is algebraically/bitwise checked against the
project's Aggregation in --check_aggregation smoke runs.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy
import torch
import torchvision
from Algorithm.FedEcho import FedEchoController, make_basis, project_update
from Algorithm.FedEchoModelView import train_in_model_view
from Algorithm.SymmetryView import ViewScheduler, verify_flip_equivariance
from models.Fed import Aggregation
from models.Nets import VGG16
from models.Update import LocalUpdate_FedAvg
from models.test import test_img
from utils.get_dataset import get_dataset
from utils.set_seed import set_random_seed


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def aggregate_add(accumulator, state, count):
    if accumulator is None:
        return {k: v.detach().clone() * count for k, v in state.items()}
    for name, value in state.items():
        accumulator[name] += value.detach() * count
    return accumulator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(algorithm="FedEchoView")
    p.add_argument("--view_mode", choices=["none", "alt"], default="alt")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--disable_control", action="store_true", help="Parity test only")
    p.add_argument("--check_aggregation", action="store_true", help="Memory-intensive smoke audit")
    # Deliberately fixed protocol: no hyperparameter search and no extra seeds.
    p.set_defaults(dataset="cifar10", model="vgg", seed=1, num_users=100,
                   frac=0.1, local_ep=5, local_bs=50, bs=256, optimizer="sgd",
                   lr=0.01, momentum=0.5, weight_decay=0.0, iid=0, noniid_case=5,
                   data_beta=0.3, generate_data=0, input_hflip=0,
                   num_channels=3, num_classes=10, num_workers=0,
                   use_project_head=0, out_dim=256, verbose=False, FP_conv=1000, FP_fc=0,
                   reset=0.015625, remethod="ori_normal")
    args = p.parse_args()
    if args.epochs < 1 or args.checkpoint_every < 1:
        p.error("Positive round counts required")
    args.output_dir = args.output_dir.resolve()
    return args


def summarize(rows):
    if not rows:
        return {"completed_rounds": 0}
    accuracies = np.array([r["test_accuracy"] for r in rows])
    def window(lo, hi):
        return float(np.mean(accuracies[lo-1:hi])) if len(rows) >= hi else None
    return {"completed_rounds": len(rows), "peak_accuracy": float(accuracies.max()),
            "peak_round": int(accuracies.argmax()+1), "last_observed_accuracy": float(accuracies[-1]),
            "mean_rounds_1_500": window(1, 500), "mean_rounds_401_500": window(401, 500),
            "mean_rounds_151_770": window(151, 770), "mean_rounds_901_1000": window(901, 1000),
            **{f"last{n}_mean": float(np.mean(accuracies[-n:])) if len(rows) >= n else None
               for n in [5, 10, 20, 50, 100]},
            "controlled_rounds": sum(r["max_offset_ratio"] > 0 for r in rows),
            "max_offset_ratio": max(r["max_offset_ratio"] for r in rows),
            "max_consensus_residual": max(r["consensus_residual"] for r in rows),
            "max_gpu_allocated_bytes": max(r["gpu_peak_bytes"] for r in rows),
            "mean_flip_fraction": float(np.mean([r["flip_fraction"] for r in rows])),
            "alt_violations": sum(r["alt_violations"] for r in rows)}


def restore_csv(path, rows, fields):
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    handle.flush()
    return handle, writer


ROUND_FIELDS = ["round", "algorithm", "test_accuracy", "round_seconds", "selected_clients",
                "global_state_sha256", "rho", "valid_components", "trajectory_norm",
                "history_coordinates", "response_pairs", "max_offset_ratio", "consensus_residual",
                "prediction_mse", "zero_prediction_mse", "gpu_peak_bytes", "views", "flip_fraction", "alt_violations"]
EVENT_FIELDS = ["round", "client_id", "task_id", "sgd_rng_sha256", "reset_seed",
                "components", "prediction", "offset_coefficients", "observed_response",
                "current_view", "previous_view", "visit_index"]


def run(args):
    os.chdir(ROOT)
    output = args.output_dir
    if (output / "config.json").exists() and not args.resume:
        raise RuntimeError("Output already exists. Use --resume; existing results are protected.")
    output.mkdir(parents=True, exist_ok=True)
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    set_random_seed(args.seed)
    train, test, users = get_dataset(args)
    model = VGG16(args).to(args.device)
    controller = FedEchoController(rank=2, epsilon=1e-8)
    controller.view_scheduler = ViewScheduler(args.view_mode, args.seed)
    equivariance_error = verify_flip_equivariance(model)
    partition = ROOT / "data/cifar10_100_noniidCase5_beta0.3.json"
    sources = ["Algorithm/FedEcho.py", "experiments/fedecho_reproduction.py",
               "Algorithm/FedEchoModelView.py", "Algorithm/SymmetryView.py",
               "experiments/fedecho_model_view.py", "models/Nets.py", "models/Update.py",
               "models/Fed.py", "models/test.py", "utils/get_dataset.py", "utils/set_seed.py"]
    config = {**vars(args), "view_equivariance_error": equivariance_error,
              "method": "FedEcho rank2 + client-local I/F model view; no reset",
              "response_coordinates": "canonical after inverse view, actual start x_i",
              "source_sha256": {f: sha_file(ROOT/f) for f in sources},
              "partition_sha256": sha_file(partition), "initial_state_sha256": state_hash(model.state_dict()),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "scipy": scipy.__version__, "python": sys.version,
              "cuda": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(args.gpu) if args.gpu >= 0 else "CPU",
              "train_transform": str(train.transform), "test_transform": str(test.transform),
              "train_samples": len(train), "test_samples": len(test),
              "client_sizes": {int(i): len(indices) for i, indices in users.items()},
              "numerical_conventions": {"rank": 2, "epsilon_trajectory": 1e-8,
                  "response_scope": "trainable parameters; original BN buffer aggregation",
                  "rho_solver": "1025 fixed brackets, bounded refinement, endpoints",
                  "dispatch_solver": "joint unit-ball dual, consensus tolerance 1e-9",
                  "sgd_randomness": "original global RNG stream, state hash per participation",
                  "failure_policy": "raise; never silently skip clients or alter definition"}}
    # JSON-normalize Path/device before comparisons on resume.
    config = json.loads(json.dumps(config, default=str))
    if args.resume and (output / "config.json").exists():
        original = json.loads((output / "config.json").read_text(encoding="utf-8"))
        for key in config:
            if key not in {"epochs", "resume", "checkpoint_every", "check_aggregation", "view_equivariance_error"}:
                if original.get(key) != config[key]:
                    raise RuntimeError(f"Resume configuration mismatch: {key}")
    else:
        atomic_json(output / "config.json", config)
    rows, events, previous, start = [], [], None, 0
    checkpoint = output / "checkpoint.pt"
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        previous, controller = saved["previous"], saved["controller"]
        rows, events, start = saved["rows"], saved["events"], saved["round"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if args.gpu >= 0:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if start > args.epochs:
        raise RuntimeError("Checkpoint exceeds requested horizon")
    round_file, round_writer = restore_csv(output/"round_metrics.csv", rows, ROUND_FIELDS)
    event_file, event_writer = restore_csv(output/"client_events.csv", events, EVENT_FIELDS)
    atomic_json(output/"status.json", {"status": "running", "pid": os.getpid(), "completed_rounds": start})
    print(f"START {args.algorithm} GPU={config['gpu_name']} rounds={args.epochs} resume_at={start}", flush=True)

    try:
        for t in range(start, args.epochs):
            began = time.perf_counter()
            if args.gpu >= 0:
                torch.cuda.reset_peak_memory_stats()
            current = model.state_dict()
            clients = np.random.choice(range(args.num_users), 10, replace=False)
            sizes = np.array([len(users[i]) for i in clients])
            weights = sizes / sizes.sum()
            if args.algorithm == "FedEchoView":
                basis, norms, radius, components = make_basis(model, current, previous)
                alpha, beta, rho, history_count = controller.dispatch(
                    clients, weights, t+1, norms, radius, components)
                if args.disable_control:
                    alpha[:] = 0
            else:
                basis, norms, radius, components = [], np.zeros(0), 0.0, []
                alpha, beta = np.zeros((10, 0)), np.zeros((10, 0))
                rho, history_count = 0.0, 0
            consensus = float(np.linalg.norm(weights @ (alpha * norms)) / radius) if components else 0.0
            ratio = float(np.max(np.linalg.norm(alpha * norms, axis=1)) / radius) if components else 0.0
            if consensus > 1e-9 or ratio > 1 + 1e-9:
                raise RuntimeError("Dispatch invariant failed")
            accumulator, projections, round_events, reference_states = None, [], [], []
            views, violations = [], 0
            for task, client in enumerate(clients):
                local_model = copy.deepcopy(model)
                reset_seed = None
                if components:
                    with torch.no_grad():
                        for name, parameter in local_model.named_parameters():
                            for a, direction in enumerate(basis):
                                parameter.add_(direction[name], alpha=float(alpha[task, a]))
                # Actual dispatched state is retained only for this client.
                actual_start = {name: parameter.detach().clone()
                                for name, parameter in local_model.named_parameters()} if components else {}
                previous_view = controller.view_scheduler.last.get(int(client))
                view, violation = controller.view_scheduler.choose(int(client), t)
                views.append(view)
                violations += int(violation)
                if violation:
                    raise RuntimeError("Client view did not alternate")
                rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
                local = LocalUpdate_FedAvg(args=args, dataset=train, idxs=users[client], dataset_test=test)
                returned = train_in_model_view(local_model, local, view)
                projections.append(project_update(returned, actual_start, basis, norms))
                accumulator = aggregate_add(accumulator, returned, int(sizes[task]))
                if args.check_aggregation:
                    reference_states.append({k: v.detach().clone() for k, v in returned.items()})
                round_events.append({"round": t+1, "client_id": int(client), "task_id": task,
                    "sgd_rng_sha256": rng_hash, "reset_seed": reset_seed,
                    "current_view": view, "previous_view": previous_view[0] if previous_view else None,
                    "visit_index": controller.view_scheduler.participation_count[int(client)],
                    "components": json.dumps(components), "prediction": json.dumps(beta[task].tolist()),
                    "offset_coefficients": json.dumps(alpha[task].tolist())})
                del local_model, returned, actual_start, local
            for key in accumulator:
                accumulator[key] = torch.div(accumulator[key], int(sizes.sum()))
            if args.check_aggregation:
                reference = Aggregation(reference_states, sizes.tolist())
                if any(not torch.equal(reference[k], accumulator[k]) for k in accumulator):
                    raise RuntimeError("Streaming aggregation != original Aggregation")
                del reference, reference_states
            if args.algorithm == "FedEchoView":
                relative = controller.observe(clients, weights, t+1, projections, components)
            else:
                relative = np.zeros((10, 0))
            mse = float(np.mean((beta - relative)**2)) if components else None
            zero_mse = float(np.mean(relative**2)) if components else None
            for j, event in enumerate(round_events):
                event["observed_response"] = json.dumps(relative[j].tolist())
                events.append(event)
                event_writer.writerow(event)
            previous = cpu_state(model) if args.algorithm == "FedEchoView" else None
            model.load_state_dict(accumulator)
            del accumulator, basis
            accuracy, _ = test_img(model, test, args)
            row = {"round": t+1, "algorithm": args.algorithm, "test_accuracy": float(accuracy),
                   "round_seconds": time.perf_counter()-began, "selected_clients": json.dumps(clients.tolist()),
                   "global_state_sha256": state_hash(model.state_dict()), "rho": rho,
                   "valid_components": json.dumps(components), "trajectory_norm": radius,
                   "history_coordinates": history_count, "response_pairs": controller.persistence.count,
                   "max_offset_ratio": ratio, "consensus_residual": consensus,
                   "prediction_mse": mse, "zero_prediction_mse": zero_mse,
                   "gpu_peak_bytes": torch.cuda.max_memory_allocated() if args.gpu >= 0 else 0,
                   "views": json.dumps(views), "flip_fraction": views.count("F")/len(views),
                   "alt_violations": violations}
            rows.append(row)
            round_writer.writerow(row)
            round_file.flush()
            event_file.flush()
            print(f"ROUND {t+1}/{args.epochs} accuracy={float(accuracy):.4f} rho={rho:.5f} "
                  f"offset={ratio:.5f} pairs={controller.persistence.count} sec={row['round_seconds']:.2f}", flush=True)
            if (t+1) % args.checkpoint_every == 0 or t+1 == args.epochs:
                saved = {"round": t+1, "model": cpu_state(model), "previous": previous,
                         "controller": controller, "rows": rows, "events": events,
                         "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                         "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if args.gpu >= 0 else []}
                temp = checkpoint.with_suffix(".tmp")
                torch.save(saved, temp)
                os.replace(temp, checkpoint)
                atomic_json(output/"summary.json", summarize(rows))
                atomic_json(output/"status.json", {"status": "running", "pid": os.getpid(), "completed_rounds": t+1})
        atomic_json(output/"summary.json", summarize(rows))
        atomic_json(output/"status.json", {"status": "completed", "pid": os.getpid(), "completed_rounds": len(rows)})
    except BaseException as error:
        atomic_json(output/"status.json", {"status": "failed", "completed_rounds": len(rows),
                    "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        round_file.close()
        event_file.close()


if __name__ == "__main__":
    run(parse_args())
