"""Paper FedEcho only: fixed seed-1 VGG/CIFAR-10/alpha=.3, no augmentation.

The baseline infrastructure and audited training loop are reused. This entrypoint
exposes no reset, model-view, decay-floor, or hyperparameter-search controls.
"""
from experiments.fedecho_reproduction import parse_args as infrastructure_args, run


def parse_args():
    # Reuse the protocol defaults, then reject baseline-only audit switches.
    args = infrastructure_args(default_epochs=1000)
    if args.algorithm != "FedEcho" or args.disable_control:
        raise ValueError("Use the infrastructure runner for baseline/parity tests")
    args.paper_entrypoint = True
    return args


if __name__ == "__main__":
    run(parse_args())
