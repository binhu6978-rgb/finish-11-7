"""FedEcho with random CLIENT MODEL views; input augmentation is forbidden."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import fedecho_model_view as base


def prepare_args(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--view_policy", choices=["random", "none"], default="random")
    policy, rest = p.parse_known_args(sys.argv[1:] if argv is None else argv)
    if "--view_mode" in rest:
        raise ValueError("Use --view_policy; this entry fixes the random model-view experiment")
    original = sys.argv
    try:
        sys.argv = [original[0], *rest]
        args = base.parse_args()
    finally:
        sys.argv = original
    if args.input_hflip != 0:
        raise ValueError("Input/data-loader augmentation is prohibited")
    args.view_mode = "rand" if policy.view_policy == "random" else "none"
    args.method_variant = "FedEcho_ModelViewRandom"
    args.data_augmentation_forbidden = True
    args.variant_entry_sha256 = base.sha_file(Path(__file__))
    return args


if __name__ == '__main__':
    base.run(prepare_args())
