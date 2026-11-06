# finish-11-7

Source-code snapshot of the local FedPhoenix-clean-finish working tree on
2026-11-07. Includes baseline algorithms, experimental implementations, launch
scripts, tests, and FedEcho reproduction notes. Datasets, partition files,
training results, raw logs, checkpoints, caches and the original Git history
are deliberately excluded.

## Current paper-only FedEcho reproduction

The paper entrypoint is `main_fedecho.py`; implementation details and the audit
are in `experiments/FEDECHO_PAPER_AUDIT.md`. Model-view and random-view files are
retained as historical experimental code. They are not part of the paper-only
FedEcho reproduction.

Fixed setting: seed=1, VGG16, CIFAR-10, Dirichlet alpha=0.3, 100 clients with 10
selected per round, 1000 rounds, local epochs=5, batch=50, SGD lr=0.01,
momentum=0.5, weight decay=0. No image augmentation, model views or kernel reset
are used by the paper-only entrypoint.

The verified local environment uses torch 2.7.1+cu128 and torchvision 0.22.1 on
an RTX 4080 SUPER. The code also imports NumPy, SciPy, pandas, scikit-learn and
Pillow. Match the project environment for trajectory reproduction.

Before training, supply CIFAR-10 locally under the loader's supported `data/`
layout, plus the original partition file:

`data/cifar10_100_noniidCase5_beta0.3.json`

Expected partition SHA256:
`ffa71f5432b8905bf13117d30fe3c981090aa3fa1e93e0a1c179be4f8bb5d882`.
It covers all 50,000 training examples exactly once across the 100 clients.
The partition is not included in this source-only export. Regenerating another
split would not reproduce the original experiment.

From this repository's root:

```bash
python -m unittest discover -s tests
python main_fedecho.py --epochs 1000 --gpu 0 --output_dir results/fedecho_paper_seed1/FedEcho
```

The complete gate/retry/report launcher is:

```bash
python scripts/run_fedecho_paper.py --output_dir results/fedecho_paper_seed1 --gpu 0 --max_retries 2
```

That launcher additionally requires the historical comparison log at
`result_other_method/cifar10/0.3/vgg_FedPhoenix.log`; it does not retrain
FedPhoenix. Supply that excluded local artifact before using the comparison
launcher. The direct training entrypoint does not need the comparison log.

Local verification passed 131 unit tests, a 5-round GPU smoke, 3-round
disabled-control/FedAvg exact parity, 10-round active dispatch coverage,
original aggregation equality and exact 5+5 checkpoint-resume parity. These
are correctness checks, not evidence that FedEcho beats FedPhoenix. The formal
1000-round experiment was still running when this snapshot was prepared.

## Snapshot provenance

`CODE_SNAPSHOT.json` records SHA256 values of the 111 copied source files before
the export-specific `.gitignore` replacement. README and the snapshot manifest
are export additions. The source working tree, original Git remote and running
training processes were left unchanged.
