# FedEcho reproduction, seed 1

Source: `6341_FedEcho_Echoes_of_the_Pas.pdf`, Section 3 and Appendices B--D.
Independent code path; existing FedPhoenix and symmetry code are unchanged.
The first experiment is VGG/CIFAR-10/Dirichlet alpha=0.3 on the existing
`data/cifar10_100_noniidCase5_beta0.3.json` partition. No image augmentation.
100 clients; 10 sampled per round; 5 local epochs; batch 50; SGD lr=.01,
momentum=.5, weight decay=0; evaluation on all 10000 test images each round.
One training seed, 1. Formal horizon 500; no automatic promotion or tuning.
User correction: train FedEcho ONLY. The comparison source is the existing
`result_other_method/cifar10/0.3/vgg_FedPhoenix.log`, restricted to rounds 1--500.
The initial fresh formal FedPhoenix process was stopped at the user's request;
its partial files remain local. The launcher never starts FedPhoenix again.

## Algorithms and geometry

FedEcho uses its own on-policy history, weighted cohort-relative local updates
measured from the ACTUAL dispatched start, pooled temporal decay estimated from
past-only consecutive valid pairs, and the Appendix D rank-two CNN basis.
The spatial direction subtracts each Conv2d input/output pair's spatial mean;
the complementary direction includes every other trainable parameter.
All nontrainable BatchNorm buffers are dispatched from the global model and
aggregated according to the original project. They are not control coordinates.
FedEcho has no random reset, horizontal view or shadow training.
FedPhoenix uses the existing kernel-reset routine, reset=1/64, FP_conv=1000,
FP_fc=0, ori_normal, and its original sample-weighted aggregation rule.

Numerical choices missing from the paper are fixed before accuracy inspection:
trajectory/component tolerance 1e-8, float64 dot-product reductions, rank-two
JOINT unit-ball/weighted-consensus projection with residual tolerance 1e-9.
The decay SSE is pooled exactly through per-gap sufficient statistics. Its
one-dimensional numerical search uses a fixed 1025-point grid and bounded
refinement of each grid local minimum, including the endpoints; no validation
or test accuracy enters estimation. No-history predictions are zero; the joint
centering can still assign these clients nonzero offsets. Invalid components
leave their histories unchanged. Response pairs are added after dispatch/training.

Streaming aggregation follows the same multiply/add/divide ordering as
`models.Fed.Aggregation`; smoke tests independently check exact tensor equality.
RNG usage retains the original global training stream. Core estimation/solving
does not consume global random numbers. Client selections and per-client SGD
RNG hashes are saved to verify paired runs. A fresh matched FedPhoenix run
was initially used to validate the current-code baseline's 20-round prefix;
it matched the stored B0 hashes exactly. The formal comparison now reads the
user-specified historical log instead. That log has no saved selections or
full config, so its paired sampling/partition identity is not asserted.
Actual software/hardware versions and source hashes are saved.

## Correctness and stopping

Tests cover an independent Dykstra solution to the dispatch problem, saturation,
weighted centering, rank-two orthogonality, actual-start displacement, temporal
estimation, invalid-history behavior, checkpoint continuation, and aggregation.
GPU gates compare disabled-control FedEcho with FedAvg and check active-history
smoke/resume. Formal training starts only after the gates pass.

Primary utility endpoint: mean test accuracy in rounds 1--500, with +1 pp over
FedPhoenix as the minimum target and +2 pp as the desired gain. Also report
rounds 401--500 mean, peak, final, last20/50/100, prediction-vs-zero MSE,
history availability, trust-bound activity and memory. A failed threshold does
not trigger a search, added seeds, or a combination with reset/augmentation.

Each run saves config/source/partition/initial-model hashes, round and client
CSVs, status/summary JSON and an atomic checkpoint every 10 rounds. Resume
restores both global models, controller/pool, all CPU/CUDA/Python/NumPy RNG,
and CSVs to the last checkpoint. Partial-round work is replayed; no client is
silently skipped. A resumed run must match saved source and protocol hashes.
The launcher allows two checkpoint retries for FedEcho runtime failures. It
never changes batch size, model, history definition or dispatch parameters.
`--adopt_pid` safely attaches to a verified existing FedEcho worker without
interrupting or duplicating it; passed gates are required when adopting.

The installed SciPy 1.18 BFGS/SLSQP paths crash with Windows DLL exception
0xc06d007f even on a trivial quadratic. No package/environment was changed.
The equivalent joint dual problem is solved with a NumPy 2x2 damped Newton
method and Dykstra projection if needed; scalar decay refinement uses golden
section search. Every result is checked for consensus/ball feasibility.

Example (from any directory, preferably using the project's fd Python):

```bash
python scripts/run_fedecho_reproduction.py --gpu 0 --output_dir results/fedecho_reproduction_seed1
```

This implementation lives in the Git-tracked clean-finish repository, using the
same original VGG/CIFAR-10/local-SGD substrate as the later our-finish server
copy. The latter's additional architecture/view adaptations are not used here.
File hashes were checked against the our-finish server copy: `models/Nets.py`,
`models/Update.py`, `models/Fed.py`, `utils/sampling.py`, and
`utils/dataset_utils.py` are identical. The fixed partition is read directly
from the clean-finish data directory and its SHA-256 is saved in each config.
