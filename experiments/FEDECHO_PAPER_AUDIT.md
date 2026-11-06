# FedEcho paper reproduction: fixed single setting

This is an algorithm reproduction, not an accuracy search. Only seed 1, VGG,
CIFAR-10, and the existing Dirichlet alpha=0.3 split are in scope. The formal
horizon is 1000 rounds, with the first 500 reported separately. No FedPhoenix
training is launched. Old runs and model-view variants remain separate artifacts.

Paper read: `C:/Users/admin/Desktop/6341_FedEcho_Echoes_of_the_Pas.pdf` (18 pages),
SHA256 `80b8fda3e1d5279c439931f986d4965d4bea1d0e8ebce67a498225657f3429ee`.
The references mention a `(3).pdf` and a remote Linux server; neither is the
available local source. The supplied local paper and actual Windows code are
used. This distinction is not resolved by assuming those copies are identical.

## A. Paper-defined algorithm

At round t, use current minus previous global trainable parameters as m_t; the
first round has zero trajectory. On each Conv2d weight tensor, subtract its H/W
mean per input/output channel pair to obtain the spatial component. The base
component is the remainder, including all non-convolutional parameters. These
two whole-model directions reconstruct m_t and are orthogonal (up to roundoff).

For valid component a, the recorded response is
`delta_i,a = <(y_i-x_i)-sum_j p_j(y_j-x_j), d_a>/||d_a||^2`.
The implementation projects each actual displacement immediately, then performs
the weighted scalar centering. It never substitutes `y_i-w_t` for `y_i-x_i`.

Keep the latest valid scalar/timestamp per client/component. Fit one shared rho
in [0,1] to all historical adjacent-valid pairs, with communication-round gaps:
`sum (delta_current-rho**gap*delta_previous)**2`. The current round is added only
after prediction/training. No pairs means rho=0. Invalid directions leave their
histories intact. There is no rho floor, moving window, or forced active control.

Predict beta from the latest response decayed by its actual age. Solve the paper's
weighted path-surrogate problem jointly in the two coordinates, subject to
weighted zero mean and a ball of radius ||m_t|| for every client. With inactive
balls, alpha is exactly `-.5*(beta-p@beta)`. Active balls require a joint convex
projection; independent clipping is not equivalent.

Dispatch `x_i=w_t+sum_a alpha_i,a*d_a`, perform the original local training, and
aggregate the complete returned state using sample-size weights. There are no
kernel resets, historical client starting models, model flips, or input augmentations.

## B. Inherited experimental substrate

- Existing `get_dataset`, `VGG16`, `LocalUpdate_FedAvg`, `test_img`, and seed helper.
- 100 clients, 10 selected without replacement per round, seed=1.
- Existing `data/cifar10_100_noniidCase5_beta0.3.json` is read, not regenerated.
  SHA256 `ffa71f5432b8905bf13117d30fe3c981090aa3fa1e93e0a1c179be4f8bb5d882`.
- CIFAR-10 has 10 output classes; input is ToTensor plus the project's Normalize.
- Local epochs=5, batch=50, eval batch=256, SGD lr=.01, momentum=.5, wd=0;
  workers=0, GPU=0. The original optimizer is recreated per local participation.
- VGG's existing BatchNorm and classifier Dropout are retained. Buffers start
  from the current global state, are excluded from control/projection, and are
  included in ordinary full-state aggregation. Dropout is not a newly added
  image augmentation.
- The streamed aggregation uses the original multiply/add/divide order and is
  checked tensor-for-tensor against `models.Fed.Aggregation`, including integer
  BN counters. It saves the memory of ten retained full local models in formal runs.
- CIFAR-100/Tiny-ImageNet issues in the reference are outside this requested run;
  their loading/labels/partitions are not silently modified.

## C. Documented engineering choices

The paper does not give a numeric trajectory tolerance. Use epsilon_traj=1e-8
(configurable for numerical validity, fixed in this reproduction). Accumulated
norms/dots and scalar optimization use float64; model training retains the
baseline dtype. Component norms are logged, so tolerance sensitivity is auditable.

The one-dimensional rho objective uses exact sufficient statistics per gap
`A=sum previous^2`, `B=sum previous*current`, `C=sum current^2`; this represents
the full pair-pool SSE exactly without storing every pair. A deterministic 1025
point bracketing grid, bounded refinement of grid minima/edge intervals, and
explicit endpoints solve the objective numerically. This is not an analytical
proof of global minimization for arbitrary data. Known-rho and boundary tests,
and the prior independent dense-grid replay, check the practical implementation.

For dispatch, normalize coordinates by component norms and ||m_t||, solve the
unit-ball dual using damped 2x2 Newton steps, and use Dykstra projection if the
consensus residual is insufficient. Both target the same unique convex primal
solution. Unit tests compare the active objective against an independent oracle.
The acceptance threshold for normalized weighted consensus is 1e-9.

No new dependency is installed. In this environment, SciPy multivariate optimizer
calls previously failed at a native DLL boundary; the explicit low-dimensional
solver avoids that runtime issue without changing the objective.

History is O(clients*2) scalar/timestamps plus O(distinct gaps) sufficient stats,
not O(clients*VGG size). Tensor geometry is streamed per parameter. Only the
previous global state, current directions, and one actual client starting state
are added to the normal local/aggregation working set. Checkpoints keep only
the latest state, every 10 rounds and initially, with all CPU/CUDA RNG states.

Output locks prevent duplicate workers for one directory. Resume checks source,
partition and configuration hashes and rolls CSVs back to the exact checkpoint.
Runtime failures can retry twice using that checkpoint; no batch, optimizer,
method, or data changes are allowed. Reporting uses a dependency-free SVG to
avoid the old plotting-library failure after completed training.

## D. What this audit establishes and leaves open

The prior 500-round paper-only run did not meet the practical accuracy target:
first-500 mean 68.21272%, versus 68.64604% in the supplied FedPhoenix log. Control
was nonzero in only 14 rounds, last at round 21. Its saved responses, predicted
decays, weighted centering, dispatch coefficients, pair pool and checkpoint
state were replayed independently in the preceding audit. They did not reveal
an incorrect sign, omitted factor 1/2, future leakage, or lost response history.
This is evidence against claiming that a simple VGG formula bug caused the result.

A real generic edge bug was found and fixed: a root-level `nn.Conv2d` has parameter
name `weight`, not `.weight`. The actual VGG's named `features.*` convolutions
were already recognized, so this correction does not explain or improve that run.
Additional fixes reject nonfinite geometry/weights and expose the documented
tolerance through the runner. No mathematical rule is changed for the actual VGG.

The fresh VGG smoke is checked against the old VGG first-10 hashes when that
archive is available. Exact agreement would confirm the reproduction reproduces
the old prefix; it would not be evidence of improved accuracy. The fresh run's
1000-round result is reported regardless of whether rho collapses or accuracy
is disappointing. A negative result is not relabeled as a successful performance
reproduction. Single-seed results have no cross-seed uncertainty estimate.

There is no unresolved choice that requires inventing a different method.
The historical FedPhoenix log lacks paired sampling/RNG/config traces, so its
accuracy comparison is explicitly historical, not an exact paired causal test.

## Files and verification

Core: `Algorithm/FedEcho.py`.
Shared original-substrate runner: `experiments/fedecho_reproduction.py`.
Paper-only entry: `main_fedecho.py` (default 1000 rounds).
Gate/retry/report launcher: `scripts/run_fedecho_paper.py`.
Tests: `tests/test_fedecho.py`, `tests/test_fedecho_paper.py`.

The launcher requires the full regression suite, 3-round disabled-control/FedAvg
exact model/client/RNG parity, a 5-round real-data GPU smoke, 10-round active
dispatch coverage, bitwise aggregation parity, and exact 5+5 resume versus an
uninterrupted 10-round run. At most two gate workers run concurrently; one formal
FedEcho setting is trained. No second algorithm is invented to fill a GPU slot.

Run from the repository:

```powershell
& D:\software\minicoonda\envs\fd\python.exe scripts\run_fedecho_paper.py --output_dir results\fedecho_paper_seed1 --gpu 0 --max_retries 2
```

Results: `results/fedecho_paper_seed1/`. Gates/config/hashes are saved before
formal training. The formal subdirectory contains round/client CSVs, config,
status, summary and latest checkpoint. Completion writes `comparison.csv`,
`comparison.json`, `accuracy_curve.svg`, and `RESULTS_SUMMARY.md`. First-500,
late-100, final, peak, last20/50/100, best rolling5/10 and 100-round stage means
are separated; only baseline rounds 1-1000 are used.
