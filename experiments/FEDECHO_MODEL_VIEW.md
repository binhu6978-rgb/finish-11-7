# FedEcho core + alternating client model view, seed 1

This is an augmented FedEcho variant, evaluated alongside the unchanged pure
FedEcho reproduction. It is not claimed to be the paper's unmodified algorithm.
No reset is added. No estimator gain, history rule, control basis or trust
radius is changed to improve accuracy.

## Core retained verbatim

`Algorithm/FedEcho.py` is reused without modification: actual-start local
displacement; same-round sample-weighted centering; online, past-only pooled
decay; client/component latest-valid histories; the CNN base/spatial rank-two
trajectory decomposition; joint weighted-zero-sum dispatch and trust ball;
invalid-component behavior; and standard local SGD/weighted aggregation.

Canonical server starts are x_i=w+Delta_i, with weighted mean w. Inside the
client training operator, a participation uses identity I or horizontal model
view F. F training is followed by inverse mapping before return. The response
is ALWAYS computed as canonical returned y_i minus canonical x_i, before
projecting onto the canonical current global trajectory. Consensus/trust
claims apply in that common coordinate system, not to the raw I/F weights.

The view policy is the existing seed-1 SHA256 initial assignment followed by
per-client alternating participations, with no RNG consumption. There is ONE
response history per client/component, shared across that client's views; no
view-conditioned bank or extra response-fitting rule is introduced. View state
is stored on the checkpointed controller, so resume preserves alternation.
The augmentation changes the effective local training operator; the underlying
`LocalUpdate_FedAvg` SGD routine is unchanged. It is equivalent to a horizontal
input-view augmentation under the tested VGG symmetry, so this is explicitly
reported as model-view augmentation, not as absence of augmentation.

## Frozen experimental protocol and comparisons

VGG/CIFAR-10; existing 100-client Dirichlet alpha .3 partition; 10 clients per
round; local epochs 5, batch 50; SGD lr .01, momentum .5, wd 0; seed 1 only;
500 rounds. Other source/config/initial-model/partition checks match the pure
FedEcho run. Input preprocessing remains the original normalize-only pipeline.
There is no random reset and no second local training pass.

FedPhoenix is read only from the user-specified
`result_other_method/cifar10/0.3/vgg_FedPhoenix.log`, using rounds 1--500.
The pure FedEcho trajectory is read from `results/fedecho_reproduction_seed1`.
No FedPhoenix process is launched. Historical baseline config/sampling cannot
be independently verified from that log and are not labeled as paired.

An isolated minimal runner fork preserves the running pure experiment and its
checkpoint source hashes. Identity-view GPU parity against the pure FedEcho
prefix, model-space-vs-input-space SGD equivalence tests, zero-sum inverse-map
tests, and active-view checkpoint/resume checks run before the formal job.
Gates run sequentially, using at most one extra GPU worker alongside pure
FedEcho. The formal variant likewise uses only that additional worker.

The primary endpoint remains first-500 mean accuracy, minimum +1 pp and desired
+2 pp versus the selected historical FedPhoenix log. Compare to pure FedEcho
as well, and report history-control activity separately from view augmentation.
No extra seed, tuning, automatic 1000-round promotion or other variant is run.

```bash
python scripts/run_fedecho_model_view.py --gpu 0
```

Outputs: `results/fedecho_model_view_seed1/`, including gates, model-view run,
round/client CSVs, checkpoint, provenance, comparison CSV/JSON, curve, and report.
