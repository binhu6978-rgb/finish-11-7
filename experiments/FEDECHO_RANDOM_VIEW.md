# FedEcho + random model-side view, seed 1

User constraint: ALL input/data-loader augmentation remains OFF. Both this
random-view experiment and the existing alternating-view experiment use the
same original ToTensor/Normalize pipeline, input_hflip=0, no RandomCrop and no
input RandomHorizontalFlip. The rejected input-HFlip trial was cancelled during
smoke tests and is excluded from comparisons. No FedPhoenix process is run.

The only difference from the existing alternating model-view run is scheduling:
instead of forcing I/F alternation for each client's successive participations,
a private checkpointed RNG selects I/F with equal probability at each client
participation. This private RNG does not consume the data shuffle or global
training RNG. Both views train on the original input tensors. The F view flips
Conv kernels, runs the unchanged original local SGD, then maps the returned
state back to canonical coordinates. This model transformation has an input-view
equivalence under the tested architecture; no claim is made that it lacks any
augmentation-like effect. The prohibited input preprocessing operations are
never enabled.

The full existing FedEcho model-view training engine and `Algorithm/FedEcho.py`
are reused unchanged. The actual-start response, online pooled temporal decay,
rank-two current-trajectory basis, joint weighted-zero-sum dispatch, trust ball,
and original weighted aggregation all remain intact. Histories are on-policy,
in canonical coordinates, with no per-view bank or predictor modification.
Model-view randomness is serialized with the controller for exact resume.

Fixed VGG/CIFAR-10/alpha .3 partition, seed 1, 100 clients, 10 per round,
5 local epochs, batch 50, SGD lr .01/momentum .5/wd 0; 500 rounds. No reset,
forced predictor strength, new seed, or search. Initial model, client selection,
and SGD RNG agree with the existing model-view run. No input tensor transformation
is inserted by this new entry point.

Gates: full regression, identity-view 3-round exact parity, private-RNG invariance,
8-round continuous/resumed equality, identical preprocessing and core hashes,
and original aggregation equality. One extra worker runs alongside the existing
alternating model-view worker; maximum two training processes total.

Compare to first 500 rounds of the user-provided
`result_other_method/cifar10/0.3/vgg_FedPhoenix.log` and the existing pure/alternating
FedEcho outputs. The primary endpoint is first-500 mean accuracy (+1 pp minimum,
+2 pp desired), with history-control activity explicitly reported separately.
The log has no full saved config/sampling, so historical pairing is not asserted.

```bash
python scripts/run_fedecho_random_view.py --gpu 0
```

Output: `results/fedecho_random_view_seed1/`.
