# Loss and Output Contract

This document is authoritative for configured losses, target domains, output
activation compatibility, normalization, and fused output gradients.

## Configuration and compatibility

`loss` is training-owned configuration in `project.conf`. Supported canonical
names are `mse`, `binary_cross_entropy`, and
`categorical_cross_entropy`. It remains part of the canonical training digest;
changing it invalidates weights and checkpoints produced for another loss.

The complete model and every dataset used by a command are validated before
execution:

- `mse` accepts any supported output activation and finite targets;
- `binary_cross_entropy` requires sigmoid output and exact binary targets
  (`0` or `1`) at every output;
- `categorical_cross_entropy` requires a softmax layer with at least two
  outputs and exactly one-hot targets per sample.

Binary cross-entropy supports one or more independent sigmoid outputs. The
version 1 evaluation protocol produces binary classification metrics only for
the existing single-output case. Soft labels are deliberately outside the
version 1 target contract and fail actionably instead of being interpreted
implicitly.

Initialization rejects an incompatible loss/output pair before filesystem
changes. Project loading, CSV import, training, validation, test evaluation,
and gradient checking apply the same contract. CSV import validates transformed
targets before replacing any managed dataset.

## Definitions and numerical paths

MSE is the output mean
`sum((prediction - target)^2) / output_count`; its output gradient includes the
same normalization.

Binary cross-entropy is the output mean. Given sigmoid logit `z` and target
`y`, its stable term is
`max(z, 0) - z*y + log1p(exp(-abs(z)))`. Backpropagation uses the fused
pre-activation gradient `(sigmoid(z) - y) / output_count`; it never divides by
a saturated probability.

Categorical cross-entropy is the per-sample class loss, not an output mean. It
is evaluated from softmax logits as
`max(z) + log(sum(exp(z - max(z)))) - z[target_class]`.
Backpropagation uses the fused pre-activation gradient `softmax(z) - y` and
does not separately multiply by the softmax Jacobian.

Training data loss, validation/early-stopping loss, coherent epoch reporting,
and
the model-backed `evaluate` command all use these logits-based formulations.
The lower-level probability-only loss API remains available for already
materialized predictions; it bounds exact zero/one probabilities for finite
diagnostic evaluation, while training and reported project evaluation do not
depend on that approximation.

All dataset means retain deterministic sample order and compensated summation.
Fused output handling changes neither parameter layout nor the ordered
cross-worker gradient reduction contract.

Regularization does not redefine any configured loss. Training reports a
separate objective equal to its coherent post-epoch data loss plus the model
penalty defined in `training-engine.md`. Validation, early stopping, test
evaluation, and the `evaluate` command continue to report and compare data loss
only, so dataset metrics do not silently acquire parameter penalties.

## Persistence compatibility

Milestone 9.1 adds no weights or checkpoint fields and no mutable optimizer
state. Existing `mse` project files retain their spelling, parsed enum,
canonical training stream, digest, training arithmetic, and persistence
compatibility exactly; they require no migration, re-import, or retraining.
New cross-entropy projects use the existing persistence versions, distinguished
by their canonical training digest.
