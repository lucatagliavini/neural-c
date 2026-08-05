# Training Observability and Evaluation

This document is authoritative for training telemetry, validation/test data,
evaluation, history export, state inspection, and early stopping.

## Dataset Ownership

`train.txt`, `validation.txt`, and `test.txt` use the same version 1 dataset
grammar and must match the model input/output widths. Training data owns model
updates. Validation data owns early-stopping decisions and is mandatory only
when early stopping is enabled. Test data is never consulted by training and
is the default input to `evaluate`.

All three files are project-owned. Commands load and validate the selected data
while holding the project lock. The training digest includes the canonical
validation dataset and early-stopping parameters only when early stopping is
enabled; test data deliberately does not affect persistence provenance.

## Progress and History

`train --report-interval N` is execution-only. Zero disables reporting. A
positive interval reports absolute epoch/target, training loss, convergence
loss, best loss, absolute/relative change, maximum pre-clipping batch gradient
norm, and clipped batch count. The target epoch is always
reported, as is an earlier patience boundary. With early stopping, convergence
means validation loss and both current and selected-best validation loss are
shown. Reporting does not alter checkpoint scheduling, numerical reduction,
project files, or canonical digests.

`--history` writes the same bounded interval observations to `history.txt`.
Fresh training replaces it; resume and refinement append. History is flushed
after each row but remains diagnostic: an open or write failure emits a warning
and does not fail training or affect recovery.
History version 1 rows accept the additive trailing `gradient_norm` and
`clipped_batches` fields. Legacy rows without them remain diagnostic input, and
newly emitted rows always include them.

`inspect --state` fully validates any present weights/checkpoint under the
shared lock. It reports cumulative completed and target epochs and, for early
stopping, the selected/best epoch, best validation loss, stale count, and
completion reason.

## Evaluation

`evaluate PROJECT [--dataset train|validation|test] [--threads N]` defaults to
`test`. Snapshot acquisition holds the shared lock through project, finalized
weights, provenance, and dataset validation; execution then uses the immutable
snapshot. Loss accumulation and sample order are deterministic across worker
counts.

Versioned output always includes dataset identity, dimensions, selected model
epoch, and mean configured loss. Exact binary targets with one sigmoid output
produce a two-class evaluation at threshold 0.5. Exact one-hot targets with a
softmax output produce multiclass argmax evaluation. Those contracts add
accuracy, a truth-major confusion matrix, and per-class precision, recall, and
F1; zero denominators produce zero. Other targets are reported as regression
without inferred classification metrics.

Configured cross-entropy loss is evaluated from model logits using the stable
definitions in `losses.md`; classification metrics continue to use the
materialized sigmoid or softmax predictions.

## Early Stopping

`early_stopping_patience` and `early_stopping_min_delta` are project settings.
Patience zero disables the feature and requires a zero delta. Positive patience
requires `validation.txt`. An epoch improves only when
`best_validation_loss - current_validation_loss > min_delta`; otherwise the
stale count advances. Training stops after `patience` consecutive non-improving
epochs.

The current model and the selected best model have distinct ownership. Version
2 checkpoints atomically persist both payloads plus best epoch/loss and stale
count, so signal interruption and `--resume` preserve the exact decision state.
Final version 2 weights contain the selected model, while metadata separately
records observed `completed_epochs`, `selected_epoch`, target, and whether the
run reached its target or stopped early. Prediction and evaluation therefore
never silently use the last, worse epoch. Additional training begins from the
selected finalized model and retains cumulative observed epoch numbering.
