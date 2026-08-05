# Training Engine Architecture

This document is authoritative for Milestone 4 backpropagation, batching, and
parallel training. Persistence and continuation remain governed by
`persistence-format.md` and `training-resume.md`.

## Backpropagation State

Each `NeuralWorkspace` owns forward pre-activations and activations plus private
backward scratch buffers for activation gradients, pre-activation gradients,
and the model-input gradient. A workspace belongs to one model and one worker;
no forward or backward call allocates per sample. A single-sample operation
runs forward, evaluates the configured loss, and fills a model-compatible
`NeuralGradient` without changing model parameters.
Output activation and targets must satisfy `losses.md`. Cross-entropy paths
evaluate logits directly and install their fused pre-activation gradient before
dense backward; MSE retains the ordinary activation Jacobian path.

## Gradient Checking

`neural_model_gradient_check` compares the analytic sample gradient with a
central finite difference for every weight and bias. The perturbation is
`epsilon * max(1, abs(parameter))`; the denominator uses the two representable
perturbed values rather than assuming an exact `2 * epsilon`. A parameter
passes when either its absolute or relative error is within tolerance.

The checker requires exclusive mutable access to the model because it installs
temporary parameter values. Every perturbation is restored before proceeding,
including failure paths, so a completed or rejected check leaves parameters
bit-identical. The supplied workspace is scratch and may change. Checks around
non-differentiable points such as ReLU at zero are expected to fail and must use
test fixtures whose perturbations remain on one differentiable branch.

## Batch Semantics

`batch_size` is training-owned configuration in `project.conf`. Zero selects
the complete dataset and preserves the historical full-batch behavior. A
positive value selects that many logical-order samples per update; values above
the dataset sample count resolve to the complete dataset. Each batch gradient
is the ordered sum of its sample gradients
divided by the actual number of samples in that batch, including a smaller
final batch. The main thread performs exactly one exclusive model update per
batch.

`NeuralBatchPlan` validates the resolved batch size and maps each batch to a
contiguous half-open logical-position range. `NeuralBatchAccumulator` accepts
gradients only in increasing logical-position order. It owns one model-shaped
sum plus one compensation buffer, can be reset between batches, and exposes
the gradient only after the declared `[begin, end)` range is complete and
finalization has divided it by that range's sample count. Failed,
out-of-order, or out-of-range additions do not advance its state. Neumaier
compensation preserves small terms without changing the required sample order;
every operation validates all results before changing either buffer.

Because batch size changes parameter updates, a positive configured value is
covered by the canonical training digest. Zero retains the legacy canonical
stream exactly. Batch size is never treated like execution-only
`thread_count`.

`shuffle` is optional training-owned configuration. Absent or zero retains the
identity source order exactly; one enables `splitmix64_fisher_yates_v1` once per
epoch. `init --shuffle` materializes one. The trainer allocates one
`NeuralSampleOrder`, resets it to `[0, sample_count)`, and applies descending
Fisher-Yates for logical positions `sample_count - 1` through 1.

The epoch-local SplitMix64 seed is derived using defined `uint64_t` arithmetic:

```text
domain = 0x6e657572616c2d73
increment = 0x9e3779b97f4a7c15
epoch_seed = splitmix64_mix(
    (training_seed XOR domain) + increment * (absolute_epoch_index + 1))
```

The absolute epoch index is zero-based and equals `completed_epochs` at the
start of a resumed range. For every Fisher-Yates bound `b`, draws below
`(0 - b) % b` are rejected before `draw % b` is taken, avoiding modulo bias
without architecture-specific integer extensions. The local stream neither
reads nor changes the model initialization RNG state.

## Regularization Objective

`l1_regularization` and `l2_regularization` are optional finite non-negative
training-owned coefficients in `project.conf`. Their defaults, including an
absent legacy property, are zero. `init --l1 V` and `init --l2 V` materialize
them. `regularize_biases` is zero or one; zero excludes every bias and is the
default, while `init --regularize-biases` includes biases and requires at least
one positive coefficient.

For model parameters `theta`, the objective associated with a data loss is:

```text
objective = mean_data_loss
          + l1_regularization * sum(abs(theta))
          + (l2_regularization / 2) * sum(theta * theta)
```

The sums traverse each layer's weights followed, only when enabled, by its
biases. The L1 subgradient is `-1`, `0`, or `1` for negative, zero, or positive
parameters. The L2 gradient is `l2_regularization * theta`; the factor one half
therefore does not appear in the gradient. Penalty accumulation is
Neumaier-compensated and all products, sums, and gradient changes are checked
before mutation. Adding regularization to a gradient is transactional.

Every batch adds the same model penalty gradient to its finalized mean data
gradient. Reported `loss` remains the mean configured data loss of the coherent
post-epoch model. Reported `objective` is that loss plus the same model's
regularization penalty. Validation and test loss remain data-only metrics;
early stopping selects by validation data loss, never by the training
objective. Zero coefficients preserve prior update arithmetic exactly and make
`objective` equal `loss`.

## Gradient Norm and Clipping

`gradient_clip_norm` is optional training-owned configuration in
`project.conf`. Zero or an absent legacy property disables clipping and
preserves the previous training digest and parameter updates exactly. A finite
positive value is the maximum L2 norm and is covered by canonical training
provenance. `init --gradient-clip-norm V` materializes the requested value.

For every batch, the coordinator first completes ordered compensated reduction
and division by the actual batch sample count. The norm then traverses every
weight followed by every bias in model layer order. A scaled sum-of-squares
calculation avoids avoidable intermediate overflow and underflow; a
non-representable final norm fails before model mutation. The value reported is
always this pre-clipping norm of the finalized mean batch gradient.

When the norm is strictly greater than the positive configured maximum, every
gradient component is transactionally multiplied by `maximum / norm`. Zero
gradients and gradients exactly at the boundary are unchanged. An epoch report
contains the maximum pre-clipping batch norm observed in that epoch and the
number of clipped batches. The final training result contains the same values
for its last completed epoch.

The complete update order is: deterministic sample reduction and mean,
regularization contribution, norm measurement, clipping, future optimizer
transformation, then the exclusive model update. Milestone 10 optimizers must
consume the already clipped gradient. Worker threads never add regularization,
calculate norms, clip, or update parameters.

## Parallel Execution

A `NeuralParallelExecutor` creates a persistent POSIX thread pool once per
training run. Pool size is the smaller of the requested thread count and
dataset sample count; each batch activates at most its own sample count.
Workers execute deterministic waves containing at most one sample per worker.
Each worker owns a `NeuralWorkerContext`, error, loss, logical position, source
sample index, and status.
`NeuralExecutionPlan` maps the current batch range to contiguous waves of
at most the effective worker count; the final wave may be smaller.

After every wave, the main thread accumulates completed gradients by increasing
logical position into the batch accumulator. Worker buffers are then reused.
This keeps gradient memory proportional to
`(pool_size + 2) * parameter_count`, including the sum and compensation
buffers. The arithmetic order remains independent of scheduling and thread
count. Workers never update the model. Errors are selected by the earliest
failing logical position and identify its source sample. Cancellation is
cooperative at task boundaries;
asynchronous thread cancellation is forbidden.

The executor is driven by one coordinator. Its model and dataset must outlive
it and remain unchanged while a batch operation is active. A successful call
returns an executor-owned mean gradient valid until the next batch call. A
failed wave contributes no gradients; a later batch call resets the
accumulator and may safely reuse the pool.

## Epoch Observation and Checkpointing

Epoch reporting evaluates the complete dataset after all updates in that epoch,
using a trainer-owned reusable workspace and output buffer. One reported value
therefore always describes one coherent model state. An optional observer is
called synchronously by the coordinator once per completed epoch, after the
model update and coherent loss evaluation. No worker task is active during the
call, so the observer may read the model but must not modify it. The report's
`completed_epochs` value and the model parameters describe the same epoch
boundary. The observer and its context are borrowed for the duration of the
training call.

The trainer does not interpret `checkpoint_interval` and performs no file I/O.
Fresh project training installs a project-layer observer that writes
`checkpoint.txt` at each positive completed-epoch multiple of the configured
interval. An interval of zero disables periodic saves; the observer still
validates epoch reports but performs no persistence work unless a graceful
stop was requested. A report outside an interval boundary likewise performs
no persistence work unless it is the first coherent boundary after a stop
request.
Rejecting any report, including because checkpoint serialization or atomic
replacement failed, fails the run immediately. The failed epoch remains
applied only to the in-memory model; no final weights are produced.

During `train`, the process installs minimal `SIGINT` and `SIGTERM` handlers.
The handler preserves the first signal number in a `volatile sig_atomic_t` and
does no allocation, locking, logging, or file I/O. At the next completed epoch
boundary, the project observer writes one atomic recovery checkpoint and then
rejects the report to stop training. A successfully checkpointed `SIGINT`
returns status 130 and `SIGTERM` returns status 143. If the emergency save
fails, training reports the persistence error and returns the ordinary runtime
failure status instead of claiming a recoverable interruption.

An observer may return the controlled-stop status after accepting a coherent
epoch. The trainer records that epoch and returns success without executing the
remaining range. The early-stopping project observer uses this only after
validation loss and best-state ownership have been updated. It also enriches
telemetry with current/best validation loss and persists current plus best
models together whenever a checkpoint is required.

## Completion

Fresh training runs exactly the configured number of epochs, with one update
per resolved batch. It is allowed only when neither
`weights.txt` nor `checkpoint.txt` exists. The final model and canonical
project digests are written atomically to `weights.txt` only after every epoch
and report succeeds. Successful finalization installs `weights.txt` before
durably removing `checkpoint.txt`; a failure before the weights rename leaves
the latest successfully installed checkpoint available. If finalization is
interrupted after the weights rename, both valid files may remain for the
resume transition to reconcile.

Refinement starts from validated final weights and runs exactly the requested
number of further updates over cumulative absolute epoch numbers. It installs
periodic or signal-requested recovery checkpoints while retaining the previous
weights as a stable baseline. Successful completion atomically replaces those
weights and then removes any checkpoint. The same epoch observer, periodic
schedule, signal handling, and deterministic reduction rules apply to fresh,
resumed, and additional training.

## Absolute Epoch Ranges

Continuation drives the same trainer with an absolute half-open epoch range.
The input boundary is the number of epochs already completed; the target is the
total completed-epoch count required by the run. Reports continue with
`completed_epochs + 1` and therefore retain the same absolute numbering used by
checkpoint metadata and periodic persistence. The range is invalid when the
target is zero or the completed boundary exceeds it.

When both boundaries are equal, the trainer performs no parameter update and
emits no epoch report. It still evaluates the complete dataset once so the
returned final loss describes the already completed model, and it resolves the
effective worker count normally. Fresh training is the range from zero to the
configured epoch count. Resume is the range from the validated checkpoint
boundary to its validated target. Additional training is the range from the
final weights' cumulative completed boundary to that boundary plus the checked
requested increment.

The public entry points are `neural_model_train` and
`neural_model_train_range`; both resolve full-batch or mini-batch behavior from
the supplied training configuration.
