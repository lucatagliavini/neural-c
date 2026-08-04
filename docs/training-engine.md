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

The internal trainer accepts a positive resolved batch size. Milestone 4 uses
the dataset sample count, making one full batch per epoch, while tests also
cover unit, partial, and incomplete final batches. Samples remain in source
order and are not shuffled. Each batch gradient is the ordered sum of its
sample gradients divided by the actual number of samples in that batch. The
main thread performs exactly one exclusive model update per batch.

`NeuralBatchPlan` validates the resolved batch size and maps each batch to a
contiguous half-open sample range. `NeuralBatchAccumulator` accepts gradients
only in increasing global sample-index order. It owns one model-shaped
sum plus one compensation buffer, can be reset between batches, and exposes
the gradient only after the declared `[begin, end)` range is complete and
finalization has divided it by that range's sample count. Failed,
out-of-order, or out-of-range additions do not advance its state. Neumaier
compensation preserves small terms without changing the required sample order;
every operation validates all results before changing either buffer.

Batch size is not yet public configuration. When exposed, it must be owned by
training configuration and its canonical digest because it changes parameter
updates. It must never be treated like execution-only `thread_count`.

## Parallel Execution

A `NeuralParallelExecutor` creates a persistent POSIX thread pool once per
training run. Pool size is the smaller of the requested thread count and
dataset sample count; each batch activates at most its own sample count.
Workers execute deterministic waves containing at most one sample per worker.
Each worker owns a `NeuralWorkerContext`, error, loss, sample index, and status.
`NeuralExecutionPlan` maps the current batch range to contiguous waves of
at most the effective worker count; the final wave may be smaller.

After every wave, the main thread accumulates completed gradients by increasing
sample index into the batch accumulator. Worker buffers are then reused. This
keeps gradient memory proportional to
`(pool_size + 2) * parameter_count`, including the sum and compensation
buffers. The arithmetic order remains independent of scheduling and thread
count. Workers never update the model. Errors are selected by the lowest
failing sample index. Cancellation is cooperative at task boundaries;
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

## Completion

Fresh training runs exactly the configured number of epochs, with one full
batch and one model update per epoch. It is allowed only when neither
`weights.txt` nor `checkpoint.txt` exists. The final model and canonical
project digests are written atomically to `weights.txt` only after every epoch
and report succeeds. Successful finalization installs `weights.txt` before
durably removing `checkpoint.txt`; a failure before the weights rename leaves
the latest successfully installed checkpoint available. If finalization is
interrupted after the weights rename, both valid files may remain for the
resume transition to reconcile. Refinement is a later Milestone 5
responsibility.

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
boundary to its validated target.
