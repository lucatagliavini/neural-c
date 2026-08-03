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

Batch size is not yet public configuration. When exposed, it must be owned by
training configuration and its canonical digest because it changes parameter
updates. It must never be treated like execution-only `thread_count`.

## Parallel Execution

A persistent POSIX thread pool is created once per training run. Its effective
worker count is the smaller of the requested thread count and sample count.
Workers execute deterministic waves containing at most one sample per worker.
Each worker owns a `NeuralWorkerContext`, error, loss, sample index, and status.

After every wave, the main thread accumulates completed gradients by increasing
sample index into the batch accumulator. Worker buffers are then reused. This
keeps gradient memory proportional to `thread_count * parameter_count`, while
the arithmetic order remains independent of scheduling and thread count.
Workers never update the model. Errors are selected by the lowest failing
sample index. Cancellation is cooperative at task boundaries; asynchronous
thread cancellation is forbidden.

## Loss and Completion

Epoch reporting evaluates the complete dataset after all updates in that epoch,
so one reported value always describes one coherent model state. Milestone 4
runs exactly the configured number of epochs without early stopping. A fresh
successful run atomically replaces `weights.txt`; failures leave prior final
weights untouched. Checkpoints, signals, resume, and refinement are Milestone
5 responsibilities.
