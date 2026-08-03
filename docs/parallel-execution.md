# Parallel Execution Contract

This document is authoritative for execution-only thread settings, ownership,
deterministic work partitioning, and gradient reduction.

## Ownership and Thread Safety

A fully constructed `NeuralModel` may be shared by concurrent forward passes
while it remains read-only. Every worker must own a distinct
`NeuralWorkerContext`, which bundles one `NeuralWorkspace` and one writable
`NeuralGradient`. A workspace, gradient, or `NeuralError` must never be written
by two threads. Worker contexts must not outlive their model.

Parameter setters, gradient application, persistence loading, and training
updates require exclusive model access. Persistence saving may read a model
only while no writer can change it. The library does not hide locks inside the
model; the future executor owns worker lifetime, joins, and the exclusive
update boundary.

## Deterministic Tasks and Reduction

The logical task plan contains exactly one task per sample, always in original
sample order. `thread_count` changes only how many workers execute those tasks;
it never changes task boundaries. Each completed task produces a private
sample gradient by copying its worker scratch gradient into the corresponding
task slot. After all workers join, `neural_gradient_reduce_ordered` adds those
gradients by increasing sample index. The batch mean is formed by scaling that
sum by `1 / sample_count` before one coordinated model update.

This correctness-first design makes floating-point grouping independent of
thread scheduling and thread count. It requires one gradient result per active
batch sample; a future bounded-memory optimization must preserve the same
logical reduction order or explicitly version the reproducibility contract.
Workers must not update weights directly, use atomic floating-point additions,
or implement asynchronous “Hogwild” training.

## Execution Configuration

`-j N` or `--threads N` selects a positive worker count for `train` and
`predict`; the default is `NEURAL_DEFAULT_THREAD_COUNT` (currently 1). The
effective worker count is capped by the sample count. This is operational
configuration: it is absent from `project.conf`, canonical digests,
`weights.txt`, and `checkpoint.txt`. Resuming with a different thread count
must produce the same logical work and reduction order.

Concurrency tests use C11 threads. `make test-thread-sanitize` builds and runs
the shared-model test with ThreadSanitizer when the host runtime supports it.
