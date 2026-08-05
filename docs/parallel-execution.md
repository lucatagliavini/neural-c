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
model. `NeuralParallelExecutor` owns worker lifetime and joins; its single
coordinator owns the exclusive update boundary between batch calls.

## Deterministic Tasks and Reduction

Each epoch creates one logical sample-order plan and partitions its positions
into configured batch ranges. Disabled shuffle uses source order; enabled
shuffle uses the deterministic absolute-epoch plan specified in
`training-engine.md`. Each range contains exactly one logical task per sample.
`thread_count`
changes only how many workers execute those tasks; it never changes batch or
task boundaries. Each completed task leaves its sample
gradient in its worker context. After each bounded execution wave, the main
thread feeds those gradients to `NeuralBatchAccumulator` by increasing logical
plan position, then reuses the worker buffers. The source indices may be in any
shuffled order. Finalization forms the batch mean
only after the declared batch range is complete, using its actual sample count
before one coordinated model update. `NeuralExecutionPlan` derives
contiguous, bounded wave ranges from the batch range and effective worker count.

This correctness-first design makes floating-point grouping independent of
thread scheduling and thread count. Neumaier compensation improves numerical
accuracy while preserving that fixed order. Storage requires one gradient per
worker plus sum and compensation gradients, remaining proportional to
`(pool_size + 2) * parameter_count`. The persistent executor drives these
waves without allocating or copying one gradient per sample.
Workers must not update weights directly, use atomic floating-point additions,
or implement asynchronous “Hogwild” training.

Gradient norm calculation and clipping occur only on the coordinator after the
ordered mean gradient is finalized. Consequently worker scheduling and count
cannot change the norm, clipping decision, scaling factor, or update order.

## Execution Configuration

`-j N` or `--threads N` selects a positive worker count for `train` and
`predict`; the default is `NEURAL_DEFAULT_THREAD_COUNT` (currently 1). The
effective worker count is capped by the sample count. This is operational
configuration: it is absent from `project.conf`, canonical digests,
`weights.txt`, and `checkpoint.txt`. Resuming with a different thread count
must produce the same logical work and reduction order.

Shuffle is not execution configuration: its enablement and algorithm identity
are training provenance. Workers receive already selected source indices and
never advance a PRNG or derive their own order.

Prediction caps its worker count by the number of input samples. Each worker
owns a model workspace, shares only the immutable loaded snapshot, and writes
disjoint output slots. Sample results are serialized in original input order;
there is no cross-sample reduction. Output is therefore byte-identical across
thread counts. See `prediction.md` for the command and snapshot contract.

Concurrency tests use POSIX threads, matching the supported Linux targets.
`make test-thread-sanitize` builds and runs the shared-model test with
ThreadSanitizer when the host runtime supports it.

On WSL2, the GCC ThreadSanitizer shadow-memory layout conflicts with the
kernel's high-entropy ASLR. The Makefile detects the Microsoft kernel and runs
only the instrumented child through `setarch <machine> -R`; system-wide ASLR
settings remain unchanged. POSIX threads are used directly because the GNU
ThreadSanitizer runtime crashes inside glibc's C11 `thrd_create` wrapper on the
validated WSL environment, while the equivalent `pthread_create` path passes.
