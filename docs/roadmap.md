# Implementation Roadmap

- **Milestone 1 — complete:** strict project parsers, validation, and `inspect`.
- **Milestone 1.1 — complete:** project `init`, repeatable CLI options, safe
  `--force`, training-mode requests, and resume/checkpoint architecture.
- **Milestone 2 — complete:** parametrized activation specifications, dynamic
  runtime model, deterministic Xavier/He initialization, workspace, and forward
  pass.
- **Milestone 3 — complete:** versioned weights/checkpoint payload parsers,
  canonical SHA-256 digests, transactional model loading, and durable atomic
  I/O.
- **Milestone 3.1 — complete:** domain-specific dense/loss/tensor primitives,
  activation derivatives, model-shaped private gradients, deterministic sample
  tasks and reduction, thread-aware worker contexts, and isolated atomic I/O.
- **Milestone 4 — complete:**
  - **4.1 — complete:** allocation-free single-sample backpropagation and
    workspace scratch buffers.
  - **4.2 — complete:** central finite-difference gradient checks with exact
    parameter restoration and activation coverage.
  - **4.3 — complete:** generic deterministic batch planning and ordered,
    transactional accumulation; public behavior remains full-batch.
  - **4.3.1 — complete:** complete batch-range invariants,
    Neumaier-compensated accumulation, and bounded execution-wave planning.
  - **4.4 — complete:** persistent pthread workers, deterministic error
    selection, reusable bounded-memory execution waves, and coordinator-only
    accumulation.
  - **4.5 — complete:** coherent epoch loss reporting, deterministic fresh
    training, end-to-end XOR convergence, and atomic final weights.
- **Milestone 5 — complete:**
  - **5.1 — complete:** project-owned periodic atomic checkpoints at completed
    epoch boundaries, zero-interval opt-out, failure propagation, and
    weights-before-checkpoint-removal finalization; permanent non-blocking
    project reader/writer locking protects command-level state transitions.
  - **5.2 — complete:** validated checkpoint resume over absolute epoch ranges,
    execution-only worker changes, and interrupted-finalization recovery.
  - **5.3 — complete:** minimal `SIGINT` and `SIGTERM` stop requests,
    one coherent atomic recovery checkpoint, resumable interruption, and
    conventional 130/143 exit statuses after successful persistence.
  - **5.4 — complete:** cumulative refinement through `--additional-epochs`,
    stable baseline weights, resumable absolute-epoch checkpoints, checked
    targets, repeated runs, and graceful interruption.
  - **5.5 — complete:** validated immutable weight snapshots, shared-lock
    loading, deterministic multi-sample parallel inference, versioned output,
    and end-to-end XOR prediction.
- **Milestone 6:** runtime validation on x86-64 and ppc64le.

Read `specification.md`, `model-runtime.md`, `training-engine.md`,
`parallel-execution.md`, `project-locking.md`, `persistence-format.md`,
`training-resume.md`, and `prediction.md` before changing execution, formats,
training state, inference, or persistence semantics.
