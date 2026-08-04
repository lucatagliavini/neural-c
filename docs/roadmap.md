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
- **Milestone 6 — complete:** cross-architecture qualification under native
  x86-64 and emulated ppc64le, without claiming native ppc64le hardware
  coverage.
  - **6.1 — complete:** define the supported native and emulated runtime matrix, fixture
    manifest, comparison rules, and failure-reporting convention.
  - **6.2 — complete:** capture the x86-64 reference results for deterministic training,
    interruption/resume, refinement, persistence round trips, and prediction.
  - **6.3 — complete:** execute the same matrix on emulated ppc64le and distinguish
    exact protocol requirements from documented floating-point tolerances.
  - **6.4 — complete:** verify bidirectional x86-64/ppc64le weights and checkpoint exchange,
    publish the qualification procedure, and add the repeatable release check.
- **Milestone 7 — complete — training observability and evaluation:**
  - **7.1 — complete:** add execution-only `--report-interval N` training telemetry, with
    zero disabling intermediate reports and positive values reporting absolute
    epoch, target, loss, absolute/relative improvement, best loss, and optional
    non-deterministic timing information. Keep it independent of checkpoint
    scheduling, persistence, and canonical digests.
  - **7.2 — complete:** specify validation and test dataset ownership, formats, provenance,
    locking, and compatibility rules without weakening finalized-weight
    validation.
  - **7.3 — complete:** add deterministic `evaluate` execution with loss, accuracy,
    confusion matrix, per-class precision/recall/F1 where the target contract
    represents classification, and machine-readable versioned output.
  - **7.4 — complete:** add optional bounded training-history export and state inspection,
    including the last finalized weights, active checkpoint target, and
    convergence history without making logs recovery-critical.
  - **7.5 — complete:** add validation-driven early stopping with `patience`, `min_delta`,
    explicit best-state ownership, resumable state, and atomic finalization of
    the selected model rather than merely the last epoch.
- **Milestone 8 — complete — scalable data interfaces:**
  - **8.1 — complete:** define a versioned multi-sample input document and extend prediction
    to consume files or standard input in bounded batches while preserving
    ordered, deterministic version 1-compatible results.
  - **8.2 — complete:** add strict CSV import with an explicit schema, locale-independent
    parsing, categorical label mapping, actionable row errors, and no implicit
    target or column guessing.
  - **8.3 — complete:** add versioned normalization/standardization metadata computed from
    training data only and applied identically to validation, test, and
    prediction inputs.
  - **8.4 — complete:** add deterministic train/validation/test splitting, including
    stratification for categorical targets and reproducible split provenance.
  - **8.4.1 — complete:** replace per-class floor sizing for new imports with
    exact global subset counts and deterministic largest-remainder
    apportionment, reserve training coverage per class, persist the split
    algorithm in preprocessing version 2, and retain version 1 loading and
    digest semantics.
  - **8.5 — complete:** define missing-value policy and reject, impute, or transform values
    only through explicit persisted configuration.
- **Milestone 9 — losses and configurable gradient descent:**
  - **9.1 — complete:** generalize the loss contract and add numerically stable binary and
    categorical cross-entropy, including the fused softmax/cross-entropy path,
    gradient checks, and compatibility validation between targets, outputs,
    activations, and loss.
  - **9.2:** expose deterministic mini-batch size as training-owned
    configuration covered by canonical provenance, including incomplete final
    batches and exact continuation semantics.
  - **9.3:** add deterministic per-epoch shuffling with a specified PRNG stream,
    stable sample-order plans across thread counts, and sufficient checkpoint
    state for exact resume.
  - **9.4:** add gradient norm reporting and configurable clipping with explicit
    ordering relative to reduction, regularization, and optimizer updates.
  - **9.5:** add L1 and L2 regularization with documented bias treatment,
    training-digest ownership, and consistent objective reporting.
- **Milestone 10 — optimizers and convergence control:**
  - **10.1:** introduce a versioned optimizer abstraction while preserving the
    existing gradient-descent behavior and deterministic parameter traversal.
  - **10.2:** add momentum and Adam with finite-state validation, checked update
    arithmetic, and exact deterministic tests.
  - **10.3:** version checkpoint persistence for optimizer buffers, schedule
    state, shuffle state, and early-stopping state; retain explicit backward
    loading rules for version 1 gradient-descent and version 2 early-stopping
    checkpoints.
  - **10.4:** add constant, step, exponential, and plateau-driven learning-rate
    schedules whose current value and next transition survive resume exactly.
  - **10.5:** integrate divergence detection, minimum-loss targets, maximum
    no-improvement limits, and clear completion reasons into training results
    and CLI output.
- **Milestone 11 — regularized and structured models:**
  - **11.1:** add deterministic training-only dropout with explicit inference
    behavior, per-run PRNG ownership, checkpoint state, and gradient checks.
  - **11.2:** add normalization layers with completely specified training and
    inference statistics and transactional persistence.
  - **11.3:** add convolution and pooling layers behind the existing model and
    workspace ownership contracts, with architecture-neutral parameter layout.
  - **11.4:** add embedding layers and categorical-index validation.
  - **11.5:** add recurrent layers only after defining sequence input, masking,
    truncation, state ownership, and deterministic batching contracts.
- **Milestone 12 — embedding, interoperability, and release readiness:**
  - **12.1:** define and test a stable library-facing inference API with explicit
    ownership, error, thread-safety, and ABI/version boundaries.
  - **12.2:** add supported model/data import and export paths without bypassing
    canonical validation or silently changing numerical semantics.
  - **12.3:** add reproducible performance and memory benchmarks for training,
    evaluation, bulk inference, and persistence on both target architectures.
  - **12.4:** publish compatibility, migration, operational recovery, and release
    documentation, then run the complete native, sanitizer, thread-sanitizer,
    cross-build, and cross-runtime qualification matrix.

## Roadmap Completion Rules

Every checkpoint begins with its authoritative contract and ownership rules.
Behavior-changing configuration belongs in project provenance; execution-only
presentation or resource controls do not. Any new mutable training state must
have an exact interruption/resume story before its implementation is complete.

Each checkpoint requires deterministic positive and negative fixtures, exact
one-versus-many-thread comparisons where applicable, native tests, ASan/UBSan,
ThreadSanitizer when supported, both architecture builds, and documentation.
Persistence or protocol changes additionally require malformed-input tests,
transactional failure tests, backward-compatibility rules, and bidirectional
x86-64/ppc64le round trips.

Read `specification.md`, `model-runtime.md`, `training-engine.md`,
`parallel-execution.md`, `project-locking.md`, `persistence-format.md`,
`training-resume.md`, `prediction.md`, `losses.md`, and
`runtime-validation.md` before
changing execution, formats, training state, inference, persistence, or
cross-architecture validation semantics.
