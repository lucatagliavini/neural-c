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
- **Milestone 4 — in progress:**
  - **4.1 — complete:** allocation-free single-sample backpropagation and
    workspace scratch buffers.
  - **4.2 — complete:** central finite-difference gradient checks with exact
    parameter restoration and activation coverage.
  - **4.3 — complete:** generic deterministic batch planning and ordered,
    transactional accumulation; public behavior remains full-batch.
  - **4.4 — next:** persistent pthread workers and bounded-memory execution
    waves.
  - **4.5:** coherent epoch loss reporting, fresh `train`, and atomic final
    weights.
- **Milestone 5:** periodic checkpoints, signals, `--resume`, refinement, and
  end-to-end XOR convergence.
- **Milestone 6:** runtime validation on x86-64 and ppc64le.

Read `specification.md`, `model-runtime.md`, `training-engine.md`,
`parallel-execution.md`, `persistence-format.md`, and `training-resume.md`
before changing execution, formats, training state, or persistence semantics.
