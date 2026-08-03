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
- **Milestone 4:** compose full-network backpropagation, finite-difference
  gradient checks, batch training, and loss reporting.
- **Milestone 5:** periodic checkpoints, signals, `--resume`, refinement, and
  end-to-end XOR convergence.
- **Milestone 6:** runtime validation on x86-64 and ppc64le.

Read `specification.md`, `model-runtime.md`, `parallel-execution.md`,
`persistence-format.md`, and `training-resume.md` before changing execution,
formats, training state, or persistence semantics.
