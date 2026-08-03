# Implementation Roadmap

- **Milestone 1 — complete:** strict project parsers, validation, and `inspect`.
- **Milestone 1.1 — complete:** project `init`, repeatable CLI options, safe
  `--force`, training-mode requests, and resume/checkpoint architecture.
- **Milestone 2 — complete:** parametrized activation specifications, dynamic
  runtime model, deterministic Xavier/He initialization, workspace, and forward
  pass.
- **Milestone 3 — next:** versioned weights/checkpoint payload parsers, canonical
  digests, and atomic I/O.
- **Milestone 4:** MSE, gradient checks, backpropagation, and training.
- **Milestone 5:** periodic checkpoints, signals, `--resume`, refinement, and
  end-to-end XOR convergence.
- **Milestone 6:** runtime validation on x86-64 and ppc64le.

Read `specification.md` and `training-resume.md` before changing formats,
training state, or persistence semantics.
