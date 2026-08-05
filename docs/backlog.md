# Development Backlog

This file records work that is ready or intentionally deferred beyond the
completed items in `roadmap.md`. The session handoff identifies the active
entry to resume.

## Next session — Milestone 9.5

**Status:** ready to start in the next session.

**Goal:** add configurable L1 and L2 regularization with explicit bias,
objective-reporting, clipping-order, provenance, and continuation semantics.

1. Specify the exact L1/L2 objective convention, coefficient scaling, and
   whether biases are excluded by default or controlled explicitly.
2. Add regularization to the finalized mean data gradient before norm
   measurement and clipping, preserving deterministic parameter traversal and
   the Milestone 9.4 update contract.
3. Define whether training, validation, and test loss report the data loss or
   regularized objective, and expose any additional metric without ambiguity.
4. Treat enabled regularization as training-owned digest provenance while
   leaving disabled legacy streams and persistence payload versions unchanged.
5. Add analytic-gradient, finite-difference, zero/disabled, bias-policy,
   clipping-interaction, worker-count, resume, and cross-runtime coverage.
6. Update authoritative contracts and qualify with `make check`,
   `make test-sanitize`,
   `make test-thread-sanitize`, and `make check-cross-runtime`.

## Deferred technical note

- Cross-entropy `evaluate` currently performs a second serial forward pass to
  obtain logits after parallel prediction computes metric inputs. The behavior
  is correct and deterministic; a combined parallel evaluation executor could
  remove that extra work in a future performance-focused checkpoint.

## Later roadmap

- Optimizer and convergence work remains grouped under Milestone 10.
