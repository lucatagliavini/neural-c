# Development Backlog

This file records work that is ready or intentionally deferred beyond the
completed items in `roadmap.md`. The session handoff identifies the active
entry to resume.

## Next checkpoint — Milestone 11.1

**Status:** ready for planning after Milestone 10 field usage.

**Goal:** add deterministic training-only dropout without weakening inference,
resume, provenance, or cross-runtime contracts.

1. Specify dropout placement, probability validation, inverted scaling, and
   the exact distinction between training and inference.
2. Define a run-owned PRNG stream independent from initialization and sample
   shuffling, with deterministic masks across worker counts.
3. Persist all continuation state needed to reproduce uninterrupted training.
4. Cover dropout identity at probability zero, gradient checks under fixed
   masks, resume equivalence, and architecture exchange.
5. Keep existing dense models and persistence files byte-compatible.

Milestones 11.2–11.5 remain ordered behind this checkpoint in `roadmap.md`.
The recommended product step before starting them is to exercise Milestone 10
on real datasets and record any interface or convergence-control friction.

## Deferred technical note

- Cross-entropy `evaluate` currently performs a second serial forward pass to
  obtain logits after parallel prediction computes metric inputs. The behavior
  is correct and deterministic; a combined parallel evaluation executor could
  remove that extra work in a future performance-focused checkpoint.
