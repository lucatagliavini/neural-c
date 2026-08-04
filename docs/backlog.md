# Development Backlog

This file records work that is ready or intentionally deferred beyond the
completed items in `roadmap.md`. The session handoff identifies the active
entry to resume.

## Next session — Milestone 9.3

**Status:** ready to start in the next session.

**Goal:** add deterministic per-epoch sample shuffling without changing
results across worker counts or breaking exact checkpoint continuation.

1. Specify the shuffle contract before implementation:
   - the exact PRNG algorithm and seed/absolute-epoch derivation;
   - Fisher-Yates traversal and unbiased bounded-index generation;
   - whether the plan is regenerated from absolute epoch identity or requires
     persisted mutable state;
   - compatibility and digest/version consequences for existing persistence.
2. Introduce an epoch sample-order plan that is independent of worker count and
   partitions the shuffled order into the existing deterministic mini-batches.
3. Preserve ordered gradient reduction according to the logical epoch plan,
   while workers remain scheduling-only execution resources.
4. Make fresh, resumed, and additional training produce the same epoch plans
   for the same absolute epoch numbers.
5. Add deterministic tests for:
   - repeated plans and distinct epoch plans;
   - one and many workers;
   - full and incomplete final batches;
   - continuous versus checkpoint-resumed training;
   - x86-64 and emulated ppc64le persistence exchange.
6. Update the authoritative training, parallel, persistence, continuation,
   runtime-validation, specification, README, roadmap, and session handoff
   documents.
7. Qualify with `make check`, `make test-sanitize`,
   `make test-thread-sanitize`, and `make check-cross-runtime`.

## Deferred technical note

- Cross-entropy `evaluate` currently performs a second serial forward pass to
  obtain logits after parallel prediction computes metric inputs. The behavior
  is correct and deterministic; a combined parallel evaluation executor could
  remove that extra work in a future performance-focused checkpoint.

## Later roadmap

- Continue with Milestone 9.4 gradient norms and clipping after 9.3 is complete.
- Milestone 9.5 adds L1/L2 regularization.
- Optimizer and convergence work remains grouped under Milestone 10.
