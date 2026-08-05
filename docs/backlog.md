# Development Backlog

This file records work that is ready or intentionally deferred beyond the
completed items in `roadmap.md`. The session handoff identifies the active
entry to resume.

## Next session — Milestone 10.1

**Status:** ready to start in the next session.

**Goal:** introduce a versioned optimizer abstraction while retaining exact
gradient-descent behavior, deterministic traversal, and current persistence.

1. Define optimizer identity, lifecycle, parameter traversal, and error
   contracts without yet adding momentum or Adam state.
2. Route the established reduced, regularized, and clipped gradient through the
   abstraction while keeping gradient descent bit-identical when selected.
3. Define project configuration and digest ownership for optimizer identity;
   preserve absent legacy configuration and existing payload versions.
4. Keep the abstraction ready for transactional mutable buffers and versioned
   checkpoint state in Milestones 10.2 and 10.3 without persisting unused data.
5. Add disabled-compatibility, traversal, failure, worker-count, resume, and
   cross-runtime coverage.
6. Update authoritative contracts and qualify with `make check`,
   `make test-sanitize`,
   `make test-thread-sanitize`, and `make check-cross-runtime`.

## Deferred technical note

- Cross-entropy `evaluate` currently performs a second serial forward pass to
  obtain logits after parallel prediction computes metric inputs. The behavior
  is correct and deterministic; a combined parallel evaluation executor could
  remove that extra work in a future performance-focused checkpoint.

## Later roadmap

- Milestone 10.2 adds momentum and Adam after the optimizer boundary is stable.
