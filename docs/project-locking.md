# Project Locking Contract

This document is authoritative for coordinating commands that access one
project directory concurrently.

## Lock File and Lifetime

Each initialized project owns a permanent `.neural-c.lock` regular file in its
project directory. The filename is the overridable
`NEURAL_DEFAULT_LOCK_FILENAME` convention. New files use mode `0600`. The file
is operational state: it is absent from canonical digests and persistence
payloads and is not an initialization-managed model, configuration, dataset,
weights, or checkpoint file.

The operating-system advisory lock on the open file description is
authoritative; the existence or contents of the file do not mean that a command
is active. Normal release closes the descriptor and never removes the file.
Keeping one inode prevents two processes from accidentally locking different
files during an unlink-and-recreate race. A failed brand-new initialization may
discard its lock file while rolling back the directory it just created.

Lock acquisition is non-blocking. Contention fails immediately with a project
busy error. Commands do not wait, retry, steal locks, inspect PIDs, or treat a
stale-looking file as permission to proceed. Filesystems used for projects must
provide the Linux `flock` shared/exclusive semantics supported by the target
platforms.

## Access Modes

Mutating operations acquire an exclusive lock before inspecting mutable
project state and retain it through their last persistence transition:

- `init --force` holds it across staging, replacement, and rollback.
- Fresh training holds it across initial state checks, all epochs, periodic
  checkpoints, final weights installation, and checkpoint removal.
- Resume follows the same rule through checkpoint validation, continuation, or
  interrupted-finalization reconciliation.
- Additional-epoch training holds it while validating baseline weights,
  running all further epochs, installing any recovery checkpoints, replacing
  final weights, and removing the checkpoint.

On `SIGINT` or `SIGTERM`, training retains the exclusive lock while reaching
the next coherent epoch boundary and installing its recovery checkpoint. The
lock is released only as the controlled training path unwinds. Abrupt process
termination instead relies on automatic descriptor closure and the atomic
persistence invariants described below.

A new `init` first claims a missing path with atomic directory creation, then
creates and acquires the directory's exclusive lock before generating files.

Read-only operations acquire a shared lock. `inspect` retains it until all
project files have been loaded, validated, and reported. Prediction will retain
it until model, project provenance, and weights have been loaded and validated;
inference may then continue from that in-memory snapshot after releasing the
lock. Multiple readers may coexist, but any reader fails immediately while a
mutating command holds the exclusive lock.

## Relationship to Atomic Persistence

Locking prevents cooperating commands from racing over logical project state.
Atomic replacement remains necessary for crash consistency and for recovery
after abrupt process termination, which releases the OS lock automatically but
may leave the documented checkpoint/finalization states on disk.
