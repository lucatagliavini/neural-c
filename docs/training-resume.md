# Training Continuation Architecture

This document is authoritative for training continuation. The exact payload
grammar is defined in `persistence-format.md`, epoch execution in
`training-engine.md`, and concurrency in `project-locking.md`.

## File Responsibilities

- `model.txt` is the immutable network architecture.
- `checkpoint.txt` is the latest recoverable state of an incomplete run.
- `weights.txt` is the completed state used for prediction or refinement.
- `.neural-c.lock` is permanent operational coordination state and is not
  persistence provenance.

Use one singular checkpoint that is atomically replaced. Historical snapshots,
if introduced later, belong in a separate `checkpoints/` directory. Never put
mutable training state in `model.txt`.

Every mutating training mode holds the exclusive project lock from its initial
state checks through final persistence. Contention fails immediately; see
`project-locking.md` for reader behavior and crash-release semantics.

## Training Modes

`train <project>` starts a fresh run only when neither persistence file exists.
`train <project> --resume` requires `checkpoint.txt` and continues to the
checkpoint target. `train <project> --additional-epochs N` requires
`weights.txt`, starts a new run from those weights, resets optimizer/run state,
and targets exactly `N` further epochs. Resume and additional epochs are
mutually exclusive.

A checkpoint always represents the boundary after a fully completed epoch.
Interrupted partial epochs are discarded and repeated. If both persistence
files remain after an interrupted finalization, resume validates both and
finishes the commit; refinement must refuse while a checkpoint exists.

## Resume Validation and State Machine

Resume acquires the exclusive project lock before examining persistence state.
It loads the project, computes canonical digests, constructs compatible runtime
models, and validates complete persistence payloads before any disk mutation.
The checkpoint target must equal the configured project epoch count in addition
to the persistence-format invariants. A different execution-only worker count
is allowed.

The valid and invalid file combinations are:

| `checkpoint.txt` | `weights.txt` | Resume action |
| --- | --- | --- |
| absent | absent | Fail: no resumable state. |
| absent | present | Fail: the project is already finalized. |
| present | absent | Continue from checkpoint, or finalize directly when already at target. |
| present | present | Validate interrupted finalization, then remove only the checkpoint. |

For interrupted finalization, weights must match all project digests and report
exactly the checkpoint target as their completed epoch count. The checkpoint
must independently be valid for the same target. When the checkpoint is also
at target, its model parameters must be bit-identical to final weights; an older
periodic checkpoint may legitimately contain earlier parameters. Any malformed
payload, provenance mismatch, epoch mismatch, or same-epoch parameter mismatch
leaves both files untouched.

After loading a checkpoint-only state, resume runs the absolute epoch range
`[completed_epochs, target_epochs)`. Periodic saves retain absolute epoch
numbering and the original target. A zero interval suppresses those periodic
replacements but retains the checkpoint that made resume possible. Successful
completion atomically installs final weights and then durably removes the
checkpoint. A failure before completion produces no final weights; an ambiguous
post-rename failure is reconciled by the two-file rule on the next resume.

## Text Metadata

The checkpoint header and metadata have this shape before the layer
payload defined from the row-major layout in `model-runtime.md`:

```text
neural-c checkpoint 1
model_digest sha256:<64 lowercase hexadecimal characters>
dataset_digest sha256:<64 lowercase hexadecimal characters>
training_digest sha256:<64 lowercase hexadecimal characters>
completed_epochs 3500
target_epochs 10000
optimizer gradient_descent
rng_state 123456789
```

`weights.txt` uses `neural-c weights 1`, includes all three provenance digests
and the completed epoch count, but excludes transient optimizer and RNG state.
Floating-point values use `DBL_DECIMAL_DIG` digits to round-trip exactly.

Resume must reject unsupported versions, malformed or non-finite values,
dimension mismatches, digest mismatches, unknown optimizers, and completed
epochs greater than the target. Digests cover canonical parsed content rather
than raw whitespace, so comments and formatting do not invalidate state.

Thread count is deliberately absent from checkpoint metadata. Resume may use a
different worker count because logical sample tasks and gradient reduction
remain ordered as specified in `parallel-execution.md`.

## Checkpoint Lifecycle

`checkpoint_interval` in `project.conf` controls periodic saves. During fresh
training, the project-layer epoch observer saves at positive completed-epoch
multiples of that interval, including the target epoch when it is an interval
boundary. A value of zero disables periodic saves, so normal completion writes
only `weights.txt`. Zero remains part of the canonical training digest. A
graceful `SIGINT` or `SIGTERM` stop still writes an explicit recovery
checkpoint because signal handling is independent of the periodic schedule.
The checkpoint contains the canonical project digests, configured target,
gradient-descent optimizer identity, current model RNG state, and the
parameters at exactly that completed epoch boundary. The trainer remains
independent of paths, intervals, and persistence policy.

Each save uses same-directory atomic replacement. A failure before rename
leaves the previous checkpoint byte-for-byte unchanged; a failure reported
after rename may leave either the previous or the newly completed payload, but
never a partially serialized destination. In either case training stops and
does not write final weights. Temporary files are cleaned up on handled
failure paths.

A graceful interrupt handler only records the first signal in a
`volatile sig_atomic_t` stop request. The training loop reaches the next
coherent epoch boundary, writes a same-directory temporary checkpoint, renames
it atomically, and stops without producing final weights. After a successful
emergency save the CLI returns 130 for `SIGINT` or 143 for `SIGTERM`; a failed
save returns the ordinary runtime failure status and reports the persistence
error. The recovery checkpoint is resumable through `train --resume`, even
when `checkpoint_interval` is zero. Successful completion atomically writes
`weights.txt` before durably removing `checkpoint.txt`. A crash can therefore
leave the previous valid checkpoint, the latest valid checkpoint, or both a
valid checkpoint and final weights for later finalization recovery.
