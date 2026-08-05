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
and targets exactly `N` further epochs. The persisted completed-epoch count is
absolute and cumulative across refinements. Resume and additional epochs are
mutually exclusive, and a target addition that would overflow `size_t` is
rejected before disk mutation.

A checkpoint always represents the boundary after a fully completed epoch.
This remains sufficient for deterministic mini-batch training because sample
order is regenerated from immutable project configuration plus the absolute
epoch index, and batch boundaries slice that logical order.
An interruption is observed only at the next completed epoch boundary; a hard
failure before that boundary persists no partial epoch, so recovery repeats
from the prior coherent checkpoint. Refinement refuses
while any checkpoint exists and directs the user to `--resume`. During active
refinement both persistence files deliberately coexist: `weights.txt` remains
the last stable baseline and `checkpoint.txt` owns the in-progress run.

## Resume Validation and State Machine

Resume acquires the exclusive project lock before examining persistence state.
It loads the project, computes canonical digests, constructs compatible runtime
models, and validates complete persistence payloads before any disk mutation.
For checkpoint-only fresh training, the target must equal the configured
project epoch count in addition to the persistence-format invariants. A
refinement checkpoint may exceed that configured count and is identified by a
validated baseline `weights.txt` whose completed count is smaller than the
checkpoint target. A different execution-only worker count is allowed.

The valid and invalid file combinations are:

| `checkpoint.txt` | `weights.txt` | Resume action |
| --- | --- | --- |
| absent | absent | Fail: no resumable state. |
| absent | present | Fail: the project is already finalized. |
| present | absent | Continue only a configured fresh run, or finalize it directly at target. |
| present | present | Validate finalization or active refinement from the epoch relationship. |

When weights report exactly the checkpoint target, they represent interrupted
finalization. The checkpoint must independently be valid for that target. When
it is also at target, its model parameters must be bit-identical to final
weights; an older periodic checkpoint may legitimately contain earlier
parameters. Resume then removes only the checkpoint without rewriting weights.

When weights report fewer epochs than the checkpoint target, they represent a
refinement baseline. Their completed count must be at least the configured
fresh-training epoch count, the checkpoint may not precede that baseline, and
same-epoch checkpoint parameters must be bit-identical to it. Resume continues
from the checkpoint model and atomically replaces the baseline only at the new
target. Weights beyond the target, incomplete baseline weights, malformed
payloads, provenance mismatches, or invalid epoch relationships leave both
files untouched.

After loading a checkpoint-only state, resume runs the absolute epoch range
`[completed_epochs, target_epochs)`. Periodic saves retain absolute epoch
numbering and the original target. A zero interval suppresses those periodic
replacements but retains the checkpoint that made resume possible. Successful
completion atomically installs final weights and then durably removes the
checkpoint. A failure before completion produces no final weights; an ambiguous
post-rename failure is reconciled by the two-file rule on the next resume.

## Refinement Transition

Additional training acquires the exclusive lock, requires valid final weights
and no checkpoint, and computes `target = completed + N` with checked
arithmetic. The original `weights.txt` remains byte-for-byte unchanged
throughout training and is therefore still a valid model if the process
crashes before the first completed epoch.

Updates run over the absolute range `[baseline, target)`. Periodic checkpoint
boundaries use the same cumulative epoch numbering as fresh training and
resume. On success, the refined model atomically replaces `weights.txt` with
the cumulative target count, then the checkpoint is durably removed. A failure
before any checkpoint leaves only the unchanged baseline and the refinement
can be restarted. A graceful signal, or a later failure after a periodic save,
leaves the baseline plus a resumable checkpoint. With `checkpoint_interval 0`,
normal refinement performs no checkpoint write and persists only the final
weights. Repeated refinements apply the same transition from the latest
completed count. Gradient descent has no additional optimizer buffers; its run
state and model RNG state are reset deterministically for each refinement.

With early stopping, the checkpoint atomically owns both the current and best
model plus the best epoch/loss and stale count. Resume restores that full state
and continues the original patience sequence. Final weights own the selected
best model and distinguish the last observed epoch from the selected epoch.
An early-finalized version 2 weights file is a valid additional-training
baseline even when its observed epoch count is below the configured target.

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

`batch_size` is deliberately different: it changes update boundaries, belongs
to `project.conf`, and is covered by the training digest. Changing it makes
existing weights and checkpoints fail provenance validation before mutation.
Enabled `shuffle` is also training-owned and digest-bound. It requires no
checkpoint field: resume of `[completed_epochs, target_epochs)` regenerates its
first plan for zero-based absolute epoch `completed_epochs`. Disabled or absent
shuffle preserves legacy source order and its canonical digest.
Positive `gradient_clip_norm` is likewise training-owned and digest-bound, so
changing it rejects existing persistence before mutation. Resume needs no new
field: every regenerated batch gradient is measured and clipped independently,
and no norm or clipping counter affects the next epoch's state.
Positive L1 or L2 regularization is training-owned and digest-bound together
with its bias policy. It needs no checkpoint field: every batch derives its
penalty gradient from the coherent model at that update boundary. Resume and
continuous execution therefore reproduce the same terms, objective, clipping
decision, and model update without mutable regularization state.

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
