# Training Continuation Architecture

This document is authoritative for future training and persistence work. The
CLI and public request types are present; weight and checkpoint payload I/O is
not implemented yet.

## File Responsibilities

- `model.txt` is the immutable network architecture.
- `checkpoint.txt` is the latest recoverable state of an incomplete run.
- `weights.txt` is the completed state used for prediction or refinement.

Use one singular checkpoint that is atomically replaced. Historical snapshots,
if introduced later, belong in a separate `checkpoints/` directory. Never put
mutable training state in `model.txt`.

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

## Planned Text Metadata

The checkpoint header and metadata will have this shape before the layer
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

`weights.txt` uses `neural-c weights 1`, includes the model digest and training
provenance, but excludes transient optimizer state. Floating-point values must
use enough decimal digits to round-trip a `double`.

Resume must reject unsupported versions, malformed or non-finite values,
dimension mismatches, digest mismatches, unknown optimizers, and completed
epochs greater than the target. Digests cover canonical parsed content rather
than raw whitespace, so comments and formatting do not invalidate state.

## Checkpoint Lifecycle

`checkpoint_interval` in `project.conf` controls periodic saves. A graceful
interrupt only sets a `sig_atomic_t` stop request; the training loop reaches an
epoch boundary, writes a same-directory temporary checkpoint, renames it
atomically, and exits. Successful completion atomically writes `weights.txt`
before removing `checkpoint.txt`. A crash therefore leaves either the previous
valid checkpoint or a state that finalization can validate and complete.
