# Project Specification

## Implementation Baseline

The framework uses C11 and `double` (`neural_real`) for all neural-network
values. Supported Linux targets are x86-64 and little-endian PowerPC 64. The
library must remain independent of third-party runtime dependencies.

## Project Files

A project directory uses the conventional filenames `model.txt`,
`project.conf`, `train.txt`, `weights.txt`, and `checkpoint.txt`, plus the
operational `.neural-c.lock`. Architecture belongs only in `model.txt`;
training parameters belong only in `project.conf`; samples belong only in
`train.txt`. Derived values such as output width and layer count must not be
repeated. Every text format starts with a versioned `neural-c` header and is
validated completely before execution. Lock ownership and command access modes
are defined in `project-locking.md`.

Real numbers use locale-independent ASCII decimal syntax: `.` is the decimal
separator and `e` or `E` introduces an optional base-10 exponent. Readers must
not interpret numbers according to the process locale, and writers must not
emit locale-specific separators.

Dense layers use the activation grammar documented in `model-runtime.md`.
Activation parameters belong only in `model.txt` and are covered by model
digests; weights and checkpoints must not duplicate them.
The version 1 persistence grammar, digest encoding, and atomic-write contract
are authoritative in `persistence-format.md`. Prediction snapshot ownership
and versioned CLI output are authoritative in `prediction.md`.

Thread count is execution-only configuration supplied by `--threads`; it must
not appear in project files, digests, weights, or checkpoints. Parallel code
must follow `parallel-execution.md` and preserve the documented deterministic
sample order.

## Defaults and Constants

Configurable compile-time defaults and resource limits must be declared in
`include/neural/defaults.h`, use the `NEURAL_DEFAULT_` prefix, and be guarded
with `#ifndef` so a build can override them consistently with `-D`. This
includes parser capacities, growth policy, conventional filenames, and bounded
message sizes. Source files must validate relationships between overridden
values at compile time where possible.

Protocol identity and version constants belong in `version.h`; public type
sizes belong in the header that defines the type. Runtime behavior such as
epochs, learning rate, loss, and random seed must remain in project files.
Defaults used by `init` may be compile-time constants, but must be materialized
in `project.conf`; training never reads hidden build defaults. Internal enum
values, bit masks, array indexes, and
mathematically intrinsic values such as zero and one do not need public macros.
Avoid duplicating literals when one declaration is authoritative.

## Project Initialization

`init` requires an input width and one or more ordered layer declarations. It
validates all options before filesystem changes and creates the three source
files through same-directory temporary files. If the target exists, the
operation fails unless `-f` or `--force` is present. Forced initialization may
replace only the conventional managed files and must preserve unrelated files.
Existing weights are removed because their shape may no longer match. Managed
files are staged for rollback so a failed commit restores the previous project.
An existing checkpoint is removed for the same compatibility reason.
Existing projects are exclusively locked before forced initialization changes
any managed file. A newly created directory is exclusively locked before its
project files are generated.

## Validation Policy

Parsers reject unknown or duplicate fields, invalid dimensions, non-finite
numbers, unsupported versions, arithmetic overflow, and inconsistent sample
widths. Failures should include the source path, line number when available,
and an actionable explanation. New behavior requires positive and negative
tests, followed by native tests, sanitizers, and both architecture builds.
