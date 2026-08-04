# Prediction Contract

This document is authoritative for loading finalized model snapshots,
executing inference, and formatting prediction CLI output.

## Command Input

`predict` accepts a project directory followed by one or more complete samples:

```sh
./neural-c.sh predict projects/xor 0 0 0 1 1 0 1 1 --threads 4
```

The positional numeric values form one flat sequence. The model input width
partitions that sequence into samples, so the example above contains four
two-input samples. At least one sample is required and trailing partial samples
are rejected. Values use the locale-independent finite decimal grammar shared
by project files. Use the option terminator when a value begins with `-`, for
example `predict PROJECT --threads 2 -- -1 0`.

Bulk input is available as `predict PROJECT --input FILE|- --batch-size N`.
The versioned document, bounded-memory behavior, explicit `?` missing token,
and all-or-nothing output rule are specified in `data-interfaces.md`.

`--threads N` is execution-only. The effective worker count is the smaller of
the positive requested count and the sample count. It never changes sample
boundaries, values, ordering, persistence, or output text.

## Snapshot Loading

Prediction acquires the project's non-blocking shared lock, loads the complete
project, computes all canonical digests, constructs the runtime model, and
loads `weights.txt` transactionally. Weights must match model, dataset, and
training provenance. Version 1 weights must report at least the configured
fresh-training epoch count; version 2 early-stopping weights instead validate
their observed, target, selected, and completion relationship. A missing,
malformed, incomplete, or incompatible weights file fails before inference.

The shared lock is retained until all project and weights data have been copied
and validated. It is then released before inference. The resulting model and
any persisted preprocessing form an immutable in-memory snapshot, so a later
training command cannot change an
already running prediction. Prediction fails immediately if a writer holds the
lock while the snapshot is being loaded.

When stable baseline weights coexist with a refinement checkpoint, prediction
may load that finalized baseline. The output's `completed_epochs` identifies
the exact snapshot version. A fresh-training checkpoint without weights is not
predictable.

## Deterministic Execution

Each prediction worker owns one model workspace and writes only its assigned
sample output range. Workers share the immutable model. Samples may execute in
parallel, but output slots and serialized lines remain in input order. No
floating-point reduction occurs across samples, so changing the thread count
produces byte-identical output. If multiple samples fail, the command reports
the lowest failing sample index.

## Version 1 Output

Successful output is a complete versioned text document:

```text
neural-c predictions 1
completed_epochs 10000
samples 4
inputs 2
outputs 1
sample 0 0.025499005160150735
sample 1 0.97114493017532899
sample 2 0.97588667050672029
sample 3 0.022639502034234268
end
```

Metadata precedes exactly one indexed `sample` line per input sample. Each line
contains exactly the model output width in model order. Indices are zero-based
and contiguous. Numeric outputs use `DBL_DECIMAL_DIG` precision and the C
numeric locale, matching persistence round-trip conventions. `end` terminates
the document. Operational worker counts are deliberately omitted so output is
identical across execution configurations.

## Version 2 Early-Stopping Output

When finalized weights use the early-stopping format, prediction emits version
2 and adds `selected_epoch`, `target_epochs`, and
`completion target|early_stopping` after `completed_epochs`. Samples are always
computed from the selected model payload. The separation makes an early stop,
or a target-reaching run whose best epoch was earlier, explicit without
changing version 1 output for existing weights.
