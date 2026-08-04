# Persistence Format

This document is authoritative for version 1 and early-stopping version 2
weights/checkpoints, canonical digests, and atomic replacement.

## Text Payload

`weights.txt` starts with:

```text
neural-c weights 1
model_digest sha256:<64 lowercase hexadecimal characters>
dataset_digest sha256:<64 lowercase hexadecimal characters>
training_digest sha256:<64 lowercase hexadecimal characters>
completed_epochs <non-negative integer>
```

`checkpoint.txt` uses `neural-c checkpoint 1`, then the same four metadata
lines followed by `target_epochs`, `optimizer gradient_descent`, and
`rng_state`. The target must be positive and not smaller than the completed
epoch count. The saved RNG state must equal the runtime model state.
Epoch counts are cumulative: refined weights may exceed the configured fresh
epoch count, and an active refinement checkpoint may target a larger absolute
count while the previous finalized weights remain beside it.

The payload then contains every layer in zero-based order:

```text
layer <index>
weights
<one double per expected weight>
biases
<one double per expected bias>
end_layer
```

One final `end` terminates the file. Counts and dimensions are deliberately
absent: the loader obtains them from `model.txt`, using the neuron-major layout
in `model-runtime.md`. It stages and validates the complete payload before
changing the runtime model. Numbers always use the locale-independent `.`
decimal separator and are written with `DBL_DECIMAL_DIG` precision.

Early-stopping final weights use `neural-c weights 2` and add
`selected_epoch`, `target_epochs`, and `completion target|early_stopping` after
`completed_epochs`. Their single payload is the selected best model, not
necessarily the model observed at `completed_epochs`. Early-stopping
checkpoints use `neural-c checkpoint 2`; after the common checkpoint metadata
they store `best_epoch`, `best_loss`, and `stale_epochs`, followed by marked
`current_model` and `best_model` payloads. Both models and decision metadata are
installed by one atomic replacement.

## Canonical Digests

SHA-256 operates on parsed values, not source text. Consequently, comments,
blank lines, and equivalent whitespace do not affect a digest. Canonical
integers are unsigned 64-bit big-endian values. Strings are a canonical length
followed by their bytes. A `double` is its IEEE 754 binary64 bit pattern encoded
as a big-endian integer; this makes x86-64 and ppc64le agree.

The model stream contains the format tag/version, input and layer counts, then
each neuron count, activation name, and activation parameters in enum order.
The dataset stream contains its dimensions and values in sample/input/output
order. The training stream contains epochs, learning rate, seed, loss, and
checkpoint interval. A zero checkpoint interval canonically represents
disabled periodic saves. When early stopping is enabled, the stream also
contains its patience/minimum delta and the canonical validation digest.
Disabled early stopping preserves the version 1 training digest exactly.
Changing any owned value changes only its corresponding digest.

## Atomic Replacement

Writers create a unique `*.tmp.XXXXXX` file beside the destination, write and
flush it, synchronize it with `fsync`, then rename it over the destination and
synchronize the parent directory. Temporary files are removed after failures.
The resulting files use owner-only permissions because weights may represent
valuable or sensitive trained state.
Atomic filesystem mechanics live in the internal `atomic_file.c`; persistence
format parsing and serialization remain isolated in `persistence.c`.
