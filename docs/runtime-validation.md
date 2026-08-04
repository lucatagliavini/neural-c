# Cross-Architecture Runtime Validation

This document defines Milestone 6 qualification for native x86-64 and
emulated ppc64le execution. It complements the execution, persistence,
continuation, locking, and prediction contracts; it does not redefine their
behavior.

## Coverage Claim

The repeatable project qualification runs x86-64 natively and ppc64le through
QEMU user-mode on the same Linux filesystem. It proves that ppc64le code is
cross-compiled, starts with its target loader and C library, passes the C and
CLI fixtures, and interoperates with x86-64 persistence files.

This is functional ppc64le runtime coverage, not native-hardware coverage.
QEMU results do not qualify PowerPC performance, host-kernel integration,
hardware-specific races, or sanitizer runtimes. Native ppc64le CI or hardware
may be added later without changing the emulated matrix.

## Host Requirements

The default x86-64 runner expects:

- `powerpc64le-linux-gnu-gcc`;
- `qemu-ppc64le` from `qemu-user`;
- `/usr/powerpc64le-linux-gnu` containing `lib/ld64.so.2` and the target C
  library;
- ordinary Linux process, signal, `/proc`, `flock`, and filesystem semantics.

Override the executable and sysroot when necessary:

```sh
make check-cross-runtime \
    PPC64LE_QEMU=/path/to/qemu-ppc64le \
    PPC64LE_SYSROOT=/path/to/ppc64le-sysroot
```

The runner invokes QEMU explicitly and does not require a writable or
registered `binfmt_misc` handler. `tests/run_ppc64le.sh` is the CLI adapter;
it uses `exec` so signals and `/proc/<pid>/fd` observations refer to the QEMU
process that owns the emulated command's file descriptors.

## Fixture Manifest

The canonical XOR project supplies the cross-runtime training fixture. Its
required source digests are:

```text
model    8da5a1f53fc59cabe5e685895a6e254d6b362ff4704f3d6b7325d98b7e4cc0d5
dataset  df97fb50ab811253bc1ebfceb93dea8679f5c337bb9995f384f47a7936991275
training 08778dfaba9557e47d924f3010e4b972935e718b994c3a65eb2f0aac5de4ab6d
```

The native baseline must train to 10,000 epochs, classify all four XOR inputs
with the documented thresholds, produce identical prediction documents with
one and four workers, survive signal interruption/resume, preserve a stable
refinement baseline, and pass transactional weights/checkpoint round trips.

## Comparison Rules

The following values are exact across architectures:

- protocol versions, dimensions, epoch counts, field order, and terminators;
- canonical SHA-256 digests;
- integer PRNG sequences and serialized binary64 round trips;
- persistence metadata and validation decisions;
- prediction output across worker counts within one runtime.

Cross-architecture floating-point results use an absolute tolerance of
`1e-14` or a relative tolerance of `1e-12`. Different target `libm`
implementations may legally produce last-bit differences in activations, so
byte equality is recorded when observed but is not the portability contract.
Every compared prediction must also satisfy the XOR classification thresholds.

## Validation Matrix

| Area | Native x86-64 | Emulated ppc64le | Cross-runtime |
| --- | --- | --- | --- |
| Parsers, model, math, gradients | Complete C suite | Same sources cross-built and executed | Canonical digests exact |
| Parallel training and prediction | One/many-worker fixtures | Same fixtures under QEMU | Ordered results compared |
| Persistence and atomic I/O | Round-trip and failure fixtures | Same fixtures under QEMU | Weights loaded both directions |
| Locking and signals | C and CLI fixtures | C and CLI fixtures | Checkpoint resume both directions |
| Resume and refinement | C and CLI fixtures | C and CLI fixtures | Finalized snapshots readable by both |

Run individual layers with:

```sh
make test-ppc64le
make test-ppc64le-cli
make test-cross-runtime
```

The complete repeatable qualification is:

```sh
make check-cross-runtime
```

`test-cross-runtime` creates isolated projects under
`build/tests/cross-runtime`, compares native and emulated baseline training,
loads each architecture's weights under the other runtime, interrupts training
on each side, and resumes the checkpoint on the other. Successful runs remove
their generated projects. Set `KEEP_CROSS_RUNTIME_ARTIFACTS=1` only while
diagnosing a failure.

The matrix also imports, trains, and predicts an independently created
softmax/categorical-cross-entropy project on both architectures. Its canonical
training digest is exact and its predictions use the floating-point tolerance
above. That fixture uses a training batch size that leaves an incomplete final
batch, covering deterministic mini-batch execution and persistence provenance
on both runtimes.
