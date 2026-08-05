# neural-c

`neural-c` is a small C11 framework for defining, training, and running neural
networks from human-readable text files. Models, datasets, configuration, and
weights live together in a project directory such as `projects/xor/`.

The numerical type is `double`. Linux builds are supported for x86-64 and
little-endian PowerPC 64.

Design rules and format ownership are recorded in
[`docs/specification.md`](docs/specification.md). Training continuation is
specified in [`docs/training-resume.md`](docs/training-resume.md), and current
milestone status is tracked in [`docs/roadmap.md`](docs/roadmap.md).
Runtime layout and activation rules are defined in
[`docs/model-runtime.md`](docs/model-runtime.md).
Weights, checkpoints, and canonical digests are defined in
[`docs/persistence-format.md`](docs/persistence-format.md).
Thread ownership and deterministic reduction are defined in
[`docs/parallel-execution.md`](docs/parallel-execution.md).
Backpropagation, batch semantics, and the persistent worker pool are defined in
[`docs/training-engine.md`](docs/training-engine.md).
Prediction snapshots and versioned output are defined in
[`docs/prediction.md`](docs/prediction.md).
Native and emulated architecture qualification is defined in
[`docs/runtime-validation.md`](docs/runtime-validation.md).
Bulk input, CSV schemas, splitting, normalization, and missing-value ownership
are defined in [`docs/data-interfaces.md`](docs/data-interfaces.md).
Loss, activation, target, and fused-gradient contracts are defined in
[`docs/losses.md`](docs/losses.md).

## Project format

Each project uses conventional filenames, so paths and dimensions are not
duplicated in configuration. `model.txt` owns the topology:

```text
neural-c model 1
input 2
dense 2 sigmoid
dense 1 sigmoid
```

Supported activations are `linear`, `sigmoid`, `tanh`, `relu`, `leaky_relu`,
`elu`, and `softmax`. Parameterized forms are explicit:

```text
dense 8 leaky_relu alpha=0.01
dense 4 elu alpha=1
```

`project.conf` contains only training parameters, while `train.txt` contains
rows in the form `0 1 -> 1`. The loader derives the output width from the final
layer and validates every dataset row before execution. Blank lines and `#`
comments are accepted; unknown or duplicate properties are rejected.

Supported losses are `mse`, `binary_cross_entropy` for sigmoid outputs with
binary targets, and `categorical_cross_entropy` for softmax outputs with
one-hot targets. Existing MSE projects remain fully compatible.

Strict CSV ingestion can build the native datasets and persisted preprocessing
without column guessing:

```sh
./neural-c.sh import-csv projects/iris iris.csv \
    --schema iris-schema.txt \
    --validation-ratio 0.2 --test-ratio 0.2 --split-seed 42 \
    --normalization standardize --missing mean
```

Categorical schemas produce a deterministic stratified split with exact global
subset counts and at least one training sample per class when feasible.
Statistics and imputation values are learned from training rows only and reused
by prediction.

## Build and test

```sh
make build-native
make build-ppc64le
make test
make test-thread-sanitize
make test-ppc64le
make test-ppc64le-cli
make test-cross-runtime
make check-cross-runtime
make verify-binaries
```

On an x86-64 host with `qemu-user`, the ppc64le cross-toolchain, and its
sysroot installed, `make check-cross-runtime` cross-builds the complete C test
suite, executes it under `qemu-ppc64le`, runs the full CLI integration suite,
and verifies bidirectional weights and checkpoint exchange with native
x86-64. See [`docs/runtime-validation.md`](docs/runtime-validation.md) for the
qualification contract and the boundary between emulated and native coverage.

## Create a project

Use one repeatable `--layer` option for every dense layer. The final layer
defines the output width:

```sh
./neural-c.sh init projects/example \
    --inputs 3 \
    --layer 5:leaky_relu:alpha=0.01 \
    --layer 2:sigmoid
```

The command creates `model.txt`, `project.conf`, and an empty `train.txt`
template. The parent directory must already exist. Training options are
optional and their documented defaults are written explicitly to
`project.conf`. If the project directory exists, initialization fails without
changes; `-f` or `--force` replaces only neural-c managed files, removes stale
weights, and preserves unrelated files.

## Resume and refinement

Fresh training validates the project, initializes the model from its configured
seed, and writes `weights.txt` atomically only after all epochs complete:

```sh
./neural-c.sh train projects/example --threads 4
```

Fresh training refuses to start if `weights.txt` or `checkpoint.txt` already
exists. Full-batch and mini-batch updates and epoch loss reporting are
deterministic across worker counts. Configure training batches during
initialization with `--batch-size N`; zero selects the full dataset, while a
positive value creates logical mini-batches with a possibly smaller final
batch. Add `--shuffle` during initialization to enable deterministic per-epoch
Fisher-Yates ordering. The order is derived from the configured seed and
absolute epoch, so worker-count changes and checkpoint resume reproduce the
same updates exactly. Existing projects without `shuffle`, and new projects
without `--shuffle`, retain source order. `--gradient-clip-norm V` enables
deterministic L2 clipping of each finalized mean batch gradient. Training
reports the maximum pre-clipping gradient norm and number of clipped batches
for the last epoch; zero, the default, disables clipping and preserves legacy
updates exactly. `--l1 V` and `--l2 V` add deterministic regularization before
norm measurement and clipping; weights are regularized by default, while
`--regularize-biases` includes biases explicitly. Progress and completion keep
data `loss` distinct from the regularized `objective`. The current
`--optimizer gradient_descent` default passes updates through the versioned
optimizer boundary while preserving established results exactly. Mutating
commands hold an exclusive, non-blocking project lock
in `.neural-c.lock`; `inspect` uses a shared lock and reports an immediate
project-busy error when training or forced initialization is active.

Resume continues a validated checkpoint and may use a different execution-only
worker count:

```sh
./neural-c.sh train projects/example --resume
```

It also safely completes interrupted finalization and refinement when both
valid persistence files remain. Start refinement from final weights with:

```sh
./neural-c.sh train projects/example --additional-epochs 2000
```

Refinement retains the prior `weights.txt` as a stable baseline until all
additional epochs complete, records cumulative epoch counts, and may be
repeated. If interrupted, continue its checkpoint with `--resume`.
The generated `checkpoint_interval` setting controls periodic saves; setting
it to `0` disables them while retaining the final `weights.txt` write.
`SIGINT` and `SIGTERM` stop at the next completed epoch and atomically save one
recovery checkpoint even when the periodic interval is disabled. After a
successful emergency save the command exits with status 130 or 143,
respectively, and the run can continue with `--resume`.
Versioned parsers, exact `double` round trips, SHA-256 compatibility checks,
and atomic replacement are already implemented for both persistence files.

Enable deterministic progress and optional diagnostic history independently of
checkpoint frequency:

```sh
./neural-c.sh train projects/example --report-interval 100 --history
```

Early stopping is configured at initialization with
`--early-stopping-patience N --early-stopping-min-delta X` and requires a
compatible `validation.txt`. Its resumable checkpoint atomically retains both
the current and selected-best model; final weights always contain the selected
model.

`train` and `predict` accept execution-only `-j N` or `--threads N`. The value
defaults to 1 and is intentionally excluded from project files, digests, and
checkpoints. Training uses a persistent pool with one private worker context
per thread and deterministic, bounded-memory gradient accumulation.

Run the architecture-appropriate executable through the launcher:

```sh
./neural-c.sh --version
./neural-c.sh inspect projects/xor
./neural-c.sh inspect projects/xor --state
./neural-c.sh evaluate projects/xor --dataset test --threads 4
```

Predict one or more samples by supplying a flat sequence partitioned by the
model input width:

```sh
./neural-c.sh predict projects/xor 0 0 0 1 1 0 1 1 --threads 4
```

For large inference jobs, use a versioned input document from a file or stdin;
`--batch-size` bounds working memory without changing output order:

```sh
./neural-c.sh predict projects/xor --input samples.txt --batch-size 1024 \
    --threads 4
```

The versioned result reports the loaded weights' cumulative epoch count and
one ordered `sample` line per input. Snapshot loading holds a shared project
lock through complete provenance validation, then inference runs from immutable
memory. Consequently, later training cannot change an in-flight prediction and
the output is identical across worker counts.

Validation/test ownership, metric contracts, reporting, history, and early
stopping are specified in
[`docs/observability-evaluation.md`](docs/observability-evaluation.md).
For a complete command-oriented workflow, optimizer and schedule examples, and
the external input/output protocol, see [`docs/usage.md`](docs/usage.md).

`inspect` loads and validates the complete project and prints its canonical
model, dataset, and training SHA-256 digests. Fresh training, resume,
refinement, and prediction are implemented. The reusable option parser
supports `--option value`, `--option=value`, short options, and `--` before
positional values that begin with `-`.
