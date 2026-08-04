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

## Build and test

```sh
make build-native
make build-ppc64le
make test
make test-thread-sanitize
make verify-binaries
```

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
exists. Full-batch updates and epoch loss reporting are deterministic across
worker counts. Mutating commands hold an exclusive, non-blocking project lock
in `.neural-c.lock`; `inspect` uses a shared lock and reports an immediate
project-busy error when training or forced initialization is active.

Resume continues a validated checkpoint and may use a different execution-only
worker count:

```sh
./neural-c.sh train projects/example --resume
```

It also safely completes interrupted finalization when both valid persistence
files remain. Refinement is reserved for the next continuation checkpoint:

```sh
./neural-c.sh train projects/example --additional-epochs 2000
```

Refinement will start from final `weights.txt`.
The generated `checkpoint_interval` setting controls periodic saves; setting
it to `0` disables them while retaining the final `weights.txt` write.
`SIGINT` and `SIGTERM` stop at the next completed epoch and atomically save one
recovery checkpoint even when the periodic interval is disabled. After a
successful emergency save the command exits with status 130 or 143,
respectively, and the run can continue with `--resume`.
Versioned parsers, exact `double` round trips, SHA-256 compatibility checks,
and atomic replacement are already implemented for both persistence files.

`train` and `predict` accept execution-only `-j N` or `--threads N`. The value
defaults to 1 and is intentionally excluded from project files, digests, and
checkpoints. Training uses a persistent pool with one private worker context
per thread and deterministic, bounded-memory gradient accumulation.

Run the architecture-appropriate executable through the launcher:

```sh
./neural-c.sh --version
./neural-c.sh inspect projects/xor
```

`inspect` loads and validates the complete project and prints its canonical
model, dataset, and training SHA-256 digests. Fresh training and resume are
implemented; `predict` and refinement remain planned. The reusable option parser
supports `--option value`, `--option=value`, short options, and `--` before
positional values that begin with `-`.
