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

The CLI reserves two validated, mutually exclusive training modes:

```sh
./neural-c.sh train projects/example --resume
./neural-c.sh train projects/example --additional-epochs 2000
```

Their execution will be implemented with the training engine. Resume will use
the singular `checkpoint.txt`; refinement will start from final `weights.txt`.
The generated `checkpoint_interval` setting controls future periodic saves.

Run the architecture-appropriate executable through the launcher:

```sh
./neural-c.sh --version
./neural-c.sh inspect projects/xor
```

`inspect` loads and validates the complete project. `train` and `predict` are
present in the CLI but will be implemented in later milestones. The reusable
option parser supports `--option value`, `--option=value`, short options, and
`--` before positional values that begin with `-`.
