# neural-c Usage Guide

This guide covers the normal workflow from source data to prediction. Commands
run from the repository root. `./neural-c.sh` selects the executable for the
host architecture; use `make build-native` first when no binary exists.

## 1. Build and inspect the CLI

```sh
make build-native
./neural-c.sh --version
./neural-c.sh --help
```

A neural-c project is a directory. Its model, training configuration, native
datasets, preprocessing metadata, weights, and recovery checkpoint remain
separate files so an external application can produce inputs and consume
outputs without linking to the C implementation.

## 2. Create a model

Declare the input width and repeat `--layer` in execution order. The last layer
defines the output width.

```sh
mkdir -p projects
./neural-c.sh init projects/example \
    --inputs 4 \
    --layer 8:relu \
    --layer 3:softmax \
    --loss categorical_cross_entropy \
    --epochs 500 \
    --batch-size 32 \
    --shuffle
```

Common output/loss pairs are:

| Task | Output layer | Loss |
| --- | --- | --- |
| Regression | `N:linear` | `mse` |
| Binary classification | `1:sigmoid` | `binary_cross_entropy` |
| Multi-class classification | `N:softmax` | `categorical_cross_entropy` |

`init` validates all options before changing files. Use `--force` only when
intentionally replacing the managed files of an existing project.

## 3. Supply data

### Native text data

Add one sample per line to `train.txt`:

```text
5.1 3.5 1.4 0.2 -> 1 0 0
6.4 3.2 4.5 1.5 -> 0 1 0
6.3 3.3 6.0 2.5 -> 0 0 1
```

The number of values on each side must match the model input and output widths.
Optional `validation.txt` and `test.txt` use the same format.

### CSV import

For CSV data, define every column explicitly. Example schema:

```text
neural-c csv-schema 1
columns 5
header yes
inputs 0 1 2 3
label 4
class setosa 1 0 0
class versicolor 0 1 0
class virginica 0 0 1
end
```

Import, split, normalize, and impute in one transactional operation:

```sh
./neural-c.sh import-csv projects/example data/flowers.csv \
    --schema data/flowers.schema \
    --validation-ratio 0.2 \
    --test-ratio 0.2 \
    --split-seed 42 \
    --normalization standardize \
    --missing mean
```

Normalization and imputation statistics are fitted from the training subset
only. They are persisted and automatically reused for prediction.

## 4. Choose optimizer and learning-rate behavior

Gradient descent is the compatibility default:

```sh
--optimizer gradient_descent
```

Momentum and Adam are configured during `init`:

```sh
# Momentum
--optimizer momentum --momentum 0.9

# Adam
--optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 \
    --adam-epsilon 1e-8
```

The base rate comes from `--learning-rate`. Available schedules are:

```sh
# Constant rate (default)
--lr-schedule constant

# Multiply by 0.5 every 50 completed epochs
--lr-schedule step --lr-decay 0.5 --lr-step-epochs 50

# Multiply by 0.99 after every completed epoch
--lr-schedule exponential --lr-decay 0.99

# Multiply by 0.5 after 10 stale loss observations
--lr-schedule plateau --lr-decay 0.5 \
    --lr-plateau-patience 10 --lr-plateau-min-delta 1e-5
```

Step and exponential transitions affect the next epoch. Plateau monitors the
coherent full training loss at epoch boundaries. Optimizer buffers, current
rate, next transition, plateau counters, and convergence counters are stored in
checkpoint version 3 and resume exactly.

## 5. Configure stopping and safety controls

All controls are optional and use the coherent full training loss:

```sh
--target-loss 0.01
--max-no-improvement-epochs 20 --no-improvement-min-delta 1e-5
--divergence-threshold 1000
```

`target_loss` and the no-improvement limit are successful completion reasons.
The CLI reports `target_epochs`, `loss_target`, `no_improvement`, or
`early_stopping`. A loss above the divergence threshold is a runtime failure:
neural-c does not publish new final weights from that run. Defaults `-1`, `0`,
and `0` respectively disable these controls.

Validation-driven early stopping is separate and selects the best validation
model:

```sh
--early-stopping-patience 10 --early-stopping-min-delta 1e-5
```

It requires `validation.txt`.

## 6. Train, monitor, resume, and refine

```sh
./neural-c.sh train projects/example --threads 4
```

Optional progress and history are execution settings and do not alter model
provenance:

```sh
./neural-c.sh train projects/example --threads 4 \
    --report-interval 10 --history
```

`checkpoint_interval` in `project.conf` controls periodic recovery snapshots.
`SIGINT` and `SIGTERM` request a checkpoint at the next coherent epoch even
when the interval is zero. Resume with any supported worker count:

```sh
./neural-c.sh train projects/example --resume --threads 2
```

After final weights exist, continue from them with a new run state:

```sh
./neural-c.sh train projects/example --additional-epochs 100
```

Fresh training refuses existing weights or checkpoints. `--resume` requires a
checkpoint; `--additional-epochs` requires finalized weights.

## 7. Inspect and evaluate

```sh
./neural-c.sh inspect projects/example
./neural-c.sh inspect projects/example --state
./neural-c.sh evaluate projects/example --dataset test --threads 4
```

`inspect` validates the project and prints canonical provenance digests.
`--state` summarizes finalized weights and any recoverable checkpoint.
`evaluate` emits a versioned machine-readable document with loss and, for
classification contracts, accuracy, confusion matrix, precision, recall, and
F1.

## 8. Predict from an external application

For a small number of samples, pass a flat sequence. It is partitioned by the
model input width:

```sh
./neural-c.sh predict projects/example 5.1 3.5 1.4 0.2
```

For bounded streaming input, use a versioned document:

```text
neural-c inputs 1
samples 2
inputs 4
sample 0 5.1 3.5 1.4 0.2
sample 1 6.3 3.3 6.0 2.5
end
```

Consume a file:

```sh
./neural-c.sh predict projects/example --input request.txt \
    --batch-size 1024 --threads 4
```

Or connect processes through standard input and output:

```sh
producer | ./neural-c.sh predict projects/example --input - >result.txt
```

Output is a strict versioned document with ordered `sample` rows. It is staged
until the complete request validates, remains deterministic across batch and
worker counts, and includes the finalized weights' epoch and completion
metadata. This is the recommended boundary for an external input/output
interpreter.

## 9. Project files at a glance

| File | Purpose |
| --- | --- |
| `model.txt` | Architecture and activations |
| `project.conf` | Training, optimizer, schedule, and stopping configuration |
| `train.txt` | Training samples in native format |
| `validation.txt` | Optional validation samples |
| `test.txt` | Optional held-out evaluation samples |
| `preprocessing.txt` | Persisted CSV transforms and provenance |
| `weights.txt` | Final immutable model snapshot |
| `checkpoint.txt` | Incomplete-run recovery state |
| `history.txt` | Optional diagnostic progress history |
| `.neural-c.lock` | Non-blocking project coordination lock |

Do not edit project-owned configuration or datasets underneath existing
weights/checkpoints: digest validation will reject the mismatch. To change a
training-owned setting, intentionally create a new project or replace the old
managed state.
