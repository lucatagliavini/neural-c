# Runtime Model Architecture

This document is authoritative for model execution and parameter layout.

## Description and Ownership

`NeuralModelSpec` is the owned, parsed description of `model.txt`.
`NeuralModel` is an opaque runtime object with a deep copy of every activation
specification, plus allocated weights and biases. `NeuralWorkspace` owns
pre-activation, activation, backward scratch, and model-input gradient buffers
for one model. A workspace must not outlive or be used with a different model.
Saved forward values and the last successful model-input gradient are available
through read-only accessors.

Each dense layer stores weights in neuron-major row order:

```text
weights[neuron_index * input_count + input_index]
```

Biases follow neuron order. Future weights and checkpoint payloads must use
this exact layout. Model construction checks every size multiplication and
addition before allocation. Public setters reject mismatched or non-finite
parameters.

## Activation Specifications

The model grammar accepts zero or more named numeric parameters:

```text
dense 8 relu
dense 8 leaky_relu alpha=0.01
dense 4 elu alpha=1
dense 3 softmax
```

`init --layer` uses colon-separated equivalents, such as
`8:leaky_relu:alpha=0.01`. Parameters are stored and emitted in canonical enum
order. Unknown, duplicate, missing, non-finite, or out-of-range values are
invalid. Supported kinds are linear, sigmoid, tanh, ReLU, leaky ReLU, ELU, and
softmax. Softmax is vector-valued; all others are scalar.

## Initialization and Execution

SplitMix64 supplies a specified random sequence, converted to `double` with 53
random bits. ReLU, leaky ReLU, and ELU use He uniform initialization. Other
activations use Xavier uniform initialization. Biases start at zero. The final
PRNG state is retained for future checkpoint provenance.

Forward execution validates buffer dimensions and finite inputs, computes
dense pre-activations through `dense.c`, then dispatches the whole layer to its
activation.
Sigmoid and softmax use numerically stable formulations. Any non-finite
intermediate is an error. Do not enable unsafe floating-point optimizations
such as `-ffast-math`.

Separate workspaces avoid allocation during forward execution and reserve
per-layer values needed by backpropagation. Forward and single-sample backward
perform no allocation after workspace and gradient construction. Numerically
sensitive tests use tolerances across architectures; PRNG integer sequence
tests are exact.

## Numerical Modules and Gradients

Numerical responsibilities are domain-specific rather than collected in a
generic `math.c`:

- `dense.c` implements neuron-major dense forward and single-sample backward.
- `activation.c` implements activation forward and Jacobian-vector products.
- `loss.c` implements MSE and its derivative.
- `tensor_ops.c` implements checked zero, ordered addition, and scaling.
- `gradient.c` owns model-shaped gradient storage, transactional addition,
  ordered reduction, and the single transactional parameter update.
- `batch.c` owns contiguous batch planning and deterministic batch-gradient
  accumulation; it does not execute samples or update model parameters.

MSE is the mean of squared output differences; its derivative is normalized by
the output count. Dense backward produces gradients for inputs, weights, and
biases without modifying the model. At non-differentiable zero, ReLU uses
derivative 0 and leaky ReLU uses the non-negative branch derivative 1. Softmax
backward computes the full Jacobian-vector product rather than an element-wise
approximation.

`neural_model_sample_gradient` performs forward evaluation, MSE evaluation,
and reverse traversal into a model-compatible `NeuralGradient`. It requires a
workspace and gradient owned by the same exact model. Output values are valid
only on success; a failed sample must never participate in batch reduction.

See `parallel-execution.md` for ownership and deterministic reduction rules.
