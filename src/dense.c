#include "neural/dense.h"

#include <math.h>
#include <stdint.h>

static int dimensions_match(size_t weight_count,
                            size_t bias_count,
                            size_t input_count,
                            size_t output_count)
{
    return input_count != 0U && output_count != 0U &&
           bias_count == output_count &&
           input_count <= SIZE_MAX / output_count &&
           weight_count == input_count * output_count;
}

int neural_dense_forward(const neural_real *weights,
                         size_t weight_count,
                         const neural_real *biases,
                         size_t bias_count,
                         const neural_real *inputs,
                         size_t input_count,
                         neural_real *outputs,
                         size_t output_count,
                         NeuralError *error)
{
    size_t neuron_index;

    if (weights == NULL || biases == NULL || inputs == NULL ||
        outputs == NULL ||
        !dimensions_match(weight_count,
                          bias_count,
                          input_count,
                          output_count)) {
        neural_error_set(error, "invalid dense forward arguments");
        return 0;
    }
    for (neuron_index = 0U;
         neuron_index < output_count;
         neuron_index++) {
        neural_real sum = biases[neuron_index];
        size_t input_index;
        size_t offset = neuron_index * input_count;

        if (!isfinite(sum)) {
            neural_error_set(error, "dense biases must be finite");
            return 0;
        }
        for (input_index = 0U; input_index < input_count; input_index++) {
            if (!isfinite(weights[offset + input_index]) ||
                !isfinite(inputs[input_index])) {
                neural_error_set(error, "dense inputs and weights must be finite");
                return 0;
            }
            sum += weights[offset + input_index] * inputs[input_index];
        }
        if (!isfinite(sum)) {
            neural_error_set(error, "dense output is not finite");
            return 0;
        }
        outputs[neuron_index] = sum;
    }
    return 1;
}

int neural_dense_backward(const neural_real *weights,
                          size_t weight_count,
                          const neural_real *inputs,
                          size_t input_count,
                          const neural_real *output_gradients,
                          size_t output_count,
                          neural_real *input_gradients,
                          size_t input_gradient_count,
                          neural_real *weight_gradients,
                          size_t weight_gradient_count,
                          neural_real *bias_gradients,
                          size_t bias_gradient_count,
                          NeuralError *error)
{
    size_t neuron_index;
    size_t input_index;

    if (weights == NULL || inputs == NULL || output_gradients == NULL ||
        input_gradients == NULL || weight_gradients == NULL ||
        bias_gradients == NULL || input_gradient_count != input_count ||
        weight_gradient_count != weight_count ||
        !dimensions_match(weight_count,
                          bias_gradient_count,
                          input_count,
                          output_count)) {
        neural_error_set(error, "invalid dense backward arguments");
        return 0;
    }
    for (neuron_index = 0U; neuron_index < output_count; neuron_index++) {
        size_t offset = neuron_index * input_count;
        neural_real output_gradient = output_gradients[neuron_index];

        if (!isfinite(output_gradient)) {
            neural_error_set(error, "dense output gradients must be finite");
            return 0;
        }
        bias_gradients[neuron_index] = output_gradient;
        for (input_index = 0U; input_index < input_count; input_index++) {
            neural_real gradient;

            if (!isfinite(weights[offset + input_index]) ||
                !isfinite(inputs[input_index])) {
                neural_error_set(error, "dense backward values must be finite");
                return 0;
            }
            gradient = output_gradient * inputs[input_index];
            if (!isfinite(gradient)) {
                neural_error_set(error, "dense weight gradient is not finite");
                return 0;
            }
            weight_gradients[offset + input_index] = gradient;
        }
    }
    for (input_index = 0U; input_index < input_count; input_index++) {
        neural_real gradient = 0.0;

        for (neuron_index = 0U;
             neuron_index < output_count;
             neuron_index++) {
            gradient += weights[neuron_index * input_count + input_index] *
                        output_gradients[neuron_index];
        }
        if (!isfinite(gradient)) {
            neural_error_set(error, "dense input gradient is not finite");
            return 0;
        }
        input_gradients[input_index] = gradient;
    }
    return 1;
}
