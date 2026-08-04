#include <math.h>
#include <stdio.h>
#include <string.h>

#include "neural/activation.h"
#include "neural/dense.h"
#include "neural/loss.h"
#include "neural/tensor_ops.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int nearly_equal(neural_real left,
                        neural_real right,
                        neural_real tolerance)
{
    return fabs(left - right) <= tolerance;
}

static void test_tensor_operations(void)
{
    neural_real values[] = {1.0, -2.0, 3.0};
    const neural_real additions[] = {0.5, 1.0, -1.0};
    NeuralError error;

    check(neural_tensor_add(values, additions, 3U, &error),
          "tensor addition must succeed");
    check(values[0] == 1.5 && values[1] == -1.0 && values[2] == 2.0,
          "tensor addition must preserve element order");
    check(neural_tensor_scale(values, 3U, 2.0, &error),
          "tensor scaling must succeed");
    check(values[0] == 3.0 && values[1] == -2.0 && values[2] == 4.0,
          "tensor scaling must apply to every element");
    check(neural_tensor_zero(values, 3U, &error),
          "tensor zeroing must succeed");
    check(values[0] == 0.0 && values[1] == 0.0 && values[2] == 0.0,
          "tensor zeroing must clear every element");
}

static void test_dense_operations(void)
{
    const neural_real weights[] = {1.0, 2.0, 3.0, 4.0};
    const neural_real biases[] = {0.5, -0.5};
    const neural_real inputs[] = {2.0, -1.0};
    const neural_real output_gradients[] = {0.25, -0.5};
    neural_real outputs[2];
    neural_real input_gradients[2];
    neural_real weight_gradients[4];
    neural_real bias_gradients[2];
    NeuralError error;

    check(neural_dense_forward(weights,
                               4U,
                               biases,
                               2U,
                               inputs,
                               2U,
                               outputs,
                               2U,
                               &error),
          "dense forward must succeed");
    check(outputs[0] == 0.5 && outputs[1] == 1.5,
          "dense forward must use neuron-major weights");
    check(neural_dense_backward(weights,
                                4U,
                                inputs,
                                2U,
                                output_gradients,
                                2U,
                                input_gradients,
                                2U,
                                weight_gradients,
                                4U,
                                bias_gradients,
                                2U,
                                &error),
          "dense backward must succeed");
    check(input_gradients[0] == -1.25 &&
              input_gradients[1] == -1.5,
          "dense backward must calculate input gradients");
    check(weight_gradients[0] == 0.5 &&
              weight_gradients[1] == -0.25 &&
              weight_gradients[2] == -1.0 &&
              weight_gradients[3] == 0.5,
          "dense backward must calculate neuron-major weight gradients");
    check(bias_gradients[0] == 0.25 && bias_gradients[1] == -0.5,
          "dense backward must calculate bias gradients");
}

static void test_loss_operations(void)
{
    const neural_real predicted[] = {1.0, 3.0};
    const neural_real expected[] = {0.0, 1.0};
    neural_real gradient[2];
    neural_real value;
    NeuralError error;

    check(neural_loss_evaluate(NEURAL_LOSS_MSE,
                               predicted,
                               expected,
                               2U,
                               &value,
                               &error),
          "MSE evaluation must succeed");
    check(value == 2.5, "MSE must be the mean squared difference");
    check(neural_loss_gradient(NEURAL_LOSS_MSE,
                               predicted,
                               expected,
                               2U,
                               gradient,
                               &error),
          "MSE gradient must succeed");
    check(gradient[0] == 1.0 && gradient[1] == 2.0,
          "MSE gradient must include output-count normalization");

    {
        const neural_real logits[] = {0.0, 0.0};
        const neural_real probabilities[] = {0.5, 0.5};
        const neural_real binary_targets[] = {0.0, 1.0};

        check(neural_loss_evaluate_with_logits(
                  NEURAL_LOSS_BINARY_CROSS_ENTROPY,
                  NEURAL_ACTIVATION_SIGMOID,
                  logits,
                  probabilities,
                  binary_targets,
                  2U,
                  &value,
                  &error) && fabs(value - log(2.0)) < 1e-15,
              "binary cross-entropy must use stable logits");
        check(neural_loss_pre_activation_gradient(
                  NEURAL_LOSS_BINARY_CROSS_ENTROPY,
                  NEURAL_ACTIVATION_SIGMOID,
                  probabilities,
                  binary_targets,
                  2U,
                  gradient,
                  &error) && gradient[0] == 0.25 && gradient[1] == -0.25,
              "sigmoid binary cross-entropy must use its fused gradient");
    }
    {
        const neural_real logits[] = {1000.0, 0.0, -1000.0};
        const neural_real probabilities[] = {1.0, 0.0, 0.0};
        const neural_real categorical_target[] = {1.0, 0.0, 0.0};
        neural_real categorical_gradient[3];

        check(neural_loss_evaluate_with_logits(
                  NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY,
                  NEURAL_ACTIVATION_SOFTMAX,
                  logits,
                  probabilities,
                  categorical_target,
                  3U,
                  &value,
                  &error) && value == 0.0,
              "categorical cross-entropy must remain finite for extreme logits");
        check(neural_loss_pre_activation_gradient(
                  NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY,
                  NEURAL_ACTIVATION_SOFTMAX,
                  probabilities,
                  categorical_target,
                  3U,
                  categorical_gradient,
                  &error) && categorical_gradient[0] == 0.0 &&
                  categorical_gradient[1] == 0.0 &&
                  categorical_gradient[2] == 0.0,
              "softmax categorical cross-entropy must use p minus y");
    }
    {
        const neural_real invalid_target[] = {0.5};
        const neural_real invalid_probability[] = {1.5};
        const neural_real binary_target[] = {1.0};

        check(!neural_loss_validate_output(
                  NEURAL_LOSS_BINARY_CROSS_ENTROPY,
                  NEURAL_ACTIVATION_LINEAR,
                  1U,
                  &error) && strstr(error.message, "sigmoid") != NULL,
              "binary cross-entropy must reject non-sigmoid outputs");
        check(!neural_loss_validate_targets(
                  NEURAL_LOSS_BINARY_CROSS_ENTROPY,
                  invalid_target,
                  1U,
                  1U,
                  &error) && strstr(error.message, "zero or one") != NULL,
              "binary cross-entropy must reject non-binary targets");
        check(!neural_loss_evaluate(NEURAL_LOSS_BINARY_CROSS_ENTROPY,
                                    invalid_probability,
                                    binary_target,
                                    1U,
                                    &value,
                                    &error) &&
                  strstr(error.message, "probabilities") != NULL,
              "cross-entropy must reject values outside probability range");
    }
}

static void test_activation_backward(void)
{
    NeuralActivationSpec sigmoid = {NEURAL_ACTIVATION_SIGMOID, 0U, NULL};
    NeuralActivationSpec softmax = {NEURAL_ACTIVATION_SOFTMAX, 0U, NULL};
    NeuralActivationSpec relu = {NEURAL_ACTIVATION_RELU, 0U, NULL};
    NeuralActivationSpec leaky = {NEURAL_ACTIVATION_LEAKY_RELU, 0U, NULL};
    NeuralActivationSpec elu = {NEURAL_ACTIVATION_ELU, 0U, NULL};
    const neural_real sigmoid_pre[] = {0.0};
    const neural_real sigmoid_output[] = {0.5};
    const neural_real sigmoid_upstream[] = {2.0};
    const neural_real softmax_pre[] = {1.0, 2.0, 3.0};
    const neural_real softmax_output[] = {0.2, 0.3, 0.5};
    const neural_real softmax_upstream[] = {1.0, -2.0, 0.5};
    const neural_real piecewise_pre[] = {-1.0, 0.0, 2.0};
    const neural_real relu_output[] = {0.0, 0.0, 2.0};
    const neural_real leaky_output[] = {-0.1, 0.0, 2.0};
    const neural_real elu_output[] = {2.0 * expm1(-1.0),
                                      0.0,
                                      2.0};
    const neural_real unit_upstream[] = {1.0, 1.0, 1.0};
    neural_real gradient[3];
    NeuralError error;

    check(neural_activation_backward(&sigmoid,
                                     sigmoid_pre,
                                     sigmoid_output,
                                     sigmoid_upstream,
                                     gradient,
                                     1U,
                                     &error),
          "sigmoid backward must succeed");
    check(gradient[0] == 0.5,
          "sigmoid backward must use y times one-minus-y");

    check(neural_activation_backward(&softmax,
                                     softmax_pre,
                                     softmax_output,
                                     softmax_upstream,
                                     gradient,
                                     3U,
                                     &error),
          "softmax Jacobian-vector product must succeed");
    check(nearly_equal(gradient[0], 0.23, 1e-15) &&
              nearly_equal(gradient[1], -0.555, 1e-15) &&
              nearly_equal(gradient[2], 0.325, 1e-15),
          "softmax backward must include cross-output coupling");

    check(neural_activation_backward(&relu,
                                     piecewise_pre,
                                     relu_output,
                                     unit_upstream,
                                     gradient,
                                     3U,
                                     &error) &&
              gradient[0] == 0.0 && gradient[1] == 0.0 &&
              gradient[2] == 1.0,
          "ReLU backward must define its zero derivative as zero");
    check(neural_activation_spec_set_parameter(
              &leaky,
              NEURAL_ACTIVATION_PARAMETER_ALPHA,
              0.1,
              &error) &&
              neural_activation_backward(&leaky,
                                         piecewise_pre,
                                         leaky_output,
                                         unit_upstream,
                                         gradient,
                                         3U,
                                         &error) &&
              gradient[0] == 0.1 && gradient[1] == 1.0 &&
              gradient[2] == 1.0,
          "leaky ReLU backward must use the non-negative branch at zero");
    check(neural_activation_spec_set_parameter(
              &elu,
              NEURAL_ACTIVATION_PARAMETER_ALPHA,
              2.0,
              &error) &&
              neural_activation_backward(&elu,
                                         piecewise_pre,
                                         elu_output,
                                         unit_upstream,
                                         gradient,
                                         3U,
                                         &error) &&
              nearly_equal(gradient[0], 2.0 * exp(-1.0), 1e-15) &&
              gradient[1] == 1.0 && gradient[2] == 1.0,
          "ELU backward must use alpha-exp on the negative branch");
    neural_activation_spec_free(&leaky);
    neural_activation_spec_free(&elu);
}

int main(void)
{
    test_tensor_operations();
    test_dense_operations();
    test_loss_operations();
    test_activation_backward();

    if (failures != 0) {
        fprintf(stderr, "%d math test(s) failed\n", failures);
        return 1;
    }
    puts("All math tests passed");
    return 0;
}
