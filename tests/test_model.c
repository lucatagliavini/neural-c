#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "neural/activation.h"
#include "neural/model.h"
#include "neural/random.h"

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

static void test_random_generator(void)
{
    NeuralRandom random;
    neural_real unit;

    neural_random_init(&random, UINT64_C(0));
    check(neural_random_next_uint64(&random) ==
              UINT64_C(0xe220a8397b1dcdaf),
          "SplitMix64 first reference value must be stable");
    check(neural_random_next_uint64(&random) ==
              UINT64_C(0x6e789e6aa1b965f4),
          "SplitMix64 second reference value must be stable");
    unit = neural_random_next_unit(&random);
    check(unit >= 0.0 && unit < 1.0,
          "random unit value must be in the half-open unit interval");
}

static void test_activations(void)
{
    NeuralError error;
    NeuralActivationSpec spec = {NEURAL_ACTIVATION_LINEAR, 0U, NULL};
    neural_real inputs[] = {-1000.0, 0.0, 1000.0};
    neural_real outputs[3];
    neural_real softmax_inputs[] = {1000.0, 999.0, 998.0};

    check(neural_activation_apply(&spec,
                                  inputs,
                                  outputs,
                                  3U,
                                  &error),
          "linear activation must apply");
    check(outputs[0] == inputs[0] && outputs[2] == inputs[2],
          "linear activation must preserve values");

    spec.kind = NEURAL_ACTIVATION_SIGMOID;
    check(neural_activation_apply(&spec,
                                  inputs,
                                  outputs,
                                  3U,
                                  &error),
          "stable sigmoid must accept extreme finite values");
    check(outputs[0] >= 0.0 && outputs[0] < 1e-100,
          "negative sigmoid extreme must approach zero");
    check(outputs[1] == 0.5 && outputs[2] == 1.0,
          "sigmoid reference values must match");

    spec.kind = NEURAL_ACTIVATION_SOFTMAX;
    check(neural_activation_apply(&spec,
                                  softmax_inputs,
                                  outputs,
                                  3U,
                                  &error),
          "stable softmax must accept large logits");
    check(nearly_equal(outputs[0] + outputs[1] + outputs[2],
                       1.0,
                       1e-15),
          "softmax probabilities must sum to one");
    check(outputs[0] > outputs[1] && outputs[1] > outputs[2],
          "softmax must preserve logit ordering");

    spec.kind = NEURAL_ACTIVATION_LEAKY_RELU;
    check(neural_activation_spec_set_parameter(
              &spec,
              NEURAL_ACTIVATION_PARAMETER_ALPHA,
              0.1,
              &error),
          "leaky ReLU alpha must be configurable");
    inputs[0] = -2.0;
    inputs[1] = 3.0;
    check(neural_activation_apply(&spec,
                                  inputs,
                                  outputs,
                                  2U,
                                  &error),
          "leaky ReLU must apply");
    check(nearly_equal(outputs[0], -0.2, 1e-15) && outputs[1] == 3.0,
          "leaky ReLU must use alpha only on negative values");
    neural_activation_spec_free(&spec);

    spec.kind = NEURAL_ACTIVATION_ELU;
    check(neural_activation_spec_set_parameter(
              &spec,
              NEURAL_ACTIVATION_PARAMETER_ALPHA,
              1.0,
              &error),
          "ELU alpha must be configurable");
    check(neural_activation_apply(&spec,
                                  inputs,
                                  outputs,
                                  2U,
                                  &error),
          "ELU must apply");
    check(nearly_equal(outputs[0], expm1(-2.0), 1e-15),
          "ELU negative branch must match its definition");
    neural_activation_spec_free(&spec);
}

static void compare_initialized_models(const NeuralModel *first,
                                       const NeuralModel *second,
                                       int expect_equal)
{
    size_t layer_index;
    int any_difference = 0;

    for (layer_index = 0U;
         layer_index < neural_model_layer_count(first);
         layer_index++) {
        size_t first_count;
        size_t second_count;
        const neural_real *first_weights =
            neural_model_layer_weights(first, layer_index, &first_count);
        const neural_real *second_weights =
            neural_model_layer_weights(second, layer_index, &second_count);
        size_t index;

        check(first_count == second_count,
              "comparable layers must have equal weight counts");
        for (index = 0U; index < first_count; index++) {
            if (first_weights[index] != second_weights[index]) {
                any_difference = 1;
            }
        }
    }
    check(expect_equal ? !any_difference : any_difference,
          expect_equal
              ? "same seed must produce identical weights"
              : "different seeds must produce different weights");
}

static void test_model_initialization(void)
{
    NeuralLayerSpec layers[] = {
        {3U, {NEURAL_ACTIVATION_RELU, 0U, NULL}},
        {2U, {NEURAL_ACTIVATION_SOFTMAX, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    NeuralModel *first = NULL;
    NeuralModel *same = NULL;
    NeuralModel *different = NULL;
    NeuralError error;
    size_t layer_index;

    check(neural_model_create(&spec, UINT64_C(42), &first, &error),
          "runtime model must be created from a valid specification");
    check(neural_model_create(&spec, UINT64_C(42), &same, &error),
          "same-seed runtime model must be created");
    check(neural_model_create(&spec, UINT64_C(43), &different, &error),
          "different-seed runtime model must be created");
    check(neural_model_input_count(first) == 2U,
          "runtime input width must match specification");
    check(neural_model_output_count(first) == 2U,
          "runtime output width must come from final layer");
    check(neural_model_parameter_count(first) == 17U,
          "runtime parameter count must include weights and biases");
    compare_initialized_models(first, same, 1);
    compare_initialized_models(first, different, 0);

    for (layer_index = 0U;
         layer_index < neural_model_layer_count(first);
         layer_index++) {
        size_t weight_count;
        size_t bias_count;
        const neural_real *weights =
            neural_model_layer_weights(first, layer_index, &weight_count);
        const neural_real *biases =
            neural_model_layer_biases(first, layer_index, &bias_count);
        neural_real limit = layer_index == 0U
                                ? sqrt(6.0 / 2.0)
                                : sqrt(6.0 / 5.0);
        size_t weight_index;
        size_t bias_index;

        for (weight_index = 0U; weight_index < weight_count; weight_index++) {
            check(isfinite(weights[weight_index]) &&
                      fabs(weights[weight_index]) <= limit,
                  "initialized weight must be finite and inside its range");
        }
        for (bias_index = 0U; bias_index < bias_count; bias_index++) {
            check(biases[bias_index] == 0.0,
                  "all initialized biases must be zero");
        }
    }
    neural_model_free(first);
    neural_model_free(same);
    neural_model_free(different);
}

static void test_known_forward_pass(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}
    };
    NeuralModelSpec spec = {2U, 1U, &layer};
    NeuralModel *model = NULL;
    NeuralModel *other_model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralWorkspace *other_workspace = NULL;
    NeuralError error;
    neural_real weights[] = {1.0, -1.0};
    neural_real biases[] = {0.0};
    neural_real inputs[] = {2.0, 1.0};
    neural_real output;
    const neural_real *stored_values;
    size_t stored_count;

    check(neural_model_create(&spec, UINT64_C(1), &model, &error),
          "known forward model must be created");
    check(neural_model_create(&spec, UINT64_C(2), &other_model, &error),
          "workspace compatibility model must be created");
    check(neural_model_set_layer_parameters(model,
                                            0U,
                                            weights,
                                            2U,
                                            biases,
                                            1U,
                                            &error),
          "known layer parameters must be accepted");
    check(neural_workspace_create(model, &workspace, &error),
          "forward workspace must be created");
    check(neural_workspace_create(other_model, &other_workspace, &error),
          "second forward workspace must be created");
    check(neural_model_forward(model,
                               workspace,
                               inputs,
                               2U,
                               &output,
                               1U,
                               &error),
          "known forward pass must succeed");
    check(nearly_equal(output, 1.0 / (1.0 + exp(-1.0)), 1e-15),
          "known forward output must match sigmoid reference");
    stored_values = neural_workspace_layer_pre_activations(workspace,
                                                           0U,
                                                           &stored_count);
    check(stored_values != NULL && stored_count == 1U &&
              stored_values[0] == 1.0,
          "workspace must expose stored pre-activations read-only");
    stored_values = neural_workspace_layer_activations(workspace,
                                                       0U,
                                                       &stored_count);
    check(stored_values != NULL && stored_count == 1U &&
              stored_values[0] == output,
          "workspace must expose stored activations read-only");
    check(!neural_model_forward(model,
                                workspace,
                                inputs,
                                1U,
                                &output,
                                1U,
                                &error),
          "forward pass must reject wrong input width");
    inputs[0] = NAN;
    check(!neural_model_forward(model,
                                workspace,
                                inputs,
                                2U,
                                &output,
                                1U,
                                &error),
          "forward pass must reject non-finite input");
    inputs[0] = 2.0;
    check(!neural_model_forward(model,
                                other_workspace,
                                inputs,
                                2U,
                                &output,
                                1U,
                                &error),
          "workspace must not be reused with a different model");

    neural_workspace_free(workspace);
    neural_workspace_free(other_workspace);
    neural_model_free(model);
    neural_model_free(other_model);
}

static void test_multilayer_forward_pass(void)
{
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}},
        {2U, {NEURAL_ACTIVATION_SOFTMAX, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralError error;
    neural_real identity_weights[] = {1.0, 0.0, 0.0, 1.0};
    neural_real biases[] = {0.0, 0.0};
    neural_real inputs[] = {2.0, 1.0};
    neural_real outputs[2];

    check(neural_model_create(&spec, UINT64_C(9), &model, &error),
          "multilayer model must be created");
    check(neural_model_set_layer_parameters(model,
                                            0U,
                                            identity_weights,
                                            4U,
                                            biases,
                                            2U,
                                            &error),
          "first identity layer must be set");
    check(neural_model_set_layer_parameters(model,
                                            1U,
                                            identity_weights,
                                            4U,
                                            biases,
                                            2U,
                                            &error),
          "second identity layer must be set");
    check(neural_workspace_create(model, &workspace, &error),
          "multilayer workspace must be created");
    check(neural_model_forward(model,
                               workspace,
                               inputs,
                               2U,
                               outputs,
                               2U,
                               &error),
          "multilayer forward pass must succeed");
    check(nearly_equal(outputs[0], 1.0 / (1.0 + exp(-1.0)), 1e-15) &&
              nearly_equal(outputs[0] + outputs[1], 1.0, 1e-15),
          "multilayer forward pass must chain linear and softmax layers");

    neural_workspace_free(workspace);
    neural_model_free(model);
}

static void test_parameterized_model_copy(void)
{
    NeuralActivationParameter parameters[] = {
        {NEURAL_ACTIVATION_PARAMETER_ALPHA, 0.1}
    };
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LEAKY_RELU, 1U, parameters}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralError error;
    neural_real weights[] = {1.0};
    neural_real biases[] = {0.0};
    neural_real input = -2.0;
    neural_real output;

    check(neural_model_create(&spec, UINT64_C(3), &model, &error),
          "parameterized runtime model must be created");
    parameters[0].value = 0.5;
    check(neural_model_set_layer_parameters(model,
                                            0U,
                                            weights,
                                            1U,
                                            biases,
                                            1U,
                                            &error),
          "parameterized layer weights must be set");
    check(neural_workspace_create(model, &workspace, &error),
          "parameterized model workspace must be created");
    check(neural_model_forward(model,
                               workspace,
                               &input,
                               1U,
                               &output,
                               1U,
                               &error),
          "parameterized model forward pass must succeed");
    check(nearly_equal(output, -0.2, 1e-15),
          "runtime model must deep-copy activation parameters");
    weights[0] = NAN;
    check(!neural_model_set_layer_parameters(model,
                                             0U,
                                             weights,
                                             1U,
                                             biases,
                                             1U,
                                             &error),
          "parameter setter must reject non-finite weights");

    neural_workspace_free(workspace);
    neural_model_free(model);
}

int main(void)
{
    test_random_generator();
    test_activations();
    test_model_initialization();
    test_known_forward_pass();
    test_multilayer_forward_pass();
    test_parameterized_model_copy();

    if (failures != 0) {
        fprintf(stderr, "%d model test(s) failed\n", failures);
        return 1;
    }
    puts("All model tests passed");
    return 0;
}
