#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "neural/activation.h"
#include "neural/gradient_check.h"
#include "neural/model.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int layer_parameters_equal(const NeuralModel *model,
                                  size_t layer_index,
                                  const neural_real *weights,
                                  size_t expected_weight_count,
                                  const neural_real *biases,
                                  size_t expected_bias_count)
{
    size_t weight_count;
    size_t bias_count;
    const neural_real *model_weights = neural_model_layer_weights(
        model,
        layer_index,
        &weight_count);
    const neural_real *model_biases = neural_model_layer_biases(
        model,
        layer_index,
        &bias_count);

    return model_weights != NULL && model_biases != NULL &&
           weight_count == expected_weight_count &&
           bias_count == expected_bias_count &&
           memcmp(model_weights,
                  weights,
                  weight_count * sizeof(*weights)) == 0 &&
           memcmp(model_biases,
                  biases,
                  bias_count * sizeof(*biases)) == 0;
}

static void run_single_activation_check(NeuralActivationKind kind,
                                        const char *description)
{
    NeuralActivationParameter alpha = {
        NEURAL_ACTIVATION_PARAMETER_ALPHA, 0.1
    };
    int parameterized = kind == NEURAL_ACTIVATION_LEAKY_RELU ||
                        kind == NEURAL_ACTIVATION_ELU;
    NeuralActivationSpec activation = {
        kind,
        parameterized ? 1U : 0U,
        parameterized ? &alpha : NULL
    };
    NeuralLayerSpec layer = {2U, activation};
    NeuralModelSpec spec = {2U, 1U, &layer};
    const neural_real weights[] = {0.3, -0.2, -0.4, 0.5};
    const neural_real biases[] = {0.7, -0.6};
    const neural_real inputs[] = {0.8, -1.1};
    const neural_real expected[] = {0.2, -0.1};
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralGradientCheckResult result;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(11), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  weights,
                                                  4U,
                                                  biases,
                                                  2U,
                                                  &error) &&
               neural_workspace_create(model, &workspace, &error);
    check(prepared, "single-activation checker fixture must be prepared");
    if (prepared) {
        check(neural_model_gradient_check(model,
                                          workspace,
                                          NEURAL_LOSS_MSE,
                                          inputs,
                                          2U,
                                          expected,
                                          2U,
                                          NULL,
                                          &result,
                                          &error) &&
                  result.passed && result.checked_parameter_count == 6U &&
                  isfinite(result.maximum_absolute_error) &&
                  isfinite(result.maximum_relative_error),
              description);
        check(layer_parameters_equal(model,
                                     0U,
                                     weights,
                                     4U,
                                     biases,
                                     2U),
              "gradient checker must exactly restore activation fixture parameters");
    }
    neural_workspace_free(workspace);
    neural_model_free(model);
}

static void test_all_activation_kinds(void)
{
    run_single_activation_check(NEURAL_ACTIVATION_LINEAR,
                                "linear gradient check must pass");
    run_single_activation_check(NEURAL_ACTIVATION_SIGMOID,
                                "sigmoid gradient check must pass");
    run_single_activation_check(NEURAL_ACTIVATION_TANH,
                                "tanh gradient check must pass");
    run_single_activation_check(NEURAL_ACTIVATION_RELU,
                                "ReLU gradient check must pass away from zero");
    run_single_activation_check(NEURAL_ACTIVATION_LEAKY_RELU,
                                "leaky ReLU gradient check must pass");
    run_single_activation_check(NEURAL_ACTIVATION_ELU,
                                "ELU gradient check must pass");
    run_single_activation_check(NEURAL_ACTIVATION_SOFTMAX,
                                "softmax gradient check must pass");
}

static void test_multilayer_gradient_check(void)
{
    NeuralLayerSpec layers[] = {
        {3U, {NEURAL_ACTIVATION_TANH, 0U, NULL}},
        {2U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    const neural_real first_weights[] = {
        0.2, -0.3,
        0.4, 0.1,
        -0.5, 0.6
    };
    const neural_real first_biases[] = {0.1, -0.2, 0.3};
    const neural_real second_weights[] = {
        0.7, -0.4, 0.2,
        -0.1, 0.5, -0.6
    };
    const neural_real second_biases[] = {0.05, -0.15};
    const neural_real inputs[] = {0.9, -0.7};
    const neural_real expected[] = {0.25, 0.75};
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralGradientCheckResult result;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(12), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  first_weights,
                                                  6U,
                                                  first_biases,
                                                  3U,
                                                  &error) &&
               neural_model_set_layer_parameters(model,
                                                  1U,
                                                  second_weights,
                                                  6U,
                                                  second_biases,
                                                  2U,
                                                  &error) &&
               neural_workspace_create(model, &workspace, &error);
    check(prepared, "multilayer gradient-check fixture must be prepared");
    if (prepared) {
        check(neural_model_gradient_check(model,
                                          workspace,
                                          NEURAL_LOSS_MSE,
                                          inputs,
                                          2U,
                                          expected,
                                          2U,
                                          NULL,
                                          &result,
                                          &error) &&
                  result.passed && result.checked_parameter_count == 17U,
              "multilayer gradient check must cover every parameter");
        check(layer_parameters_equal(model,
                                     0U,
                                     first_weights,
                                     6U,
                                     first_biases,
                                     3U) &&
                  layer_parameters_equal(model,
                                         1U,
                                         second_weights,
                                         6U,
                                         second_biases,
                                         2U),
              "multilayer gradient check must restore every layer exactly");
    }
    neural_workspace_free(workspace);
    neural_model_free(model);
}

static void test_cross_entropy_gradient_checks(void)
{
    NeuralLayerSpec binary_layer = {
        1U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}
    };
    NeuralLayerSpec categorical_layer = {
        3U, {NEURAL_ACTIVATION_SOFTMAX, 0U, NULL}
    };
    NeuralModelSpec binary_spec = {2U, 1U, &binary_layer};
    NeuralModelSpec categorical_spec = {2U, 1U, &categorical_layer};
    const neural_real binary_inputs[] = {0.4, -0.7};
    const neural_real binary_target[] = {1.0};
    const neural_real categorical_inputs[] = {-0.2, 0.9};
    const neural_real categorical_target[] = {0.0, 1.0, 0.0};
    NeuralModel *binary_model = NULL;
    NeuralModel *categorical_model = NULL;
    NeuralWorkspace *binary_workspace = NULL;
    NeuralWorkspace *categorical_workspace = NULL;
    NeuralGradientCheckResult result;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&binary_spec, UINT64_C(21),
                                   &binary_model, &error) &&
               neural_workspace_create(binary_model,
                                       &binary_workspace,
                                       &error) &&
               neural_model_create(&categorical_spec, UINT64_C(22),
                                   &categorical_model, &error) &&
               neural_workspace_create(categorical_model,
                                       &categorical_workspace,
                                       &error);
    check(prepared, "cross-entropy gradient-check fixtures must be prepared");
    if (prepared) {
        check(neural_model_gradient_check(
                  binary_model,
                  binary_workspace,
                  NEURAL_LOSS_BINARY_CROSS_ENTROPY,
                  binary_inputs,
                  2U,
                  binary_target,
                  1U,
                  NULL,
                  &result,
                  &error) && result.passed,
              "sigmoid binary cross-entropy gradient check must pass");
        check(neural_model_gradient_check(
                  categorical_model,
                  categorical_workspace,
                  NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY,
                  categorical_inputs,
                  2U,
                  categorical_target,
                  3U,
                  NULL,
                  &result,
                  &error) && result.passed,
              "softmax categorical cross-entropy gradient check must pass");
    }
    neural_workspace_free(categorical_workspace);
    neural_model_free(categorical_model);
    neural_workspace_free(binary_workspace);
    neural_model_free(binary_model);
}

static void test_nondifferentiable_mismatch(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_RELU, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    const neural_real weights[] = {0.0};
    const neural_real biases[] = {0.0};
    const neural_real input = 1.0;
    const neural_real expected = -1.0;
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralGradientCheckResult result;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(13), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  weights,
                                                  1U,
                                                  biases,
                                                  1U,
                                                  &error) &&
               neural_workspace_create(model, &workspace, &error);
    check(prepared, "ReLU kink fixture must be prepared");
    if (prepared) {
        check(!neural_model_gradient_check(model,
                                           workspace,
                                           NEURAL_LOSS_MSE,
                                           &input,
                                           1U,
                                           &expected,
                                           1U,
                                           NULL,
                                           &result,
                                           &error) &&
                  !result.passed && result.checked_parameter_count == 2U &&
                  strstr(error.message, "gradient check failed") != NULL,
              "gradient checker must report a ReLU kink mismatch");
        check(layer_parameters_equal(model,
                                     0U,
                                     weights,
                                     1U,
                                     biases,
                                     1U),
              "failed gradient check must restore model parameters exactly");
    }
    neural_workspace_free(workspace);
    neural_model_free(model);
}

static void test_invalid_arguments(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    const neural_real weights[] = {0.5};
    const neural_real biases[] = {0.25};
    const neural_real input = 1.0;
    const neural_real expected = 0.0;
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralGradientCheckConfig config;
    NeuralGradientCheckResult result;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(14), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  weights,
                                                  1U,
                                                  biases,
                                                  1U,
                                                  &error) &&
               neural_workspace_create(model, &workspace, &error);
    check(prepared, "invalid gradient-check fixture must be prepared");
    neural_gradient_check_config_default(&config);
    check(config.epsilon > 0.0 && config.absolute_tolerance >= 0.0 &&
              config.relative_tolerance >= 0.0,
          "default gradient-check configuration must be valid");
    if (prepared) {
        config.epsilon = 0.0;
        check(!neural_model_gradient_check(model,
                                           workspace,
                                           NEURAL_LOSS_MSE,
                                           &input,
                                           1U,
                                           &expected,
                                           1U,
                                           &config,
                                           &result,
                                           &error) &&
                  result.checked_parameter_count == 0U,
              "gradient checker must reject zero epsilon");
        neural_gradient_check_config_default(&config);
        check(!neural_model_gradient_check(model,
                                           workspace,
                                           NEURAL_LOSS_MSE,
                                           &input,
                                           0U,
                                           &expected,
                                           1U,
                                           &config,
                                           &result,
                                           &error),
              "gradient checker must reject mismatched dimensions");
        check(layer_parameters_equal(model,
                                     0U,
                                     weights,
                                     1U,
                                     biases,
                                     1U),
              "invalid checks must leave model parameters unchanged");
    }
    neural_workspace_free(workspace);
    neural_model_free(model);
}

int main(void)
{
    test_all_activation_kinds();
    test_multilayer_gradient_check();
    test_cross_entropy_gradient_checks();
    test_nondifferentiable_mismatch();
    test_invalid_arguments();

    if (failures != 0) {
        fprintf(stderr, "%d gradient-check test(s) failed\n", failures);
        return 1;
    }
    puts("All gradient-check tests passed");
    return 0;
}
