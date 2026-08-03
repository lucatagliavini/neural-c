#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "neural/activation.h"
#include "neural/backprop.h"
#include "neural/gradient.h"
#include "neural/model.h"

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

static int array_nearly_equal(const neural_real *actual,
                              const neural_real *expected,
                              size_t count,
                              neural_real tolerance)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        if (!nearly_equal(actual[index], expected[index], tolerance)) {
            return 0;
        }
    }
    return 1;
}

static void check_layer_gradient(NeuralGradient *gradient,
                                 size_t layer_index,
                                 const neural_real *expected_weights,
                                 size_t expected_weight_count,
                                 const neural_real *expected_biases,
                                 size_t expected_bias_count,
                                 neural_real tolerance,
                                 const char *description)
{
    size_t weight_count;
    size_t bias_count;
    const neural_real *weights = neural_gradient_layer_weights(
        gradient,
        layer_index,
        &weight_count);
    const neural_real *biases = neural_gradient_layer_biases(
        gradient,
        layer_index,
        &bias_count);

    check(weights != NULL && biases != NULL &&
              weight_count == expected_weight_count &&
              bias_count == expected_bias_count &&
              array_nearly_equal(weights,
                                 expected_weights,
                                 weight_count,
                                 tolerance) &&
              array_nearly_equal(biases,
                                 expected_biases,
                                 bias_count,
                                 tolerance),
          description);
}

static void test_linear_multilayer_gradient(void)
{
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}},
        {1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    const neural_real first_weights[] = {1.0, 2.0, -1.0, 0.5};
    const neural_real first_biases[] = {0.5, -0.5};
    const neural_real second_weights[] = {2.0, -1.0};
    const neural_real second_biases[] = {0.25};
    const neural_real inputs[] = {2.0, -1.0};
    const neural_real expected[] = {1.25};
    const neural_real expected_first_weight_gradient[] = {
        24.0, -12.0, -12.0, 6.0
    };
    const neural_real expected_first_bias_gradient[] = {12.0, -6.0};
    const neural_real expected_second_weight_gradient[] = {3.0, -18.0};
    const neural_real expected_second_bias_gradient[] = {6.0};
    const neural_real expected_input_gradient[] = {18.0, 21.0};
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralGradient *gradient = NULL;
    NeuralError error;
    neural_real loss_value = 0.0;
    const neural_real *input_gradient;
    const neural_real *unchanged_weights;
    size_t input_gradient_count;
    size_t unchanged_weight_count;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(1), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  first_weights,
                                                  4U,
                                                  first_biases,
                                                  2U,
                                                  &error) &&
               neural_model_set_layer_parameters(model,
                                                  1U,
                                                  second_weights,
                                                  2U,
                                                  second_biases,
                                                  1U,
                                                  &error) &&
               neural_workspace_create(model, &workspace, &error) &&
               neural_gradient_create(model, &gradient, &error);
    check(prepared, "linear backward fixture must be prepared");
    if (!prepared) {
        neural_gradient_free(gradient);
        neural_workspace_free(workspace);
        neural_model_free(model);
        return;
    }
    check(neural_model_sample_gradient(model,
                                       workspace,
                                       gradient,
                                       NEURAL_LOSS_MSE,
                                       inputs,
                                       2U,
                                       expected,
                                       1U,
                                       &loss_value,
                                       &error),
          "linear multilayer sample gradient must succeed");
    check(loss_value == 9.0,
          "linear multilayer loss must match the analytic value");
    check_layer_gradient(gradient,
                         0U,
                         expected_first_weight_gradient,
                         4U,
                         expected_first_bias_gradient,
                         2U,
                         0.0,
                         "first linear layer gradient must be exact");
    check_layer_gradient(gradient,
                         1U,
                         expected_second_weight_gradient,
                         2U,
                         expected_second_bias_gradient,
                         1U,
                         0.0,
                         "second linear layer gradient must be exact");
    input_gradient = neural_workspace_input_gradients(workspace,
                                                       &input_gradient_count);
    check(input_gradient_count == 2U &&
              array_nearly_equal(input_gradient,
                                 expected_input_gradient,
                                 2U,
                                 0.0),
          "model input gradient must be retained in the workspace");
    unchanged_weights = neural_model_layer_weights(model,
                                                    0U,
                                                    &unchanged_weight_count);
    check(unchanged_weight_count == 4U &&
              array_nearly_equal(unchanged_weights,
                                 first_weights,
                                 4U,
                                 0.0),
          "sample gradient must not update model parameters");
    check(neural_model_sample_gradient(model,
                                       workspace,
                                       gradient,
                                       NEURAL_LOSS_MSE,
                                       inputs,
                                       2U,
                                       expected,
                                       1U,
                                       &loss_value,
                                       &error) &&
              loss_value == 9.0,
          "workspace and gradient buffers must be reusable per sample");

    neural_gradient_free(gradient);
    neural_workspace_free(workspace);
    neural_model_free(model);
}

static void test_parameterized_activation_gradient(void)
{
    NeuralActivationParameter alpha = {
        NEURAL_ACTIVATION_PARAMETER_ALPHA, 0.1
    };
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_LEAKY_RELU, 1U, &alpha}},
        {1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    const neural_real first_weights[] = {1.0, 0.0, 0.0, 1.0};
    const neural_real first_biases[] = {0.0, 0.0};
    const neural_real second_weights[] = {1.0, 2.0};
    const neural_real second_biases[] = {0.0};
    const neural_real inputs[] = {-2.0, 3.0};
    const neural_real expected[] = {0.8};
    const neural_real expected_first_weight_gradient[] = {
        -2.0, 3.0, -40.0, 60.0
    };
    const neural_real expected_first_bias_gradient[] = {1.0, 20.0};
    const neural_real expected_second_weight_gradient[] = {-2.0, 30.0};
    const neural_real expected_second_bias_gradient[] = {10.0};
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralGradient *gradient = NULL;
    NeuralError error;
    neural_real loss_value;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(2), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  first_weights,
                                                  4U,
                                                  first_biases,
                                                  2U,
                                                  &error) &&
               neural_model_set_layer_parameters(model,
                                                  1U,
                                                  second_weights,
                                                  2U,
                                                  second_biases,
                                                  1U,
                                                  &error) &&
               neural_workspace_create(model, &workspace, &error) &&
               neural_gradient_create(model, &gradient, &error);
    check(prepared, "parameterized backward fixture must be prepared");
    if (prepared) {
        check(neural_model_sample_gradient(model,
                                           workspace,
                                           gradient,
                                           NEURAL_LOSS_MSE,
                                           inputs,
                                           2U,
                                           expected,
                                           1U,
                                           &loss_value,
                                           &error) &&
                  nearly_equal(loss_value, 25.0, 1e-14),
              "leaky ReLU sample gradient must compute its loss");
        check_layer_gradient(
            gradient,
            0U,
            expected_first_weight_gradient,
            4U,
            expected_first_bias_gradient,
            2U,
            1e-14,
            "leaky ReLU derivative must reach the first layer");
        check_layer_gradient(
            gradient,
            1U,
            expected_second_weight_gradient,
            2U,
            expected_second_bias_gradient,
            1U,
            1e-14,
            "leaky ReLU activations must feed the output gradient");
    }

    neural_gradient_free(gradient);
    neural_workspace_free(workspace);
    neural_model_free(model);
}

static void test_softmax_gradient(void)
{
    NeuralLayerSpec layer = {
        2U, {NEURAL_ACTIVATION_SOFTMAX, 0U, NULL}
    };
    NeuralModelSpec spec = {2U, 1U, &layer};
    const neural_real weights[] = {1.0, 0.0, 0.0, 1.0};
    const neural_real biases[] = {0.0, 0.0};
    const neural_real inputs[] = {1.0, 0.0};
    const neural_real expected[] = {0.0, 1.0};
    NeuralModel *model = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralGradient *gradient = NULL;
    NeuralError error;
    neural_real loss_value;
    neural_real probability = exp(1.0) / (exp(1.0) + 1.0);
    neural_real delta = 2.0 * probability * probability *
                        (1.0 - probability);
    neural_real expected_weight_gradient[] = {delta, 0.0, -delta, 0.0};
    neural_real expected_bias_gradient[] = {delta, -delta};
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(3), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  weights,
                                                  4U,
                                                  biases,
                                                  2U,
                                                  &error) &&
               neural_workspace_create(model, &workspace, &error) &&
               neural_gradient_create(model, &gradient, &error);
    check(prepared, "softmax backward fixture must be prepared");
    if (prepared) {
        check(neural_model_sample_gradient(model,
                                           workspace,
                                           gradient,
                                           NEURAL_LOSS_MSE,
                                           inputs,
                                           2U,
                                           expected,
                                           2U,
                                           &loss_value,
                                           &error) &&
                  nearly_equal(loss_value,
                               probability * probability,
                               1e-15),
              "softmax sample gradient must compute vector MSE");
        check_layer_gradient(gradient,
                             0U,
                             expected_weight_gradient,
                             4U,
                             expected_bias_gradient,
                             2U,
                             1e-15,
                             "softmax Jacobian-vector product must reach dense parameters");
    }

    neural_gradient_free(gradient);
    neural_workspace_free(workspace);
    neural_model_free(model);
}

static void test_compatibility_failures(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    NeuralModel *model = NULL;
    NeuralModel *other = NULL;
    NeuralWorkspace *workspace = NULL;
    NeuralWorkspace *other_workspace = NULL;
    NeuralGradient *gradient = NULL;
    NeuralGradient *other_gradient = NULL;
    NeuralError error;
    neural_real input = 1.0;
    neural_real expected = 0.0;
    neural_real loss_value;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(4), &model, &error) &&
               neural_model_create(&spec, UINT64_C(5), &other, &error) &&
               neural_workspace_create(model, &workspace, &error) &&
               neural_workspace_create(other, &other_workspace, &error) &&
               neural_gradient_create(model, &gradient, &error) &&
               neural_gradient_create(other, &other_gradient, &error);
    check(prepared, "compatibility fixtures must be prepared");
    if (prepared) {
        check(!neural_model_sample_gradient(model,
                                            other_workspace,
                                            gradient,
                                            NEURAL_LOSS_MSE,
                                            &input,
                                            1U,
                                            &expected,
                                            1U,
                                            &loss_value,
                                            &error),
              "sample gradient must reject another model's workspace");
        check(!neural_model_sample_gradient(model,
                                            workspace,
                                            other_gradient,
                                            NEURAL_LOSS_MSE,
                                            &input,
                                            1U,
                                            &expected,
                                            1U,
                                            &loss_value,
                                            &error),
              "sample gradient must reject another model's gradient");
        check(!neural_model_sample_gradient(model,
                                            workspace,
                                            gradient,
                                            NEURAL_LOSS_MSE,
                                            &input,
                                            0U,
                                            &expected,
                                            1U,
                                            &loss_value,
                                            &error),
              "sample gradient must reject wrong input dimensions");
        expected = NAN;
        check(!neural_model_sample_gradient(model,
                                            workspace,
                                            gradient,
                                            NEURAL_LOSS_MSE,
                                            &input,
                                            1U,
                                            &expected,
                                            1U,
                                            &loss_value,
                                            &error),
              "sample gradient must reject non-finite expected values");
    }

    neural_gradient_free(other_gradient);
    neural_gradient_free(gradient);
    neural_workspace_free(other_workspace);
    neural_workspace_free(workspace);
    neural_model_free(other);
    neural_model_free(model);
}

int main(void)
{
    test_linear_multilayer_gradient();
    test_parameterized_activation_gradient();
    test_softmax_gradient();
    test_compatibility_failures();

    if (failures != 0) {
        fprintf(stderr, "%d backpropagation test(s) failed\n", failures);
        return 1;
    }
    puts("All backpropagation tests passed");
    return 0;
}
