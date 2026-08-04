#include "neural/gradient_check.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/backprop.h"
#include "neural/defaults.h"
#include "neural/gradient.h"

typedef struct {
    NeuralModel *model;
    NeuralWorkspace *workspace;
    NeuralLoss loss;
    const neural_real *inputs;
    size_t input_count;
    const neural_real *expected;
    size_t expected_count;
    neural_real *outputs;
    const NeuralGradientCheckConfig *config;
    NeuralGradientCheckResult *result;
    neural_real worst_score;
} GradientCheckContext;

void neural_gradient_check_config_default(NeuralGradientCheckConfig *config)
{
    if (config != NULL) {
        config->epsilon = NEURAL_DEFAULT_GRADIENT_CHECK_EPSILON;
        config->absolute_tolerance =
            NEURAL_DEFAULT_GRADIENT_CHECK_ABSOLUTE_TOLERANCE;
        config->relative_tolerance =
            NEURAL_DEFAULT_GRADIENT_CHECK_RELATIVE_TOLERANCE;
    }
}

static int config_validate(const NeuralGradientCheckConfig *config,
                           NeuralError *error)
{
    if (config == NULL || !isfinite(config->epsilon) ||
        config->epsilon <= 0.0 ||
        !isfinite(config->absolute_tolerance) ||
        config->absolute_tolerance < 0.0 ||
        !isfinite(config->relative_tolerance) ||
        config->relative_tolerance < 0.0) {
        neural_error_set(error, "invalid gradient-check configuration");
        return 0;
    }
    return 1;
}

static int evaluate_loss(GradientCheckContext *context,
                         neural_real *value,
                         NeuralError *error)
{
    return neural_model_forward(context->model,
                                context->workspace,
                                context->inputs,
                                context->input_count,
                                context->outputs,
                                context->expected_count,
                                error) &&
           neural_loss_evaluate_with_logits(
               context->loss,
               neural_model_output_activation(context->model),
               neural_workspace_layer_pre_activations(
                   context->workspace,
                   neural_model_layer_count(context->model) - 1U,
                   NULL),
               context->outputs,
               context->expected,
               context->expected_count,
               value,
               error);
}

static int install_layer(GradientCheckContext *context,
                         size_t layer_index,
                         const neural_real *weights,
                         size_t weight_count,
                         const neural_real *biases,
                         size_t bias_count,
                         NeuralError *error)
{
    return neural_model_set_layer_parameters(context->model,
                                             layer_index,
                                             weights,
                                             weight_count,
                                             biases,
                                             bias_count,
                                             error);
}

static void record_comparison(GradientCheckContext *context,
                              size_t layer_index,
                              NeuralParameterKind kind,
                              size_t parameter_index,
                              neural_real analytic,
                              neural_real numeric)
{
    neural_real absolute_error = fabs(analytic - numeric);
    neural_real scale = fmax(fabs(analytic), fabs(numeric));
    neural_real relative_error = scale == 0.0
                                     ? 0.0
                                     : absolute_error / scale;
    neural_real absolute_score =
        context->config->absolute_tolerance == 0.0
            ? (absolute_error == 0.0 ? 0.0 : INFINITY)
            : absolute_error / context->config->absolute_tolerance;
    neural_real relative_score =
        context->config->relative_tolerance == 0.0
            ? (relative_error == 0.0 ? 0.0 : INFINITY)
            : relative_error / context->config->relative_tolerance;
    neural_real comparison_score = fmin(absolute_score, relative_score);
    NeuralGradientCheckResult *result = context->result;

    result->checked_parameter_count++;
    if (absolute_error > result->maximum_absolute_error) {
        result->maximum_absolute_error = absolute_error;
    }
    if (relative_error > result->maximum_relative_error) {
        result->maximum_relative_error = relative_error;
    }
    if (comparison_score >= context->worst_score) {
        context->worst_score = comparison_score;
        result->worst_layer_index = layer_index;
        result->worst_parameter_kind = kind;
        result->worst_parameter_index = parameter_index;
        result->worst_analytic_value = analytic;
        result->worst_numeric_value = numeric;
    }
    if (absolute_error > context->config->absolute_tolerance &&
        relative_error > context->config->relative_tolerance) {
        result->passed = 0;
    }
}

static int check_parameter(GradientCheckContext *context,
                           size_t layer_index,
                           neural_real *weights,
                           size_t weight_count,
                           neural_real *biases,
                           size_t bias_count,
                           NeuralParameterKind kind,
                           size_t parameter_index,
                           neural_real analytic,
                           NeuralError *error)
{
    neural_real *parameters = kind == NEURAL_PARAMETER_WEIGHT
                                  ? weights
                                  : biases;
    neural_real original = parameters[parameter_index];
    neural_real step = context->config->epsilon * fmax(1.0, fabs(original));
    neural_real positive = original + step;
    neural_real negative = original - step;
    neural_real positive_loss = 0.0;
    neural_real negative_loss = 0.0;
    neural_real numeric;
    int evaluated;

    if (!isfinite(step) || !isfinite(positive) || !isfinite(negative) ||
        positive == original || negative == original || positive == negative) {
        neural_error_set(error,
                         "unable to perturb layer %zu parameter %zu",
                         layer_index,
                         parameter_index);
        return 0;
    }

    parameters[parameter_index] = positive;
    evaluated = install_layer(context,
                              layer_index,
                              weights,
                              weight_count,
                              biases,
                              bias_count,
                              error) &&
                evaluate_loss(context, &positive_loss, error);
    if (!evaluated) {
        parameters[parameter_index] = original;
        if (!install_layer(context,
                           layer_index,
                           weights,
                           weight_count,
                           biases,
                           bias_count,
                           error)) {
            neural_error_set(error,
                             "unable to restore layer %zu after gradient-check failure",
                             layer_index);
        }
        return 0;
    }

    parameters[parameter_index] = negative;
    evaluated = install_layer(context,
                              layer_index,
                              weights,
                              weight_count,
                              biases,
                              bias_count,
                              error) &&
                evaluate_loss(context, &negative_loss, error);
    parameters[parameter_index] = original;
    if (!install_layer(context,
                       layer_index,
                       weights,
                       weight_count,
                       biases,
                       bias_count,
                       error)) {
        neural_error_set(error,
                         "unable to restore layer %zu after gradient check",
                         layer_index);
        return 0;
    }
    if (!evaluated) {
        return 0;
    }

    numeric = (positive_loss - negative_loss) / (positive - negative);
    if (!isfinite(numeric)) {
        neural_error_set(error,
                         "numeric gradient is not finite at layer %zu parameter %zu",
                         layer_index,
                         parameter_index);
        return 0;
    }
    record_comparison(context,
                      layer_index,
                      kind,
                      parameter_index,
                      analytic,
                      numeric);
    return 1;
}

static int check_layer(GradientCheckContext *context,
                       const NeuralGradient *analytic_gradient,
                       size_t layer_index,
                       NeuralError *error)
{
    const neural_real *model_weights;
    const neural_real *model_biases;
    const neural_real *analytic_weights;
    const neural_real *analytic_biases;
    neural_real *weights = NULL;
    neural_real *biases = NULL;
    size_t weight_count;
    size_t bias_count;
    size_t analytic_weight_count;
    size_t analytic_bias_count;
    size_t parameter_index;
    int success = 0;

    model_weights = neural_model_layer_weights(context->model,
                                               layer_index,
                                               &weight_count);
    model_biases = neural_model_layer_biases(context->model,
                                             layer_index,
                                             &bias_count);
    analytic_weights = neural_gradient_layer_weights_const(
        analytic_gradient,
        layer_index,
        &analytic_weight_count);
    analytic_biases = neural_gradient_layer_biases_const(
        analytic_gradient,
        layer_index,
        &analytic_bias_count);
    if (model_weights == NULL || model_biases == NULL ||
        analytic_weights == NULL || analytic_biases == NULL ||
        weight_count != analytic_weight_count ||
        bias_count != analytic_bias_count ||
        weight_count > SIZE_MAX / sizeof(*weights) ||
        bias_count > SIZE_MAX / sizeof(*biases)) {
        neural_error_set(error,
                         "invalid gradient-check dimensions at layer %zu",
                         layer_index);
        return 0;
    }
    weights = malloc(weight_count * sizeof(*weights));
    biases = malloc(bias_count * sizeof(*biases));
    if (weights == NULL || biases == NULL) {
        neural_error_set(error,
                         "unable to allocate gradient-check layer %zu",
                         layer_index);
        goto cleanup;
    }
    memcpy(weights, model_weights, weight_count * sizeof(*weights));
    memcpy(biases, model_biases, bias_count * sizeof(*biases));

    for (parameter_index = 0U;
         parameter_index < weight_count;
         parameter_index++) {
        if (!check_parameter(context,
                             layer_index,
                             weights,
                             weight_count,
                             biases,
                             bias_count,
                             NEURAL_PARAMETER_WEIGHT,
                             parameter_index,
                             analytic_weights[parameter_index],
                             error)) {
            goto cleanup;
        }
    }
    for (parameter_index = 0U;
         parameter_index < bias_count;
         parameter_index++) {
        if (!check_parameter(context,
                             layer_index,
                             weights,
                             weight_count,
                             biases,
                             bias_count,
                             NEURAL_PARAMETER_BIAS,
                             parameter_index,
                             analytic_biases[parameter_index],
                             error)) {
            goto cleanup;
        }
    }
    success = 1;

cleanup:
    free(weights);
    free(biases);
    return success;
}

int neural_model_gradient_check(NeuralModel *model,
                                NeuralWorkspace *workspace,
                                NeuralLoss loss,
                                const neural_real *inputs,
                                size_t input_count,
                                const neural_real *expected,
                                size_t expected_count,
                                const NeuralGradientCheckConfig *config,
                                NeuralGradientCheckResult *result,
                                NeuralError *error)
{
    NeuralGradientCheckConfig defaults;
    const NeuralGradientCheckConfig *effective_config = config;
    NeuralGradient *analytic_gradient = NULL;
    GradientCheckContext context;
    neural_real analytic_loss;
    size_t output_count;
    size_t layer_index;
    int success = 0;

    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (model == NULL || workspace == NULL || inputs == NULL ||
        expected == NULL || result == NULL) {
        neural_error_set(error, "invalid gradient-check arguments");
        return 0;
    }
    if (effective_config == NULL) {
        neural_gradient_check_config_default(&defaults);
        effective_config = &defaults;
    }
    if (!config_validate(effective_config, error)) {
        return 0;
    }
    output_count = neural_model_output_count(model);
    if (input_count != neural_model_input_count(model) ||
        expected_count != output_count ||
        output_count > SIZE_MAX / sizeof(*context.outputs)) {
        neural_error_set(error,
                         "gradient-check dimensions do not match model");
        return 0;
    }
    context.outputs = malloc(output_count * sizeof(*context.outputs));
    if (context.outputs == NULL ||
        !neural_gradient_create(model, &analytic_gradient, error)) {
        free(context.outputs);
        return 0;
    }
    context.model = model;
    context.workspace = workspace;
    context.loss = loss;
    context.inputs = inputs;
    context.input_count = input_count;
    context.expected = expected;
    context.expected_count = expected_count;
    context.config = effective_config;
    context.result = result;
    context.worst_score = -1.0;
    result->passed = 1;

    if (!neural_model_sample_gradient(model,
                                      workspace,
                                      analytic_gradient,
                                      loss,
                                      inputs,
                                      input_count,
                                      expected,
                                      expected_count,
                                      &analytic_loss,
                                      error)) {
        goto cleanup;
    }
    for (layer_index = 0U;
         layer_index < neural_model_layer_count(model);
         layer_index++) {
        if (!check_layer(&context,
                         analytic_gradient,
                         layer_index,
                         error)) {
            goto cleanup;
        }
    }
    if (!result->passed) {
        neural_error_set(
            error,
            "gradient check failed at layer %zu %s %zu: analytic %.*g, numeric %.*g",
            result->worst_layer_index,
            result->worst_parameter_kind == NEURAL_PARAMETER_WEIGHT
                ? "weight"
                : "bias",
            result->worst_parameter_index,
            DBL_DECIMAL_DIG,
            result->worst_analytic_value,
            DBL_DECIMAL_DIG,
            result->worst_numeric_value);
        goto cleanup;
    }
    success = 1;

cleanup:
    neural_gradient_free(analytic_gradient);
    free(context.outputs);
    return success;
}
