#include "neural/gradient.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "compensated_sum.h"
#include "neural/tensor_ops.h"
#include "gradient_internal.h"

typedef struct {
    size_t weight_count;
    size_t bias_count;
    neural_real *weights;
    neural_real *biases;
} NeuralGradientLayer;

struct NeuralGradient {
    const NeuralModel *model;
    size_t layer_count;
    NeuralGradientLayer *layers;
};

static void free_parameter_arrays(neural_real **weights,
                                  neural_real **biases,
                                  size_t layer_count)
{
    size_t layer_index;

    for (layer_index = 0U; layer_index < layer_count; layer_index++) {
        free(weights == NULL ? NULL : weights[layer_index]);
        free(biases == NULL ? NULL : biases[layer_index]);
    }
    free(weights);
    free(biases);
}

void neural_gradient_free(NeuralGradient *gradient)
{
    size_t layer_index;

    if (gradient == NULL) {
        return;
    }
    if (gradient->layers != NULL) {
        for (layer_index = 0U;
             layer_index < gradient->layer_count;
             layer_index++) {
            free(gradient->layers[layer_index].weights);
            free(gradient->layers[layer_index].biases);
        }
    }
    free(gradient->layers);
    free(gradient);
}

int neural_gradient_create(const NeuralModel *model,
                           NeuralGradient **gradient,
                           NeuralError *error)
{
    NeuralGradient *created;
    size_t layer_index;

    if (model == NULL || gradient == NULL) {
        neural_error_set(error, "model and gradient output are required");
        return 0;
    }
    *gradient = NULL;
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate model gradient");
        return 0;
    }
    created->model = model;
    created->layer_count = neural_model_layer_count(model);
    created->layers = calloc(created->layer_count, sizeof(*created->layers));
    if (created->layers == NULL) {
        neural_error_set(error, "unable to allocate gradient layers");
        neural_gradient_free(created);
        return 0;
    }
    for (layer_index = 0U;
         layer_index < created->layer_count;
         layer_index++) {
        NeuralGradientLayer *layer = &created->layers[layer_index];

        (void)neural_model_layer_weights(model,
                                         layer_index,
                                         &layer->weight_count);
        (void)neural_model_layer_biases(model,
                                        layer_index,
                                        &layer->bias_count);
        if (layer->weight_count > SIZE_MAX / sizeof(*layer->weights) ||
            layer->bias_count > SIZE_MAX / sizeof(*layer->biases)) {
            neural_error_set(error, "gradient dimensions overflow");
            neural_gradient_free(created);
            return 0;
        }
        layer->weights = calloc(layer->weight_count,
                                sizeof(*layer->weights));
        layer->biases = calloc(layer->bias_count,
                               sizeof(*layer->biases));
        if (layer->weights == NULL || layer->biases == NULL) {
            neural_error_set(error,
                             "unable to allocate gradient layer %zu",
                             layer_index);
            neural_gradient_free(created);
            return 0;
        }
    }
    *gradient = created;
    return 1;
}

int neural_gradient_zero(NeuralGradient *gradient, NeuralError *error)
{
    size_t layer_index;

    if (gradient == NULL) {
        neural_error_set(error, "gradient is required");
        return 0;
    }
    for (layer_index = 0U;
         layer_index < gradient->layer_count;
         layer_index++) {
        NeuralGradientLayer *layer = &gradient->layers[layer_index];

        if (!neural_tensor_zero(layer->weights,
                                layer->weight_count,
                                error) ||
            !neural_tensor_zero(layer->biases,
                                layer->bias_count,
                                error)) {
            return 0;
        }
    }
    return 1;
}

neural_real *neural_gradient_layer_weights(NeuralGradient *gradient,
                                           size_t layer_index,
                                           size_t *count)
{
    if (count != NULL) {
        *count = 0U;
    }
    if (gradient == NULL || layer_index >= gradient->layer_count) {
        return NULL;
    }
    if (count != NULL) {
        *count = gradient->layers[layer_index].weight_count;
    }
    return gradient->layers[layer_index].weights;
}

neural_real *neural_gradient_layer_biases(NeuralGradient *gradient,
                                          size_t layer_index,
                                          size_t *count)
{
    if (count != NULL) {
        *count = 0U;
    }
    if (gradient == NULL || layer_index >= gradient->layer_count) {
        return NULL;
    }
    if (count != NULL) {
        *count = gradient->layers[layer_index].bias_count;
    }
    return gradient->layers[layer_index].biases;
}

const neural_real *neural_gradient_layer_weights_const(
    const NeuralGradient *gradient,
    size_t layer_index,
    size_t *count)
{
    if (count != NULL) {
        *count = 0U;
    }
    if (gradient == NULL || layer_index >= gradient->layer_count) {
        return NULL;
    }
    if (count != NULL) {
        *count = gradient->layers[layer_index].weight_count;
    }
    return gradient->layers[layer_index].weights;
}

const neural_real *neural_gradient_layer_biases_const(
    const NeuralGradient *gradient,
    size_t layer_index,
    size_t *count)
{
    if (count != NULL) {
        *count = 0U;
    }
    if (gradient == NULL || layer_index >= gradient->layer_count) {
        return NULL;
    }
    if (count != NULL) {
        *count = gradient->layers[layer_index].bias_count;
    }
    return gradient->layers[layer_index].biases;
}

int neural_gradient_copy(NeuralGradient *destination,
                         const NeuralGradient *source,
                         NeuralError *error)
{
    size_t layer_index;

    if (destination == NULL || source == NULL ||
        destination->model != source->model ||
        destination->layer_count != source->layer_count) {
        neural_error_set(error, "gradient copy requires the same model");
        return 0;
    }
    for (layer_index = 0U;
         layer_index < source->layer_count;
         layer_index++) {
        const NeuralGradientLayer *layer = &source->layers[layer_index];
        size_t index;

        for (index = 0U; index < layer->weight_count; index++) {
            if (!isfinite(layer->weights[index])) {
                neural_error_set(error, "weight gradients must be finite");
                return 0;
            }
        }
        for (index = 0U; index < layer->bias_count; index++) {
            if (!isfinite(layer->biases[index])) {
                neural_error_set(error, "bias gradients must be finite");
                return 0;
            }
        }
    }
    for (layer_index = 0U;
         layer_index < source->layer_count;
         layer_index++) {
        const NeuralGradientLayer *source_layer = &source->layers[layer_index];
        NeuralGradientLayer *destination_layer =
            &destination->layers[layer_index];

        memcpy(destination_layer->weights,
               source_layer->weights,
               source_layer->weight_count * sizeof(*source_layer->weights));
        memcpy(destination_layer->biases,
               source_layer->biases,
               source_layer->bias_count * sizeof(*source_layer->biases));
    }
    return 1;
}

int neural_gradient_is_compatible(const NeuralGradient *gradient,
                                  const NeuralModel *model)
{
    return gradient != NULL && model != NULL && gradient->model == model;
}

int neural_gradient_add(NeuralGradient *destination,
                        const NeuralGradient *source,
                        NeuralError *error)
{
    size_t layer_index;

    if (destination == NULL || source == NULL ||
        destination->model != source->model ||
        destination->layer_count != source->layer_count) {
        neural_error_set(error, "gradient belongs to a different model");
        return 0;
    }
    for (layer_index = 0U;
         layer_index < destination->layer_count;
         layer_index++) {
        NeuralGradientLayer *destination_layer =
            &destination->layers[layer_index];
        const NeuralGradientLayer *source_layer = &source->layers[layer_index];
        size_t index;

        if (destination_layer->weight_count != source_layer->weight_count ||
            destination_layer->bias_count != source_layer->bias_count) {
            neural_error_set(error, "gradient dimensions do not match");
            return 0;
        }
        for (index = 0U; index < destination_layer->weight_count; index++) {
            if (!isfinite(destination_layer->weights[index]) ||
                !isfinite(source_layer->weights[index]) ||
                !isfinite(destination_layer->weights[index] +
                          source_layer->weights[index])) {
                neural_error_set(error, "summed weight gradient is not finite");
                return 0;
            }
        }
        for (index = 0U; index < destination_layer->bias_count; index++) {
            if (!isfinite(destination_layer->biases[index]) ||
                !isfinite(source_layer->biases[index]) ||
                !isfinite(destination_layer->biases[index] +
                          source_layer->biases[index])) {
                neural_error_set(error, "summed bias gradient is not finite");
                return 0;
            }
        }
    }
    for (layer_index = 0U;
         layer_index < destination->layer_count;
         layer_index++) {
        NeuralGradientLayer *destination_layer =
            &destination->layers[layer_index];
        const NeuralGradientLayer *source_layer = &source->layers[layer_index];
        size_t index;

        for (index = 0U; index < destination_layer->weight_count; index++) {
            destination_layer->weights[index] += source_layer->weights[index];
        }
        for (index = 0U; index < destination_layer->bias_count; index++) {
            destination_layer->biases[index] += source_layer->biases[index];
        }
    }
    return 1;
}

static int compensated_gradients_are_compatible(
    const NeuralGradient *sum,
    const NeuralGradient *compensation,
    const NeuralGradient *addend,
    NeuralError *error)
{
    if (sum == NULL || compensation == NULL || addend == NULL ||
        sum == compensation || sum == addend || compensation == addend ||
        sum->model != compensation->model || sum->model != addend->model ||
        sum->layer_count != compensation->layer_count ||
        sum->layer_count != addend->layer_count) {
        neural_error_set(error,
                         "compensated gradients require one model and distinct buffers");
        return 0;
    }
    return 1;
}

int neural_gradient_accumulate_compensated(
    NeuralGradient *sum,
    NeuralGradient *compensation,
    const NeuralGradient *addend,
    NeuralError *error)
{
    size_t layer_index;

    if (!compensated_gradients_are_compatible(sum,
                                              compensation,
                                              addend,
                                              error)) {
        return 0;
    }
    for (layer_index = 0U; layer_index < sum->layer_count; layer_index++) {
        NeuralGradientLayer *sum_layer = &sum->layers[layer_index];
        NeuralGradientLayer *compensation_layer =
            &compensation->layers[layer_index];
        const NeuralGradientLayer *addend_layer = &addend->layers[layer_index];
        size_t index;
        neural_real ignored_sum;
        neural_real ignored_compensation;

        if (sum_layer->weight_count != compensation_layer->weight_count ||
            sum_layer->weight_count != addend_layer->weight_count ||
            sum_layer->bias_count != compensation_layer->bias_count ||
            sum_layer->bias_count != addend_layer->bias_count) {
            neural_error_set(error,
                             "compensated gradient dimensions do not match");
            return 0;
        }
        for (index = 0U; index < sum_layer->weight_count; index++) {
            if (!neural_compensated_add(sum_layer->weights[index],
                                        compensation_layer->weights[index],
                                        addend_layer->weights[index],
                                        &ignored_sum,
                                        &ignored_compensation)) {
                neural_error_set(error,
                                 "compensated weight sum is not finite");
                return 0;
            }
        }
        for (index = 0U; index < sum_layer->bias_count; index++) {
            if (!neural_compensated_add(sum_layer->biases[index],
                                        compensation_layer->biases[index],
                                        addend_layer->biases[index],
                                        &ignored_sum,
                                        &ignored_compensation)) {
                neural_error_set(error,
                                 "compensated bias sum is not finite");
                return 0;
            }
        }
    }
    for (layer_index = 0U; layer_index < sum->layer_count; layer_index++) {
        NeuralGradientLayer *sum_layer = &sum->layers[layer_index];
        NeuralGradientLayer *compensation_layer =
            &compensation->layers[layer_index];
        const NeuralGradientLayer *addend_layer = &addend->layers[layer_index];
        size_t index;

        for (index = 0U; index < sum_layer->weight_count; index++) {
            (void)neural_compensated_add(
                sum_layer->weights[index],
                compensation_layer->weights[index],
                addend_layer->weights[index],
                &sum_layer->weights[index],
                &compensation_layer->weights[index]);
        }
        for (index = 0U; index < sum_layer->bias_count; index++) {
            (void)neural_compensated_add(
                sum_layer->biases[index],
                compensation_layer->biases[index],
                addend_layer->biases[index],
                &sum_layer->biases[index],
                &compensation_layer->biases[index]);
        }
    }
    return 1;
}

int neural_gradient_finish_compensated(NeuralGradient *sum,
                                       const NeuralGradient *compensation,
                                       neural_real factor,
                                       NeuralError *error)
{
    size_t layer_index;

    if (sum == NULL || compensation == NULL || sum == compensation ||
        sum->model != compensation->model ||
        sum->layer_count != compensation->layer_count || !isfinite(factor)) {
        neural_error_set(error, "invalid compensated gradient finalization");
        return 0;
    }
    for (layer_index = 0U; layer_index < sum->layer_count; layer_index++) {
        NeuralGradientLayer *sum_layer = &sum->layers[layer_index];
        const NeuralGradientLayer *compensation_layer =
            &compensation->layers[layer_index];
        size_t index;

        if (sum_layer->weight_count != compensation_layer->weight_count ||
            sum_layer->bias_count != compensation_layer->bias_count) {
            neural_error_set(error,
                             "compensated gradient dimensions do not match");
            return 0;
        }
        for (index = 0U; index < sum_layer->weight_count; index++) {
            neural_real combined = sum_layer->weights[index] +
                                   compensation_layer->weights[index];

            if (!isfinite(sum_layer->weights[index]) ||
                !isfinite(compensation_layer->weights[index]) ||
                !isfinite(combined) || !isfinite(combined * factor)) {
                neural_error_set(error,
                                 "final compensated weight is not finite");
                return 0;
            }
        }
        for (index = 0U; index < sum_layer->bias_count; index++) {
            neural_real combined = sum_layer->biases[index] +
                                   compensation_layer->biases[index];

            if (!isfinite(sum_layer->biases[index]) ||
                !isfinite(compensation_layer->biases[index]) ||
                !isfinite(combined) || !isfinite(combined * factor)) {
                neural_error_set(error,
                                 "final compensated bias is not finite");
                return 0;
            }
        }
    }
    for (layer_index = 0U; layer_index < sum->layer_count; layer_index++) {
        NeuralGradientLayer *sum_layer = &sum->layers[layer_index];
        const NeuralGradientLayer *compensation_layer =
            &compensation->layers[layer_index];
        size_t index;

        for (index = 0U; index < sum_layer->weight_count; index++) {
            sum_layer->weights[index] =
                (sum_layer->weights[index] +
                 compensation_layer->weights[index]) * factor;
        }
        for (index = 0U; index < sum_layer->bias_count; index++) {
            sum_layer->biases[index] =
                (sum_layer->biases[index] +
                 compensation_layer->biases[index]) * factor;
        }
    }
    return 1;
}

int neural_gradient_reduce_ordered(
    NeuralGradient *destination,
    NeuralGradient *const *sample_gradients,
    size_t sample_count,
    NeuralError *error)
{
    NeuralGradient *accumulator = NULL;
    NeuralGradient *compensation = NULL;
    size_t sample_index;

    if (destination == NULL || sample_gradients == NULL ||
        sample_count == 0U) {
        neural_error_set(error, "ordered gradient reduction requires samples");
        return 0;
    }
    if (!neural_gradient_create(destination->model, &accumulator, error) ||
        !neural_gradient_create(destination->model, &compensation, error)) {
        neural_gradient_free(accumulator);
        return 0;
    }
    for (sample_index = 0U; sample_index < sample_count; sample_index++) {
        if (sample_gradients[sample_index] == destination ||
            !neural_gradient_accumulate_compensated(
                accumulator,
                compensation,
                sample_gradients[sample_index],
                error)) {
            if (sample_gradients[sample_index] == destination) {
                neural_error_set(error,
                                 "reduction destination cannot be an input");
            }
            neural_gradient_free(compensation);
            neural_gradient_free(accumulator);
            return 0;
        }
    }
    if (!neural_gradient_finish_compensated(accumulator,
                                            compensation,
                                            1.0,
                                            error) ||
        !neural_gradient_copy(destination, accumulator, error)) {
        neural_gradient_free(compensation);
        neural_gradient_free(accumulator);
        return 0;
    }
    neural_gradient_free(compensation);
    neural_gradient_free(accumulator);
    return 1;
}

int neural_gradient_scale(NeuralGradient *gradient,
                          neural_real factor,
                          NeuralError *error)
{
    size_t layer_index;

    if (gradient == NULL || !isfinite(factor)) {
        neural_error_set(error, "invalid gradient scaling arguments");
        return 0;
    }
    for (layer_index = 0U;
         layer_index < gradient->layer_count;
         layer_index++) {
        const NeuralGradientLayer *layer = &gradient->layers[layer_index];
        size_t index;

        for (index = 0U; index < layer->weight_count; index++) {
            if (!isfinite(layer->weights[index]) ||
                !isfinite(layer->weights[index] * factor)) {
                neural_error_set(error, "scaled weight gradient is not finite");
                return 0;
            }
        }
        for (index = 0U; index < layer->bias_count; index++) {
            if (!isfinite(layer->biases[index]) ||
                !isfinite(layer->biases[index] * factor)) {
                neural_error_set(error, "scaled bias gradient is not finite");
                return 0;
            }
        }
    }
    for (layer_index = 0U;
         layer_index < gradient->layer_count;
         layer_index++) {
        NeuralGradientLayer *layer = &gradient->layers[layer_index];

        if (!neural_tensor_scale(layer->weights,
                                 layer->weight_count,
                                 factor,
                                 error) ||
            !neural_tensor_scale(layer->biases,
                                 layer->bias_count,
                                 factor,
                                 error)) {
            return 0;
        }
    }
    return 1;
}

static int norm_accumulate(neural_real value,
                           neural_real *scale,
                           neural_real *sum_squares)
{
    neural_real magnitude;
    neural_real ratio;

    if (!isfinite(value)) {
        return 0;
    }
    magnitude = fabs(value);
    if (magnitude == 0.0) {
        return 1;
    }
    if (*scale < magnitude) {
        ratio = *scale / magnitude;
        *sum_squares = 1.0 + *sum_squares * ratio * ratio;
        *scale = magnitude;
    } else {
        ratio = magnitude / *scale;
        *sum_squares += ratio * ratio;
    }
    return isfinite(*scale) && isfinite(*sum_squares);
}

int neural_gradient_norm(const NeuralGradient *gradient,
                         neural_real *norm,
                         NeuralError *error)
{
    neural_real scale = 0.0;
    neural_real sum_squares = 1.0;
    neural_real result;
    size_t layer_index;

    if (gradient == NULL || norm == NULL) {
        neural_error_set(error, "gradient and norm output are required");
        return 0;
    }
    for (layer_index = 0U;
         layer_index < gradient->layer_count;
         layer_index++) {
        const NeuralGradientLayer *layer = &gradient->layers[layer_index];
        size_t index;

        for (index = 0U; index < layer->weight_count; index++) {
            if (!norm_accumulate(layer->weights[index],
                                 &scale,
                                 &sum_squares)) {
                neural_error_set(error, "weight gradient norm is not finite");
                return 0;
            }
        }
        for (index = 0U; index < layer->bias_count; index++) {
            if (!norm_accumulate(layer->biases[index],
                                 &scale,
                                 &sum_squares)) {
                neural_error_set(error, "bias gradient norm is not finite");
                return 0;
            }
        }
    }
    result = scale == 0.0 ? 0.0 : scale * sqrt(sum_squares);
    if (!isfinite(result)) {
        neural_error_set(error, "gradient norm exceeds the finite range");
        return 0;
    }
    *norm = result;
    return 1;
}

int neural_gradient_clip_norm(NeuralGradient *gradient,
                              neural_real maximum_norm,
                              neural_real *original_norm,
                              int *clipped,
                              NeuralError *error)
{
    neural_real norm;
    int did_clip = 0;

    if (gradient == NULL || original_norm == NULL || clipped == NULL ||
        !isfinite(maximum_norm) || maximum_norm < 0.0) {
        neural_error_set(error, "invalid gradient clipping arguments");
        return 0;
    }
    if (!neural_gradient_norm(gradient, &norm, error)) {
        return 0;
    }
    if (maximum_norm > 0.0 && norm > maximum_norm) {
        if (!neural_gradient_scale(gradient, maximum_norm / norm, error)) {
            return 0;
        }
        did_clip = 1;
    }
    *original_norm = norm;
    *clipped = did_clip;
    return 1;
}

static int regularization_arguments_validate(
    const NeuralModel *model,
    neural_real l1_coefficient,
    neural_real l2_coefficient,
    int include_biases,
    NeuralError *error)
{
    if (model == NULL || !isfinite(l1_coefficient) ||
        l1_coefficient < 0.0 || !isfinite(l2_coefficient) ||
        l2_coefficient < 0.0 ||
        (include_biases != 0 && include_biases != 1)) {
        neural_error_set(error, "invalid regularization arguments");
        return 0;
    }
    return 1;
}

static int regularization_gradient_term(neural_real parameter,
                                        neural_real l1_coefficient,
                                        neural_real l2_coefficient,
                                        neural_real *term)
{
    neural_real l1_term;
    neural_real l2_term;
    neural_real result;

    if (!isfinite(parameter)) {
        return 0;
    }
    l1_term = parameter > 0.0
                  ? l1_coefficient
                  : (parameter < 0.0 ? -l1_coefficient : 0.0);
    l2_term = l2_coefficient * parameter;
    result = l1_term + l2_term;
    if (!isfinite(l2_term) || !isfinite(result)) {
        return 0;
    }
    *term = result;
    return 1;
}

static int regularization_gradient_array_validate(
    const neural_real *parameters,
    neural_real *gradients,
    size_t count,
    neural_real l1_coefficient,
    neural_real l2_coefficient)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        neural_real term = 0.0;

        if (!isfinite(gradients[index]) ||
            !regularization_gradient_term(parameters[index],
                                          l1_coefficient,
                                          l2_coefficient,
                                          &term) ||
            !isfinite(gradients[index] + term)) {
            return 0;
        }
    }
    return 1;
}

static void regularization_gradient_array_add(
    const neural_real *parameters,
    neural_real *gradients,
    size_t count,
    neural_real l1_coefficient,
    neural_real l2_coefficient)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        neural_real term = 0.0;

        (void)regularization_gradient_term(parameters[index],
                                           l1_coefficient,
                                           l2_coefficient,
                                           &term);
        gradients[index] += term;
    }
}

int neural_gradient_add_regularization(NeuralGradient *gradient,
                                       const NeuralModel *model,
                                       neural_real l1_coefficient,
                                       neural_real l2_coefficient,
                                       int include_biases,
                                       NeuralError *error)
{
    size_t layer_index;

    if (gradient == NULL ||
        !regularization_arguments_validate(model,
                                           l1_coefficient,
                                           l2_coefficient,
                                           include_biases,
                                           error) ||
        !neural_gradient_is_compatible(gradient, model)) {
        if (error != NULL && error->message[0] == '\0') {
            neural_error_set(error, "regularization gradient is incompatible");
        }
        return 0;
    }
    for (layer_index = 0U;
         layer_index < gradient->layer_count;
         layer_index++) {
        NeuralGradientLayer *gradient_layer = &gradient->layers[layer_index];
        const neural_real *weights;
        const neural_real *biases;
        size_t weight_count;
        size_t bias_count;

        weights = neural_model_layer_weights(model,
                                             layer_index,
                                             &weight_count);
        biases = neural_model_layer_biases(model,
                                           layer_index,
                                           &bias_count);
        if (weights == NULL || biases == NULL ||
            weight_count != gradient_layer->weight_count ||
            bias_count != gradient_layer->bias_count ||
            !regularization_gradient_array_validate(
                weights,
                gradient_layer->weights,
                weight_count,
                l1_coefficient,
                l2_coefficient) ||
            (include_biases != 0 &&
             !regularization_gradient_array_validate(
                 biases,
                 gradient_layer->biases,
                 bias_count,
                 l1_coefficient,
                 l2_coefficient))) {
            neural_error_set(error,
                             "regularization gradient is not finite");
            return 0;
        }
    }
    for (layer_index = 0U;
         layer_index < gradient->layer_count;
         layer_index++) {
        NeuralGradientLayer *gradient_layer = &gradient->layers[layer_index];
        const neural_real *weights;
        const neural_real *biases;
        size_t weight_count;
        size_t bias_count;

        weights = neural_model_layer_weights(model,
                                             layer_index,
                                             &weight_count);
        biases = neural_model_layer_biases(model,
                                           layer_index,
                                           &bias_count);
        regularization_gradient_array_add(weights,
                                          gradient_layer->weights,
                                          weight_count,
                                          l1_coefficient,
                                          l2_coefficient);
        if (include_biases != 0) {
            regularization_gradient_array_add(
                biases,
                gradient_layer->biases,
                bias_count,
                l1_coefficient,
                l2_coefficient);
        }
    }
    return 1;
}

static int regularization_penalty_array(
    const neural_real *parameters,
    size_t count,
    neural_real l1_coefficient,
    neural_real l2_coefficient,
    neural_real *sum,
    neural_real *compensation)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        neural_real magnitude;
        neural_real l1_term;
        neural_real l2_term;
        neural_real term;

        if (!isfinite(parameters[index])) {
            return 0;
        }
        magnitude = fabs(parameters[index]);
        l1_term = l1_coefficient * magnitude;
        if (magnitude < 2.0) {
            l2_term = (l2_coefficient * (0.5 * magnitude)) * magnitude;
        } else {
            l2_term = (l2_coefficient * magnitude) * (0.5 * magnitude);
        }
        term = l1_term + l2_term;
        if (!isfinite(l1_term) || !isfinite(l2_term) || !isfinite(term) ||
            !neural_compensated_add(*sum,
                                    *compensation,
                                    term,
                                    sum,
                                    compensation)) {
            return 0;
        }
    }
    return 1;
}

int neural_model_regularization_penalty(const NeuralModel *model,
                                        neural_real l1_coefficient,
                                        neural_real l2_coefficient,
                                        int include_biases,
                                        neural_real *penalty,
                                        NeuralError *error)
{
    neural_real sum = 0.0;
    neural_real compensation = 0.0;
    neural_real result;
    size_t layer_index;

    if (penalty == NULL ||
        !regularization_arguments_validate(model,
                                           l1_coefficient,
                                           l2_coefficient,
                                           include_biases,
                                           error)) {
        if (error != NULL && error->message[0] == '\0') {
            neural_error_set(error, "regularization penalty output is required");
        }
        return 0;
    }
    for (layer_index = 0U;
         layer_index < neural_model_layer_count(model);
         layer_index++) {
        const neural_real *weights;
        const neural_real *biases;
        size_t weight_count;
        size_t bias_count;

        weights = neural_model_layer_weights(model,
                                             layer_index,
                                             &weight_count);
        biases = neural_model_layer_biases(model,
                                           layer_index,
                                           &bias_count);
        if (weights == NULL || biases == NULL ||
            !regularization_penalty_array(weights,
                                          weight_count,
                                          l1_coefficient,
                                          l2_coefficient,
                                          &sum,
                                          &compensation) ||
            (include_biases != 0 &&
             !regularization_penalty_array(biases,
                                           bias_count,
                                           l1_coefficient,
                                           l2_coefficient,
                                           &sum,
                                           &compensation))) {
            neural_error_set(error,
                             "regularization penalty is not finite");
            return 0;
        }
    }
    result = sum + compensation;
    if (!isfinite(result) || result < 0.0) {
        neural_error_set(error, "regularization penalty is not finite");
        return 0;
    }
    *penalty = result;
    return 1;
}

int neural_model_apply_gradient(NeuralModel *model,
                                const NeuralGradient *gradient,
                                neural_real learning_rate,
                                NeuralError *error)
{
    neural_real **new_weights = NULL;
    neural_real **new_biases = NULL;
    size_t layer_count;
    size_t layer_index;
    int success = 0;

    if (model == NULL || gradient == NULL || gradient->model != model ||
        !isfinite(learning_rate) || learning_rate <= 0.0) {
        neural_error_set(error, "invalid model gradient update arguments");
        return 0;
    }
    layer_count = gradient->layer_count;
    new_weights = calloc(layer_count, sizeof(*new_weights));
    new_biases = calloc(layer_count, sizeof(*new_biases));
    if (new_weights == NULL || new_biases == NULL) {
        neural_error_set(error, "unable to allocate gradient update staging");
        goto cleanup;
    }
    for (layer_index = 0U; layer_index < layer_count; layer_index++) {
        const NeuralGradientLayer *gradient_layer =
            &gradient->layers[layer_index];
        const neural_real *weights;
        const neural_real *biases;
        size_t weight_count;
        size_t bias_count;
        size_t index;

        weights = neural_model_layer_weights(model,
                                             layer_index,
                                             &weight_count);
        biases = neural_model_layer_biases(model,
                                           layer_index,
                                           &bias_count);
        new_weights[layer_index] = malloc(weight_count * sizeof(**new_weights));
        new_biases[layer_index] = malloc(bias_count * sizeof(**new_biases));
        if (new_weights[layer_index] == NULL ||
            new_biases[layer_index] == NULL) {
            neural_error_set(error,
                             "unable to allocate update for layer %zu",
                             layer_index);
            goto cleanup;
        }
        for (index = 0U; index < weight_count; index++) {
            new_weights[layer_index][index] =
                weights[index] -
                learning_rate * gradient_layer->weights[index];
            if (!isfinite(new_weights[layer_index][index])) {
                neural_error_set(error,
                                 "weight update is not finite at layer %zu",
                                 layer_index);
                goto cleanup;
            }
        }
        for (index = 0U; index < bias_count; index++) {
            new_biases[layer_index][index] =
                biases[index] -
                learning_rate * gradient_layer->biases[index];
            if (!isfinite(new_biases[layer_index][index])) {
                neural_error_set(error,
                                 "bias update is not finite at layer %zu",
                                 layer_index);
                goto cleanup;
            }
        }
    }
    for (layer_index = 0U; layer_index < layer_count; layer_index++) {
        const NeuralGradientLayer *gradient_layer =
            &gradient->layers[layer_index];

        if (!neural_model_set_layer_parameters(
                model,
                layer_index,
                new_weights[layer_index],
                gradient_layer->weight_count,
                new_biases[layer_index],
                gradient_layer->bias_count,
                error)) {
            goto cleanup;
        }
    }
    success = 1;

cleanup:
    free_parameter_arrays(new_weights, new_biases, layer_count);
    return success;
}
