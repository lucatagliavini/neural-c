#include "neural/gradient.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/tensor_ops.h"

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

int neural_gradient_reduce_ordered(
    NeuralGradient *destination,
    NeuralGradient *const *sample_gradients,
    size_t sample_count,
    NeuralError *error)
{
    NeuralGradient *accumulator = NULL;
    size_t sample_index;
    size_t layer_index;

    if (destination == NULL || sample_gradients == NULL ||
        sample_count == 0U) {
        neural_error_set(error, "ordered gradient reduction requires samples");
        return 0;
    }
    if (!neural_gradient_create(destination->model, &accumulator, error)) {
        return 0;
    }
    for (sample_index = 0U; sample_index < sample_count; sample_index++) {
        if (sample_gradients[sample_index] == destination ||
            !neural_gradient_add(accumulator,
                                 sample_gradients[sample_index],
                                 error)) {
            if (sample_gradients[sample_index] == destination) {
                neural_error_set(error,
                                 "reduction destination cannot be an input");
            }
            neural_gradient_free(accumulator);
            return 0;
        }
    }
    for (layer_index = 0U;
         layer_index < destination->layer_count;
         layer_index++) {
        NeuralGradientLayer *destination_layer =
            &destination->layers[layer_index];
        const NeuralGradientLayer *accumulator_layer =
            &accumulator->layers[layer_index];

        memcpy(destination_layer->weights,
               accumulator_layer->weights,
               destination_layer->weight_count *
                   sizeof(*destination_layer->weights));
        memcpy(destination_layer->biases,
               accumulator_layer->biases,
               destination_layer->bias_count *
                   sizeof(*destination_layer->biases));
    }
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
