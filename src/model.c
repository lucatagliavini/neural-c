#include "neural/model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/random.h"

typedef struct {
    size_t input_count;
    size_t neuron_count;
    size_t weight_count;
    NeuralActivationSpec activation;
    neural_real *weights;
    neural_real *biases;
} NeuralRuntimeLayer;

struct NeuralModel {
    size_t input_count;
    size_t layer_count;
    size_t parameter_count;
    uint64_t random_state;
    NeuralRuntimeLayer *layers;
};

struct NeuralWorkspace {
    const NeuralModel *model;
    size_t layer_count;
    neural_real **pre_activations;
    neural_real **activations;
};

static int checked_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0U && right > SIZE_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int checked_add(size_t left, size_t right, size_t *result)
{
    if (right > SIZE_MAX - left) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static neural_real initialization_limit(const NeuralRuntimeLayer *layer)
{
    neural_real denominator;

    if (neural_activation_initializer(layer->activation.kind) ==
        NEURAL_INITIALIZER_HE_UNIFORM) {
        denominator = (neural_real)layer->input_count;
    } else {
        denominator = (neural_real)layer->input_count +
                      (neural_real)layer->neuron_count;
    }
    return sqrt(6.0 / denominator);
}

void neural_model_free(NeuralModel *model)
{
    size_t layer_index;

    if (model == NULL) {
        return;
    }
    for (layer_index = 0U; layer_index < model->layer_count; layer_index++) {
        NeuralRuntimeLayer *layer = &model->layers[layer_index];
        neural_activation_spec_free(&layer->activation);
        free(layer->weights);
        free(layer->biases);
    }
    free(model->layers);
    free(model);
}

int neural_model_create(const NeuralModelSpec *spec,
                        uint64_t seed,
                        NeuralModel **model,
                        NeuralError *error)
{
    NeuralModel *created;
    NeuralRandom random;
    size_t previous_count;
    size_t layer_index;

    if (model == NULL) {
        neural_error_set(error, "model output pointer is required");
        return 0;
    }
    *model = NULL;
    if (!neural_model_spec_validate(spec, error)) {
        return 0;
    }
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate runtime model");
        return 0;
    }
    created->layers = calloc(spec->layer_count, sizeof(*created->layers));
    if (created->layers == NULL) {
        neural_error_set(error, "unable to allocate runtime layers");
        neural_model_free(created);
        return 0;
    }
    created->input_count = spec->input_count;
    created->layer_count = spec->layer_count;
    previous_count = spec->input_count;
    neural_random_init(&random, seed);

    for (layer_index = 0U; layer_index < spec->layer_count; layer_index++) {
        const NeuralLayerSpec *source = &spec->layers[layer_index];
        NeuralRuntimeLayer *layer = &created->layers[layer_index];
        size_t parameter_count;
        size_t weight_index;
        neural_real limit;

        layer->input_count = previous_count;
        layer->neuron_count = source->neuron_count;
        if (!checked_multiply(layer->input_count,
                              layer->neuron_count,
                              &layer->weight_count) ||
            layer->weight_count > SIZE_MAX / sizeof(*layer->weights)) {
            neural_error_set(error,
                             "runtime layer %zu dimensions overflow",
                             layer_index);
            neural_model_free(created);
            return 0;
        }
        if (!checked_add(layer->weight_count,
                         layer->neuron_count,
                         &parameter_count) ||
            !checked_add(created->parameter_count,
                         parameter_count,
                         &created->parameter_count)) {
            neural_error_set(error, "model parameter count overflows size_t");
            neural_model_free(created);
            return 0;
        }
        if (!neural_activation_spec_copy(&source->activation,
                                         &layer->activation,
                                         error)) {
            neural_model_free(created);
            return 0;
        }
        layer->weights = malloc(layer->weight_count * sizeof(*layer->weights));
        layer->biases = calloc(layer->neuron_count, sizeof(*layer->biases));
        if (layer->weights == NULL || layer->biases == NULL) {
            neural_error_set(error,
                             "unable to allocate parameters for layer %zu",
                             layer_index);
            neural_model_free(created);
            return 0;
        }
        limit = initialization_limit(layer);
        if (!isfinite(limit) || limit <= 0.0) {
            neural_error_set(error,
                             "invalid initialization range for layer %zu",
                             layer_index);
            neural_model_free(created);
            return 0;
        }
        for (weight_index = 0U;
             weight_index < layer->weight_count;
             weight_index++) {
            neural_real unit = neural_random_next_unit(&random);
            layer->weights[weight_index] = (2.0 * unit - 1.0) * limit;
        }
        previous_count = layer->neuron_count;
    }
    created->random_state = random.state;
    *model = created;
    return 1;
}

size_t neural_model_input_count(const NeuralModel *model)
{
    return model == NULL ? 0U : model->input_count;
}

size_t neural_model_output_count(const NeuralModel *model)
{
    if (model == NULL || model->layer_count == 0U) {
        return 0U;
    }
    return model->layers[model->layer_count - 1U].neuron_count;
}

size_t neural_model_layer_count(const NeuralModel *model)
{
    return model == NULL ? 0U : model->layer_count;
}

size_t neural_model_parameter_count(const NeuralModel *model)
{
    return model == NULL ? 0U : model->parameter_count;
}

size_t neural_model_layer_input_count(const NeuralModel *model,
                                      size_t layer_index)
{
    if (model == NULL || layer_index >= model->layer_count) {
        return 0U;
    }
    return model->layers[layer_index].input_count;
}

size_t neural_model_layer_neuron_count(const NeuralModel *model,
                                       size_t layer_index)
{
    if (model == NULL || layer_index >= model->layer_count) {
        return 0U;
    }
    return model->layers[layer_index].neuron_count;
}

uint64_t neural_model_random_state(const NeuralModel *model)
{
    return model == NULL ? UINT64_C(0) : model->random_state;
}

const neural_real *neural_model_layer_weights(const NeuralModel *model,
                                              size_t layer_index,
                                              size_t *count)
{
    if (count != NULL) {
        *count = 0U;
    }
    if (model == NULL || layer_index >= model->layer_count) {
        return NULL;
    }
    if (count != NULL) {
        *count = model->layers[layer_index].weight_count;
    }
    return model->layers[layer_index].weights;
}

const neural_real *neural_model_layer_biases(const NeuralModel *model,
                                             size_t layer_index,
                                             size_t *count)
{
    if (count != NULL) {
        *count = 0U;
    }
    if (model == NULL || layer_index >= model->layer_count) {
        return NULL;
    }
    if (count != NULL) {
        *count = model->layers[layer_index].neuron_count;
    }
    return model->layers[layer_index].biases;
}

int neural_model_set_layer_parameters(NeuralModel *model,
                                      size_t layer_index,
                                      const neural_real *weights,
                                      size_t weight_count,
                                      const neural_real *biases,
                                      size_t bias_count,
                                      NeuralError *error)
{
    NeuralRuntimeLayer *layer;
    size_t index;

    if (model == NULL || layer_index >= model->layer_count ||
        weights == NULL || biases == NULL) {
        neural_error_set(error, "invalid layer parameter arguments");
        return 0;
    }
    layer = &model->layers[layer_index];
    if (weight_count != layer->weight_count ||
        bias_count != layer->neuron_count) {
        neural_error_set(error,
                         "layer %zu parameter dimensions do not match",
                         layer_index);
        return 0;
    }
    for (index = 0U; index < weight_count; index++) {
        if (!isfinite(weights[index])) {
            neural_error_set(error, "layer weights must be finite");
            return 0;
        }
    }
    for (index = 0U; index < bias_count; index++) {
        if (!isfinite(biases[index])) {
            neural_error_set(error, "layer biases must be finite");
            return 0;
        }
    }
    memcpy(layer->weights, weights, weight_count * sizeof(*weights));
    memcpy(layer->biases, biases, bias_count * sizeof(*biases));
    return 1;
}

void neural_workspace_free(NeuralWorkspace *workspace)
{
    size_t layer_index;

    if (workspace == NULL) {
        return;
    }
    for (layer_index = 0U;
         layer_index < workspace->layer_count;
         layer_index++) {
        if (workspace->pre_activations != NULL) {
            free(workspace->pre_activations[layer_index]);
        }
        if (workspace->activations != NULL) {
            free(workspace->activations[layer_index]);
        }
    }
    free(workspace->pre_activations);
    free(workspace->activations);
    free(workspace);
}

int neural_workspace_create(const NeuralModel *model,
                            NeuralWorkspace **workspace,
                            NeuralError *error)
{
    NeuralWorkspace *created;
    size_t layer_index;

    if (model == NULL || workspace == NULL) {
        neural_error_set(error, "model and workspace output are required");
        return 0;
    }
    *workspace = NULL;
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate workspace");
        return 0;
    }
    created->model = model;
    created->layer_count = model->layer_count;
    created->pre_activations = calloc(model->layer_count,
                                      sizeof(*created->pre_activations));
    created->activations = calloc(model->layer_count,
                                  sizeof(*created->activations));
    if (created->pre_activations == NULL || created->activations == NULL) {
        neural_error_set(error, "unable to allocate workspace layers");
        neural_workspace_free(created);
        return 0;
    }
    for (layer_index = 0U; layer_index < model->layer_count; layer_index++) {
        size_t count = model->layers[layer_index].neuron_count;
        created->pre_activations[layer_index] =
            calloc(count, sizeof(**created->pre_activations));
        created->activations[layer_index] =
            calloc(count, sizeof(**created->activations));
        if (created->pre_activations[layer_index] == NULL ||
            created->activations[layer_index] == NULL) {
            neural_error_set(error,
                             "unable to allocate workspace for layer %zu",
                             layer_index);
            neural_workspace_free(created);
            return 0;
        }
    }
    *workspace = created;
    return 1;
}

int neural_model_forward(const NeuralModel *model,
                         NeuralWorkspace *workspace,
                         const neural_real *inputs,
                         size_t input_count,
                         neural_real *outputs,
                         size_t output_count,
                         NeuralError *error)
{
    const neural_real *previous_values;
    size_t previous_count;
    size_t layer_index;
    size_t index;

    if (model == NULL || workspace == NULL || inputs == NULL ||
        outputs == NULL || workspace->model != model) {
        neural_error_set(error, "invalid or incompatible forward-pass buffers");
        return 0;
    }
    if (input_count != model->input_count ||
        output_count != neural_model_output_count(model)) {
        neural_error_set(error, "forward-pass dimensions do not match model");
        return 0;
    }
    for (index = 0U; index < input_count; index++) {
        if (!isfinite(inputs[index])) {
            neural_error_set(error, "model inputs must be finite");
            return 0;
        }
    }
    previous_values = inputs;
    previous_count = input_count;
    for (layer_index = 0U; layer_index < model->layer_count; layer_index++) {
        const NeuralRuntimeLayer *layer = &model->layers[layer_index];
        neural_real *pre_activations =
            workspace->pre_activations[layer_index];
        size_t neuron_index;

        if (previous_count != layer->input_count) {
            neural_error_set(error, "runtime layer chain is inconsistent");
            return 0;
        }
        for (neuron_index = 0U;
             neuron_index < layer->neuron_count;
             neuron_index++) {
            neural_real sum = layer->biases[neuron_index];
            size_t input_index;
            size_t weight_offset = neuron_index * layer->input_count;

            for (input_index = 0U;
                 input_index < layer->input_count;
                 input_index++) {
                sum += layer->weights[weight_offset + input_index] *
                       previous_values[input_index];
            }
            if (!isfinite(sum)) {
                neural_error_set(error,
                                 "pre-activation is not finite at layer %zu",
                                 layer_index);
                return 0;
            }
            pre_activations[neuron_index] = sum;
        }
        if (!neural_activation_apply(
                &layer->activation,
                pre_activations,
                workspace->activations[layer_index],
                layer->neuron_count,
                error)) {
            return 0;
        }
        previous_values = workspace->activations[layer_index];
        previous_count = layer->neuron_count;
    }
    memcpy(outputs, previous_values, output_count * sizeof(*outputs));
    return 1;
}
