#include "neural/backprop.h"

#include "neural/activation.h"
#include "neural/dense.h"
#include "model_internal.h"

int neural_model_sample_gradient(const NeuralModel *model,
                                 NeuralWorkspace *workspace,
                                 NeuralGradient *gradient,
                                 NeuralLoss loss,
                                 const neural_real *inputs,
                                 size_t input_count,
                                 const neural_real *expected,
                                 size_t expected_count,
                                 neural_real *loss_value,
                                 NeuralError *error)
{
    size_t output_count;
    size_t layer_index;

    if (model == NULL || workspace == NULL || gradient == NULL ||
        inputs == NULL || expected == NULL || loss_value == NULL ||
        workspace->model != model ||
        !neural_gradient_is_compatible(gradient, model)) {
        neural_error_set(error,
                         "invalid or incompatible sample-gradient arguments");
        return 0;
    }
    output_count = neural_model_output_count(model);
    if (input_count != model->input_count || expected_count != output_count) {
        neural_error_set(error,
                         "sample-gradient dimensions do not match model");
        return 0;
    }
    if (!neural_model_forward(
            model,
            workspace,
            inputs,
            input_count,
            workspace->activation_gradients[model->layer_count - 1U],
            output_count,
            error) ||
        !neural_loss_evaluate(loss,
                              workspace->activations[model->layer_count - 1U],
                              expected,
                              output_count,
                              loss_value,
                              error) ||
        !neural_loss_gradient(
            loss,
            workspace->activations[model->layer_count - 1U],
            expected,
            output_count,
            workspace->activation_gradients[model->layer_count - 1U],
            error)) {
        return 0;
    }

    for (layer_index = model->layer_count; layer_index-- > 0U;) {
        const NeuralRuntimeLayer *layer = &model->layers[layer_index];
        const neural_real *layer_inputs = layer_index == 0U
                                              ? inputs
                                              : workspace->activations[
                                                    layer_index - 1U];
        neural_real *input_gradients = layer_index == 0U
                                           ? workspace->input_gradients
                                           : workspace->activation_gradients[
                                                 layer_index - 1U];
        neural_real *weight_gradients;
        neural_real *bias_gradients;
        size_t weight_count;
        size_t bias_count;

        weight_gradients = neural_gradient_layer_weights(gradient,
                                                          layer_index,
                                                          &weight_count);
        bias_gradients = neural_gradient_layer_biases(gradient,
                                                      layer_index,
                                                      &bias_count);
        if (weight_gradients == NULL || bias_gradients == NULL ||
            weight_count != layer->weight_count ||
            bias_count != layer->neuron_count) {
            neural_error_set(error,
                             "sample gradient does not match layer %zu",
                             layer_index);
            return 0;
        }
        if (!neural_activation_backward(
                &layer->activation,
                workspace->pre_activations[layer_index],
                workspace->activations[layer_index],
                workspace->activation_gradients[layer_index],
                workspace->pre_activation_gradients[layer_index],
                layer->neuron_count,
                error) ||
            !neural_dense_backward(
                layer->weights,
                layer->weight_count,
                layer_inputs,
                layer->input_count,
                workspace->pre_activation_gradients[layer_index],
                layer->neuron_count,
                input_gradients,
                layer->input_count,
                weight_gradients,
                weight_count,
                bias_gradients,
                bias_count,
                error)) {
            return 0;
        }
    }
    return 1;
}
