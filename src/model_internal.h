#ifndef NEURAL_INTERNAL_MODEL_H
#define NEURAL_INTERNAL_MODEL_H

#include "neural/model.h"

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
    neural_real **activation_gradients;
    neural_real **pre_activation_gradients;
    neural_real *input_gradients;
};

#endif
