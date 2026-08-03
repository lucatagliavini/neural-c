#ifndef NEURAL_MODEL_H
#define NEURAL_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "neural/error.h"
#include "neural/project.h"
#include "neural/types.h"

typedef struct NeuralModel NeuralModel;
typedef struct NeuralWorkspace NeuralWorkspace;

int neural_model_create(const NeuralModelSpec *spec,
                        uint64_t seed,
                        NeuralModel **model,
                        NeuralError *error);
void neural_model_free(NeuralModel *model);

size_t neural_model_input_count(const NeuralModel *model);
size_t neural_model_output_count(const NeuralModel *model);
size_t neural_model_layer_count(const NeuralModel *model);
size_t neural_model_parameter_count(const NeuralModel *model);
size_t neural_model_layer_input_count(const NeuralModel *model,
                                      size_t layer_index);
size_t neural_model_layer_neuron_count(const NeuralModel *model,
                                       size_t layer_index);
uint64_t neural_model_random_state(const NeuralModel *model);

const neural_real *neural_model_layer_weights(const NeuralModel *model,
                                              size_t layer_index,
                                              size_t *count);
const neural_real *neural_model_layer_biases(const NeuralModel *model,
                                             size_t layer_index,
                                             size_t *count);
int neural_model_set_layer_parameters(NeuralModel *model,
                                      size_t layer_index,
                                      const neural_real *weights,
                                      size_t weight_count,
                                      const neural_real *biases,
                                      size_t bias_count,
                                      NeuralError *error);

int neural_workspace_create(const NeuralModel *model,
                            NeuralWorkspace **workspace,
                            NeuralError *error);
void neural_workspace_free(NeuralWorkspace *workspace);

int neural_model_forward(const NeuralModel *model,
                         NeuralWorkspace *workspace,
                         const neural_real *inputs,
                         size_t input_count,
                         neural_real *outputs,
                         size_t output_count,
                         NeuralError *error);

#endif
