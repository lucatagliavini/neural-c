#ifndef NEURAL_GRADIENT_H
#define NEURAL_GRADIENT_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/model.h"
#include "neural/types.h"

typedef struct NeuralGradient NeuralGradient;

int neural_gradient_create(const NeuralModel *model,
                           NeuralGradient **gradient,
                           NeuralError *error);
void neural_gradient_free(NeuralGradient *gradient);
int neural_gradient_zero(NeuralGradient *gradient, NeuralError *error);
int neural_gradient_copy(NeuralGradient *destination,
                         const NeuralGradient *source,
                         NeuralError *error);
int neural_gradient_is_compatible(const NeuralGradient *gradient,
                                  const NeuralModel *model);

neural_real *neural_gradient_layer_weights(NeuralGradient *gradient,
                                           size_t layer_index,
                                           size_t *count);
neural_real *neural_gradient_layer_biases(NeuralGradient *gradient,
                                          size_t layer_index,
                                          size_t *count);

int neural_gradient_reduce_ordered(
    NeuralGradient *destination,
    NeuralGradient *const *sample_gradients,
    size_t sample_count,
    NeuralError *error);
int neural_gradient_scale(NeuralGradient *gradient,
                          neural_real factor,
                          NeuralError *error);
int neural_model_apply_gradient(NeuralModel *model,
                                const NeuralGradient *gradient,
                                neural_real learning_rate,
                                NeuralError *error);

#endif
