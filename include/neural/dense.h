#ifndef NEURAL_DENSE_H
#define NEURAL_DENSE_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/types.h"

int neural_dense_forward(const neural_real *weights,
                         size_t weight_count,
                         const neural_real *biases,
                         size_t bias_count,
                         const neural_real *inputs,
                         size_t input_count,
                         neural_real *outputs,
                         size_t output_count,
                         NeuralError *error);

int neural_dense_backward(const neural_real *weights,
                          size_t weight_count,
                          const neural_real *inputs,
                          size_t input_count,
                          const neural_real *output_gradients,
                          size_t output_count,
                          neural_real *input_gradients,
                          size_t input_gradient_count,
                          neural_real *weight_gradients,
                          size_t weight_gradient_count,
                          neural_real *bias_gradients,
                          size_t bias_gradient_count,
                          NeuralError *error);

#endif
