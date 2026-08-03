#ifndef NEURAL_TENSOR_OPS_H
#define NEURAL_TENSOR_OPS_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/types.h"

int neural_tensor_zero(neural_real *values,
                       size_t count,
                       NeuralError *error);
int neural_tensor_add(neural_real *destination,
                      const neural_real *source,
                      size_t count,
                      NeuralError *error);
int neural_tensor_scale(neural_real *values,
                        size_t count,
                        neural_real factor,
                        NeuralError *error);

#endif
