#ifndef NEURAL_LOSS_H
#define NEURAL_LOSS_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/project.h"
#include "neural/types.h"

int neural_loss_evaluate(NeuralLoss loss,
                         const neural_real *predicted,
                         const neural_real *expected,
                         size_t count,
                         neural_real *value,
                         NeuralError *error);
int neural_loss_gradient(NeuralLoss loss,
                         const neural_real *predicted,
                         const neural_real *expected,
                         size_t count,
                         neural_real *gradient,
                         NeuralError *error);

#endif
