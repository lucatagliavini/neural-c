#ifndef NEURAL_BACKPROP_H
#define NEURAL_BACKPROP_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/gradient.h"
#include "neural/loss.h"
#include "neural/model.h"
#include "neural/types.h"

int neural_model_sample_gradient(const NeuralModel *model,
                                 NeuralWorkspace *workspace,
                                 NeuralGradient *gradient,
                                 NeuralLoss loss,
                                 const neural_real *inputs,
                                 size_t input_count,
                                 const neural_real *expected,
                                 size_t expected_count,
                                 neural_real *loss_value,
                                 NeuralError *error);

#endif
