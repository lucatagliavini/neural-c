#ifndef NEURAL_GRADIENT_INTERNAL_H
#define NEURAL_GRADIENT_INTERNAL_H

#include "neural/error.h"
#include "neural/gradient.h"
#include "neural/types.h"

int neural_gradient_accumulate_compensated(
    NeuralGradient *sum,
    NeuralGradient *compensation,
    const NeuralGradient *addend,
    NeuralError *error);
int neural_gradient_finish_compensated(NeuralGradient *sum,
                                       const NeuralGradient *compensation,
                                       neural_real factor,
                                       NeuralError *error);

#endif
