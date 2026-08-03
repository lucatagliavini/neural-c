#ifndef NEURAL_GRADIENT_CHECK_H
#define NEURAL_GRADIENT_CHECK_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/loss.h"
#include "neural/model.h"
#include "neural/types.h"

typedef enum {
    NEURAL_PARAMETER_WEIGHT,
    NEURAL_PARAMETER_BIAS
} NeuralParameterKind;

typedef struct {
    neural_real epsilon;
    neural_real absolute_tolerance;
    neural_real relative_tolerance;
} NeuralGradientCheckConfig;

typedef struct {
    int passed;
    size_t checked_parameter_count;
    neural_real maximum_absolute_error;
    neural_real maximum_relative_error;
    size_t worst_layer_index;
    NeuralParameterKind worst_parameter_kind;
    size_t worst_parameter_index;
    neural_real worst_analytic_value;
    neural_real worst_numeric_value;
} NeuralGradientCheckResult;

void neural_gradient_check_config_default(NeuralGradientCheckConfig *config);

int neural_model_gradient_check(NeuralModel *model,
                                NeuralWorkspace *workspace,
                                NeuralLoss loss,
                                const neural_real *inputs,
                                size_t input_count,
                                const neural_real *expected,
                                size_t expected_count,
                                const NeuralGradientCheckConfig *config,
                                NeuralGradientCheckResult *result,
                                NeuralError *error);

#endif
