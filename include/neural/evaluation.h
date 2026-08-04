#ifndef NEURAL_EVALUATION_H
#define NEURAL_EVALUATION_H

#include <stddef.h>

#include "neural/activation.h"
#include "neural/error.h"
#include "neural/project.h"
#include "neural/model.h"
#include "neural/types.h"

typedef struct {
    size_t sample_count;
    size_t output_count;
    size_t class_count;
    size_t correct_count;
    neural_real loss;
    neural_real accuracy;
    size_t *confusion;
    neural_real *precision;
    neural_real *recall;
    neural_real *f1;
    int is_classification;
} NeuralEvaluationResult;

int neural_evaluation_compute(NeuralLoss loss,
                              NeuralActivationKind output_activation,
                              const neural_real *predicted,
                              const neural_real *expected,
                              size_t sample_count,
                              size_t output_count,
                              NeuralEvaluationResult *result,
                              NeuralError *error);
void neural_evaluation_result_free(NeuralEvaluationResult *result);

int neural_model_evaluate_dataset_loss(const NeuralModel *model,
                                       NeuralWorkspace *workspace,
                                       neural_real *predicted,
                                       const NeuralDataset *dataset,
                                       NeuralLoss loss,
                                       neural_real *value,
                                       NeuralError *error);

#endif
