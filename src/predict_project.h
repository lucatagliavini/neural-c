#ifndef NEURAL_INTERNAL_PREDICT_PROJECT_H
#define NEURAL_INTERNAL_PREDICT_PROJECT_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/model.h"
#include "neural/parallel.h"
#include "neural/types.h"

typedef struct {
    NeuralModel *model;
    size_t input_count;
    size_t output_count;
    size_t completed_epochs;
} NeuralPredictionSnapshot;

#define NEURAL_PREDICTION_SNAPSHOT_INITIALIZER {NULL, 0U, 0U, 0U}

int neural_project_prediction_load(const char *directory,
                                   NeuralPredictionSnapshot *snapshot,
                                   NeuralError *error);
void neural_prediction_snapshot_free(NeuralPredictionSnapshot *snapshot);

int neural_prediction_run(const NeuralPredictionSnapshot *snapshot,
                          const neural_real *inputs,
                          size_t sample_count,
                          const NeuralExecutionConfig *execution,
                          neural_real *outputs,
                          size_t *worker_count,
                          NeuralError *error);

#endif
