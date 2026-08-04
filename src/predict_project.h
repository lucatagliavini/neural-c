#ifndef NEURAL_INTERNAL_PREDICT_PROJECT_H
#define NEURAL_INTERNAL_PREDICT_PROJECT_H

#include <stddef.h>

#include "neural/checkpoint.h"
#include "neural/error.h"
#include "neural/model.h"
#include "neural/parallel.h"
#include "neural/project.h"
#include "neural/types.h"

typedef struct {
    NeuralModel *model;
    NeuralPreprocessing preprocessing;
    int has_preprocessing;
    size_t input_count;
    size_t output_count;
    size_t completed_epochs;
    size_t selected_epoch;
    size_t target_epochs;
    NeuralCompletionReason completion_reason;
    size_t format_version;
} NeuralPredictionSnapshot;

#define NEURAL_PREDICTION_SNAPSHOT_INITIALIZER \
    {NULL, {0}, 0, 0U, 0U, 0U, 0U, 0U, NEURAL_COMPLETION_TARGET, 0U}

typedef struct {
    NeuralPredictionSnapshot prediction;
    NeuralDataset dataset;
    NeuralLoss loss;
    NeuralActivationKind output_activation;
} NeuralEvaluationSnapshot;

#define NEURAL_EVALUATION_SNAPSHOT_INITIALIZER \
    {NEURAL_PREDICTION_SNAPSHOT_INITIALIZER, {0U, 0U, 0U, NULL, NULL}, \
     NEURAL_LOSS_MSE, NEURAL_ACTIVATION_LINEAR}

int neural_project_prediction_load(const char *directory,
                                   NeuralPredictionSnapshot *snapshot,
                                   NeuralError *error);
void neural_prediction_snapshot_free(NeuralPredictionSnapshot *snapshot);

int neural_project_evaluation_load(const char *directory,
                                   const char *dataset_filename,
                                   NeuralEvaluationSnapshot *snapshot,
                                   NeuralError *error);
void neural_evaluation_snapshot_free(NeuralEvaluationSnapshot *snapshot);

int neural_prediction_run(const NeuralPredictionSnapshot *snapshot,
                          const neural_real *inputs,
                          size_t sample_count,
                          const NeuralExecutionConfig *execution,
                          neural_real *outputs,
                          size_t *worker_count,
                          NeuralError *error);
int neural_prediction_prepare_inputs(const NeuralPredictionSnapshot *snapshot,
                                     neural_real *inputs,
                                     size_t sample_count,
                                     NeuralError *error);

#endif
