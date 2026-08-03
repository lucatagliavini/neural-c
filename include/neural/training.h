#ifndef NEURAL_TRAINING_H
#define NEURAL_TRAINING_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/model.h"
#include "neural/parallel.h"
#include "neural/project.h"
#include "neural/types.h"

typedef enum {
    NEURAL_TRAIN_FRESH,
    NEURAL_TRAIN_RESUME,
    NEURAL_TRAIN_ADDITIONAL
} NeuralTrainingMode;

typedef struct {
    NeuralTrainingMode mode;
    size_t additional_epochs;
} NeuralTrainingRequest;

typedef struct {
    size_t completed_epochs;
    neural_real loss;
} NeuralEpochReport;

typedef int (*NeuralEpochObserver)(const NeuralEpochReport *report,
                                   void *context,
                                   NeuralError *error);

typedef struct {
    size_t completed_epochs;
    size_t worker_count;
    neural_real final_loss;
} NeuralTrainingResult;

int neural_training_request_validate(const NeuralTrainingRequest *request,
                                     NeuralError *error);
const char *neural_training_mode_name(NeuralTrainingMode mode);

int neural_model_train_full_batch(
    NeuralModel *model,
    const NeuralDataset *dataset,
    const NeuralTrainingConfig *training,
    const NeuralExecutionConfig *execution,
    NeuralEpochObserver observer,
    void *observer_context,
    NeuralTrainingResult *result,
    NeuralError *error);

#endif
