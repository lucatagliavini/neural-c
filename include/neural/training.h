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
    neural_real objective;
    size_t target_epochs;
    int has_validation_loss;
    neural_real validation_loss;
    neural_real best_validation_loss;
    int stopped_early;
    neural_real max_gradient_norm;
    size_t clipped_batch_count;
} NeuralEpochReport;

#define NEURAL_EPOCH_OBSERVER_ERROR 0
#define NEURAL_EPOCH_OBSERVER_CONTINUE 1
#define NEURAL_EPOCH_OBSERVER_STOP 2

typedef int (*NeuralEpochObserver)(const NeuralEpochReport *report,
                                   void *context,
                                   NeuralError *error);

typedef struct {
    size_t completed_epochs;
    size_t worker_count;
    neural_real final_loss;
    neural_real final_objective;
    neural_real final_max_gradient_norm;
    size_t clipped_batch_count;
} NeuralTrainingResult;

int neural_training_request_validate(const NeuralTrainingRequest *request,
                                     NeuralError *error);
const char *neural_training_mode_name(NeuralTrainingMode mode);

int neural_model_train(
    NeuralModel *model,
    const NeuralDataset *dataset,
    const NeuralTrainingConfig *training,
    const NeuralExecutionConfig *execution,
    NeuralEpochObserver observer,
    void *observer_context,
    NeuralTrainingResult *result,
    NeuralError *error);

int neural_model_train_range(
    NeuralModel *model,
    const NeuralDataset *dataset,
    const NeuralTrainingConfig *training,
    const NeuralExecutionConfig *execution,
    size_t completed_epochs,
    size_t target_epochs,
    NeuralEpochObserver observer,
    void *observer_context,
    NeuralTrainingResult *result,
    NeuralError *error);

#endif
