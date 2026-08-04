#include "neural/training.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "neural/evaluation.h"

int neural_training_request_validate(const NeuralTrainingRequest *request,
                                     NeuralError *error)
{
    if (request == NULL) {
        neural_error_set(error, "training request is required");
        return 0;
    }
    switch (request->mode) {
    case NEURAL_TRAIN_FRESH:
    case NEURAL_TRAIN_RESUME:
        if (request->additional_epochs != 0U) {
            neural_error_set(error,
                             "additional epochs are valid only in additional mode");
            return 0;
        }
        return 1;
    case NEURAL_TRAIN_ADDITIONAL:
        if (request->additional_epochs == 0U) {
            neural_error_set(error,
                             "additional epochs must be a positive integer");
            return 0;
        }
        return 1;
    }
    neural_error_set(error, "unknown training mode");
    return 0;
}

const char *neural_training_mode_name(NeuralTrainingMode mode)
{
    switch (mode) {
    case NEURAL_TRAIN_FRESH:
        return "fresh";
    case NEURAL_TRAIN_RESUME:
        return "resume";
    case NEURAL_TRAIN_ADDITIONAL:
        return "additional";
    }
    return "unknown";
}

int neural_model_train_full_batch_range(
    NeuralModel *model,
    const NeuralDataset *dataset,
    const NeuralTrainingConfig *training,
    const NeuralExecutionConfig *execution,
    size_t completed_epochs,
    size_t target_epochs,
    NeuralEpochObserver observer,
    void *observer_context,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralParallelExecutor *executor = NULL;
    NeuralWorkspace *evaluation_workspace = NULL;
    neural_real *predicted = NULL;
    NeuralTrainingResult completed = {0U, 0U, 0.0};
    size_t output_count;
    size_t epoch_index;
    int success = 0;

    neural_error_clear(error);
    if (result == NULL || model == NULL || dataset == NULL ||
        training == NULL || execution == NULL) {
        neural_error_set(error, "training arguments and result are required");
        return 0;
    }
    *result = completed;
    if (target_epochs == 0U || completed_epochs > target_epochs) {
        neural_error_set(error, "training epoch range is invalid");
        return 0;
    }
    if (!neural_training_config_validate(training, error) ||
        !neural_parallel_executor_create(model,
                                         dataset,
                                         training->loss,
                                         execution,
                                         &executor,
                                         error) ||
        !neural_workspace_create(model, &evaluation_workspace, error)) {
        goto cleanup;
    }
    output_count = neural_model_output_count(model);
    if (output_count > SIZE_MAX / sizeof(*predicted)) {
        neural_error_set(error, "training output dimensions overflow");
        goto cleanup;
    }
    predicted = malloc(output_count * sizeof(*predicted));
    if (predicted == NULL) {
        neural_error_set(error, "unable to allocate training evaluation output");
        goto cleanup;
    }
    completed.worker_count =
        neural_parallel_executor_worker_count(executor);
    if (completed_epochs == target_epochs) {
        if (!neural_model_evaluate_dataset_loss(model,
                                   evaluation_workspace,
                                   predicted,
                                   dataset,
                                   training->loss,
                                   &completed.final_loss,
                                   error)) {
            goto cleanup;
        }
        completed.completed_epochs = completed_epochs;
        *result = completed;
        success = 1;
        goto cleanup;
    }
    for (epoch_index = completed_epochs; epoch_index < target_epochs;) {
        const NeuralGradient *gradient;
        NeuralEpochReport report = {0};

        if (!neural_parallel_executor_batch_gradient(
                executor,
                0U,
                dataset->sample_count,
                &gradient,
                error) ||
            !neural_model_apply_gradient(model,
                                         gradient,
                                         training->learning_rate,
                                         error) ||
            !neural_model_evaluate_dataset_loss(model,
                                   evaluation_workspace,
                                   predicted,
                                   dataset,
                                   training->loss,
                                   &report.loss,
                                   error)) {
            goto cleanup;
        }
        report.completed_epochs = epoch_index + 1U;
        report.target_epochs = target_epochs;
        if (observer != NULL) {
            int observer_status = observer(&report, observer_context, error);

            if (observer_status == NEURAL_EPOCH_OBSERVER_STOP) {
                completed.completed_epochs = report.completed_epochs;
                completed.final_loss = report.loss;
                epoch_index = report.completed_epochs;
                break;
            }
            if (observer_status != NEURAL_EPOCH_OBSERVER_CONTINUE) {
            if (error != NULL && error->message[0] == '\0') {
                neural_error_set(error,
                                 "epoch observer rejected epoch %zu",
                                 report.completed_epochs);
            }
            goto cleanup;
            }
        }
        completed.completed_epochs = report.completed_epochs;
        completed.final_loss = report.loss;
        epoch_index = report.completed_epochs;
    }
    *result = completed;
    success = 1;

cleanup:
    free(predicted);
    neural_workspace_free(evaluation_workspace);
    neural_parallel_executor_free(executor);
    return success;
}

int neural_model_train_full_batch(
    NeuralModel *model,
    const NeuralDataset *dataset,
    const NeuralTrainingConfig *training,
    const NeuralExecutionConfig *execution,
    NeuralEpochObserver observer,
    void *observer_context,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    size_t target_epochs = training == NULL ? 0U : training->epochs;

    return neural_model_train_full_batch_range(model,
                                               dataset,
                                               training,
                                               execution,
                                               0U,
                                               target_epochs,
                                               observer,
                                               observer_context,
                                               result,
                                               error);
}
