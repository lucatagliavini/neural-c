#include "neural/training.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "neural/loss.h"
#include "compensated_sum.h"

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

static int compensated_add_scalar(neural_real *sum,
                                  neural_real *compensation,
                                  neural_real value,
                                  NeuralError *error)
{
    neural_real new_sum;
    neural_real new_compensation;

    if (sum == NULL || compensation == NULL ||
        !neural_compensated_add(*sum,
                                *compensation,
                                value,
                                &new_sum,
                                &new_compensation)) {
        neural_error_set(error, "loss accumulation requires finite values");
        return 0;
    }
    *sum = new_sum;
    *compensation = new_compensation;
    return 1;
}

static int evaluate_dataset_loss(const NeuralModel *model,
                                 NeuralWorkspace *workspace,
                                 neural_real *predicted,
                                 const NeuralDataset *dataset,
                                 NeuralLoss loss,
                                 neural_real *value,
                                 NeuralError *error)
{
    neural_real sum = 0.0;
    neural_real compensation = 0.0;
    size_t sample_index;

    for (sample_index = 0U;
         sample_index < dataset->sample_count;
         sample_index++) {
        const neural_real *inputs =
            dataset->inputs + sample_index * dataset->input_count;
        const neural_real *expected =
            dataset->outputs + sample_index * dataset->output_count;
        neural_real sample_loss;

        if (!neural_model_forward(model,
                                  workspace,
                                  inputs,
                                  dataset->input_count,
                                  predicted,
                                  dataset->output_count,
                                  error) ||
            !neural_loss_evaluate(loss,
                                  predicted,
                                  expected,
                                  dataset->output_count,
                                  &sample_loss,
                                  error) ||
            !compensated_add_scalar(&sum,
                                    &compensation,
                                    sample_loss,
                                    error)) {
            return 0;
        }
    }
    *value = (sum + compensation) /
             (neural_real)dataset->sample_count;
    if (!isfinite(*value)) {
        neural_error_set(error, "dataset mean loss is not finite");
        return 0;
    }
    return 1;
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
    for (epoch_index = 0U; epoch_index < training->epochs;) {
        const NeuralGradient *gradient;
        NeuralEpochReport report;

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
            !evaluate_dataset_loss(model,
                                   evaluation_workspace,
                                   predicted,
                                   dataset,
                                   training->loss,
                                   &report.loss,
                                   error)) {
            goto cleanup;
        }
        report.completed_epochs = epoch_index + 1U;
        if (observer != NULL &&
            !observer(&report, observer_context, error)) {
            if (error != NULL && error->message[0] == '\0') {
                neural_error_set(error,
                                 "epoch observer rejected epoch %zu",
                                 report.completed_epochs);
            }
            goto cleanup;
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
