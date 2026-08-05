#include "neural/training.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "neural/batch.h"
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
    NeuralError *error)
{
    NeuralParallelExecutor *executor = NULL;
    NeuralSampleOrder *sample_order = NULL;
    NeuralGradient *clipped_gradient = NULL;
    NeuralWorkspace *evaluation_workspace = NULL;
    neural_real *predicted = NULL;
    NeuralTrainingResult completed = {0U, 0U, 0.0, 0.0, 0U};
    NeuralBatchPlan batch_plan;
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
        !neural_sample_order_create(dataset->sample_count,
                                    &sample_order,
                                    error) ||
        !neural_workspace_create(model, &evaluation_workspace, error)) {
        goto cleanup;
    }
    if (!neural_batch_plan_create(
            dataset->sample_count,
            training->batch_size == 0U ||
                    training->batch_size > dataset->sample_count
                ? dataset->sample_count
                : training->batch_size,
            &batch_plan,
            error)) {
        goto cleanup;
    }
    if (training->gradient_clip_norm > 0.0 &&
        !neural_gradient_create(model, &clipped_gradient, error)) {
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
        NeuralEpochReport report = {0};
        size_t batch_index;

        if (!neural_sample_order_prepare(sample_order,
                                         training->seed,
                                         (uint64_t)epoch_index,
                                         training->shuffle,
                                         error)) {
            goto cleanup;
        }
        for (batch_index = 0U;
             batch_index < batch_plan.batch_count;
             batch_index++) {
            const NeuralGradient *gradient;
            const NeuralGradient *update_gradient;
            neural_real gradient_norm;
            int clipped = 0;
            size_t sample_begin;
            size_t sample_end;

            if (!neural_batch_plan_range(&batch_plan,
                                         batch_index,
                                         &sample_begin,
                                         &sample_end,
                                         error) ||
                !neural_parallel_executor_ordered_batch_gradient(
                    executor,
                    sample_order,
                    sample_begin,
                    sample_end,
                    &gradient,
                    error)) {
                goto cleanup;
            }
            update_gradient = gradient;
            if (training->gradient_clip_norm > 0.0) {
                if (!neural_gradient_copy(clipped_gradient,
                                          gradient,
                                          error) ||
                    !neural_gradient_clip_norm(
                        clipped_gradient,
                        training->gradient_clip_norm,
                        &gradient_norm,
                        &clipped,
                        error)) {
                    goto cleanup;
                }
                update_gradient = clipped_gradient;
            } else if (!neural_gradient_norm(gradient,
                                             &gradient_norm,
                                             error)) {
                goto cleanup;
            }
            if (gradient_norm > report.max_gradient_norm) {
                report.max_gradient_norm = gradient_norm;
            }
            if (clipped) {
                report.clipped_batch_count++;
            }
            if (!neural_model_apply_gradient(model,
                                             update_gradient,
                                             training->learning_rate,
                                             error)) {
                goto cleanup;
            }
        }
        if (!neural_model_evaluate_dataset_loss(model,
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
                completed.final_max_gradient_norm =
                    report.max_gradient_norm;
                completed.clipped_batch_count =
                    report.clipped_batch_count;
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
        completed.final_max_gradient_norm = report.max_gradient_norm;
        completed.clipped_batch_count = report.clipped_batch_count;
        epoch_index = report.completed_epochs;
    }
    *result = completed;
    success = 1;

cleanup:
    free(predicted);
    neural_workspace_free(evaluation_workspace);
    neural_gradient_free(clipped_gradient);
    neural_sample_order_free(sample_order);
    neural_parallel_executor_free(executor);
    return success;
}

int neural_model_train(
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

    return neural_model_train_range(model,
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
