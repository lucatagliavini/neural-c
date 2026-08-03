#include "neural/parallel.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/backprop.h"
#include "neural/batch.h"

typedef struct {
    struct NeuralParallelExecutor *executor;
    NeuralWorkerContext *context;
    pthread_t thread;
    size_t sample_index;
    neural_real loss;
    NeuralError error;
    int assigned;
    int success;
    int started;
} NeuralExecutorWorker;

struct NeuralParallelExecutor {
    const NeuralModel *model;
    const NeuralDataset *dataset;
    NeuralLoss loss;
    NeuralExecutorWorker *workers;
    size_t worker_count;
    size_t pending_worker_count;
    NeuralBatchAccumulator *accumulator;
    pthread_mutex_t mutex;
    pthread_cond_t work_available;
    pthread_cond_t work_complete;
    int mutex_initialized;
    int work_available_initialized;
    int work_complete_initialized;
    int stopping;
    int broken;
};

static int executor_dataset_validate(const NeuralModel *model,
                                     const NeuralDataset *dataset,
                                     NeuralLoss loss,
                                     NeuralError *error)
{
    size_t input_value_count;
    size_t output_value_count;
    size_t index;

    if (model == NULL || dataset == NULL || dataset->sample_count == 0U ||
        dataset->inputs == NULL || dataset->outputs == NULL ||
        dataset->input_count != neural_model_input_count(model) ||
        dataset->output_count != neural_model_output_count(model) ||
        loss != NEURAL_LOSS_MSE ||
        dataset->input_count > SIZE_MAX / dataset->sample_count ||
        dataset->output_count > SIZE_MAX / dataset->sample_count) {
        neural_error_set(error,
                         "executor dataset must match a non-empty model");
        return 0;
    }
    input_value_count = dataset->sample_count * dataset->input_count;
    output_value_count = dataset->sample_count * dataset->output_count;
    for (index = 0U; index < input_value_count; index++) {
        if (!isfinite(dataset->inputs[index])) {
            neural_error_set(error, "executor inputs must be finite");
            return 0;
        }
    }
    for (index = 0U; index < output_value_count; index++) {
        if (!isfinite(dataset->outputs[index])) {
            neural_error_set(error, "executor outputs must be finite");
            return 0;
        }
    }
    return 1;
}

static void *executor_worker_run(void *opaque)
{
    NeuralExecutorWorker *worker = opaque;
    NeuralParallelExecutor *executor = worker->executor;

    (void)pthread_mutex_lock(&executor->mutex);
    for (;;) {
        size_t sample_index;
        const neural_real *inputs;
        const neural_real *expected;

        while (!executor->stopping && !worker->assigned) {
            (void)pthread_cond_wait(&executor->work_available,
                                    &executor->mutex);
        }
        if (executor->stopping) {
            break;
        }
        sample_index = worker->sample_index;
        worker->assigned = 0;
        (void)pthread_mutex_unlock(&executor->mutex);

        inputs = executor->dataset->inputs +
                 sample_index * executor->dataset->input_count;
        expected = executor->dataset->outputs +
                   sample_index * executor->dataset->output_count;
        neural_error_clear(&worker->error);
        worker->success = neural_model_sample_gradient(
            executor->model,
            neural_worker_context_workspace(worker->context),
            neural_worker_context_gradient(worker->context),
            executor->loss,
            inputs,
            executor->dataset->input_count,
            expected,
            executor->dataset->output_count,
            &worker->loss,
            &worker->error);

        (void)pthread_mutex_lock(&executor->mutex);
        executor->pending_worker_count--;
        if (executor->pending_worker_count == 0U) {
            (void)pthread_cond_signal(&executor->work_complete);
        }
    }
    (void)pthread_mutex_unlock(&executor->mutex);
    return NULL;
}

static void executor_stop_workers(NeuralParallelExecutor *executor)
{
    size_t worker_index;

    if (executor->mutex_initialized) {
        (void)pthread_mutex_lock(&executor->mutex);
        executor->stopping = 1;
        if (executor->work_available_initialized) {
            (void)pthread_cond_broadcast(&executor->work_available);
        }
        (void)pthread_mutex_unlock(&executor->mutex);
    }
    for (worker_index = 0U;
         worker_index < executor->worker_count;
         worker_index++) {
        if (executor->workers != NULL &&
            executor->workers[worker_index].started) {
            (void)pthread_join(executor->workers[worker_index].thread, NULL);
            executor->workers[worker_index].started = 0;
        }
    }
}

void neural_parallel_executor_free(NeuralParallelExecutor *executor)
{
    size_t worker_index;

    if (executor == NULL) {
        return;
    }
    executor_stop_workers(executor);
    if (executor->workers != NULL) {
        for (worker_index = 0U;
             worker_index < executor->worker_count;
             worker_index++) {
            neural_worker_context_free(
                executor->workers[worker_index].context);
        }
    }
    neural_batch_accumulator_free(executor->accumulator);
    if (executor->work_complete_initialized) {
        (void)pthread_cond_destroy(&executor->work_complete);
    }
    if (executor->work_available_initialized) {
        (void)pthread_cond_destroy(&executor->work_available);
    }
    if (executor->mutex_initialized) {
        (void)pthread_mutex_destroy(&executor->mutex);
    }
    free(executor->workers);
    free(executor);
}

int neural_parallel_executor_create(
    const NeuralModel *model,
    const NeuralDataset *dataset,
    NeuralLoss loss,
    const NeuralExecutionConfig *config,
    NeuralParallelExecutor **executor,
    NeuralError *error)
{
    NeuralParallelExecutor *created;
    size_t worker_index;
    int code;

    if (executor == NULL) {
        neural_error_set(error, "parallel executor output is required");
        return 0;
    }
    *executor = NULL;
    if (!executor_dataset_validate(model, dataset, loss, error) ||
        !neural_execution_config_validate(config, error)) {
        return 0;
    }
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate parallel executor");
        return 0;
    }
    created->model = model;
    created->dataset = dataset;
    created->loss = loss;
    created->worker_count = config->thread_count < dataset->sample_count
                                ? config->thread_count
                                : dataset->sample_count;
    if (created->worker_count > SIZE_MAX / sizeof(*created->workers)) {
        neural_error_set(error, "parallel worker count overflows memory size");
        neural_parallel_executor_free(created);
        return 0;
    }
    created->workers = calloc(created->worker_count,
                              sizeof(*created->workers));
    if (created->workers == NULL) {
        neural_error_set(error, "unable to allocate parallel workers");
        neural_parallel_executor_free(created);
        return 0;
    }
    code = pthread_mutex_init(&created->mutex, NULL);
    if (code != 0) {
        neural_error_set(error, "unable to initialize worker mutex: %s",
                         strerror(code));
        neural_parallel_executor_free(created);
        return 0;
    }
    created->mutex_initialized = 1;
    code = pthread_cond_init(&created->work_available, NULL);
    if (code != 0) {
        neural_error_set(error, "unable to initialize work condition: %s",
                         strerror(code));
        neural_parallel_executor_free(created);
        return 0;
    }
    created->work_available_initialized = 1;
    code = pthread_cond_init(&created->work_complete, NULL);
    if (code != 0) {
        neural_error_set(error, "unable to initialize completion condition: %s",
                         strerror(code));
        neural_parallel_executor_free(created);
        return 0;
    }
    created->work_complete_initialized = 1;
    if (!neural_batch_accumulator_create(model,
                                         &created->accumulator,
                                         error)) {
        neural_parallel_executor_free(created);
        return 0;
    }
    for (worker_index = 0U;
         worker_index < created->worker_count;
         worker_index++) {
        NeuralExecutorWorker *worker = &created->workers[worker_index];

        worker->executor = created;
        if (!neural_worker_context_create(model, &worker->context, error)) {
            neural_parallel_executor_free(created);
            return 0;
        }
    }
    for (worker_index = 0U;
         worker_index < created->worker_count;
         worker_index++) {
        NeuralExecutorWorker *worker = &created->workers[worker_index];

        code = pthread_create(&worker->thread,
                              NULL,
                              executor_worker_run,
                              worker);
        if (code != 0) {
            neural_error_set(error, "unable to create worker %zu: %s",
                             worker_index, strerror(code));
            neural_parallel_executor_free(created);
            return 0;
        }
        worker->started = 1;
    }
    *executor = created;
    return 1;
}

size_t neural_parallel_executor_worker_count(
    const NeuralParallelExecutor *executor)
{
    return executor == NULL ? 0U : executor->worker_count;
}

static int executor_dispatch_wave(NeuralParallelExecutor *executor,
                                  size_t sample_begin,
                                  size_t sample_end,
                                  NeuralError *error)
{
    size_t wave_sample_count = sample_end - sample_begin;
    size_t worker_index;
    int code;

    code = pthread_mutex_lock(&executor->mutex);
    if (code != 0) {
        executor->broken = 1;
        neural_error_set(error, "unable to lock worker pool: %s",
                         strerror(code));
        return 0;
    }
    executor->pending_worker_count = wave_sample_count;
    for (worker_index = 0U;
         worker_index < wave_sample_count;
         worker_index++) {
        NeuralExecutorWorker *worker = &executor->workers[worker_index];

        worker->sample_index = sample_begin + worker_index;
        worker->assigned = 1;
        worker->success = 0;
    }
    code = pthread_cond_broadcast(&executor->work_available);
    if (code == 0) {
        while (executor->pending_worker_count != 0U) {
            code = pthread_cond_wait(&executor->work_complete,
                                     &executor->mutex);
            if (code != 0) {
                break;
            }
        }
    }
    (void)pthread_mutex_unlock(&executor->mutex);
    if (code != 0) {
        executor->broken = 1;
        neural_error_set(error, "worker-pool synchronization failed: %s",
                         strerror(code));
        return 0;
    }
    for (worker_index = 0U;
         worker_index < wave_sample_count;
         worker_index++) {
        const NeuralExecutorWorker *worker =
            &executor->workers[worker_index];

        if (!worker->success) {
            neural_error_set(error,
                             "sample %zu failed: %s",
                             worker->sample_index,
                             worker->error.message[0] == '\0'
                                 ? "worker reported no detail"
                                 : worker->error.message);
            return 0;
        }
    }
    for (worker_index = 0U;
         worker_index < wave_sample_count;
         worker_index++) {
        NeuralExecutorWorker *worker = &executor->workers[worker_index];

        if (!neural_batch_accumulator_add(
                executor->accumulator,
                worker->sample_index,
                neural_worker_context_gradient(worker->context),
                error)) {
            return 0;
        }
    }
    return 1;
}

int neural_parallel_executor_batch_gradient(
    NeuralParallelExecutor *executor,
    size_t sample_begin,
    size_t sample_end,
    const NeuralGradient **gradient,
    NeuralError *error)
{
    NeuralExecutionConfig config;
    NeuralExecutionPlan plan;
    size_t wave_index;

    if (gradient == NULL) {
        neural_error_set(error, "batch gradient output is required");
        return 0;
    }
    *gradient = NULL;
    if (executor == NULL || sample_begin >= sample_end ||
        sample_end > executor->dataset->sample_count) {
        neural_error_set(error, "batch range is outside the executor dataset");
        return 0;
    }
    if (executor->broken) {
        neural_error_set(error, "parallel executor is no longer usable");
        return 0;
    }
    config.thread_count = executor->worker_count;
    if (!neural_execution_plan_create(sample_begin,
                                      sample_end,
                                      &config,
                                      &plan,
                                      error) ||
        !neural_batch_accumulator_reset(executor->accumulator,
                                        sample_begin,
                                        sample_end,
                                        error)) {
        return 0;
    }
    for (wave_index = 0U; wave_index < plan.wave_count; wave_index++) {
        size_t wave_begin;
        size_t wave_end;

        if (!neural_execution_plan_wave_range(&plan,
                                              wave_index,
                                              &wave_begin,
                                              &wave_end,
                                              error) ||
            !executor_dispatch_wave(executor,
                                    wave_begin,
                                    wave_end,
                                    error)) {
            return 0;
        }
    }
    if (!neural_batch_accumulator_finalize(executor->accumulator, error)) {
        return 0;
    }
    *gradient = neural_batch_accumulator_gradient(executor->accumulator);
    return *gradient != NULL;
}
