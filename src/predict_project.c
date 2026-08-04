#define _POSIX_C_SOURCE 200809L

#include "predict_project.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/defaults.h"
#include "neural/digest.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "path.h"
#include "project_lock.h"

typedef struct {
    const NeuralPredictionSnapshot *snapshot;
    const neural_real *inputs;
    neural_real *outputs;
    size_t sample_count;
    size_t worker_index;
    size_t worker_count;
    size_t failed_sample;
    NeuralWorkspace *workspace;
    NeuralError error;
    int success;
} PredictionWorker;

void neural_prediction_snapshot_free(NeuralPredictionSnapshot *snapshot)
{
    if (snapshot != NULL) {
        neural_model_free(snapshot->model);
        snapshot->model = NULL;
        snapshot->input_count = 0U;
        snapshot->output_count = 0U;
        snapshot->completed_epochs = 0U;
    }
}

int neural_project_prediction_load(const char *directory,
                                   NeuralPredictionSnapshot *snapshot,
                                   NeuralError *error)
{
    NeuralPredictionSnapshot loaded =
        NEURAL_PREDICTION_SNAPSHOT_INITIALIZER;
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralWeightsMetadata metadata;
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    char *weights_path = NULL;
    int project_loaded = 0;
    int success = 0;

    neural_error_clear(error);
    if (directory == NULL || directory[0] == '\0' || snapshot == NULL) {
        neural_error_set(error, "prediction project and snapshot are required");
        return 0;
    }
    *snapshot = loaded;
    if (!neural_project_lock_acquire(directory,
                                     NEURAL_PROJECT_LOCK_SHARED,
                                     &project_lock,
                                     error)) {
        goto cleanup;
    }
    weights_path = neural_path_join(directory,
                                    NEURAL_DEFAULT_WEIGHTS_FILENAME,
                                    error);
    if (weights_path == NULL ||
        !neural_project_load(directory, &project, error)) {
        goto cleanup;
    }
    project_loaded = 1;
    if (!neural_project_digests_compute(&project, &digests, error) ||
        !neural_model_create(&project.model,
                             project.training.seed,
                             &loaded.model,
                             error) ||
        !neural_weights_load(weights_path,
                             loaded.model,
                             &digests,
                             &metadata,
                             error)) {
        goto cleanup;
    }
    if (metadata.completed_epochs < project.training.epochs) {
        neural_error_set(error,
                         "final weights epochs %zu precede configured epochs %zu",
                         metadata.completed_epochs,
                         project.training.epochs);
        goto cleanup;
    }
    loaded.input_count = neural_model_input_count(loaded.model);
    loaded.output_count = neural_model_output_count(loaded.model);
    loaded.completed_epochs = metadata.completed_epochs;
    *snapshot = loaded;
    loaded.model = NULL;
    success = 1;

cleanup:
    neural_model_free(loaded.model);
    if (project_loaded) {
        neural_project_free(&project);
    }
    free(weights_path);
    neural_project_lock_release(&project_lock);
    return success;
}

static void *prediction_worker_run(void *context)
{
    PredictionWorker *worker = context;
    size_t sample_index = worker->worker_index;

    worker->success = 1;
    worker->failed_sample = SIZE_MAX;
    while (sample_index < worker->sample_count) {
        const neural_real *sample_inputs =
            worker->inputs + sample_index * worker->snapshot->input_count;
        neural_real *sample_outputs =
            worker->outputs + sample_index * worker->snapshot->output_count;

        if (!neural_model_forward(worker->snapshot->model,
                                  worker->workspace,
                                  sample_inputs,
                                  worker->snapshot->input_count,
                                  sample_outputs,
                                  worker->snapshot->output_count,
                                  &worker->error)) {
            worker->success = 0;
            worker->failed_sample = sample_index;
            break;
        }
        if (worker->sample_count - sample_index <= worker->worker_count) {
            break;
        }
        sample_index += worker->worker_count;
    }
    return NULL;
}

int neural_prediction_run(const NeuralPredictionSnapshot *snapshot,
                          const neural_real *inputs,
                          size_t sample_count,
                          const NeuralExecutionConfig *execution,
                          neural_real *outputs,
                          size_t *worker_count,
                          NeuralError *error)
{
    NeuralExecutionPlan plan = {0U, 0U, 0U, 0U};
    PredictionWorker *workers = NULL;
    pthread_t *threads = NULL;
    size_t created_threads = 0U;
    size_t worker_index;
    size_t failed_worker = SIZE_MAX;
    int join_failed = 0;
    int success = 0;

    neural_error_clear(error);
    if (worker_count != NULL) {
        *worker_count = 0U;
    }
    if (snapshot == NULL || snapshot->model == NULL ||
        snapshot->input_count == 0U || snapshot->output_count == 0U ||
        inputs == NULL || sample_count == 0U || execution == NULL ||
        outputs == NULL || worker_count == NULL) {
        neural_error_set(error, "prediction inputs and outputs are required");
        return 0;
    }
    if (sample_count > SIZE_MAX / snapshot->input_count ||
        sample_count > SIZE_MAX / snapshot->output_count) {
        neural_error_set(error, "prediction sample dimensions are too large");
        return 0;
    }
    if (!neural_execution_plan_create(0U,
                                      sample_count,
                                      execution,
                                      &plan,
                                      error)) {
        return 0;
    }
    if (plan.worker_count > SIZE_MAX / sizeof(*workers) ||
        plan.worker_count > SIZE_MAX / sizeof(*threads)) {
        neural_error_set(error, "prediction worker count is too large");
        return 0;
    }
    workers = calloc(plan.worker_count, sizeof(*workers));
    threads = calloc(plan.worker_count, sizeof(*threads));
    if (workers == NULL || threads == NULL) {
        neural_error_set(error, "unable to allocate prediction workers");
        goto cleanup;
    }
    for (worker_index = 0U;
         worker_index < plan.worker_count;
         worker_index++) {
        workers[worker_index].snapshot = snapshot;
        workers[worker_index].inputs = inputs;
        workers[worker_index].outputs = outputs;
        workers[worker_index].sample_count = sample_count;
        workers[worker_index].worker_index = worker_index;
        workers[worker_index].worker_count = plan.worker_count;
        if (!neural_workspace_create(snapshot->model,
                                     &workers[worker_index].workspace,
                                     error)) {
            goto cleanup;
        }
    }
    for (worker_index = 0U;
         worker_index < plan.worker_count;
         worker_index++) {
        int create_status = pthread_create(&threads[worker_index],
                                           NULL,
                                           prediction_worker_run,
                                           &workers[worker_index]);

        if (create_status != 0) {
            neural_error_set(error,
                             "unable to create prediction worker: %s",
                             strerror(create_status));
            goto cleanup;
        }
        created_threads++;
    }
    for (worker_index = 0U; worker_index < created_threads; worker_index++) {
        int join_status = pthread_join(threads[worker_index], NULL);

        if (join_status != 0) {
            if (!join_failed) {
                neural_error_set(error,
                                 "unable to join prediction worker: %s",
                                 strerror(join_status));
            }
            join_failed = 1;
        }
    }
    created_threads = 0U;
    if (join_failed) {
        goto cleanup;
    }
    for (worker_index = 0U;
         worker_index < plan.worker_count;
         worker_index++) {
        if (!workers[worker_index].success &&
            (failed_worker == SIZE_MAX ||
             workers[worker_index].failed_sample <
                 workers[failed_worker].failed_sample)) {
            failed_worker = worker_index;
        }
    }
    if (failed_worker != SIZE_MAX) {
        neural_error_set(error,
                         "prediction sample %zu failed: %s",
                         workers[failed_worker].failed_sample,
                         workers[failed_worker].error.message);
        goto cleanup;
    }
    *worker_count = plan.worker_count;
    success = 1;

cleanup:
    while (created_threads > 0U) {
        created_threads--;
        (void)pthread_join(threads[created_threads], NULL);
    }
    if (workers != NULL) {
        for (worker_index = 0U;
             worker_index < plan.worker_count;
             worker_index++) {
            neural_workspace_free(workers[worker_index].workspace);
        }
    }
    free(threads);
    free(workers);
    return success;
}
