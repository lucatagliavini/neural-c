#define _POSIX_C_SOURCE 200809L

#include "train_project.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "neural/defaults.h"
#include "neural/digest.h"
#include "neural/model.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "atomic_file.h"
#include "path.h"
#include "project_checkpoint.h"
#include "project_lock.h"

static int persistence_path_is_absent(const char *path, NeuralError *error)
{
    struct stat status;

    if (lstat(path, &status) == 0) {
        neural_error_set(error,
                         "%s already exists; fresh training requires no saved state",
                         path);
        return 0;
    }
    if (errno != ENOENT) {
        neural_error_set(error,
                         "%s: unable to inspect saved state: %s",
                         path,
                         strerror(errno));
        return 0;
    }
    return 1;
}

static int persistence_path_exists(const char *path,
                                   int *exists,
                                   NeuralError *error)
{
    struct stat status;

    if (lstat(path, &status) == 0) {
        *exists = 1;
        return 1;
    }
    if (errno == ENOENT) {
        *exists = 0;
        return 1;
    }
    neural_error_set(error,
                     "%s: unable to inspect saved state: %s",
                     path,
                     strerror(errno));
    return 0;
}

static int models_have_equal_parameters(const NeuralModel *left,
                                        const NeuralModel *right)
{
    size_t layer_index;

    if (neural_model_layer_count(left) != neural_model_layer_count(right)) {
        return 0;
    }
    for (layer_index = 0U;
         layer_index < neural_model_layer_count(left);
         layer_index++) {
        size_t left_weight_count;
        size_t right_weight_count;
        size_t left_bias_count;
        size_t right_bias_count;
        const neural_real *left_weights = neural_model_layer_weights(
            left, layer_index, &left_weight_count);
        const neural_real *right_weights = neural_model_layer_weights(
            right, layer_index, &right_weight_count);
        const neural_real *left_biases = neural_model_layer_biases(
            left, layer_index, &left_bias_count);
        const neural_real *right_biases = neural_model_layer_biases(
            right, layer_index, &right_bias_count);

        if (left_weights == NULL || right_weights == NULL ||
            left_biases == NULL || right_biases == NULL ||
            left_weight_count != right_weight_count ||
            left_bias_count != right_bias_count ||
            memcmp(left_weights,
                   right_weights,
                   left_weight_count * sizeof(*left_weights)) != 0 ||
            memcmp(left_biases,
                   right_biases,
                   left_bias_count * sizeof(*left_biases)) != 0) {
            return 0;
        }
    }
    return 1;
}

int neural_project_train_fresh_controlled(
    const char *directory,
    const NeuralExecutionConfig *execution,
    const volatile sig_atomic_t *stop_request,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralProject project;
    NeuralModel *model = NULL;
    NeuralProjectDigests digests;
    NeuralWeightsMetadata metadata;
    NeuralProjectCheckpointObserver checkpoint_observer = {0};
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralTrainingResult completed = {0U, 0U, 0.0};
    char *weights_path = NULL;
    char *checkpoint_path = NULL;
    int project_loaded = 0;
    int success = 0;

    neural_error_clear(error);
    if (directory == NULL || directory[0] == '\0' || execution == NULL ||
        result == NULL) {
        neural_error_set(error, "fresh project training arguments are required");
        return 0;
    }
    *result = completed;
    if (interrupted_signal != NULL) {
        *interrupted_signal = 0;
    }
    weights_path = neural_path_join(directory,
                                    NEURAL_DEFAULT_WEIGHTS_FILENAME,
                                    error);
    checkpoint_path = neural_path_join(directory,
                                       NEURAL_DEFAULT_CHECKPOINT_FILENAME,
                                       error);
    if (weights_path == NULL || checkpoint_path == NULL ||
        !neural_project_lock_acquire(directory,
                                     NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                     &project_lock,
                                     error) ||
        !persistence_path_is_absent(weights_path, error) ||
        !persistence_path_is_absent(checkpoint_path, error) ||
        !neural_project_load(directory, &project, error)) {
        goto cleanup;
    }
    project_loaded = 1;
    if (!neural_project_digests_compute(&project, &digests, error) ||
        !neural_model_create(&project.model,
                             project.training.seed,
                             &model,
                             error) ||
        !neural_project_checkpoint_observer_initialize(
            &checkpoint_observer,
            checkpoint_path,
            model,
            &digests,
            project.training.checkpoint_interval,
            project.training.epochs,
            error)) {
        goto cleanup;
    }
    neural_project_checkpoint_observer_set_stop_request(&checkpoint_observer,
                                                        stop_request);
    if (!neural_model_train_full_batch(model,
                                       &project.dataset,
                                       &project.training,
                                       execution,
                                       neural_project_checkpoint_observe,
                                       &checkpoint_observer,
                                       &completed,
                                       error)) {
        goto cleanup;
    }
    if (!persistence_path_is_absent(weights_path, error)) {
        goto cleanup;
    }
    metadata.completed_epochs = completed.completed_epochs;
    metadata.digests = digests;
    if (!neural_weights_save_atomic(weights_path,
                                    model,
                                    &metadata,
                                    error) ||
        !neural_atomic_file_remove(checkpoint_path, 1, error)) {
        goto cleanup;
    }
    *result = completed;
    success = 1;

cleanup:
    if (interrupted_signal != NULL) {
        *interrupted_signal = checkpoint_observer.interrupted_signal;
    }
    neural_project_lock_release(&project_lock);
    neural_model_free(model);
    if (project_loaded) {
        neural_project_free(&project);
    }
    free(checkpoint_path);
    free(weights_path);
    return success;
}

int neural_project_train_fresh(const char *directory,
                               const NeuralExecutionConfig *execution,
                               NeuralTrainingResult *result,
                               NeuralError *error)
{
    return neural_project_train_fresh_controlled(directory,
                                                 execution,
                                                 NULL,
                                                 NULL,
                                                 result,
                                                 error);
}

int neural_project_train_resume_controlled(
    const char *directory,
    const NeuralExecutionConfig *execution,
    const volatile sig_atomic_t *stop_request,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralProject project;
    NeuralProjectDigests digests;
    NeuralCheckpointMetadata checkpoint_metadata;
    NeuralWeightsMetadata weights_metadata;
    NeuralWeightsMetadata final_metadata;
    NeuralProjectCheckpointObserver checkpoint_observer = {0};
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralTrainingResult completed = {0U, 0U, 0.0};
    NeuralModel *checkpoint_model = NULL;
    NeuralModel *weights_model = NULL;
    char *weights_path = NULL;
    char *checkpoint_path = NULL;
    int checkpoint_exists = 0;
    int weights_exists = 0;
    int project_loaded = 0;
    int success = 0;

    neural_error_clear(error);
    if (directory == NULL || directory[0] == '\0' || execution == NULL ||
        result == NULL) {
        neural_error_set(error, "resume project training arguments are required");
        return 0;
    }
    *result = completed;
    if (interrupted_signal != NULL) {
        *interrupted_signal = 0;
    }
    weights_path = neural_path_join(directory,
                                    NEURAL_DEFAULT_WEIGHTS_FILENAME,
                                    error);
    checkpoint_path = neural_path_join(directory,
                                       NEURAL_DEFAULT_CHECKPOINT_FILENAME,
                                       error);
    if (weights_path == NULL || checkpoint_path == NULL ||
        !neural_project_lock_acquire(directory,
                                     NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                     &project_lock,
                                     error) ||
        !persistence_path_exists(checkpoint_path,
                                 &checkpoint_exists,
                                 error) ||
        !persistence_path_exists(weights_path, &weights_exists, error)) {
        goto cleanup;
    }
    if (!checkpoint_exists) {
        neural_error_set(error,
                         weights_exists
                             ? "resume requires checkpoint.txt; project is already finalized"
                             : "resume requires checkpoint.txt");
        goto cleanup;
    }
    if (!neural_project_load(directory, &project, error)) {
        goto cleanup;
    }
    project_loaded = 1;
    if (!neural_project_digests_compute(&project, &digests, error) ||
        !neural_model_create(&project.model,
                             project.training.seed,
                             &checkpoint_model,
                             error) ||
        !neural_checkpoint_load(checkpoint_path,
                                checkpoint_model,
                                &digests,
                                &checkpoint_metadata,
                                error)) {
        goto cleanup;
    }
    if (checkpoint_metadata.target_epochs != project.training.epochs) {
        neural_error_set(error,
                         "checkpoint target %zu does not match configured epochs %zu",
                         checkpoint_metadata.target_epochs,
                         project.training.epochs);
        goto cleanup;
    }
    if (weights_exists) {
        if (!neural_model_create(&project.model,
                                 project.training.seed,
                                 &weights_model,
                                 error) ||
            !neural_weights_load(weights_path,
                                 weights_model,
                                 &digests,
                                 &weights_metadata,
                                 error)) {
            goto cleanup;
        }
        if (weights_metadata.completed_epochs !=
            checkpoint_metadata.target_epochs) {
            neural_error_set(
                error,
                "final weights epochs %zu do not match checkpoint target %zu",
                weights_metadata.completed_epochs,
                checkpoint_metadata.target_epochs);
            goto cleanup;
        }
        if (checkpoint_metadata.completed_epochs ==
                checkpoint_metadata.target_epochs &&
            !models_have_equal_parameters(checkpoint_model, weights_model)) {
            neural_error_set(
                error,
                "completed checkpoint parameters do not match final weights");
            goto cleanup;
        }
        if (!neural_model_train_full_batch_range(
                weights_model,
                &project.dataset,
                &project.training,
                execution,
                weights_metadata.completed_epochs,
                weights_metadata.completed_epochs,
                NULL,
                NULL,
                &completed,
                error) ||
            !neural_atomic_file_remove(checkpoint_path, 0, error)) {
            goto cleanup;
        }
        *result = completed;
        success = 1;
        goto cleanup;
    }
    if (!neural_project_checkpoint_observer_initialize(
            &checkpoint_observer,
            checkpoint_path,
            checkpoint_model,
            &digests,
            project.training.checkpoint_interval,
            checkpoint_metadata.target_epochs,
            error)) {
        goto cleanup;
    }
    neural_project_checkpoint_observer_set_stop_request(&checkpoint_observer,
                                                        stop_request);
    if (!neural_model_train_full_batch_range(
            checkpoint_model,
            &project.dataset,
            &project.training,
            execution,
            checkpoint_metadata.completed_epochs,
            checkpoint_metadata.target_epochs,
            neural_project_checkpoint_observe,
            &checkpoint_observer,
            &completed,
            error) ||
        !persistence_path_is_absent(weights_path, error)) {
        goto cleanup;
    }
    final_metadata.completed_epochs = completed.completed_epochs;
    final_metadata.digests = digests;
    if (!neural_weights_save_atomic(weights_path,
                                    checkpoint_model,
                                    &final_metadata,
                                    error) ||
        !neural_atomic_file_remove(checkpoint_path, 0, error)) {
        goto cleanup;
    }
    *result = completed;
    success = 1;

cleanup:
    if (interrupted_signal != NULL) {
        *interrupted_signal = checkpoint_observer.interrupted_signal;
    }
    neural_project_lock_release(&project_lock);
    neural_model_free(weights_model);
    neural_model_free(checkpoint_model);
    if (project_loaded) {
        neural_project_free(&project);
    }
    free(checkpoint_path);
    free(weights_path);
    return success;
}

int neural_project_train_resume(const char *directory,
                                const NeuralExecutionConfig *execution,
                                NeuralTrainingResult *result,
                                NeuralError *error)
{
    return neural_project_train_resume_controlled(directory,
                                                  execution,
                                                  NULL,
                                                  NULL,
                                                  result,
                                                  error);
}
