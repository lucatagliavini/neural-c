#define _POSIX_C_SOURCE 200809L

#include "train_project.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "neural/defaults.h"
#include "neural/digest.h"
#include "neural/evaluation.h"
#include "neural/gradient.h"
#include "neural/model.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "atomic_file.h"
#include "path.h"
#include "project_checkpoint.h"
#include "project_lock.h"

static int create_training_optimizer(
    const NeuralModel *model,
    const NeuralTrainingConfig *training,
    NeuralOptimizer **optimizer,
    NeuralError *error)
{
    NeuralOptimizerOptions options;

    options.kind = training->optimizer;
    options.momentum = training->momentum;
    options.adam_beta1 = training->adam_beta1;
    options.adam_beta2 = training->adam_beta2;
    options.adam_epsilon = training->adam_epsilon;
    options.learning_rate = training->learning_rate;
    options.schedule = training->learning_rate_schedule;
    options.schedule_decay = training->learning_rate_decay;
    options.schedule_step_epochs = training->learning_rate_step_epochs;
    options.schedule_plateau_patience =
        training->learning_rate_plateau_patience;
    options.schedule_plateau_min_delta =
        training->learning_rate_plateau_min_delta;
    options.divergence_threshold = training->divergence_threshold;
    options.target_loss = training->target_loss;
    options.no_improvement_epochs = training->max_no_improvement_epochs;
    options.no_improvement_min_delta = training->no_improvement_min_delta;
    return neural_optimizer_create(model, &options, optimizer, error);
}

static int save_completed_weights(
    const char *path,
    const NeuralModel *model,
    const NeuralProjectDigests *digests,
    size_t target_epochs,
    const NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralWeightsMetadata metadata = {0};

    metadata.completed_epochs = result->completed_epochs;
    metadata.digests = *digests;
    if (result->completion_reason == NEURAL_TRAINING_TARGET_EPOCHS) {
        return neural_weights_save_atomic(path, model, &metadata, error);
    }
    metadata.selected_epoch = result->completed_epochs;
    metadata.target_epochs = target_epochs;
    metadata.format_version = 2U;
    if (result->completion_reason == NEURAL_TRAINING_LOSS_TARGET) {
        metadata.completion_reason = NEURAL_COMPLETION_LOSS_TARGET;
    } else if (result->completion_reason ==
               NEURAL_TRAINING_NO_IMPROVEMENT) {
        metadata.completion_reason = NEURAL_COMPLETION_NO_IMPROVEMENT;
    } else {
        neural_error_set(error, "training completion reason is invalid");
        return 0;
    }
    return neural_early_weights_save_atomic(path, model, &metadata, error);
}

typedef struct {
    NeuralProjectCheckpointObserver *checkpoint;
    NeuralEpochObserver observer;
    void *observer_context;
} NeuralProjectTrainingObserver;

typedef struct {
    const char *checkpoint_path;
    NeuralModel *current_model;
    NeuralModel *best_model;
    const NeuralDataset *validation;
    NeuralLoss loss;
    NeuralWorkspace *workspace;
    neural_real *predicted;
    size_t interval;
    size_t patience;
    neural_real min_delta;
    const volatile sig_atomic_t *stop_request;
    int interrupted_signal;
    int has_best;
    NeuralCheckpointMetadata metadata;
    NeuralEpochObserver observer;
    void *observer_context;
} NeuralEarlyStoppingObserver;

static int model_objective(const NeuralModel *model,
                           const NeuralTrainingConfig *training,
                           neural_real loss,
                           neural_real *objective,
                           NeuralError *error)
{
    neural_real penalty;

    if (!neural_model_regularization_penalty(
            model,
            training->l1_regularization,
            training->l2_regularization,
            training->regularize_biases,
            &penalty,
            error) ||
        !isfinite(loss + penalty)) {
        if (error != NULL && error->message[0] == '\0') {
            neural_error_set(error, "training objective is not finite");
        }
        return 0;
    }
    *objective = loss + penalty;
    return 1;
}

static int copy_model_parameters(NeuralModel *destination,
                                 const NeuralModel *source,
                                 NeuralError *error)
{
    size_t layer_index;

    if (destination == NULL || source == NULL ||
        neural_model_layer_count(destination) !=
            neural_model_layer_count(source)) {
        neural_error_set(error, "model parameter copy is incompatible");
        return 0;
    }
    for (layer_index = 0U;
         layer_index < neural_model_layer_count(source);
         layer_index++) {
        const neural_real *weights;
        const neural_real *biases;
        size_t weight_count;
        size_t bias_count;

        weights = neural_model_layer_weights(source,
                                             layer_index,
                                             &weight_count);
        biases = neural_model_layer_biases(source,
                                           layer_index,
                                           &bias_count);
        if (weights == NULL || biases == NULL ||
            !neural_model_set_layer_parameters(destination,
                                               layer_index,
                                               weights,
                                               weight_count,
                                               biases,
                                               bias_count,
                                               error)) {
            return 0;
        }
    }
    return 1;
}

static int observe_early_stopping(const NeuralEpochReport *report,
                                  void *context,
                                  NeuralError *error)
{
    NeuralEarlyStoppingObserver *observer = context;
    NeuralEpochReport enriched;
    neural_real validation_loss;
    sig_atomic_t stop_signal;
    int improved;
    int should_checkpoint;

    if (report == NULL || observer == NULL ||
        !neural_model_evaluate_dataset_loss(observer->current_model,
                                            observer->workspace,
                                            observer->predicted,
                                            observer->validation,
                                            observer->loss,
                                            &validation_loss,
                                            error)) {
        return NEURAL_EPOCH_OBSERVER_ERROR;
    }
    improved = !observer->has_best ||
               observer->metadata.best_loss - validation_loss >
                   observer->min_delta;
    if (improved) {
        if (!copy_model_parameters(observer->best_model,
                                   observer->current_model,
                                   error)) {
            return NEURAL_EPOCH_OBSERVER_ERROR;
        }
        observer->metadata.best_loss = validation_loss;
        observer->metadata.best_epoch = report->completed_epochs;
        observer->metadata.stale_epochs = 0U;
        observer->has_best = 1;
    } else {
        observer->metadata.stale_epochs++;
    }
    enriched = *report;
    enriched.has_validation_loss = 1;
    enriched.validation_loss = validation_loss;
    enriched.best_validation_loss = observer->metadata.best_loss;
    enriched.stopped_early =
        observer->metadata.stale_epochs >= observer->patience;
    stop_signal = observer->stop_request == NULL
                      ? 0
                      : *observer->stop_request;
    should_checkpoint = stop_signal != 0 || enriched.stopped_early ||
        (observer->interval != 0U &&
         report->completed_epochs % observer->interval == 0U);
    if (should_checkpoint) {
        observer->metadata.completed_epochs = report->completed_epochs;
        observer->metadata.rng_state =
            neural_model_random_state(observer->current_model);
        if ((observer->metadata.optimizer !=
                 NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
             report->optimizer == NULL) ||
            !neural_early_checkpoint_save_atomic_with_optimizer(
                observer->checkpoint_path,
                observer->current_model,
                observer->best_model,
                report->optimizer,
                &observer->metadata,
                error)) {
            return NEURAL_EPOCH_OBSERVER_ERROR;
        }
    }
    if (observer->observer != NULL &&
        observer->observer(&enriched,
                           observer->observer_context,
                           error) != NEURAL_EPOCH_OBSERVER_CONTINUE) {
        return NEURAL_EPOCH_OBSERVER_ERROR;
    }
    if (stop_signal != 0) {
        observer->interrupted_signal = (int)stop_signal;
        neural_error_set(error,
                         "training interrupted by signal %d after epoch %zu; "
                         "checkpoint saved",
                         observer->interrupted_signal,
                         report->completed_epochs);
        return NEURAL_EPOCH_OBSERVER_ERROR;
    }
    return enriched.stopped_early
               ? NEURAL_EPOCH_OBSERVER_STOP
               : NEURAL_EPOCH_OBSERVER_CONTINUE;
}

static int initialize_early_observer(
    NeuralEarlyStoppingObserver *observer,
    const char *checkpoint_path,
    NeuralModel *current_model,
    NeuralModel *best_model,
    const NeuralProject *project,
    const NeuralProjectDigests *digests,
    size_t target_epochs,
    const volatile sig_atomic_t *stop_request,
    NeuralEpochObserver user_observer,
    void *user_context,
    NeuralError *error)
{
    size_t output_count;

    memset(observer, 0, sizeof(*observer));
    observer->checkpoint_path = checkpoint_path;
    observer->current_model = current_model;
    observer->best_model = best_model;
    observer->validation = &project->validation;
    observer->loss = project->training.loss;
    observer->interval = project->training.checkpoint_interval;
    observer->patience = project->training.early_stopping_patience;
    observer->min_delta = project->training.early_stopping_min_delta;
    observer->stop_request = stop_request;
    observer->observer = user_observer;
    observer->observer_context = user_context;
    observer->metadata.target_epochs = target_epochs;
    observer->metadata.optimizer = project->training.optimizer;
    observer->metadata.digests = *digests;
    observer->metadata.format_version = 2U;
    if (!neural_workspace_create(current_model,
                                 &observer->workspace,
                                 error)) {
        return 0;
    }
    output_count = neural_model_output_count(current_model);
    if (output_count > SIZE_MAX / sizeof(*observer->predicted)) {
        neural_error_set(error, "validation output dimensions overflow");
        neural_workspace_free(observer->workspace);
        observer->workspace = NULL;
        return 0;
    }
    observer->predicted = malloc(output_count * sizeof(*observer->predicted));
    if (observer->predicted == NULL) {
        neural_error_set(error, "unable to allocate validation output");
        neural_workspace_free(observer->workspace);
        observer->workspace = NULL;
        return 0;
    }
    return 1;
}

static void free_early_observer(NeuralEarlyStoppingObserver *observer)
{
    if (observer != NULL) {
        free(observer->predicted);
        neural_workspace_free(observer->workspace);
        observer->predicted = NULL;
        observer->workspace = NULL;
    }
}

static int train_early_stopping_range(
    const NeuralProject *project,
    const NeuralProjectDigests *digests,
    const NeuralExecutionConfig *execution,
    NeuralModel *current_model,
    NeuralModel *best_model,
    NeuralOptimizer *optimizer,
    const char *weights_path,
    const char *checkpoint_path,
    size_t completed_epochs,
    size_t target_epochs,
    size_t baseline_selected_epoch,
    const NeuralCheckpointMetadata *resumed_state,
    const volatile sig_atomic_t *stop_request,
    NeuralEpochObserver user_observer,
    void *user_context,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralEarlyStoppingObserver observer;
    NeuralWeightsMetadata weights = {0};
    NeuralTrainingResult completed = {0};
    int initialized = 0;
    int success = 0;

    if (!initialize_early_observer(&observer,
                                   checkpoint_path,
                                   current_model,
                                   best_model,
                                   project,
                                   digests,
                                   target_epochs,
                                   stop_request,
                                   user_observer,
                                   user_context,
                                   error)) {
        goto cleanup;
    }
    initialized = 1;
    if (resumed_state != NULL) {
        observer.metadata = *resumed_state;
        observer.has_best = 1;
    } else if (completed_epochs != 0U) {
        if (!copy_model_parameters(best_model, current_model, error) ||
            !neural_model_evaluate_dataset_loss(
                current_model,
                observer.workspace,
                observer.predicted,
                &project->validation,
                project->training.loss,
                &observer.metadata.best_loss,
                error)) {
            goto cleanup;
        }
        observer.metadata.best_epoch = baseline_selected_epoch;
        observer.metadata.stale_epochs = 0U;
        observer.has_best = 1;
    }
    if ((optimizer == NULL &&
         !neural_model_train_range(current_model,
                                   &project->dataset,
                                   &project->training,
                                   execution,
                                   completed_epochs,
                                   target_epochs,
                                   observe_early_stopping,
                                   &observer,
                                   &completed,
                                   error)) ||
        (optimizer != NULL &&
         !neural_model_train_range_with_optimizer(
             current_model,
             &project->dataset,
             &project->training,
             execution,
             optimizer,
             completed_epochs,
             target_epochs,
             observe_early_stopping,
             &observer,
             &completed,
             error))) {
        goto cleanup;
    }
    if (!observer.has_best) {
        neural_error_set(error, "early stopping produced no selectable model");
        goto cleanup;
    }
    weights.completed_epochs = completed.completed_epochs;
    weights.digests = *digests;
    weights.selected_epoch = observer.metadata.best_epoch;
    weights.target_epochs = target_epochs;
    if (completed.completion_reason == NEURAL_TRAINING_TARGET_EPOCHS) {
        weights.completion_reason = NEURAL_COMPLETION_TARGET;
    } else if (completed.completion_reason ==
               NEURAL_TRAINING_LOSS_TARGET) {
        weights.completion_reason = NEURAL_COMPLETION_LOSS_TARGET;
    } else if (completed.completion_reason ==
               NEURAL_TRAINING_NO_IMPROVEMENT) {
        weights.completion_reason = NEURAL_COMPLETION_NO_IMPROVEMENT;
    } else {
        weights.completion_reason = NEURAL_COMPLETION_EARLY_STOPPING;
    }
    weights.format_version = 2U;
    if (!copy_model_parameters(current_model, best_model, error) ||
        !neural_model_evaluate_dataset_loss(current_model,
                                            observer.workspace,
                                            observer.predicted,
                                            &project->dataset,
                                            project->training.loss,
                                            &completed.final_loss,
                                            error) ||
        !model_objective(current_model,
                         &project->training,
                         completed.final_loss,
                         &completed.final_objective,
                         error) ||
        !neural_early_weights_save_atomic(weights_path,
                                          current_model,
                                          &weights,
                                          error) ||
        !neural_atomic_file_remove(checkpoint_path, 1, error)) {
        goto cleanup;
    }
    *result = completed;
    success = 1;

cleanup:
    if (interrupted_signal != NULL && initialized) {
        *interrupted_signal = observer.interrupted_signal;
    }
    if (initialized) {
        free_early_observer(&observer);
    }
    return success;
}

static int observe_project_training(const NeuralEpochReport *report,
                                    void *context,
                                    NeuralError *error)
{
    NeuralProjectTrainingObserver *observer = context;

    if (observer == NULL || observer->checkpoint == NULL ||
        !neural_project_checkpoint_observe(report,
                                           observer->checkpoint,
                                           error)) {
        return 0;
    }
    return observer->observer == NULL ||
           observer->observer(report, observer->observer_context, error);
}

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
    NeuralEpochObserver observer,
    void *observer_context,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralProject project;
    NeuralModel *model = NULL;
    NeuralModel *best_model = NULL;
    NeuralProjectDigests digests;
    NeuralProjectCheckpointObserver checkpoint_observer = {0};
    NeuralProjectTrainingObserver training_observer = {
        &checkpoint_observer, observer, observer_context
    };
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralTrainingResult completed = {0};
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
                             error)) {
        goto cleanup;
    }
    if (project.training.early_stopping_patience != 0U) {
        if (!neural_model_create(&project.model,
                                 project.training.seed,
                                 &best_model,
                                 error) ||
            !train_early_stopping_range(&project,
                                        &digests,
                                        execution,
                                        model,
                                        best_model,
                                        NULL,
                                        weights_path,
                                        checkpoint_path,
                                        0U,
                                        project.training.epochs,
                                        0U,
                                        NULL,
                                        stop_request,
                                        observer,
                                        observer_context,
                                        interrupted_signal,
                                        &completed,
                                        error)) {
            goto cleanup;
        }
        *result = completed;
        success = 1;
        goto cleanup;
    }
    if (!neural_project_checkpoint_observer_initialize(
            &checkpoint_observer,
            checkpoint_path,
            model,
            &digests,
            project.training.optimizer,
            project.training.checkpoint_interval,
            project.training.epochs,
            error)) {
        goto cleanup;
    }
    neural_project_checkpoint_observer_set_stop_request(&checkpoint_observer,
                                                        stop_request);
    if (!neural_model_train(model,
                                       &project.dataset,
                                       &project.training,
                                       execution,
                                       observe_project_training,
                                       &training_observer,
                                       &completed,
                                       error)) {
        goto cleanup;
    }
    if (!persistence_path_is_absent(weights_path, error)) {
        goto cleanup;
    }
    if (!save_completed_weights(weights_path,
                                model,
                                &digests,
                                project.training.epochs,
                                &completed,
                                error) ||
        !neural_atomic_file_remove(checkpoint_path, 1, error)) {
        goto cleanup;
    }
    *result = completed;
    success = 1;

cleanup:
    if (interrupted_signal != NULL && *interrupted_signal == 0) {
        *interrupted_signal = checkpoint_observer.interrupted_signal;
    }
    neural_project_lock_release(&project_lock);
    neural_model_free(best_model);
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
                                                 NULL,
                                                 NULL,
                                                 result,
                                                 error);
}

int neural_project_train_resume_controlled(
    const char *directory,
    const NeuralExecutionConfig *execution,
    const volatile sig_atomic_t *stop_request,
    NeuralEpochObserver observer,
    void *observer_context,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralProject project;
    NeuralProjectDigests digests;
    NeuralCheckpointMetadata checkpoint_metadata;
    NeuralWeightsMetadata weights_metadata;
    NeuralProjectCheckpointObserver checkpoint_observer = {0};
    NeuralProjectTrainingObserver training_observer = {
        &checkpoint_observer, observer, observer_context
    };
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralTrainingResult completed = {0};
    NeuralModel *checkpoint_model = NULL;
    NeuralModel *best_model = NULL;
    NeuralModel *weights_model = NULL;
    NeuralOptimizer *optimizer = NULL;
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
        !create_training_optimizer(checkpoint_model,
                                   &project.training,
                                   &optimizer,
                                   error)) {
        goto cleanup;
    }
    if (project.training.early_stopping_patience != 0U) {
        if (!neural_model_create(&project.model,
                                 project.training.seed,
                                 &best_model,
                                 error) ||
            !neural_early_checkpoint_load_with_optimizer(
                checkpoint_path,
                checkpoint_model,
                best_model,
                optimizer,
                &digests,
                &checkpoint_metadata,
                error)) {
            goto cleanup;
        }
    } else if (!neural_checkpoint_load_with_optimizer(
                   checkpoint_path,
                   checkpoint_model,
                   optimizer,
                   &digests,
                   &checkpoint_metadata,
                   error)) {
        goto cleanup;
    }
    if (checkpoint_metadata.optimizer != project.training.optimizer) {
        neural_error_set(error,
                         "checkpoint optimizer '%s' does not match configured optimizer '%s'",
                         neural_optimizer_name(checkpoint_metadata.optimizer),
                         neural_optimizer_name(project.training.optimizer));
        goto cleanup;
    }
    if (!weights_exists &&
        checkpoint_metadata.target_epochs != project.training.epochs) {
        neural_error_set(error,
                         "checkpoint target %zu does not match configured epochs %zu",
                         checkpoint_metadata.target_epochs,
                         project.training.epochs);
        goto cleanup;
    }
    if (project.training.early_stopping_patience != 0U) {
        if (weights_exists) {
            if (!neural_model_create(&project.model,
                                     project.training.seed,
                                     &weights_model,
                                     error) ||
                !neural_weights_load(weights_path,
                                     weights_model,
                                     &digests,
                                     &weights_metadata,
                                     error) ||
                weights_metadata.completed_epochs >
                    checkpoint_metadata.completed_epochs) {
                if (error != NULL && error->message[0] == '\0') {
                    neural_error_set(error,
                                     "checkpoint precedes finalized baseline");
                }
                goto cleanup;
            }
        }
        if (!train_early_stopping_range(
                &project,
                &digests,
                execution,
                checkpoint_model,
                best_model,
                optimizer,
                weights_path,
                checkpoint_path,
                checkpoint_metadata.completed_epochs,
                checkpoint_metadata.target_epochs,
                checkpoint_metadata.best_epoch,
                &checkpoint_metadata,
                stop_request,
                observer,
                observer_context,
                interrupted_signal,
                &completed,
                error)) {
            goto cleanup;
        }
        *result = completed;
        success = 1;
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
        if (weights_metadata.completed_epochs < project.training.epochs) {
            neural_error_set(
                error,
                "final weights epochs %zu precede configured epochs %zu",
                weights_metadata.completed_epochs,
                project.training.epochs);
            goto cleanup;
        }
        if (weights_metadata.completed_epochs >
            checkpoint_metadata.target_epochs) {
            neural_error_set(
                error,
                "final weights epochs %zu exceed checkpoint target %zu",
                weights_metadata.completed_epochs,
                checkpoint_metadata.target_epochs);
            goto cleanup;
        }
        if (weights_metadata.completed_epochs ==
            checkpoint_metadata.target_epochs) {
            if (checkpoint_metadata.completed_epochs ==
                    checkpoint_metadata.target_epochs &&
                !models_have_equal_parameters(checkpoint_model,
                                               weights_model)) {
                neural_error_set(
                    error,
                    "completed checkpoint parameters do not match final weights");
                goto cleanup;
            }
            if (!neural_model_train_range(
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
        if (checkpoint_metadata.completed_epochs <
            weights_metadata.completed_epochs) {
            neural_error_set(
                error,
                "checkpoint epochs %zu precede refinement baseline %zu",
                checkpoint_metadata.completed_epochs,
                weights_metadata.completed_epochs);
            goto cleanup;
        }
        if (checkpoint_metadata.completed_epochs ==
                weights_metadata.completed_epochs &&
            !models_have_equal_parameters(checkpoint_model, weights_model)) {
            neural_error_set(
                error,
                "refinement checkpoint parameters do not match baseline weights");
            goto cleanup;
        }
    }
    if (!neural_project_checkpoint_observer_initialize(
            &checkpoint_observer,
            checkpoint_path,
            checkpoint_model,
            &digests,
            project.training.optimizer,
            project.training.checkpoint_interval,
            checkpoint_metadata.target_epochs,
            error)) {
        goto cleanup;
    }
    neural_project_checkpoint_observer_set_stop_request(&checkpoint_observer,
                                                        stop_request);
    if (!neural_model_train_range_with_optimizer(
            checkpoint_model,
            &project.dataset,
            &project.training,
            execution,
            optimizer,
            checkpoint_metadata.completed_epochs,
            checkpoint_metadata.target_epochs,
            observe_project_training,
            &training_observer,
            &completed,
            error) ||
        (!weights_exists &&
         !persistence_path_is_absent(weights_path, error))) {
        goto cleanup;
    }
    if (!save_completed_weights(weights_path,
                                checkpoint_model,
                                &digests,
                                checkpoint_metadata.target_epochs,
                                &completed,
                                error) ||
        !neural_atomic_file_remove(checkpoint_path, 0, error)) {
        goto cleanup;
    }
    *result = completed;
    success = 1;

cleanup:
    if (interrupted_signal != NULL && *interrupted_signal == 0) {
        *interrupted_signal = checkpoint_observer.interrupted_signal;
    }
    neural_project_lock_release(&project_lock);
    neural_model_free(best_model);
    neural_model_free(weights_model);
    neural_optimizer_free(optimizer);
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
                                                  NULL,
                                                  NULL,
                                                  result,
                                                  error);
}

int neural_project_train_additional_controlled(
    const char *directory,
    const NeuralExecutionConfig *execution,
    size_t additional_epochs,
    const volatile sig_atomic_t *stop_request,
    NeuralEpochObserver observer,
    void *observer_context,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error)
{
    NeuralProject project;
    NeuralProjectDigests digests;
    NeuralWeightsMetadata weights_metadata;
    NeuralProjectCheckpointObserver checkpoint_observer = {0};
    NeuralProjectTrainingObserver training_observer = {
        &checkpoint_observer, observer, observer_context
    };
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralTrainingResult completed = {0};
    NeuralModel *model = NULL;
    NeuralModel *best_model = NULL;
    char *weights_path = NULL;
    char *checkpoint_path = NULL;
    size_t target_epochs;
    int checkpoint_exists = 0;
    int weights_exists = 0;
    int project_loaded = 0;
    int success = 0;

    neural_error_clear(error);
    if (directory == NULL || directory[0] == '\0' || execution == NULL ||
        additional_epochs == 0U || result == NULL) {
        neural_error_set(error,
                         "additional project training arguments are required");
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
    if (checkpoint_exists) {
        neural_error_set(error,
                         "additional training requires no checkpoint.txt; "
                         "use --resume");
        goto cleanup;
    }
    if (!weights_exists) {
        neural_error_set(error,
                         "additional training requires finalized weights.txt");
        goto cleanup;
    }
    if (!neural_project_load(directory, &project, error)) {
        goto cleanup;
    }
    project_loaded = 1;
    if (!neural_project_digests_compute(&project, &digests, error) ||
        !neural_model_create(&project.model,
                             project.training.seed,
                             &model,
                             error) ||
        !neural_weights_load(weights_path,
                             model,
                             &digests,
                             &weights_metadata,
                             error)) {
        goto cleanup;
    }
    if (weights_metadata.completed_epochs < project.training.epochs &&
        weights_metadata.format_version != 2U) {
        neural_error_set(error,
                         "final weights epochs %zu precede configured epochs %zu",
                         weights_metadata.completed_epochs,
                         project.training.epochs);
        goto cleanup;
    }
    if (additional_epochs > SIZE_MAX - weights_metadata.completed_epochs) {
        neural_error_set(error,
                         "additional epoch target exceeds supported range");
        goto cleanup;
    }
    target_epochs = weights_metadata.completed_epochs + additional_epochs;
    if (project.training.early_stopping_patience != 0U) {
        if (!neural_model_create(&project.model,
                                 project.training.seed,
                                 &best_model,
                                 error) ||
            !train_early_stopping_range(
                &project,
                &digests,
                execution,
                model,
                best_model,
                NULL,
                weights_path,
                checkpoint_path,
                weights_metadata.completed_epochs,
                target_epochs,
                weights_metadata.selected_epoch,
                NULL,
                stop_request,
                observer,
                observer_context,
                interrupted_signal,
                &completed,
                error)) {
            goto cleanup;
        }
        *result = completed;
        success = 1;
        goto cleanup;
    }
    if (!neural_project_checkpoint_observer_initialize(
            &checkpoint_observer,
            checkpoint_path,
            model,
            &digests,
            project.training.optimizer,
            project.training.checkpoint_interval,
            target_epochs,
            error)) {
        goto cleanup;
    }
    neural_project_checkpoint_observer_set_stop_request(&checkpoint_observer,
                                                        stop_request);
    if (!neural_model_train_range(
            model,
            &project.dataset,
            &project.training,
            execution,
            weights_metadata.completed_epochs,
            target_epochs,
            observe_project_training,
            &training_observer,
            &completed,
            error)) {
        goto cleanup;
    }
    if (!save_completed_weights(weights_path,
                                model,
                                &digests,
                                target_epochs,
                                &completed,
                                error) ||
        !neural_atomic_file_remove(checkpoint_path, 1, error)) {
        goto cleanup;
    }
    *result = completed;
    success = 1;

cleanup:
    if (interrupted_signal != NULL && *interrupted_signal == 0) {
        *interrupted_signal = checkpoint_observer.interrupted_signal;
    }
    neural_project_lock_release(&project_lock);
    neural_model_free(best_model);
    neural_model_free(model);
    if (project_loaded) {
        neural_project_free(&project);
    }
    free(checkpoint_path);
    free(weights_path);
    return success;
}

int neural_project_train_additional(const char *directory,
                                    const NeuralExecutionConfig *execution,
                                    size_t additional_epochs,
                                    NeuralTrainingResult *result,
                                    NeuralError *error)
{
    return neural_project_train_additional_controlled(directory,
                                                      execution,
                                                      additional_epochs,
                                                      NULL,
                                                      NULL,
                                                      NULL,
                                                      NULL,
                                                      result,
                                                      error);
}
