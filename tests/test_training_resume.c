#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/digest.h"
#include "neural/model.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "neural/training.h"
#include "../src/train_project.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static void project_path(char *path,
                         size_t capacity,
                         const char *directory,
                         const char *name)
{
    (void)snprintf(path, capacity, "%s/%s", directory, name);
}

static int write_text(const char *path, const char *text)
{
    FILE *stream = fopen(path, "wb");
    int success;

    if (stream == NULL) {
        return 0;
    }
    success = fputs(text, stream) != EOF;
    if (fclose(stream) != 0) {
        success = 0;
    }
    return success;
}

static int copy_file(const char *source_path, const char *destination_path)
{
    FILE *source = fopen(source_path, "rb");
    FILE *destination = NULL;
    unsigned char buffer[4096];
    int success = 0;

    if (source == NULL) {
        return 0;
    }
    destination = fopen(destination_path, "wb");
    if (destination == NULL) {
        (void)fclose(source);
        return 0;
    }
    for (;;) {
        size_t count = fread(buffer, 1U, sizeof(buffer), source);

        if (count != 0U && fwrite(buffer, 1U, count, destination) != count) {
            break;
        }
        if (count < sizeof(buffer)) {
            success = feof(source) != 0 && ferror(source) == 0;
            break;
        }
    }
    if (fclose(destination) != 0 || fclose(source) != 0) {
        success = 0;
    }
    return success;
}

static int files_equal(const char *left_path, const char *right_path)
{
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    int equal = 1;

    if (left == NULL || right == NULL) {
        equal = 0;
    } else {
        for (;;) {
            int left_byte = fgetc(left);
            int right_byte = fgetc(right);

            if (left_byte != right_byte) {
                equal = 0;
                break;
            }
            if (left_byte == EOF) {
                if (ferror(left) != 0 || ferror(right) != 0) {
                    equal = 0;
                }
                break;
            }
        }
    }
    if (right != NULL && fclose(right) != 0) {
        equal = 0;
    }
    if (left != NULL && fclose(left) != 0) {
        equal = 0;
    }
    return equal;
}

static int path_exists(const char *path)
{
    struct stat status;

    return lstat(path, &status) == 0;
}

static void cleanup_project(const char *directory)
{
    static const char *const names[] = {
        "model.txt",
        "project.conf",
        "train.txt",
        "validation.txt",
        "weights.txt",
        "weights.before",
        "checkpoint.txt",
        "checkpoint.before",
        ".neural-c.lock"
    };
    size_t index;

    (void)chmod(directory, 0700);
    for (index = 0U; index < sizeof(names) / sizeof(names[0]); index++) {
        char path[256];

        project_path(path, sizeof(path), directory, names[index]);
        (void)remove(path);
    }
    (void)rmdir(directory);
}

static int prepare_early_project(const char *directory)
{
    static const char model_text[] =
        "neural-c model 1\n\ninput 1\ndense 2 sigmoid\ndense 1 sigmoid\n";
    static const char dataset_text[] =
        "neural-c dataset 1\n\n0 -> 0\n1 -> 1\n2 -> 1\n3 -> 0\n";
    static const char config_text[] =
        "neural-c project 1\n\nepochs 20\nlearning_rate 0.25\n"
        "seed 42\nloss mse\ncheckpoint_interval 0\n"
        "early_stopping_patience 3\n"
        "early_stopping_min_delta 100\n";
    char path[256];

    cleanup_project(directory);
    if (mkdir(directory, 0700) != 0) {
        return 0;
    }
    project_path(path, sizeof(path), directory, "model.txt");
    if (!write_text(path, model_text)) {
        return 0;
    }
    project_path(path, sizeof(path), directory, "project.conf");
    if (!write_text(path, config_text)) {
        return 0;
    }
    project_path(path, sizeof(path), directory, "train.txt");
    if (!write_text(path, dataset_text)) {
        return 0;
    }
    project_path(path, sizeof(path), directory, "validation.txt");
    return write_text(path, dataset_text);
}

static int prepare_project(const char *directory,
                           size_t epochs,
                           size_t checkpoint_interval)
{
    static const char model_text[] =
        "neural-c model 1\n\ninput 1\ndense 2 sigmoid\ndense 1 sigmoid\n";
    static const char dataset_text[] =
        "neural-c dataset 1\n\n0 -> 0\n1 -> 1\n2 -> 1\n3 -> 0\n";
    char config_text[256];
    char model_path[256];
    char config_path[256];
    char dataset_path[256];

    cleanup_project(directory);
    if (mkdir(directory, 0700) != 0) {
        return 0;
    }
    project_path(model_path, sizeof(model_path), directory, "model.txt");
    project_path(config_path, sizeof(config_path), directory, "project.conf");
    project_path(dataset_path, sizeof(dataset_path), directory, "train.txt");
    (void)snprintf(config_text,
                   sizeof(config_text),
                   "neural-c project 1\n\nepochs %zu\nlearning_rate 0.25\n"
                   "seed 42\nloss mse\ncheckpoint_interval %zu\n"
                   "batch_size 3\n",
                   epochs,
                   checkpoint_interval);
    return write_text(model_path, model_text) &&
           write_text(config_path, config_text) &&
           write_text(dataset_path, dataset_text);
}

static int train_state(const char *directory,
                       size_t completed_epochs,
                       NeuralProject *project,
                       NeuralProjectDigests *digests,
                       NeuralModel **model,
                       NeuralError *error)
{
    NeuralExecutionConfig execution = {1U};
    NeuralTrainingResult result;

    memset(project, 0, sizeof(*project));
    *model = NULL;
    return neural_project_load(directory, project, error) &&
           neural_project_digests_compute(project, digests, error) &&
           neural_model_create(&project->model,
                               project->training.seed,
                               model,
                               error) &&
           neural_model_train_range(*model,
                                               &project->dataset,
                                               &project->training,
                                               &execution,
                                               0U,
                                               completed_epochs,
                                               NULL,
                                               NULL,
                                               &result,
                                               error);
}

static int save_checkpoint_state(const char *directory,
                                 size_t trained_epochs,
                                 size_t completed_epochs,
                                 size_t target_epochs,
                                 NeuralError *error)
{
    NeuralProject project;
    NeuralProjectDigests digests;
    NeuralCheckpointMetadata metadata;
    NeuralModel *model = NULL;
    char path[256];
    int success;

    if (!train_state(directory,
                     trained_epochs,
                     &project,
                     &digests,
                     &model,
                     error)) {
        neural_model_free(model);
        return 0;
    }
    metadata.completed_epochs = completed_epochs;
    metadata.target_epochs = target_epochs;
    metadata.rng_state = neural_model_random_state(model);
    metadata.optimizer = NEURAL_OPTIMIZER_GRADIENT_DESCENT;
    metadata.digests = digests;
    project_path(path, sizeof(path), directory, "checkpoint.txt");
    success = neural_checkpoint_save_atomic(path, model, &metadata, error);
    neural_model_free(model);
    neural_project_free(&project);
    return success;
}

static int save_weights_state(const char *directory,
                              size_t trained_epochs,
                              size_t completed_epochs,
                              NeuralError *error)
{
    NeuralProject project;
    NeuralProjectDigests digests;
    NeuralWeightsMetadata metadata;
    NeuralModel *model = NULL;
    char path[256];
    int success;

    if (!train_state(directory,
                     trained_epochs,
                     &project,
                     &digests,
                     &model,
                     error)) {
        neural_model_free(model);
        return 0;
    }
    metadata.completed_epochs = completed_epochs;
    metadata.digests = digests;
    project_path(path, sizeof(path), directory, "weights.txt");
    success = neural_weights_save_atomic(path, model, &metadata, error);
    neural_model_free(model);
    neural_project_free(&project);
    return success;
}

static int models_equal(const NeuralModel *left, const NeuralModel *right)
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

        if (left_weight_count != right_weight_count ||
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

static void test_resume_matches_continuous_training(void)
{
    static const char *const directory = "build/tests/resume-equivalence";
    NeuralExecutionConfig parallel = {4U};
    NeuralTrainingResult resumed_result;
    NeuralTrainingResult expected_result;
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralWeightsMetadata weights_metadata;
    NeuralModel *resumed_model = NULL;
    NeuralModel *expected_model = NULL;
    NeuralError error;
    char checkpoint_path[256];
    char weights_path[256];
    int prepared;

    prepared = prepare_project(directory, 6U, 2U) &&
               save_checkpoint_state(directory, 2U, 2U, 6U, &error);
    check(prepared, "resume equivalence fixture must be prepared");
    if (prepared) {
        check(neural_project_train_resume(directory,
                                          &parallel,
                                          &resumed_result,
                                          &error) &&
                  resumed_result.completed_epochs == 6U &&
                  resumed_result.worker_count == 4U,
              "resume must continue to target with a different worker count");
        project_path(checkpoint_path,
                     sizeof(checkpoint_path),
                     directory,
                     "checkpoint.txt");
        project_path(weights_path,
                     sizeof(weights_path),
                     directory,
                     "weights.txt");
        check(!path_exists(checkpoint_path) && path_exists(weights_path),
              "successful resume must leave only final weights");
        check(neural_project_load(directory, &project, &error) &&
                  neural_project_digests_compute(&project, &digests, &error) &&
                  neural_model_create(&project.model,
                                      project.training.seed,
                                      &resumed_model,
                                      &error) &&
                  neural_model_create(&project.model,
                                      project.training.seed,
                                      &expected_model,
                                      &error) &&
                  neural_weights_load(weights_path,
                                      resumed_model,
                                      &digests,
                                      &weights_metadata,
                                      &error) &&
                  neural_model_train(expected_model,
                                                &project.dataset,
                                                &project.training,
                                                &parallel,
                                                NULL,
                                                NULL,
                                                &expected_result,
                                                &error) &&
                  weights_metadata.completed_epochs == 6U &&
                  models_equal(resumed_model, expected_model) &&
                  resumed_result.final_loss == expected_result.final_loss,
              "resumed weights must be bit-identical to continuous training");
        check(!neural_project_train_resume(directory,
                                           &parallel,
                                           &resumed_result,
                                           &error) &&
                  strstr(error.message, "already finalized") != NULL,
              "resume without a checkpoint must reject a finalized project");
    }
    neural_model_free(expected_model);
    neural_model_free(resumed_model);
    neural_project_free(&project);
    cleanup_project(directory);
}

static void test_interrupted_finalization_reconciliation(void)
{
    static const char *const directory = "build/tests/resume-finalization";
    NeuralExecutionConfig execution = {2U};
    NeuralTrainingResult result;
    NeuralError error;
    char checkpoint_path[256];
    char weights_path[256];
    char before_path[256];
    int prepared;

    project_path(checkpoint_path,
                 sizeof(checkpoint_path),
                 directory,
                 "checkpoint.txt");
    project_path(weights_path, sizeof(weights_path), directory, "weights.txt");
    project_path(before_path,
                 sizeof(before_path),
                 directory,
                 "weights.before");
    prepared = prepare_project(directory, 4U, 2U) &&
               save_checkpoint_state(directory, 2U, 2U, 4U, &error) &&
               save_weights_state(directory, 4U, 4U, &error) &&
               copy_file(weights_path, before_path);
    check(prepared, "interrupted finalization fixture must be prepared");
    if (prepared) {
        check(neural_project_train_resume(directory,
                                          &execution,
                                          &result,
                                          &error) &&
                  result.completed_epochs == 4U &&
                  !path_exists(checkpoint_path) &&
                  files_equal(weights_path, before_path),
              "resume must reconcile valid final weights without rewriting them");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 0U) &&
               save_checkpoint_state(directory, 4U, 4U, 4U, &error);
    check(prepared, "already-complete checkpoint fixture must be prepared");
    if (prepared) {
        check(neural_project_train_resume(directory,
                                          &execution,
                                          &result,
                                          &error) &&
                  result.completed_epochs == 4U &&
                  path_exists(weights_path) &&
                  !path_exists(checkpoint_path),
              "an already-complete checkpoint must finalize without updates");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 2U) &&
               save_checkpoint_state(directory, 3U, 4U, 4U, &error) &&
               save_weights_state(directory, 4U, 4U, &error) &&
               copy_file(checkpoint_path, before_path);
    check(prepared, "mismatched completed-state fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_resume(directory,
                                           &execution,
                                           &result,
                                           &error) &&
                  strstr(error.message, "do not match final weights") != NULL &&
                  path_exists(weights_path) &&
                  files_equal(checkpoint_path, before_path),
              "same-epoch parameter mismatches must leave both files untouched");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 2U) &&
               save_checkpoint_state(directory, 2U, 2U, 4U, &error) &&
               save_weights_state(directory, 3U, 3U, &error);
    check(prepared, "weights epoch mismatch fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_resume(directory,
                                           &execution,
                                           &result,
                                           &error) &&
                  strstr(error.message, "precede configured epochs") != NULL &&
                  path_exists(checkpoint_path) && path_exists(weights_path),
              "weights epoch mismatches must leave interrupted state untouched");
    }
    cleanup_project(directory);
}

static void test_additional_matches_continuous_training(void)
{
    static const char *const directory = "build/tests/additional-equivalence";
    NeuralExecutionConfig parallel = {4U};
    NeuralTrainingResult result;
    NeuralTrainingResult expected_result;
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralWeightsMetadata metadata;
    NeuralModel *refined_model = NULL;
    NeuralModel *expected_model = NULL;
    NeuralError error;
    char weights_path[256];
    char checkpoint_path[256];
    int prepared;

    project_path(weights_path, sizeof(weights_path), directory, "weights.txt");
    project_path(checkpoint_path,
                 sizeof(checkpoint_path),
                 directory,
                 "checkpoint.txt");
    prepared = prepare_project(directory, 4U, 2U) &&
               save_weights_state(directory, 4U, 4U, &error);
    check(prepared, "additional equivalence fixture must be prepared");
    if (prepared) {
        check(neural_project_train_additional(directory,
                                              &parallel,
                                              2U,
                                              &result,
                                              &error) &&
                  result.completed_epochs == 6U &&
                  result.worker_count == 4U &&
                  path_exists(weights_path) &&
                  !path_exists(checkpoint_path),
              "additional training must finalize exactly the requested epochs");
        check(neural_project_load(directory, &project, &error) &&
                  neural_project_digests_compute(&project, &digests, &error) &&
                  neural_model_create(&project.model,
                                      project.training.seed,
                                      &refined_model,
                                      &error) &&
                  neural_model_create(&project.model,
                                      project.training.seed,
                                      &expected_model,
                                      &error) &&
                  neural_weights_load(weights_path,
                                      refined_model,
                                      &digests,
                                      &metadata,
                                      &error) &&
                  neural_model_train_range(expected_model,
                                                      &project.dataset,
                                                      &project.training,
                                                      &parallel,
                                                      0U,
                                                      6U,
                                                      NULL,
                                                      NULL,
                                                      &expected_result,
                                                      &error) &&
                  metadata.completed_epochs == 6U &&
                  models_equal(refined_model, expected_model) &&
                  result.final_loss == expected_result.final_loss,
              "refinement must match continuous absolute-epoch training");
        check(neural_project_train_additional(directory,
                                              &parallel,
                                              1U,
                                              &result,
                                              &error) &&
                  result.completed_epochs == 7U,
              "additional training must support repeated refinements");
    }
    neural_model_free(expected_model);
    neural_model_free(refined_model);
    neural_project_free(&project);
    cleanup_project(directory);
}

static void test_additional_interruption_and_resume(void)
{
    static const char *const directory = "build/tests/additional-interrupt";
    NeuralExecutionConfig execution = {3U};
    NeuralTrainingResult result;
    NeuralError error;
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralCheckpointMetadata checkpoint_metadata;
    NeuralWeightsMetadata weights_metadata;
    NeuralModel *checkpoint_model = NULL;
    NeuralModel *weights_model = NULL;
    volatile sig_atomic_t stop_request = SIGTERM;
    int interrupted_signal = 0;
    char weights_path[256];
    char before_path[256];
    char checkpoint_path[256];
    int prepared;

    project_path(weights_path, sizeof(weights_path), directory, "weights.txt");
    project_path(before_path,
                 sizeof(before_path),
                 directory,
                 "weights.before");
    project_path(checkpoint_path,
                 sizeof(checkpoint_path),
                 directory,
                 "checkpoint.txt");
    prepared = prepare_project(directory, 4U, 0U) &&
               save_weights_state(directory, 4U, 4U, &error) &&
               copy_file(weights_path, before_path);
    check(prepared, "additional interruption fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_additional_controlled(
                  directory,
                  &execution,
                  2U,
                  &stop_request,
                  NULL,
                  NULL,
                  &interrupted_signal,
                  &result,
                  &error) &&
                  interrupted_signal == SIGTERM &&
                  strstr(error.message, "checkpoint saved") != NULL &&
                  files_equal(weights_path, before_path) &&
                  path_exists(checkpoint_path),
              "interrupted refinement must preserve baseline weights and save recovery");
        check(neural_project_load(directory, &project, &error) &&
                  neural_project_digests_compute(&project, &digests, &error) &&
                  neural_model_create(&project.model,
                                      project.training.seed,
                                      &checkpoint_model,
                                      &error) &&
                  neural_checkpoint_load(checkpoint_path,
                                         checkpoint_model,
                                         &digests,
                                         &checkpoint_metadata,
                                         &error) &&
                  checkpoint_metadata.completed_epochs == 5U &&
                  checkpoint_metadata.target_epochs == 6U,
              "interrupted refinement checkpoint must use absolute epochs");
        check(neural_project_train_resume(directory,
                                          &execution,
                                          &result,
                                          &error) &&
                  result.completed_epochs == 6U &&
                  !path_exists(checkpoint_path) &&
                  neural_model_create(&project.model,
                                      project.training.seed,
                                      &weights_model,
                                      &error) &&
                  neural_weights_load(weights_path,
                                      weights_model,
                                      &digests,
                                      &weights_metadata,
                                      &error) &&
                  weights_metadata.completed_epochs == 6U &&
                  models_equal(checkpoint_model, weights_model) == 0,
              "resume must finish an interrupted refinement and replace baseline");
    }
    neural_model_free(weights_model);
    neural_model_free(checkpoint_model);
    neural_project_free(&project);
    cleanup_project(directory);
}

static void test_additional_state_validation(void)
{
    static const char *const directory = "build/tests/additional-validation";
    NeuralExecutionConfig execution = {2U};
    NeuralTrainingResult result;
    NeuralError error;
    char weights_path[256];
    char before_path[256];
    char checkpoint_path[256];
    int prepared;

    project_path(weights_path, sizeof(weights_path), directory, "weights.txt");
    project_path(before_path,
                 sizeof(before_path),
                 directory,
                 "weights.before");
    project_path(checkpoint_path,
                 sizeof(checkpoint_path),
                 directory,
                 "checkpoint.txt");

    prepared = prepare_project(directory, 4U, 2U);
    check(prepared, "missing weights fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_additional(directory,
                                               &execution,
                                               1U,
                                               &result,
                                               &error) &&
                  strstr(error.message, "requires finalized weights") != NULL &&
                  !path_exists(checkpoint_path),
              "additional training must require final weights");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 2U) &&
               save_weights_state(directory, 4U, 4U, &error) &&
               save_checkpoint_state(directory, 4U, 4U, 6U, &error) &&
               copy_file(weights_path, before_path);
    check(prepared, "existing checkpoint fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_additional(directory,
                                               &execution,
                                               1U,
                                               &result,
                                               &error) &&
                  strstr(error.message, "use --resume") != NULL &&
                  files_equal(weights_path, before_path) &&
                  path_exists(checkpoint_path),
              "additional training must refuse an existing recovery run");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 0U) &&
               save_weights_state(directory, 4U, SIZE_MAX, &error) &&
               copy_file(weights_path, before_path);
    check(prepared, "additional overflow fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_additional(directory,
                                               &execution,
                                               1U,
                                               &result,
                                               &error) &&
                  strstr(error.message, "exceeds supported range") != NULL &&
                  files_equal(weights_path, before_path) &&
                  !path_exists(checkpoint_path),
              "additional target overflow must fail before disk mutation");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 0U) &&
               save_weights_state(directory, 3U, 3U, &error);
    check(prepared, "incomplete final weights fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_additional(directory,
                                               &execution,
                                               1U,
                                               &result,
                                               &error) &&
                  strstr(error.message, "precede configured epochs") != NULL &&
                  !path_exists(checkpoint_path),
              "additional training must reject incomplete final weights");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 0U) &&
               save_weights_state(directory, 4U, 4U, &error);
    check(prepared, "zero-interval additional fixture must be prepared");
    if (prepared) {
        check(neural_project_train_additional(directory,
                                              &execution,
                                              1U,
                                              &result,
                                              &error) &&
                  result.completed_epochs == 5U &&
                  !path_exists(checkpoint_path),
              "zero interval must avoid checkpoints during normal refinement");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 0U) &&
               save_weights_state(directory, 4U, 4U, &error) &&
               save_checkpoint_state(directory, 3U, 3U, 6U, &error);
    check(prepared, "checkpoint before refinement baseline fixture must prepare");
    if (prepared) {
        check(!neural_project_train_resume(directory,
                                           &execution,
                                           &result,
                                           &error) &&
                  strstr(error.message, "precede refinement baseline") != NULL &&
                  path_exists(weights_path) && path_exists(checkpoint_path),
              "resume must reject checkpoints before the refinement baseline");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 0U) &&
               save_weights_state(directory, 4U, 4U, &error) &&
               save_checkpoint_state(directory, 3U, 4U, 6U, &error);
    check(prepared, "mismatched refinement baseline fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_resume(directory,
                                           &execution,
                                           &result,
                                           &error) &&
                  strstr(error.message,
                         "do not match baseline weights") != NULL &&
                  path_exists(weights_path) && path_exists(checkpoint_path),
              "resume must reject same-epoch refinement parameter mismatches");
    }
    cleanup_project(directory);
}

static void test_target_and_write_failures(void)
{
    static const char *const directory = "build/tests/resume-failures";
    NeuralExecutionConfig execution = {2U};
    NeuralTrainingResult result;
    NeuralError error;
    char checkpoint_path[256];
    char weights_path[256];
    char before_path[256];
    char lock_path[256];
    int prepared;

    project_path(checkpoint_path,
                 sizeof(checkpoint_path),
                 directory,
                 "checkpoint.txt");
    project_path(weights_path, sizeof(weights_path), directory, "weights.txt");
    project_path(before_path,
                 sizeof(before_path),
                 directory,
                 "checkpoint.before");
    project_path(lock_path,
                 sizeof(lock_path),
                 directory,
                 ".neural-c.lock");
    prepared = prepare_project(directory, 4U, 2U) &&
               save_checkpoint_state(directory, 2U, 2U, 5U, &error);
    check(prepared, "checkpoint target mismatch fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_resume(directory,
                                           &execution,
                                           &result,
                                           &error) &&
                  strstr(error.message, "does not match configured epochs") != NULL &&
                  path_exists(checkpoint_path) && !path_exists(weights_path),
              "checkpoint target mismatches must fail before disk mutation");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 4U, 0U) &&
               save_checkpoint_state(directory, 2U, 2U, 4U, &error);
    check(prepared, "disabled periodic resume fixture must be prepared");
    if (prepared) {
        check(neural_project_train_resume(directory,
                                          &execution,
                                          &result,
                                          &error) &&
                  result.completed_epochs == 4U &&
                  path_exists(weights_path) &&
                  !path_exists(checkpoint_path),
              "resume must complete when periodic checkpoints are disabled");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 3U, 0U) &&
               save_checkpoint_state(directory, 2U, 2U, 3U, &error) &&
               copy_file(checkpoint_path, before_path) &&
               write_text(lock_path, "") && chmod(lock_path, 0600) == 0 &&
               chmod(directory, 0500) == 0;
    check(prepared, "final weights failure fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_resume(directory,
                                           &execution,
                                           &result,
                                           &error),
              "final weights write failure must reject resume");
        (void)chmod(directory, 0700);
        check(!path_exists(weights_path) &&
                  files_equal(checkpoint_path, before_path),
              "final weights failure must preserve the resumable checkpoint");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 3U, 1U) &&
               save_checkpoint_state(directory, 1U, 1U, 3U, &error) &&
               copy_file(checkpoint_path, before_path) &&
               write_text(lock_path, "") && chmod(lock_path, 0600) == 0 &&
               chmod(directory, 0500) == 0;
    check(prepared, "periodic resume failure fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_resume(directory,
                                           &execution,
                                           &result,
                                           &error),
              "periodic checkpoint write failure must reject resume");
        (void)chmod(directory, 0700);
        check(!path_exists(weights_path) &&
                  files_equal(checkpoint_path, before_path),
              "periodic failure must preserve the prior checkpoint");
    }
    cleanup_project(directory);
}

static void test_early_stopping_resume_selects_best(void)
{
    static const char *const directory = "build/tests/early-resume";
    NeuralExecutionConfig execution = {2U};
    volatile sig_atomic_t stop_request = SIGINT;
    NeuralTrainingResult result;
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralWeightsMetadata metadata;
    NeuralModel *model = NULL;
    NeuralError error;
    char checkpoint_path[256];
    char weights_path[256];
    int interrupted_signal = 0;
    int prepared = prepare_early_project(directory);

    project_path(checkpoint_path,
                 sizeof(checkpoint_path),
                 directory,
                 "checkpoint.txt");
    project_path(weights_path, sizeof(weights_path), directory, "weights.txt");
    check(prepared, "early-stopping resume fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_fresh_controlled(directory,
                                                     &execution,
                                                     &stop_request,
                                                     NULL,
                                                     NULL,
                                                     &interrupted_signal,
                                                     &result,
                                                     &error) &&
                  interrupted_signal == SIGINT &&
                  path_exists(checkpoint_path) &&
                  !path_exists(weights_path),
              "interrupted early stopping must save a resumable checkpoint");
        stop_request = 0;
        interrupted_signal = 0;
        check(neural_project_train_resume_controlled(directory,
                                                     &execution,
                                                     &stop_request,
                                                     NULL,
                                                     NULL,
                                                     &interrupted_signal,
                                                     &result,
                                                     &error) &&
                  result.completed_epochs == 4U &&
                  path_exists(weights_path) &&
                  !path_exists(checkpoint_path),
              "resumed early stopping must stop at the same patience boundary");
        check(neural_project_load(directory, &project, &error) &&
                  neural_project_digests_compute(&project, &digests, &error) &&
                  neural_model_create(&project.model,
                                      project.training.seed,
                                      &model,
                                      &error) &&
                  neural_weights_load(weights_path,
                                      model,
                                      &digests,
                                      &metadata,
                                      &error) &&
                  metadata.format_version == 2U &&
                  metadata.completed_epochs == 4U &&
                  metadata.selected_epoch == 1U &&
                  metadata.target_epochs == 20U &&
                  metadata.completion_reason ==
                      NEURAL_COMPLETION_EARLY_STOPPING,
              "final early-stopping weights must identify the selected epoch");
    }
    neural_model_free(model);
    neural_project_free(&project);
    cleanup_project(directory);
}

int main(void)
{
    test_resume_matches_continuous_training();
    test_interrupted_finalization_reconciliation();
    test_target_and_write_failures();
    test_additional_matches_continuous_training();
    test_additional_interruption_and_resume();
    test_additional_state_validation();
    test_early_stopping_resume_selects_best();

    if (failures != 0) {
        fprintf(stderr, "%d training-resume test(s) failed\n", failures);
        return 1;
    }
    puts("All training-resume tests passed");
    return 0;
}
