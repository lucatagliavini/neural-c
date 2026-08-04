#define _POSIX_C_SOURCE 200809L

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/digest.h"
#include "neural/model.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "../src/predict_project.h"
#include "../src/project_lock.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
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

static void cleanup_project(const char *directory)
{
    static const char *const names[] = {
        "model.txt",
        "project.conf",
        "train.txt",
        "weights.txt",
        "checkpoint.txt",
        ".neural-c.lock"
    };
    size_t index;

    for (index = 0U; index < sizeof(names) / sizeof(names[0]); index++) {
        char path[256];

        (void)snprintf(path, sizeof(path), "%s/%s", directory, names[index]);
        (void)remove(path);
    }
    (void)rmdir(directory);
}

static int prepare_project(const char *directory, size_t completed_epochs)
{
    static const char model_text[] =
        "neural-c model 1\n\ninput 2\ndense 1 linear\n";
    static const char config_text[] =
        "neural-c project 1\n\nepochs 1\nlearning_rate 0.25\n"
        "seed 42\nloss mse\ncheckpoint_interval 0\n";
    static const char dataset_text[] =
        "neural-c dataset 1\n\n0 0 -> 0\n";
    static const neural_real layer_weights[] = {2.0, -1.0};
    static const neural_real layer_biases[] = {0.5};
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralWeightsMetadata metadata;
    NeuralModel *model = NULL;
    NeuralError error;
    char model_path[256];
    char config_path[256];
    char dataset_path[256];
    char weights_path[256];
    int success = 0;

    cleanup_project(directory);
    if (mkdir(directory, 0700) != 0) {
        return 0;
    }
    (void)snprintf(model_path,
                   sizeof(model_path),
                   "%s/model.txt",
                   directory);
    (void)snprintf(config_path,
                   sizeof(config_path),
                   "%s/project.conf",
                   directory);
    (void)snprintf(dataset_path,
                   sizeof(dataset_path),
                   "%s/train.txt",
                   directory);
    (void)snprintf(weights_path,
                   sizeof(weights_path),
                   "%s/weights.txt",
                   directory);
    if (!write_text(model_path, model_text) ||
        !write_text(config_path, config_text) ||
        !write_text(dataset_path, dataset_text) ||
        !neural_project_load(directory, &project, &error) ||
        !neural_project_digests_compute(&project, &digests, &error) ||
        !neural_model_create(&project.model,
                             project.training.seed,
                             &model,
                             &error) ||
        !neural_model_set_layer_parameters(model,
                                           0U,
                                           layer_weights,
                                           2U,
                                           layer_biases,
                                           1U,
                                           &error)) {
        goto cleanup;
    }
    metadata.completed_epochs = completed_epochs;
    metadata.digests = digests;
    success = neural_weights_save_atomic(weights_path,
                                         model,
                                         &metadata,
                                         &error);

cleanup:
    neural_model_free(model);
    neural_project_free(&project);
    return success;
}

static void test_prediction_snapshot_and_parallel_execution(void)
{
    static const char *const directory = "build/tests/prediction";
    static const neural_real inputs[] = {0.0, 0.0, 1.0, 0.0, 1.0, 3.0};
    static const neural_real expected[] = {0.5, 2.5, -0.5};
    NeuralPredictionSnapshot snapshot =
        NEURAL_PREDICTION_SNAPSHOT_INITIALIZER;
    NeuralExecutionConfig sequential = {1U};
    NeuralExecutionConfig parallel = {8U};
    NeuralProjectLock writer = NEURAL_PROJECT_LOCK_INITIALIZER;
    neural_real sequential_outputs[3] = {0.0, 0.0, 0.0};
    neural_real parallel_outputs[3] = {0.0, 0.0, 0.0};
    NeuralError error;
    size_t worker_count = 0U;
    int prepared = prepare_project(directory, 1U);

    check(prepared, "prediction fixture must be prepared");
    if (prepared) {
        check(neural_project_prediction_load(directory, &snapshot, &error) &&
                  snapshot.input_count == 2U &&
                  snapshot.output_count == 1U &&
                  snapshot.completed_epochs == 1U,
              "prediction must load a validated immutable snapshot");
        check(neural_project_lock_acquire(directory,
                                          NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                          &writer,
                                          &error),
              "prediction loading must release its shared project lock");
        check(neural_prediction_run(&snapshot,
                                    inputs,
                                    3U,
                                    &sequential,
                                    sequential_outputs,
                                    &worker_count,
                                    &error) &&
                  worker_count == 1U &&
                  memcmp(sequential_outputs,
                         expected,
                         sizeof(expected)) == 0,
              "snapshot inference must continue from memory after lock release");
        neural_project_lock_release(&writer);
        check(neural_prediction_run(&snapshot,
                                    inputs,
                                    3U,
                                    &parallel,
                                    parallel_outputs,
                                    &worker_count,
                                    &error) &&
                  worker_count == 3U &&
                  memcmp(parallel_outputs,
                         sequential_outputs,
                         sizeof(parallel_outputs)) == 0,
              "parallel prediction must preserve sample order and exact output");
    }
    neural_project_lock_release(&writer);
    neural_prediction_snapshot_free(&snapshot);
    cleanup_project(directory);
}

static void test_prediction_validation_and_errors(void)
{
    static const char *const directory = "build/tests/prediction-errors";
    NeuralPredictionSnapshot snapshot =
        NEURAL_PREDICTION_SNAPSHOT_INITIALIZER;
    NeuralExecutionConfig execution = {2U};
    NeuralProjectLock writer = NEURAL_PROJECT_LOCK_INITIALIZER;
    neural_real failing_inputs[] = {0.0, 0.0, DBL_MAX, 0.0};
    neural_real outputs[2] = {0.0, 0.0};
    NeuralError error;
    size_t worker_count = 99U;
    char weights_path[256];
    char config_path[256];
    int prepared;

    (void)snprintf(weights_path,
                   sizeof(weights_path),
                   "%s/weights.txt",
                   directory);
    (void)snprintf(config_path,
                   sizeof(config_path),
                   "%s/project.conf",
                   directory);

    prepared = prepare_project(directory, 1U);
    check(prepared, "prediction error fixture must be prepared");
    if (prepared) {
        check(neural_project_lock_acquire(directory,
                                          NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                          &writer,
                                          &error) &&
                  !neural_project_prediction_load(directory,
                                                  &snapshot,
                                                  &error) &&
                  strstr(error.message, "project is busy") != NULL,
              "prediction must fail immediately while a writer holds the lock");
        neural_project_lock_release(&writer);
        check(neural_project_prediction_load(directory, &snapshot, &error) &&
                  !neural_prediction_run(&snapshot,
                                         failing_inputs,
                                         2U,
                                         &execution,
                                         outputs,
                                         &worker_count,
                                         &error) &&
                  worker_count == 0U &&
                  strstr(error.message, "prediction sample 1 failed") != NULL,
              "prediction must report the lowest failing sample deterministically");
        neural_prediction_snapshot_free(&snapshot);
        (void)remove(weights_path);
        check(!neural_project_prediction_load(directory, &snapshot, &error) &&
                  strstr(error.message, "weights.txt") != NULL,
              "prediction must require finalized weights");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 0U);
    check(prepared, "incomplete prediction weights fixture must be prepared");
    if (prepared) {
        check(!neural_project_prediction_load(directory, &snapshot, &error) &&
                  strstr(error.message, "precede configured epochs") != NULL,
              "prediction must reject incomplete final weights");
    }
    cleanup_project(directory);

    prepared = prepare_project(directory, 1U) &&
               write_text(config_path,
                          "neural-c project 1\n\nepochs 1\n"
                          "learning_rate 0.5\nseed 42\nloss mse\n"
                          "checkpoint_interval 0\n");
    check(prepared, "prediction digest mismatch fixture must be prepared");
    if (prepared) {
        check(!neural_project_prediction_load(directory, &snapshot, &error) &&
                  strstr(error.message, "digest does not match") != NULL,
              "prediction must reject incompatible project provenance");
    }
    neural_prediction_snapshot_free(&snapshot);
    cleanup_project(directory);
}

int main(void)
{
    test_prediction_snapshot_and_parallel_execution();
    test_prediction_validation_and_errors();

    if (failures != 0) {
        fprintf(stderr, "%d prediction test(s) failed\n", failures);
        return 1;
    }
    puts("All prediction tests passed");
    return 0;
}
