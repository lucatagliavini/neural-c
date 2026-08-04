#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/digest.h"
#include "neural/model.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "../src/project_checkpoint.h"
#include "../src/train_project.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int path_exists(const char *path)
{
    struct stat status;

    return lstat(path, &status) == 0;
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
        fclose(source);
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
    if (fclose(destination) != 0) {
        success = 0;
    }
    if (fclose(source) != 0) {
        success = 0;
    }
    return success;
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

static void test_interval_boundaries_and_payload(void)
{
    static const char *const path =
        "build/tests/periodic-checkpoint.txt";
    static const char *const comparison_path =
        "build/tests/periodic-checkpoint-comparison.txt";
    static const char *const unchanged_path =
        "build/tests/periodic-checkpoint-unchanged.txt";
    static const char *const disabled_path =
        "build/tests/periodic-checkpoint-disabled.txt";
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralProjectCheckpointObserver observer;
    NeuralProjectCheckpointObserver comparison_observer;
    NeuralProjectCheckpointObserver disabled_observer;
    NeuralCheckpointMetadata loaded;
    NeuralModel *model = NULL;
    NeuralModel *loaded_model = NULL;
    NeuralEpochReport report = {0};
    volatile sig_atomic_t stop_request = 0;
    NeuralError error;
    int prepared;

    report.completed_epochs = 1U;
    report.loss = 0.25;
    report.target_epochs = 5U;

    (void)remove(path);
    (void)remove(comparison_path);
    (void)remove(unchanged_path);
    (void)remove(disabled_path);
    prepared = neural_project_load("projects/xor", &project, &error) &&
               neural_project_digests_compute(&project, &digests, &error) &&
               neural_model_create(&project.model,
                                   project.training.seed,
                                   &model,
                                   &error) &&
               neural_model_create(&project.model,
                                   project.training.seed + UINT64_C(1),
                                   &loaded_model,
                                   &error) &&
               neural_project_checkpoint_observer_initialize(
                   &observer,
                   path,
                   model,
                   &digests,
                   2U,
                   5U,
                   &error);
    check(prepared, "periodic checkpoint fixture must be prepared");
    if (prepared) {
        check(neural_project_checkpoint_observe(&report, &observer, &error) &&
                  !path_exists(path),
              "epochs outside the interval must not create a checkpoint");

        report.completed_epochs = 2U;
        check(neural_project_checkpoint_observe(&report, &observer, &error) &&
                  path_exists(path),
              "an interval boundary must create a checkpoint");
        check(neural_checkpoint_load(path,
                                     loaded_model,
                                     &digests,
                                     &loaded,
                                     &error) &&
                  loaded.completed_epochs == 2U &&
                  loaded.target_epochs == 5U &&
                  loaded.optimizer == NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
                  loaded.rng_state == neural_model_random_state(model),
              "checkpoint metadata must describe the observed epoch");

        check(copy_file(path, unchanged_path),
              "checkpoint comparison copy must be created");
        report.completed_epochs = 3U;
        check(neural_project_checkpoint_observe(&report, &observer, &error) &&
                  files_equal(path, unchanged_path),
              "a non-boundary report must leave the checkpoint unchanged");

        report.completed_epochs = 4U;
        check(neural_project_checkpoint_observe(&report, &observer, &error) &&
                  neural_checkpoint_load(path,
                                         loaded_model,
                                         &digests,
                                         &loaded,
                                         &error) &&
                  loaded.completed_epochs == 4U,
              "later interval boundaries must atomically replace metadata");
        check(neural_project_checkpoint_observer_initialize(
                  &comparison_observer,
                  comparison_path,
                  model,
                  &digests,
                  2U,
                  5U,
                  &error) &&
                  neural_project_checkpoint_observe(&report,
                                                    &comparison_observer,
                                                    &error) &&
                  files_equal(path, comparison_path),
              "equivalent checkpoint observations must be byte deterministic");

        report.completed_epochs = 5U;
        check(neural_project_checkpoint_observer_initialize(
                  &disabled_observer,
                  disabled_path,
                  model,
                  &digests,
                  0U,
                  5U,
                  &error) &&
                  neural_project_checkpoint_observe(&report,
                                                    &disabled_observer,
                                                    &error) &&
                  !path_exists(disabled_path),
              "zero interval must perform no periodic persistence");
        stop_request = SIGTERM;
        neural_project_checkpoint_observer_set_stop_request(
            &disabled_observer,
            &stop_request);
        check(!neural_project_checkpoint_observe(&report,
                                                 &disabled_observer,
                                                 &error) &&
                  disabled_observer.interrupted_signal == SIGTERM &&
                  path_exists(disabled_path) &&
                  strstr(error.message, "checkpoint saved") != NULL &&
                  neural_checkpoint_load(disabled_path,
                                         loaded_model,
                                         &digests,
                                         &loaded,
                                         &error) &&
                  loaded.completed_epochs == 5U,
              "a stop request must force one checkpoint when periodic saves "
              "are disabled");

        report.completed_epochs = 6U;
        check(!neural_project_checkpoint_observe(&report, &observer, &error) &&
                  strstr(error.message, "report is invalid") != NULL,
              "reports beyond the target must be rejected");
    }
    neural_model_free(loaded_model);
    neural_model_free(model);
    neural_project_free(&project);
    (void)remove(disabled_path);
    (void)remove(unchanged_path);
    (void)remove(comparison_path);
    (void)remove(path);
}

static void test_write_failure_preserves_prior_checkpoint(void)
{
    static const char *const directory =
        "build/tests/checkpoint-write-failure";
    static const char *const path =
        "build/tests/checkpoint-write-failure/checkpoint.txt";
    static const char *const prior_path =
        "build/tests/checkpoint-write-failure/checkpoint.prior";
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralProjectCheckpointObserver observer;
    NeuralModel *model = NULL;
    NeuralEpochReport report = {0};
    NeuralError error;
    int prepared;

    report.completed_epochs = 2U;
    report.loss = 0.25;
    report.target_epochs = 4U;

    (void)chmod(directory, 0700);
    (void)remove(prior_path);
    (void)remove(path);
    if (mkdir(directory, 0700) != 0 && errno != EEXIST) {
        check(0, "checkpoint failure directory must be created");
        return;
    }
    prepared = neural_project_load("projects/xor", &project, &error) &&
               neural_project_digests_compute(&project, &digests, &error) &&
               neural_model_create(&project.model,
                                   project.training.seed,
                                   &model,
                                   &error) &&
               neural_project_checkpoint_observer_initialize(
                   &observer,
                   path,
                   model,
                   &digests,
                   2U,
                   4U,
                   &error) &&
               neural_project_checkpoint_observe(&report, &observer, &error) &&
               copy_file(path, prior_path);
    check(prepared, "prior checkpoint fixture must be prepared");
    if (prepared) {
        check(chmod(directory, 0500) == 0,
              "checkpoint directory must become read-only");
        report.completed_epochs = 4U;
        check(!neural_project_checkpoint_observe(&report, &observer, &error),
              "checkpoint write errors must reject the epoch report");
        check(chmod(directory, 0700) == 0,
              "checkpoint directory permissions must be restored");
        check(files_equal(path, prior_path),
              "a pre-rename write failure must preserve the prior checkpoint");
    }
    (void)chmod(directory, 0700);
    neural_model_free(model);
    neural_project_free(&project);
    (void)remove(prior_path);
    (void)remove(path);
    (void)rmdir(directory);
}

static void test_payload_is_deterministic_across_workers(void)
{
    static const char *const serial_path =
        "build/tests/checkpoint-serial.txt";
    static const char *const parallel_path =
        "build/tests/checkpoint-parallel.txt";
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralProjectCheckpointObserver serial_observer;
    NeuralProjectCheckpointObserver parallel_observer;
    NeuralExecutionConfig serial_execution = {1U};
    NeuralExecutionConfig parallel_execution = {4U};
    NeuralTrainingResult serial_result;
    NeuralTrainingResult parallel_result;
    NeuralModel *serial_model = NULL;
    NeuralModel *parallel_model = NULL;
    NeuralError error;
    int prepared;

    (void)remove(serial_path);
    (void)remove(parallel_path);
    prepared = neural_project_load("projects/xor", &project, &error);
    if (prepared) {
        project.training.epochs = 2U;
        project.training.checkpoint_interval = 2U;
        prepared = neural_project_digests_compute(&project, &digests, &error) &&
                   neural_model_create(&project.model,
                                       project.training.seed,
                                       &serial_model,
                                       &error) &&
                   neural_model_create(&project.model,
                                       project.training.seed,
                                       &parallel_model,
                                       &error) &&
                   neural_project_checkpoint_observer_initialize(
                       &serial_observer,
                       serial_path,
                       serial_model,
                       &digests,
                       project.training.checkpoint_interval,
                       project.training.epochs,
                       &error) &&
                   neural_project_checkpoint_observer_initialize(
                       &parallel_observer,
                       parallel_path,
                       parallel_model,
                       &digests,
                       project.training.checkpoint_interval,
                       project.training.epochs,
                       &error);
    }
    check(prepared, "worker-determinism checkpoint fixture must be prepared");
    if (prepared) {
        check(neural_model_train(
                  serial_model,
                  &project.dataset,
                  &project.training,
                  &serial_execution,
                  neural_project_checkpoint_observe,
                  &serial_observer,
                  &serial_result,
                  &error) &&
                  neural_model_train(
                      parallel_model,
                      &project.dataset,
                      &project.training,
                      &parallel_execution,
                      neural_project_checkpoint_observe,
                      &parallel_observer,
                      &parallel_result,
                      &error) &&
                  files_equal(serial_path, parallel_path),
              "checkpoint payloads must be byte-identical across workers");
    }
    neural_model_free(parallel_model);
    neural_model_free(serial_model);
    neural_project_free(&project);
    (void)remove(parallel_path);
    (void)remove(serial_path);
}

static void test_project_checkpoint_modes(void)
{
    static const char *const directory =
        "build/tests/checkpoint-project-failure";
    static const char *const model_path =
        "build/tests/checkpoint-project-failure/model.txt";
    static const char *const config_path =
        "build/tests/checkpoint-project-failure/project.conf";
    static const char *const dataset_path =
        "build/tests/checkpoint-project-failure/train.txt";
    static const char *const checkpoint_path =
        "build/tests/checkpoint-project-failure/checkpoint.txt";
    static const char *const weights_path =
        "build/tests/checkpoint-project-failure/weights.txt";
    static const char *const lock_path =
        "build/tests/checkpoint-project-failure/.neural-c.lock";
    static const char model_text[] =
        "neural-c model 1\n\ninput 1\ndense 1 linear\n";
    static const char config_text[] =
        "neural-c project 1\n\nepochs 2\nlearning_rate 0.1\n"
        "seed 7\nloss mse\ncheckpoint_interval 1\n";
    static const char disabled_config_text[] =
        "neural-c project 1\n\nepochs 2\nlearning_rate 0.1\n"
        "seed 7\nloss mse\ncheckpoint_interval 0\n";
    static const char dataset_text[] =
        "neural-c dataset 1\n\n1 -> 0\n";
    NeuralExecutionConfig execution = {1U};
    NeuralTrainingResult result;
    NeuralError error;
    int prepared;

    (void)chmod(directory, 0700);
    (void)remove(weights_path);
    (void)remove(checkpoint_path);
    (void)remove(lock_path);
    (void)remove(dataset_path);
    (void)remove(config_path);
    (void)remove(model_path);
    if (mkdir(directory, 0700) != 0 && errno != EEXIST) {
        check(0, "project failure directory must be created");
        return;
    }
    prepared = write_text(model_path, model_text) &&
               write_text(config_path, config_text) &&
               write_text(dataset_path, dataset_text) &&
               write_text(lock_path, "") &&
               chmod(directory, 0500) == 0;
    check(prepared, "project checkpoint failure fixture must be prepared");
    if (prepared) {
        check(!neural_project_train_fresh(directory,
                                          &execution,
                                          &result,
                                          &error) &&
                  strstr(error.message,
                         "unable to create temporary file") != NULL,
              "project training must propagate checkpoint write failures");
        check(!path_exists(weights_path) && !path_exists(checkpoint_path),
              "checkpoint failure must not create final or partial state");
    }
    (void)chmod(directory, 0700);
    if (prepared) {
        check(write_text(config_path, disabled_config_text) &&
                  neural_project_train_fresh(directory,
                                             &execution,
                                             &result,
                                             &error) &&
                  result.completed_epochs == 2U &&
                  path_exists(weights_path) &&
                  !path_exists(checkpoint_path),
              "disabled periodic checkpoints must write only final weights");
    }
    (void)remove(weights_path);
    (void)remove(checkpoint_path);
    (void)remove(lock_path);
    (void)remove(dataset_path);
    (void)remove(config_path);
    (void)remove(model_path);
    (void)rmdir(directory);
}

int main(void)
{
    test_interval_boundaries_and_payload();
    test_write_failure_preserves_prior_checkpoint();
    test_payload_is_deterministic_across_workers();
    test_project_checkpoint_modes();

    if (failures != 0) {
        fprintf(stderr, "%d checkpoint-observer test(s) failed\n", failures);
        return 1;
    }
    puts("All checkpoint-observer tests passed");
    return 0;
}
