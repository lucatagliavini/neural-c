#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/data_import.h"
#include "neural/project.h"

static int failures;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int write_text(const char *path, const char *text)
{
    FILE *stream = fopen(path, "w");
    int success = stream != NULL && fputs(text, stream) != EOF;
    if (stream != NULL && fclose(stream) != 0) success = 0;
    return success;
}

static void cleanup(const char *directory)
{
    static const char *names[] = {
        "model.txt", "project.conf", "train.txt", "validation.txt",
        "test.txt", "preprocessing.txt", "weights.txt", ".neural-c.lock"
    };
    size_t index;
    for (index = 0U; index < sizeof(names) / sizeof(names[0]); index++) {
        char path[256];
        (void)snprintf(path, sizeof(path), "%s/%s", directory, names[index]);
        (void)remove(path);
        (void)rmdir(path);
    }
    (void)rmdir(directory);
}

static int prepare(const char *directory)
{
    char path[256];

    cleanup(directory);
    if (mkdir(directory, 0700) != 0) return 0;
    (void)snprintf(path, sizeof(path), "%s/model.txt", directory);
    if (!write_text(path,
                    "neural-c model 1\ninput 4\ndense 3 softmax\n")) return 0;
    (void)snprintf(path, sizeof(path), "%s/project.conf", directory);
    if (!write_text(path,
                    "neural-c project 1\nepochs 2\nlearning_rate 0.1\n"
                    "seed 42\nloss mse\ncheckpoint_interval 0\n"
                    "early_stopping_patience 0\n"
                    "early_stopping_min_delta 0\n")) return 0;
    (void)snprintf(path, sizeof(path), "%s/train.txt", directory);
    return write_text(path, "neural-c dataset 1\n0 0 0 0 -> 1 0 0\n");
}

static void test_import_split_and_preprocessing(void)
{
    const char *directory = "build/tests/data-import";
    NeuralDataImportConfig config = {
        "tests/fixtures/iris-schema.txt",
        1.0 / 6.0,
        1.0 / 6.0,
        42U,
        NEURAL_NORMALIZATION_STANDARDIZE,
        NEURAL_MISSING_MEAN
    };
    NeuralDataImportResult result;
    NeuralProject project = {0};
    NeuralDataset test = {0};
    NeuralError error;
    char test_path[256];
    neural_real missing[] = {NAN, NAN, NAN, NAN};

    check(prepare(directory), "CSV import project must be prepared");
    check(neural_data_import_csv(directory,
                                 "tests/fixtures/iris-small.csv",
                                 &config,
                                 &result,
                                 &error) &&
              result.total_samples == 18U &&
              result.training_samples == 12U &&
              result.validation_samples == 3U &&
              result.test_samples == 3U && result.stratified,
          "CSV import must create deterministic stratified subsets");
    check(neural_project_load(directory, &project, &error) &&
              project.dataset.sample_count == 12U &&
              project.has_preprocessing &&
              project.preprocessing.missing_policy == NEURAL_MISSING_MEAN &&
              project.preprocessing.normalization ==
                  NEURAL_NORMALIZATION_STANDARDIZE,
          "project must reload persisted training-only preprocessing");
    (void)snprintf(test_path, sizeof(test_path), "%s/test.txt", directory);
    check(neural_dataset_load(test_path, 4U, 3U, &test, &error) &&
              test.sample_count == 3U,
          "test subset must use the native finite dataset format");
    check(neural_preprocessing_apply(&project.preprocessing,
                                     missing,
                                     1U,
                                     &error) &&
              missing[0] == 0.0 && missing[1] == 0.0 &&
              missing[2] == 0.0 && missing[3] == 0.0,
          "mean imputation must be persisted and reused before normalization");
    neural_dataset_free(&test);
    neural_project_free(&project);
    cleanup(directory);
}

static void test_reject_and_weights_guard(void)
{
    const char *directory = "build/tests/data-import-errors";
    NeuralDataImportConfig config = {
        "tests/fixtures/iris-schema.txt", 0.0, 0.0, 1U,
        NEURAL_NORMALIZATION_NONE, NEURAL_MISSING_REJECT
    };
    NeuralDataImportResult result;
    NeuralError error;
    char path[256];

    check(prepare(directory), "CSV import error project must be prepared");
    check(!neural_data_import_csv(directory,
                                  "tests/fixtures/iris-small.csv",
                                  &config,
                                  &result,
                                  &error) &&
              strstr(error.message, "missing training input") != NULL,
          "reject policy must fail on an explicit missing value");
    config.missing_policy = NEURAL_MISSING_MEAN;
    (void)snprintf(path, sizeof(path), "%s/weights.txt", directory);
    check(write_text(path, "not weights\n") &&
              !neural_data_import_csv(directory,
                                      "tests/fixtures/iris-small.csv",
                                      &config,
                                      &result,
                                      &error) &&
              strstr(error.message, "invalidate weights") != NULL,
          "import must refuse to invalidate a trained project silently");
    cleanup(directory);
}

static void test_transaction_rollback(void)
{
    const char *directory = "build/tests/data-import-rollback";
    NeuralDataImportConfig config = {
        "tests/fixtures/iris-schema.txt", 1.0 / 6.0, 0.0, 9U,
        NEURAL_NORMALIZATION_STANDARDIZE, NEURAL_MISSING_MEAN
    };
    NeuralDataImportResult result;
    NeuralDataset original = {0};
    NeuralError error;
    char path[256];

    check(prepare(directory), "rollback project must be prepared");
    (void)snprintf(path, sizeof(path), "%s/validation.txt", directory);
    check(mkdir(path, 0700) == 0,
          "rollback fixture must block validation replacement");
    check(!neural_data_import_csv(directory,
                                  "tests/fixtures/iris-small.csv",
                                  &config,
                                  &result,
                                  &error),
          "multi-file import must fail when a destination is not regular");
    (void)snprintf(path, sizeof(path), "%s/train.txt", directory);
    check(neural_dataset_load(path, 4U, 3U, &original, &error) &&
              original.sample_count == 1U && original.inputs[0] == 0.0,
          "failed import must restore the original training dataset");
    neural_dataset_free(&original);
    cleanup(directory);
}

int main(void)
{
    test_import_split_and_preprocessing();
    test_reject_and_weights_guard();
    test_transaction_rollback();
    if (failures != 0) {
        fprintf(stderr, "%d data import test(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    puts("data import tests passed");
    return EXIT_SUCCESS;
}
