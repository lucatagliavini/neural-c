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
              project.preprocessing.format_version == 2U &&
              project.preprocessing.split_algorithm ==
                  NEURAL_SPLIT_GLOBAL_LARGEST_REMAINDER_V1 &&
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

static void test_small_global_stratified_split(void)
{
    const char *directory = "build/tests/data-import-small-split";
    NeuralDataImportConfig config = {
        "tests/fixtures/iris-schema.txt", 0.2, 0.2, 17U,
        NEURAL_NORMALIZATION_NONE, NEURAL_MISSING_REJECT
    };
    NeuralDataImportResult first;
    NeuralDataImportResult second;
    NeuralDataset validation = {0};
    NeuralDataset test = {0};
    NeuralError error;
    char validation_path[256];
    char test_path[256];
    char first_validation[2048];
    FILE *stream;
    size_t bytes;

    check(prepare(directory), "small split project must be prepared");
    check(neural_data_import_csv(directory,
                                 "tests/fixtures/iris-small-split.csv",
                                 &config,
                                 &first,
                                 &error) &&
              first.total_samples == 12U &&
              first.training_samples == 8U &&
              first.validation_samples == 2U &&
              first.test_samples == 2U,
          "global quotas must represent small stratified datasets");
    (void)snprintf(validation_path, sizeof(validation_path),
                   "%s/validation.txt", directory);
    (void)snprintf(test_path, sizeof(test_path), "%s/test.txt", directory);
    check(neural_dataset_load(validation_path, 4U, 3U, &validation, &error) &&
              validation.sample_count == 2U &&
              neural_dataset_load(test_path, 4U, 3U, &test, &error) &&
              test.sample_count == 2U,
          "small split subsets must remain valid native datasets");
    stream = fopen(validation_path, "rb");
    bytes = stream == NULL ? 0U :
        fread(first_validation, 1U, sizeof(first_validation), stream);
    if (stream != NULL) (void)fclose(stream);
    check(bytes != 0U && bytes < sizeof(first_validation),
          "small validation split must be readable for determinism check");
    check(neural_data_import_csv(directory,
                                 "tests/fixtures/iris-small-split.csv",
                                 &config,
                                 &second,
                                 &error) &&
              memcmp(&first, &second, sizeof(first)) == 0,
          "repeated split metadata must be deterministic");
    stream = fopen(validation_path, "rb");
    if (stream != NULL) {
        char repeated[2048];
        size_t repeated_bytes = fread(repeated, 1U, sizeof(repeated), stream);

        check(repeated_bytes == bytes &&
                  memcmp(repeated, first_validation, bytes) == 0,
              "repeated stratified assignment must be byte-identical");
        (void)fclose(stream);
    } else {
        check(0, "repeated validation split must be readable");
    }
    neural_dataset_free(&test);
    neural_dataset_free(&validation);
    cleanup(directory);
}

static void test_preprocessing_v1_compatibility(void)
{
    const char *path = "build/tests/preprocessing-v1.txt";
    NeuralPreprocessing preprocessing = {0};
    NeuralError error;
    FILE *stream;
    char header[128] = "";

    check(write_text(path,
        "neural-c preprocessing 1\n"
        "inputs 1\nnormalization none\nmissing reject\n"
        "source_digest sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "schema_digest sha256:1111111111111111111111111111111111111111111111111111111111111111\n"
        "split_seed 1\nvalidation_ratio 0\ntest_ratio 0\n"
        "stratified no\nfeature 0 offset 0 scale 1 impute 0\nend\n") &&
              neural_preprocessing_load(path, &preprocessing, &error) &&
              preprocessing.format_version == 1U &&
              preprocessing.split_algorithm ==
                  NEURAL_SPLIT_PER_CLASS_FLOOR_V1,
          "preprocessing version 1 must retain legacy split semantics");
    check(neural_preprocessing_save_atomic(path, &preprocessing, &error),
          "preprocessing version 1 must remain writable");
    stream = fopen(path, "r");
    if (stream != NULL) {
        if (fgets(header, sizeof(header), stream) == NULL) {
            header[0] = '\0';
        }
        (void)fclose(stream);
    }
    check(strcmp(header, "neural-c preprocessing 1\n") == 0,
          "version 1 preprocessing round trip must retain its header");
    neural_preprocessing_free(&preprocessing);
    check(write_text(path,
        "neural-c preprocessing 2\n"
        "inputs 1\nnormalization none\nmissing reject\n"
        "source_digest sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "schema_digest sha256:1111111111111111111111111111111111111111111111111111111111111111\n"
        "split_seed 1\nvalidation_ratio 0\ntest_ratio 0\n"
        "stratified no\nsplit_algorithm unknown\n"
        "feature 0 offset 0 scale 1 impute 0\nend\n") &&
              !neural_preprocessing_load(path, &preprocessing, &error) &&
              strstr(error.message, "split algorithm") != NULL,
          "version 2 preprocessing must reject unknown split algorithms");
    (void)remove(path);
}

static void test_imbalanced_split_training_reserve(void)
{
    const char *directory = "build/tests/data-import-imbalanced";
    NeuralDataImportConfig config = {
        "tests/fixtures/iris-schema.txt", 0.4, 0.3, 23U,
        NEURAL_NORMALIZATION_MINMAX, NEURAL_MISSING_REJECT
    };
    NeuralDataImportResult result;
    NeuralProject project = {0};
    NeuralError error;
    size_t class_counts[3] = {0U, 0U, 0U};
    size_t sample;

    check(prepare(directory), "imbalanced split project must be prepared");
    check(neural_data_import_csv(directory,
                                 "tests/fixtures/iris-imbalanced-split.csv",
                                 &config,
                                 &result,
                                 &error) &&
              result.training_samples == 5U &&
              result.validation_samples == 4U &&
              result.test_samples == 3U,
          "imbalanced split must meet exact global subset counts");
    check(neural_project_load(directory, &project, &error),
          "imbalanced imported project must reload");
    for (sample = 0U; sample < project.dataset.sample_count; sample++) {
        size_t output;

        for (output = 0U; output < 3U; output++) {
            if (project.dataset.outputs[sample * 3U + output] == 1.0) {
                class_counts[output]++;
            }
        }
    }
    check(class_counts[0] >= 1U && class_counts[1] >= 1U &&
              class_counts[2] >= 1U,
          "imbalanced split must reserve training data for every class");
    neural_project_free(&project);
    config.validation_ratio = 0.5;
    config.test_ratio = 0.4;
    check(!neural_data_import_csv(directory,
                                  "tests/fixtures/iris-imbalanced-split.csv",
                                  &config,
                                  &result,
                                  &error) &&
              strstr(error.message, "one training sample per class") != NULL,
          "infeasible holdout quotas must fail instead of dropping a class");
    check(neural_project_load(directory, &project, &error) &&
              project.dataset.sample_count == 5U,
          "failed infeasible split must preserve the prior imported project");
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
    test_small_global_stratified_split();
    test_preprocessing_v1_compatibility();
    test_imbalanced_split_training_reserve();
    test_reject_and_weights_guard();
    test_transaction_rollback();
    if (failures != 0) {
        fprintf(stderr, "%d data import test(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    puts("data import tests passed");
    return EXIT_SUCCESS;
}
