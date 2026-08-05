#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural/digest.h"
#include "neural/model.h"
#include "neural/persistence.h"
#include "neural/project.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static void check_error_contains(const NeuralError *error,
                                 const char *expected,
                                 const char *description)
{
    check(strstr(error->message, expected) != NULL, description);
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
        const neural_real *left_weights;
        const neural_real *right_weights;
        const neural_real *left_biases;
        const neural_real *right_biases;
        size_t left_weight_count;
        size_t right_weight_count;
        size_t left_bias_count;
        size_t right_bias_count;

        left_weights = neural_model_layer_weights(left,
                                                  layer_index,
                                                  &left_weight_count);
        right_weights = neural_model_layer_weights(right,
                                                   layer_index,
                                                   &right_weight_count);
        left_biases = neural_model_layer_biases(left,
                                                layer_index,
                                                &left_bias_count);
        right_biases = neural_model_layer_biases(right,
                                                 layer_index,
                                                 &right_bias_count);
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

static int copy_replacing_line(const char *source_path,
                               const char *destination_path,
                               const char *prefix,
                               const char *replacement)
{
    FILE *source = fopen(source_path, "r");
    FILE *destination;
    char *line = NULL;
    size_t capacity = 0U;
    int replaced = 0;
    int success = 0;

    if (source == NULL) {
        return 0;
    }
    destination = fopen(destination_path, "w");
    if (destination == NULL) {
        (void)fclose(source);
        return 0;
    }
    while (getline(&line, &capacity, source) >= 0) {
        if (!replaced && strncmp(line, prefix, strlen(prefix)) == 0) {
            if (fprintf(destination, "%s\n", replacement) < 0) {
                goto cleanup;
            }
            replaced = 1;
        } else if (fputs(line, destination) == EOF) {
            goto cleanup;
        }
    }
    if (ferror(source) != 0 || !replaced) {
        goto cleanup;
    }
    success = 1;

cleanup:
    free(line);
    if (fclose(source) != 0) {
        success = 0;
    }
    if (fclose(destination) != 0) {
        success = 0;
    }
    return success;
}

static int append_text(const char *path, const char *text)
{
    FILE *stream = fopen(path, "a");
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

static int copy_replacing_after(const char *source_path,
                                const char *destination_path,
                                const char *marker,
                                const char *replacement)
{
    FILE *source = fopen(source_path, "r");
    FILE *destination;
    char *line = NULL;
    size_t capacity = 0U;
    int replace_next = 0;
    int replaced = 0;
    int success = 0;

    if (source == NULL) {
        return 0;
    }
    destination = fopen(destination_path, "w");
    if (destination == NULL) {
        (void)fclose(source);
        return 0;
    }
    while (getline(&line, &capacity, source) >= 0) {
        if (replace_next) {
            if (fprintf(destination, "%s\n", replacement) < 0) {
                goto cleanup;
            }
            replace_next = 0;
            replaced = 1;
        } else {
            if (fputs(line, destination) == EOF) {
                goto cleanup;
            }
            if (!replaced && strcmp(line, marker) == 0) {
                replace_next = 1;
            }
        }
    }
    if (ferror(source) != 0 || !replaced) {
        goto cleanup;
    }
    success = 1;

cleanup:
    free(line);
    if (fclose(source) != 0) {
        success = 0;
    }
    if (fclose(destination) != 0) {
        success = 0;
    }
    return success;
}

static void test_sha256(void)
{
    char digest[NEURAL_SHA256_TEXT_CAPACITY];
    NeuralError error;

    check(neural_sha256_hex(NULL, 0U, digest, &error),
          "SHA-256 must accept an empty input");
    check(strcmp(digest,
                 "e3b0c44298fc1c149afbf4c8996fb924"
                 "27ae41e4649b934ca495991b7852b855") == 0,
          "SHA-256 empty-input vector must match");
    check(neural_sha256_hex("abc", 3U, digest, &error),
          "SHA-256 must hash a byte buffer");
    check(strcmp(digest,
                 "ba7816bf8f01cfea414140de5dae2223"
                 "b00361a396177a9cb410ff61f20015ad") == 0,
          "SHA-256 abc vector must match");
}

static void test_project_digests(NeuralProject *project,
                                 NeuralProjectDigests *digests)
{
    NeuralProjectDigests repeated;
    NeuralProjectDigests changed;
    neural_real original_input;
    size_t original_epochs;
    size_t original_batch_size;
    int original_shuffle;
    neural_real original_gradient_clip_norm;
    NeuralLoss original_loss;
    NeuralError error;

    check(neural_project_digests_compute(project, digests, &error),
          "valid project digests must compute");
    check(neural_project_digests_compute(project, &repeated, &error),
          "project digests must be repeatable");
    check(memcmp(digests, &repeated, sizeof(*digests)) == 0,
          "project digests must be deterministic");
    check(strcmp(digests->model,
                 "8da5a1f53fc59cabe5e685895a6e254d"
                 "6b362ff4704f3d6b7325d98b7e4cc0d5") == 0 &&
              strcmp(digests->dataset,
                     "df97fb50ab811253bc1ebfceb93dea86"
                     "79f5c337bb9995f384f47a7936991275") == 0 &&
              strcmp(digests->training,
                     "08778dfaba9557e47d924f3010e4b972"
                     "935e718b994c3a65eb2f0aac5de4ab6d") == 0,
          "XOR canonical digests must remain architecture-independent");

    original_input = project->dataset.inputs[0];
    project->dataset.inputs[0] = original_input + 0.25;
    check(neural_project_digests_compute(project, &changed, &error) &&
              strcmp(changed.dataset, digests->dataset) != 0 &&
              strcmp(changed.model, digests->model) == 0,
          "dataset changes must affect only its canonical digest");
    project->dataset.inputs[0] = original_input;

    original_epochs = project->training.epochs;
    project->training.epochs++;
    check(neural_project_digests_compute(project, &changed, &error) &&
              strcmp(changed.training, digests->training) != 0 &&
              strcmp(changed.dataset, digests->dataset) == 0,
          "training changes must affect only its canonical digest");
    project->training.epochs = original_epochs;

    original_batch_size = project->training.batch_size;
    project->training.batch_size = 3U;
    check(neural_project_digests_compute(project, &changed, &error) &&
              strcmp(changed.training, digests->training) != 0 &&
              strcmp(changed.dataset, digests->dataset) == 0 &&
              strcmp(changed.model, digests->model) == 0,
          "mini-batch size must affect only training provenance");
    project->training.batch_size = original_batch_size;

    original_shuffle = project->training.shuffle;
    project->training.shuffle = 1;
    check(neural_project_digests_compute(project, &changed, &error) &&
              strcmp(changed.training, digests->training) != 0 &&
              strcmp(changed.dataset, digests->dataset) == 0 &&
              strcmp(changed.model, digests->model) == 0,
          "epoch shuffle must affect only training provenance");
    project->training.shuffle = original_shuffle;

    original_gradient_clip_norm = project->training.gradient_clip_norm;
    project->training.gradient_clip_norm = 0.75;
    check(neural_project_digests_compute(project, &changed, &error) &&
              strcmp(changed.training, digests->training) != 0 &&
              strcmp(changed.dataset, digests->dataset) == 0 &&
              strcmp(changed.model, digests->model) == 0,
          "gradient clipping must affect only training provenance");
    project->training.gradient_clip_norm = original_gradient_clip_norm;

    original_loss = project->training.loss;
    project->training.loss = NEURAL_LOSS_BINARY_CROSS_ENTROPY;
    check(neural_project_digests_compute(project, &changed, &error) &&
              strcmp(changed.training, digests->training) != 0,
          "a compatible cross-entropy must change training provenance");
    project->training.loss = NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY;
    check(!neural_project_digests_compute(project, &changed, &error) &&
              strstr(error.message, "softmax") != NULL,
          "digest computation must reject incompatible loss contracts");
    project->training.loss = original_loss;
}

static void test_weights_round_trip(const NeuralProject *project,
                                    const NeuralProjectDigests *digests)
{
    static const char *const path = "build/tests/persistence-weights.txt";
    static const char *const corrupt_path =
        "build/tests/persistence-weights-corrupt.txt";
    NeuralWeightsMetadata saved = {0};
    NeuralWeightsMetadata loaded;
    NeuralModel *source = NULL;
    NeuralModel *destination = NULL;
    NeuralModel *unchanged = NULL;
    NeuralModel *unchanged_reference = NULL;
    NeuralProjectDigests wrong_digests;
    NeuralError error;

    saved.completed_epochs = 10000U;
    saved.digests = *digests;
    (void)remove(path);
    (void)remove(corrupt_path);
    check(neural_model_create(&project->model,
                              project->training.seed,
                              &source,
                              &error),
          "source model must be created for weights persistence");
    check(neural_model_create(&project->model,
                              project->training.seed + UINT64_C(1),
                              &destination,
                              &error),
          "destination model must be created for weights persistence");
    check(append_text(path, "obsolete\n"),
          "weights replacement fixture must be created");
    check(neural_weights_save_atomic(path, source, &saved, &error),
          "weights must save atomically");
    check(neural_weights_load(path,
                              destination,
                              digests,
                              &loaded,
                              &error),
          "valid weights must load");
    check(models_have_equal_parameters(source, destination),
          "weights must round-trip bit exactly");
    check(loaded.completed_epochs == saved.completed_epochs &&
              memcmp(&loaded.digests,
                     &saved.digests,
                     sizeof(loaded.digests)) == 0,
          "weights metadata must round-trip");

    check(neural_model_create(&project->model,
                              project->training.seed + UINT64_C(2),
                              &unchanged,
                              &error),
          "comparison model must be created");
    check(neural_model_create(&project->model,
                              project->training.seed + UINT64_C(2),
                              &unchanged_reference,
                              &error),
          "comparison reference model must be created");
    wrong_digests = *digests;
    wrong_digests.model[0] = wrong_digests.model[0] == '0' ? '1' : '0';
    check(!neural_weights_load(path,
                               unchanged,
                               &wrong_digests,
                               &loaded,
                               &error),
          "weights with a mismatched model digest must be rejected");
    check_error_contains(&error,
                         "model digest does not match",
                         "digest mismatch must be actionable");
    check(models_have_equal_parameters(unchanged, unchanged_reference),
          "digest failures must leave model parameters unchanged");

    check(copy_replacing_line(path,
                              corrupt_path,
                              "weights\n",
                              "weights"),
          "corrupt weights fixture must be copied");
    check(append_text(corrupt_path, "unexpected\n"),
          "corrupt weights fixture must gain trailing content");
    check(!neural_weights_load(corrupt_path,
                               unchanged,
                               digests,
                               &loaded,
                               &error),
          "trailing persistence content must be rejected");
    check_error_contains(&error,
                         "unexpected content after end",
                         "trailing-content error must be actionable");
    check(models_have_equal_parameters(unchanged, unchanged_reference),
          "parse failures must not partially replace model parameters");

    check(copy_replacing_after(path,
                               corrupt_path,
                               "weights\n",
                               "nan"),
          "non-finite weights fixture must be created");
    check(!neural_weights_load(corrupt_path,
                               unchanged,
                               digests,
                               &loaded,
                               &error),
          "non-finite persisted weights must be rejected");
    check_error_contains(&error,
                         "invalid finite weight value",
                         "non-finite weight error must be actionable");
    check(models_have_equal_parameters(unchanged, unchanged_reference),
          "non-finite values must not mutate model parameters");

    check(copy_replacing_line(path,
                              corrupt_path,
                              "end_layer\n",
                              "0"),
          "excess parameter fixture must be created");
    check(!neural_weights_load(corrupt_path,
                               unchanged,
                               digests,
                               &loaded,
                               &error),
          "excess persisted parameters must be rejected");
    check_error_contains(&error,
                         "expected 'end_layer'",
                         "excess parameter error must identify layer boundary");
    check(models_have_equal_parameters(unchanged, unchanged_reference),
          "dimension failures must not mutate model parameters");

    check(copy_replacing_line(path,
                              corrupt_path,
                              "neural-c weights ",
                              "neural-c weights 3"),
          "unsupported-version fixture must be created");
    check(!neural_weights_load(corrupt_path,
                               unchanged,
                               digests,
                               &loaded,
                               &error),
          "unsupported persistence versions must be rejected");
    check_error_contains(&error,
                         "expected 'neural-c weights 1'",
                         "unsupported-version error must be actionable");

    neural_model_free(unchanged_reference);
    neural_model_free(unchanged);
    neural_model_free(destination);
    neural_model_free(source);
    (void)remove(corrupt_path);
    (void)remove(path);
}

static void test_checkpoint_round_trip(const NeuralProject *project,
                                       const NeuralProjectDigests *digests)
{
    static const char *const path = "build/tests/persistence-checkpoint.txt";
    static const char *const corrupt_path =
        "build/tests/persistence-checkpoint-corrupt.txt";
    NeuralCheckpointMetadata saved = {0};
    NeuralCheckpointMetadata loaded;
    NeuralModel *source = NULL;
    NeuralModel *destination = NULL;
    NeuralError error;

    saved.completed_epochs = 3500U;
    saved.target_epochs = 10000U;
    saved.rng_state = UINT64_C(123456789);
    saved.optimizer = NEURAL_OPTIMIZER_GRADIENT_DESCENT;
    saved.digests = *digests;
    (void)remove(path);
    (void)remove(corrupt_path);
    check(neural_model_create(&project->model,
                              project->training.seed,
                              &source,
                              &error) &&
              neural_model_set_random_state(source,
                                            saved.rng_state,
                                            &error),
          "checkpoint source model must be prepared");
    check(neural_model_create(&project->model,
                              project->training.seed + UINT64_C(9),
                              &destination,
                              &error),
          "checkpoint destination model must be prepared");
    check(neural_checkpoint_save_atomic(path, source, &saved, &error),
          "checkpoint must save atomically");
    check(neural_checkpoint_load(path,
                                 destination,
                                 digests,
                                 &loaded,
                                 &error),
          "valid checkpoint must load");
    check(models_have_equal_parameters(source, destination),
          "checkpoint parameters must round-trip bit exactly");
    check(neural_model_random_state(destination) == saved.rng_state,
          "checkpoint RNG state must be restored");
    check(loaded.completed_epochs == saved.completed_epochs &&
              loaded.target_epochs == saved.target_epochs &&
              loaded.optimizer == saved.optimizer,
          "checkpoint metadata must round-trip");

    check(copy_replacing_line(path,
                              corrupt_path,
                              "target_epochs ",
                              "target_epochs 100"),
          "invalid checkpoint fixture must be created");
    check(!neural_checkpoint_load(corrupt_path,
                                  destination,
                                  digests,
                                  &loaded,
                                  &error),
          "completed epochs above target must be rejected");
    check_error_contains(&error,
                         "epoch boundaries",
                         "invalid checkpoint epochs must be actionable");

    neural_model_free(destination);
    neural_model_free(source);
    (void)remove(corrupt_path);
    (void)remove(path);
}

static void test_invalid_save(const NeuralProject *project,
                              const NeuralProjectDigests *digests)
{
    NeuralCheckpointMetadata invalid = {0};
    NeuralModel *model = NULL;
    NeuralError error;

    invalid.completed_epochs = 2U;
    invalid.target_epochs = 1U;
    invalid.optimizer = NEURAL_OPTIMIZER_GRADIENT_DESCENT;
    invalid.digests = *digests;
    check(neural_model_create(&project->model,
                              project->training.seed,
                              &model,
                              &error),
          "model must be created for invalid-save test");
    check(!neural_checkpoint_save_atomic(
              "build/tests/invalid-checkpoint.txt",
              model,
              &invalid,
              &error),
          "invalid checkpoint metadata must fail before filesystem changes");
    check_error_contains(&error,
                         "epoch boundaries",
                         "invalid-save error must explain epoch boundaries");
    neural_model_free(model);
    (void)remove("build/tests/invalid-checkpoint.txt");
}

int main(void)
{
    NeuralProject project = {0};
    NeuralProjectDigests digests;
    NeuralError error;

    test_sha256();
    check(neural_project_load("projects/xor", &project, &error),
          "XOR project must load for persistence tests");
    if (failures == 0) {
        test_project_digests(&project, &digests);
        test_weights_round_trip(&project, &digests);
        test_checkpoint_round_trip(&project, &digests);
        test_invalid_save(&project, &digests);
    }
    neural_project_free(&project);

    if (failures != 0) {
        fprintf(stderr, "%d persistence test(s) failed\n", failures);
        return 1;
    }
    puts("All persistence tests passed");
    return 0;
}
