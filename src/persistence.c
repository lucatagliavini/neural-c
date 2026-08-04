#define _POSIX_C_SOURCE 200809L

#include "neural/persistence.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural/defaults.h"
#include "neural/parse.h"
#include "neural/version.h"
#include "atomic_file.h"

enum persistence_kind {
    PERSISTENCE_WEIGHTS,
    PERSISTENCE_CHECKPOINT
};

typedef struct {
    FILE *stream;
    const char *path;
    char *line;
    size_t line_capacity;
    size_t line_number;
    char *tokens[4];
    size_t token_count;
} PersistenceReader;

typedef struct {
    neural_real **weights;
    neural_real **biases;
    size_t layer_count;
} ParameterStaging;

typedef struct {
    const char *path;
    enum persistence_kind kind;
    const NeuralModel *model;
    const NeuralModel *best_model;
    const NeuralWeightsMetadata *weights;
    const NeuralCheckpointMetadata *checkpoint;
    int early_format;
} PersistenceWriteRequest;

static const char *persistence_kind_name(enum persistence_kind kind)
{
    return kind == PERSISTENCE_WEIGHTS ? "weights" : "checkpoint";
}

static int is_lower_hex_digest(const char *digest)
{
    size_t index;

    if (digest == NULL || strlen(digest) != NEURAL_SHA256_HEX_LENGTH) {
        return 0;
    }
    for (index = 0U; index < NEURAL_SHA256_HEX_LENGTH; index++) {
        unsigned char character = (unsigned char)digest[index];

        if (isdigit(character) == 0 &&
            !(character >= (unsigned char)'a' &&
              character <= (unsigned char)'f')) {
            return 0;
        }
    }
    return 1;
}

static int validate_digests(const NeuralProjectDigests *digests,
                            NeuralError *error)
{
    if (digests == NULL ||
        !is_lower_hex_digest(digests->model) ||
        !is_lower_hex_digest(digests->dataset) ||
        !is_lower_hex_digest(digests->training)) {
        neural_error_set(error,
                         "persistence digests must be 64 lowercase hexadecimal characters");
        return 0;
    }
    return 1;
}

static int validate_metadata(enum persistence_kind kind,
                             const NeuralWeightsMetadata *weights,
                             const NeuralCheckpointMetadata *checkpoint,
                             NeuralError *error)
{
    if (kind == PERSISTENCE_WEIGHTS) {
        if (weights == NULL) {
            neural_error_set(error, "weights metadata is required");
            return 0;
        }
        return validate_digests(&weights->digests, error);
    }
    if (checkpoint == NULL) {
        neural_error_set(error, "checkpoint metadata is required");
        return 0;
    }
    if (!validate_digests(&checkpoint->digests, error)) {
        return 0;
    }
    if (checkpoint->target_epochs == 0U ||
        checkpoint->completed_epochs > checkpoint->target_epochs) {
        neural_error_set(error,
                         "checkpoint epoch boundaries are invalid");
        return 0;
    }
    if (checkpoint->optimizer != NEURAL_OPTIMIZER_GRADIENT_DESCENT) {
        neural_error_set(error, "unknown checkpoint optimizer");
        return 0;
    }
    return 1;
}

static int write_digest(FILE *stream, const char *name, const char *digest)
{
    return fprintf(stream, "%s sha256:%s\n", name, digest) >= 0;
}

static int write_payload(FILE *stream, const NeuralModel *model)
{
    size_t layer_index;

    for (layer_index = 0U;
         layer_index < neural_model_layer_count(model);
         layer_index++) {
        const neural_real *weights;
        const neural_real *biases;
        size_t weight_count;
        size_t bias_count;
        size_t index;

        weights = neural_model_layer_weights(model,
                                             layer_index,
                                             &weight_count);
        biases = neural_model_layer_biases(model,
                                          layer_index,
                                          &bias_count);
        if (weights == NULL || biases == NULL ||
            fprintf(stream, "layer %zu\nweights\n", layer_index) < 0) {
            return 0;
        }
        for (index = 0U; index < weight_count; index++) {
            if (fprintf(stream, "%.*g\n", DBL_DECIMAL_DIG, weights[index]) < 0) {
                return 0;
            }
        }
        if (fputs("biases\n", stream) == EOF) {
            return 0;
        }
        for (index = 0U; index < bias_count; index++) {
            if (fprintf(stream, "%.*g\n", DBL_DECIMAL_DIG, biases[index]) < 0) {
                return 0;
            }
        }
        if (fputs("end_layer\n", stream) == EOF) {
            return 0;
        }
    }
    return fputs("end\n", stream) != EOF;
}

static int write_persistence(FILE *stream,
                             enum persistence_kind kind,
                             const NeuralModel *model,
                             const NeuralWeightsMetadata *weights,
                             const NeuralCheckpointMetadata *checkpoint)
{
    const NeuralProjectDigests *digests =
        kind == PERSISTENCE_WEIGHTS ? &weights->digests : &checkpoint->digests;
    size_t completed_epochs = kind == PERSISTENCE_WEIGHTS
                                  ? weights->completed_epochs
                                  : checkpoint->completed_epochs;

    if (fprintf(stream,
                "%s %s %d\n",
                NEURAL_FORMAT_MAGIC,
                persistence_kind_name(kind),
                NEURAL_FORMAT_VERSION) < 0 ||
        !write_digest(stream, "model_digest", digests->model) ||
        !write_digest(stream, "dataset_digest", digests->dataset) ||
        !write_digest(stream, "training_digest", digests->training) ||
        fprintf(stream, "completed_epochs %zu\n", completed_epochs) < 0) {
        return 0;
    }
    if (kind == PERSISTENCE_CHECKPOINT &&
        fprintf(stream,
                "target_epochs %zu\n"
                "optimizer gradient_descent\n"
                "rng_state %" PRIu64 "\n",
                checkpoint->target_epochs,
                checkpoint->rng_state) < 0) {
        return 0;
    }
    return write_payload(stream, model);
}

static int write_early_persistence(FILE *stream,
                                   const PersistenceWriteRequest *request)
{
    const NeuralProjectDigests *digests = request->kind == PERSISTENCE_WEIGHTS
        ? &request->weights->digests : &request->checkpoint->digests;
    size_t completed_epochs = request->kind == PERSISTENCE_WEIGHTS
        ? request->weights->completed_epochs
        : request->checkpoint->completed_epochs;

    if (fprintf(stream,
                "%s %s 2\n",
                NEURAL_FORMAT_MAGIC,
                persistence_kind_name(request->kind)) < 0 ||
        !write_digest(stream, "model_digest", digests->model) ||
        !write_digest(stream, "dataset_digest", digests->dataset) ||
        !write_digest(stream, "training_digest", digests->training) ||
        fprintf(stream, "completed_epochs %zu\n", completed_epochs) < 0) {
        return 0;
    }
    if (request->kind == PERSISTENCE_WEIGHTS) {
        const char *reason = request->weights->completion_reason ==
                                     NEURAL_COMPLETION_EARLY_STOPPING
                                 ? "early_stopping" : "target";

        return fprintf(stream,
                       "selected_epoch %zu\n"
                       "target_epochs %zu\n"
                       "completion %s\n",
                       request->weights->selected_epoch,
                       request->weights->target_epochs,
                       reason) >= 0 &&
               write_payload(stream, request->model);
    }
    if (request->best_model == NULL ||
        fprintf(stream,
                "target_epochs %zu\n"
                "optimizer gradient_descent\n"
                "rng_state %" PRIu64 "\n"
                "best_epoch %zu\n"
                "best_loss %.*g\n"
                "stale_epochs %zu\n"
                "current_model\n",
                request->checkpoint->target_epochs,
                request->checkpoint->rng_state,
                request->checkpoint->best_epoch,
                DBL_DECIMAL_DIG,
                request->checkpoint->best_loss,
                request->checkpoint->stale_epochs) < 0 ||
        !write_payload(stream, request->model) ||
        fputs("best_model\n", stream) == EOF) {
        return 0;
    }
    return write_payload(stream, request->best_model);
}

static int write_with_c_locale(FILE *stream,
                               void *opaque,
                               NeuralError *error)
{
    const PersistenceWriteRequest *request = opaque;
    locale_t c_locale = (locale_t)0;
    locale_t previous_locale = (locale_t)0;
    int written;

    c_locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    if (c_locale == (locale_t)0) {
        neural_error_set(error,
                         "%s: unable to create numeric locale: %s",
                         request->path,
                         strerror(errno));
        return 0;
    }
    previous_locale = uselocale(c_locale);
    if (previous_locale == (locale_t)0) {
        neural_error_set(error,
                         "%s: unable to select numeric locale: %s",
                         request->path,
                         strerror(errno));
        freelocale(c_locale);
        return 0;
    }
    if (request->early_format) {
        written = write_early_persistence(stream, request);
    } else {
        written = write_persistence(stream,
                                    request->kind,
                                    request->model,
                                    request->weights,
                                    request->checkpoint);
    }
    if (uselocale(previous_locale) == (locale_t)0) {
        neural_error_set(error,
                         "%s: unable to restore numeric locale: %s",
                         request->path,
                         strerror(errno));
        return 0;
    }
    freelocale(c_locale);
    if (!written) {
        neural_error_set(error,
                         "%s: unable to serialize persistence payload",
                         request->path);
        return 0;
    }
    return 1;
}

static int save_atomic(const char *path,
                       enum persistence_kind kind,
                       const NeuralModel *model,
                       const NeuralWeightsMetadata *weights,
                       const NeuralCheckpointMetadata *checkpoint,
                       NeuralError *error)
{
    PersistenceWriteRequest request;

    neural_error_clear(error);
    if (path == NULL || path[0] == '\0' || model == NULL) {
        neural_error_set(error, "persistence path and model are required");
        return 0;
    }
    if (!validate_metadata(kind, weights, checkpoint, error)) {
        return 0;
    }
    if (kind == PERSISTENCE_CHECKPOINT &&
        checkpoint->rng_state != neural_model_random_state(model)) {
        neural_error_set(error,
                         "checkpoint RNG state does not match the runtime model");
        return 0;
    }
    request.path = path;
    request.kind = kind;
    request.model = model;
    request.best_model = NULL;
    request.weights = weights;
    request.checkpoint = checkpoint;
    request.early_format = 0;
    return neural_atomic_file_write(path,
                                    write_with_c_locale,
                                    &request,
                                    error);
}

static int reader_error(PersistenceReader *reader,
                        NeuralError *error,
                        const char *message)
{
    neural_error_set(error,
                     "%s:%zu: %s",
                     reader->path,
                     reader->line_number,
                     message);
    return -1;
}

static int reader_next(PersistenceReader *reader, NeuralError *error)
{
    for (;;) {
        ssize_t length = getline(&reader->line,
                                 &reader->line_capacity,
                                 reader->stream);
        char *cursor;

        if (length < 0) {
            if (ferror(reader->stream) != 0) {
                reader->line_number++;
                return reader_error(reader, error, "unable to read file");
            }
            return 0;
        }
        reader->line_number++;
        if ((size_t)length > NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH) {
            return reader_error(reader, error, "line exceeds maximum length");
        }
        while (length > 0 &&
               (reader->line[(size_t)length - 1U] == '\n' ||
                reader->line[(size_t)length - 1U] == '\r')) {
            length--;
            reader->line[(size_t)length] = '\0';
        }
        reader->token_count = 0U;
        cursor = reader->line;
        while (*cursor != '\0') {
            char *start;

            while (isspace((unsigned char)*cursor) != 0) {
                cursor++;
            }
            if (*cursor == '\0' || *cursor == '#') {
                break;
            }
            if (reader->token_count ==
                sizeof(reader->tokens) / sizeof(reader->tokens[0])) {
                return reader_error(reader, error, "too many tokens on line");
            }
            start = cursor;
            while (*cursor != '\0' && *cursor != '#' &&
                   isspace((unsigned char)*cursor) == 0) {
                cursor++;
            }
            reader->tokens[reader->token_count++] = start;
            if (*cursor == '#') {
                *cursor = '\0';
                break;
            }
            if (*cursor != '\0') {
                *cursor = '\0';
                cursor++;
            }
        }
        if (reader->token_count != 0U) {
            return 1;
        }
    }
}

static int require_line(PersistenceReader *reader,
                        size_t token_count,
                        const char *first_token,
                        NeuralError *error)
{
    int status = reader_next(reader, error);

    if (status < 0) {
        return 0;
    }
    if (status == 0) {
        neural_error_set(error,
                         "%s:%zu: expected '%s' before end of file",
                         reader->path,
                         reader->line_number + 1U,
                         first_token);
        return 0;
    }
    if (reader->token_count != token_count ||
        strcmp(reader->tokens[0], first_token) != 0) {
        neural_error_set(error,
                         "%s:%zu: expected '%s'",
                         reader->path,
                         reader->line_number,
                         first_token);
        return 0;
    }
    return 1;
}

static int parse_digest_line(PersistenceReader *reader,
                             const char *name,
                             char output[NEURAL_SHA256_TEXT_CAPACITY],
                             NeuralError *error)
{
    static const char prefix[] = "sha256:";
    const char *encoded;

    if (!require_line(reader, 2U, name, error)) {
        return 0;
    }
    encoded = reader->tokens[1];
    if (strncmp(encoded, prefix, sizeof(prefix) - 1U) != 0 ||
        !is_lower_hex_digest(encoded + sizeof(prefix) - 1U)) {
        neural_error_set(error,
                         "%s:%zu: %s must use sha256 followed by 64 lowercase hexadecimal characters",
                         reader->path,
                         reader->line_number,
                         name);
        return 0;
    }
    memcpy(output,
           encoded + sizeof(prefix) - 1U,
           NEURAL_SHA256_TEXT_CAPACITY);
    return 1;
}

static int parse_size_line(PersistenceReader *reader,
                           const char *name,
                           size_t *value,
                           NeuralError *error)
{
    if (!require_line(reader, 2U, name, error)) {
        return 0;
    }
    if (!neural_parse_size(reader->tokens[1], value)) {
        neural_error_set(error,
                         "%s:%zu: %s must be a non-negative integer",
                         reader->path,
                         reader->line_number,
                         name);
        return 0;
    }
    return 1;
}

static int parse_uint64_line(PersistenceReader *reader,
                             const char *name,
                             uint64_t *value,
                             NeuralError *error)
{
    if (!require_line(reader, 2U, name, error)) {
        return 0;
    }
    if (!neural_parse_uint64(reader->tokens[1], value)) {
        neural_error_set(error,
                         "%s:%zu: %s must be an unsigned 64-bit integer",
                         reader->path,
                         reader->line_number,
                         name);
        return 0;
    }
    return 1;
}

static int parse_real_line(PersistenceReader *reader,
                           const char *name,
                           neural_real *value,
                           NeuralError *error)
{
    if (!require_line(reader, 2U, name, error)) {
        return 0;
    }
    if (!neural_parse_real(reader->tokens[1], value)) {
        neural_error_set(error,
                         "%s:%zu: %s must be a finite number",
                         reader->path,
                         reader->line_number,
                         name);
        return 0;
    }
    return 1;
}

static int parse_header(PersistenceReader *reader,
                        enum persistence_kind kind,
                        size_t *format_version,
                        NeuralError *error)
{
    int status = reader_next(reader, error);
    size_t version;

    if (status <= 0) {
        if (status == 0) {
            neural_error_set(error, "%s: empty persistence file", reader->path);
        }
        return 0;
    }
    if (reader->token_count != 3U ||
        strcmp(reader->tokens[0], NEURAL_FORMAT_MAGIC) != 0 ||
        strcmp(reader->tokens[1], persistence_kind_name(kind)) != 0 ||
        !neural_parse_size(reader->tokens[2], &version) ||
        (version != (size_t)NEURAL_FORMAT_VERSION && version != 2U)) {
        neural_error_set(error,
                         "%s:%zu: expected '%s %s %d'",
                         reader->path,
                         reader->line_number,
                         NEURAL_FORMAT_MAGIC,
                         persistence_kind_name(kind),
                         NEURAL_FORMAT_VERSION);
        return 0;
    }
    *format_version = version;
    return 1;
}

static int parse_metadata(PersistenceReader *reader,
                          enum persistence_kind kind,
                          NeuralWeightsMetadata *weights,
                          NeuralCheckpointMetadata *checkpoint,
                          size_t format_version,
                          NeuralError *error)
{
    NeuralProjectDigests *digests = kind == PERSISTENCE_WEIGHTS
                                         ? &weights->digests
                                         : &checkpoint->digests;
    size_t *completed_epochs = kind == PERSISTENCE_WEIGHTS
                                   ? &weights->completed_epochs
                                   : &checkpoint->completed_epochs;

    if (!parse_digest_line(reader, "model_digest", digests->model, error) ||
        !parse_digest_line(reader, "dataset_digest", digests->dataset, error) ||
        !parse_digest_line(reader,
                           "training_digest",
                           digests->training,
                           error) ||
        !parse_size_line(reader,
                         "completed_epochs",
                         completed_epochs,
                         error)) {
        return 0;
    }
    if (kind == PERSISTENCE_CHECKPOINT) {
        if (!parse_size_line(reader,
                             "target_epochs",
                             &checkpoint->target_epochs,
                             error) ||
            !require_line(reader, 2U, "optimizer", error)) {
            return 0;
        }
        if (strcmp(reader->tokens[1], "gradient_descent") != 0) {
            neural_error_set(error,
                             "%s:%zu: unknown optimizer '%s'",
                             reader->path,
                             reader->line_number,
                             reader->tokens[1]);
            return 0;
        }
        checkpoint->optimizer = NEURAL_OPTIMIZER_GRADIENT_DESCENT;
        if (!parse_uint64_line(reader,
                               "rng_state",
                               &checkpoint->rng_state,
                               error)) {
            return 0;
        }
    }
    if (format_version == 2U && kind == PERSISTENCE_WEIGHTS) {
        if (!parse_size_line(reader,
                             "selected_epoch",
                             &weights->selected_epoch,
                             error) ||
            !parse_size_line(reader,
                             "target_epochs",
                             &weights->target_epochs,
                             error) ||
            !require_line(reader, 2U, "completion", error)) {
            return 0;
        }
        if (strcmp(reader->tokens[1], "target") == 0) {
            weights->completion_reason = NEURAL_COMPLETION_TARGET;
        } else if (strcmp(reader->tokens[1], "early_stopping") == 0) {
            weights->completion_reason = NEURAL_COMPLETION_EARLY_STOPPING;
        } else {
            neural_error_set(error,
                             "%s:%zu: unknown completion reason '%s'",
                             reader->path,
                             reader->line_number,
                             reader->tokens[1]);
            return 0;
        }
        weights->format_version = format_version;
        if (weights->target_epochs == 0U ||
            weights->selected_epoch == 0U ||
            weights->selected_epoch > weights->completed_epochs ||
            weights->completed_epochs > weights->target_epochs ||
            (weights->completion_reason != NEURAL_COMPLETION_TARGET &&
             weights->completion_reason !=
                 NEURAL_COMPLETION_EARLY_STOPPING) ||
            (weights->completion_reason == NEURAL_COMPLETION_TARGET &&
             weights->completed_epochs != weights->target_epochs) ||
            (weights->completion_reason ==
                 NEURAL_COMPLETION_EARLY_STOPPING &&
             weights->completed_epochs >= weights->target_epochs)) {
            neural_error_set(error,
                             "%s:%zu: weights epoch boundaries are invalid",
                             reader->path,
                             reader->line_number);
            return 0;
        }
    }
    if (format_version == 1U && kind == PERSISTENCE_WEIGHTS) {
        weights->selected_epoch = weights->completed_epochs;
        weights->target_epochs = weights->completed_epochs;
        weights->completion_reason = NEURAL_COMPLETION_TARGET;
        weights->format_version = format_version;
    }
    {
        NeuralError metadata_error;

        if (!validate_metadata(kind,
                               weights,
                               checkpoint,
                               &metadata_error)) {
            neural_error_set(error,
                             "%s:%zu: %s",
                             reader->path,
                             reader->line_number,
                             metadata_error.message);
            return 0;
        }
    }
    return 1;
}

static int verify_digests(const char *path,
                          const NeuralProjectDigests *actual,
                          const NeuralProjectDigests *expected,
                          NeuralError *error)
{
    if (strcmp(actual->model, expected->model) != 0) {
        neural_error_set(error, "%s: model digest does not match", path);
        return 0;
    }
    if (strcmp(actual->dataset, expected->dataset) != 0) {
        neural_error_set(error, "%s: dataset digest does not match", path);
        return 0;
    }
    if (strcmp(actual->training, expected->training) != 0) {
        neural_error_set(error, "%s: training digest does not match", path);
        return 0;
    }
    return 1;
}

static void staging_free(ParameterStaging *staging)
{
    size_t layer_index;

    if (staging == NULL) {
        return;
    }
    for (layer_index = 0U; layer_index < staging->layer_count; layer_index++) {
        free(staging->weights[layer_index]);
        free(staging->biases[layer_index]);
    }
    free(staging->weights);
    free(staging->biases);
    memset(staging, 0, sizeof(*staging));
}

static int staging_create(const NeuralModel *model,
                          ParameterStaging *staging,
                          NeuralError *error)
{
    size_t layer_index;

    memset(staging, 0, sizeof(*staging));
    staging->layer_count = neural_model_layer_count(model);
    staging->weights = calloc(staging->layer_count,
                              sizeof(*staging->weights));
    staging->biases = calloc(staging->layer_count,
                             sizeof(*staging->biases));
    if (staging->weights == NULL || staging->biases == NULL) {
        neural_error_set(error, "unable to allocate parameter staging");
        staging_free(staging);
        return 0;
    }
    for (layer_index = 0U;
         layer_index < staging->layer_count;
         layer_index++) {
        size_t weight_count;
        size_t bias_count;

        (void)neural_model_layer_weights(model, layer_index, &weight_count);
        (void)neural_model_layer_biases(model, layer_index, &bias_count);
        if (weight_count > SIZE_MAX / sizeof(**staging->weights) ||
            bias_count > SIZE_MAX / sizeof(**staging->biases)) {
            neural_error_set(error, "persistence parameter dimensions overflow");
            staging_free(staging);
            return 0;
        }
        staging->weights[layer_index] =
            malloc(weight_count * sizeof(**staging->weights));
        staging->biases[layer_index] =
            malloc(bias_count * sizeof(**staging->biases));
        if (staging->weights[layer_index] == NULL ||
            staging->biases[layer_index] == NULL) {
            neural_error_set(error, "unable to allocate staged parameters");
            staging_free(staging);
            return 0;
        }
    }
    return 1;
}

static int parse_real_values(PersistenceReader *reader,
                             neural_real *values,
                             size_t count,
                             const char *description,
                             NeuralError *error)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        int status = reader_next(reader, error);

        if (status <= 0) {
            if (status == 0) {
                neural_error_set(error,
                                 "%s:%zu: missing %s value %zu",
                                 reader->path,
                                 reader->line_number + 1U,
                                 description,
                                 index);
            }
            return 0;
        }
        if (reader->token_count != 1U ||
            !neural_parse_real(reader->tokens[0], &values[index])) {
            neural_error_set(error,
                             "%s:%zu: invalid finite %s value",
                             reader->path,
                             reader->line_number,
                             description);
            return 0;
        }
    }
    return 1;
}

static int parse_payload(PersistenceReader *reader,
                         const NeuralModel *model,
                         ParameterStaging *staging,
                         int require_end_of_file,
                         NeuralError *error)
{
    size_t layer_index;

    for (layer_index = 0U;
         layer_index < staging->layer_count;
         layer_index++) {
        size_t parsed_layer_index;
        size_t weight_count;
        size_t bias_count;

        if (!require_line(reader, 2U, "layer", error) ||
            !neural_parse_size(reader->tokens[1], &parsed_layer_index) ||
            parsed_layer_index != layer_index) {
            neural_error_set(error,
                             "%s:%zu: expected layer %zu",
                             reader->path,
                             reader->line_number,
                             layer_index);
            return 0;
        }
        (void)neural_model_layer_weights(model, layer_index, &weight_count);
        (void)neural_model_layer_biases(model, layer_index, &bias_count);
        if (!require_line(reader, 1U, "weights", error) ||
            !parse_real_values(reader,
                               staging->weights[layer_index],
                               weight_count,
                               "weight",
                               error) ||
            !require_line(reader, 1U, "biases", error) ||
            !parse_real_values(reader,
                               staging->biases[layer_index],
                               bias_count,
                               "bias",
                               error) ||
            !require_line(reader, 1U, "end_layer", error)) {
            return 0;
        }
    }
    if (!require_line(reader, 1U, "end", error)) {
        return 0;
    }
    if (require_end_of_file) {
        int status = reader_next(reader, error);

        if (status < 0) {
            return 0;
        }
        if (status != 0) {
            neural_error_set(error,
                             "%s:%zu: unexpected content after end",
                             reader->path,
                             reader->line_number);
            return 0;
        }
    }
    return 1;
}

static int apply_staging(NeuralModel *model,
                         const ParameterStaging *staging,
                         NeuralError *error)
{
    size_t layer_index;

    for (layer_index = 0U; layer_index < staging->layer_count; layer_index++) {
        size_t weight_count;
        size_t bias_count;

        (void)neural_model_layer_weights(model, layer_index, &weight_count);
        (void)neural_model_layer_biases(model, layer_index, &bias_count);
        if (!neural_model_set_layer_parameters(model,
                                               layer_index,
                                               staging->weights[layer_index],
                                               weight_count,
                                               staging->biases[layer_index],
                                               bias_count,
                                               error)) {
            return 0;
        }
    }
    return 1;
}

static int load_persistence(const char *path,
                            enum persistence_kind kind,
                            NeuralModel *model,
                            const NeuralProjectDigests *expected_digests,
                            NeuralWeightsMetadata *weights,
                            NeuralCheckpointMetadata *checkpoint,
                            NeuralError *error)
{
    PersistenceReader reader = {0};
    ParameterStaging staging;
    NeuralWeightsMetadata loaded_weights = {0};
    NeuralCheckpointMetadata loaded_checkpoint = {0};
    const NeuralProjectDigests *loaded_digests;
    size_t format_version = 0U;
    int success = 0;

    neural_error_clear(error);
    if (path == NULL || path[0] == '\0' || model == NULL ||
        expected_digests == NULL ||
        (kind == PERSISTENCE_WEIGHTS && weights == NULL) ||
        (kind == PERSISTENCE_CHECKPOINT && checkpoint == NULL)) {
        neural_error_set(error, "invalid persistence load arguments");
        return 0;
    }
    if (!validate_digests(expected_digests, error) ||
        !staging_create(model, &staging, error)) {
        return 0;
    }
    reader.path = path;
    reader.stream = fopen(path, "r");
    if (reader.stream == NULL) {
        neural_error_set(error, "%s: unable to open file: %s", path, strerror(errno));
        goto cleanup;
    }
    if (!parse_header(&reader, kind, &format_version, error)) {
        goto cleanup;
    }
    if (kind == PERSISTENCE_CHECKPOINT && format_version != 1U) {
        neural_error_set(error,
                         "%s:%zu: expected '%s checkpoint %d'",
                         path,
                         reader.line_number,
                         NEURAL_FORMAT_MAGIC,
                         NEURAL_FORMAT_VERSION);
        goto cleanup;
    }
    if (!parse_metadata(&reader,
                        kind,
                        &loaded_weights,
                        &loaded_checkpoint,
                        format_version,
                        error)) {
        goto cleanup;
    }
    loaded_digests = kind == PERSISTENCE_WEIGHTS
                         ? &loaded_weights.digests
                         : &loaded_checkpoint.digests;
    if (!verify_digests(path, loaded_digests, expected_digests, error) ||
        !parse_payload(&reader, model, &staging, 1, error)) {
        goto cleanup;
    }
    if (fclose(reader.stream) != 0) {
        reader.stream = NULL;
        neural_error_set(error,
                         "%s: unable to close file: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    reader.stream = NULL;
    if (!apply_staging(model, &staging, error)) {
        goto cleanup;
    }
    if (kind == PERSISTENCE_CHECKPOINT &&
        !neural_model_set_random_state(model,
                                       loaded_checkpoint.rng_state,
                                       error)) {
        goto cleanup;
    }
    if (kind == PERSISTENCE_WEIGHTS) {
        *weights = loaded_weights;
    } else {
        *checkpoint = loaded_checkpoint;
    }
    success = 1;

cleanup:
    free(reader.line);
    if (reader.stream != NULL) {
        (void)fclose(reader.stream);
    }
    staging_free(&staging);
    return success;
}

int neural_weights_save_atomic(const char *path,
                               const NeuralModel *model,
                               const NeuralWeightsMetadata *metadata,
                               NeuralError *error)
{
    return save_atomic(path,
                       PERSISTENCE_WEIGHTS,
                       model,
                       metadata,
                       NULL,
                       error);
}

int neural_weights_load(const char *path,
                        NeuralModel *model,
                        const NeuralProjectDigests *expected_digests,
                        NeuralWeightsMetadata *metadata,
                        NeuralError *error)
{
    return load_persistence(path,
                            PERSISTENCE_WEIGHTS,
                            model,
                            expected_digests,
                            metadata,
                            NULL,
                            error);
}

int neural_checkpoint_save_atomic(const char *path,
                                  const NeuralModel *model,
                                  const NeuralCheckpointMetadata *metadata,
                                  NeuralError *error)
{
    return save_atomic(path,
                       PERSISTENCE_CHECKPOINT,
                       model,
                       NULL,
                       metadata,
                       error);
}

int neural_checkpoint_load(const char *path,
                           NeuralModel *model,
                           const NeuralProjectDigests *expected_digests,
                           NeuralCheckpointMetadata *metadata,
                           NeuralError *error)
{
    return load_persistence(path,
                            PERSISTENCE_CHECKPOINT,
                            model,
                            expected_digests,
                            NULL,
                            metadata,
                            error);
}

static int save_early_atomic(const char *path,
                             enum persistence_kind kind,
                             const NeuralModel *model,
                             const NeuralModel *best_model,
                             const NeuralWeightsMetadata *weights,
                             const NeuralCheckpointMetadata *checkpoint,
                             NeuralError *error)
{
    PersistenceWriteRequest request;

    neural_error_clear(error);
    if (path == NULL || path[0] == '\0' || model == NULL ||
        (kind == PERSISTENCE_CHECKPOINT && best_model == NULL)) {
        neural_error_set(error, "early persistence path and models are required");
        return 0;
    }
    if (!validate_metadata(kind, weights, checkpoint, error)) {
        return 0;
    }
    if (kind == PERSISTENCE_WEIGHTS) {
        if (weights->format_version != 2U ||
            weights->target_epochs == 0U ||
            weights->selected_epoch == 0U ||
            weights->selected_epoch > weights->completed_epochs ||
            weights->completed_epochs > weights->target_epochs ||
            (weights->completion_reason == NEURAL_COMPLETION_TARGET &&
             weights->completed_epochs != weights->target_epochs) ||
            (weights->completion_reason ==
                 NEURAL_COMPLETION_EARLY_STOPPING &&
             weights->completed_epochs >= weights->target_epochs)) {
            neural_error_set(error, "early-stopping weights metadata is invalid");
            return 0;
        }
    } else if (checkpoint->format_version != 2U ||
               checkpoint->best_epoch == 0U ||
               checkpoint->best_epoch > checkpoint->completed_epochs ||
               checkpoint->stale_epochs > checkpoint->completed_epochs ||
               !isfinite(checkpoint->best_loss) ||
               checkpoint->rng_state != neural_model_random_state(model)) {
        neural_error_set(error, "early-stopping checkpoint metadata is invalid");
        return 0;
    }
    request.path = path;
    request.kind = kind;
    request.model = model;
    request.best_model = best_model;
    request.weights = weights;
    request.checkpoint = checkpoint;
    request.early_format = 1;
    return neural_atomic_file_write(path, write_with_c_locale, &request, error);
}

int neural_early_weights_save_atomic(
    const char *path,
    const NeuralModel *selected_model,
    const NeuralWeightsMetadata *metadata,
    NeuralError *error)
{
    return save_early_atomic(path,
                             PERSISTENCE_WEIGHTS,
                             selected_model,
                             NULL,
                             metadata,
                             NULL,
                             error);
}

int neural_early_checkpoint_save_atomic(
    const char *path,
    const NeuralModel *current_model,
    const NeuralModel *best_model,
    const NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    return save_early_atomic(path,
                             PERSISTENCE_CHECKPOINT,
                             current_model,
                             best_model,
                             NULL,
                             metadata,
                             error);
}

int neural_early_checkpoint_load(
    const char *path,
    NeuralModel *current_model,
    NeuralModel *best_model,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    PersistenceReader reader = {0};
    ParameterStaging current_staging = {0};
    ParameterStaging best_staging = {0};
    NeuralWeightsMetadata unused_weights = {0};
    NeuralCheckpointMetadata loaded = {0};
    size_t format_version = 0U;
    int success = 0;

    neural_error_clear(error);
    if (path == NULL || current_model == NULL || best_model == NULL ||
        expected_digests == NULL || metadata == NULL ||
        !validate_digests(expected_digests, error) ||
        !staging_create(current_model, &current_staging, error) ||
        !staging_create(best_model, &best_staging, error)) {
        goto cleanup;
    }
    reader.path = path;
    reader.stream = fopen(path, "r");
    if (reader.stream == NULL) {
        neural_error_set(error, "%s: unable to open file: %s", path, strerror(errno));
        goto cleanup;
    }
    if (!parse_header(&reader,
                      PERSISTENCE_CHECKPOINT,
                      &format_version,
                      error) ||
        format_version != 2U ||
        !parse_metadata(&reader,
                        PERSISTENCE_CHECKPOINT,
                        &unused_weights,
                        &loaded,
                        format_version,
                        error) ||
        !parse_size_line(&reader, "best_epoch", &loaded.best_epoch, error) ||
        !parse_real_line(&reader, "best_loss", &loaded.best_loss, error) ||
        !parse_size_line(&reader, "stale_epochs", &loaded.stale_epochs, error) ||
        !require_line(&reader, 1U, "current_model", error) ||
        !verify_digests(path, &loaded.digests, expected_digests, error) ||
        !parse_payload(&reader,
                       current_model,
                       &current_staging,
                       0,
                       error) ||
        !require_line(&reader, 1U, "best_model", error) ||
        !parse_payload(&reader, best_model, &best_staging, 1, error)) {
        goto cleanup;
    }
    if (fclose(reader.stream) != 0) {
        reader.stream = NULL;
        neural_error_set(error,
                         "%s: unable to close file: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    reader.stream = NULL;
    loaded.format_version = format_version;
    if (loaded.best_epoch > loaded.completed_epochs ||
        loaded.stale_epochs > loaded.completed_epochs ||
        !isfinite(loaded.best_loss) ||
        !apply_staging(current_model, &current_staging, error) ||
        !apply_staging(best_model, &best_staging, error) ||
        !neural_model_set_random_state(current_model,
                                       loaded.rng_state,
                                       error)) {
        if (error != NULL && error->message[0] == '\0') {
            neural_error_set(error, "%s: invalid early-stopping state", path);
        }
        goto cleanup;
    }
    *metadata = loaded;
    success = 1;

cleanup:
    free(reader.line);
    if (reader.stream != NULL) {
        (void)fclose(reader.stream);
    }
    staging_free(&best_staging);
    staging_free(&current_staging);
    return success;
}
