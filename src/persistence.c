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
#include "neural/gradient.h"
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
    const NeuralOptimizer *optimizer;
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
    if (strcmp(neural_optimizer_name(checkpoint->optimizer), "unknown") == 0) {
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

static int write_gradient_payload(FILE *stream,
                                  const NeuralModel *model,
                                  const NeuralGradient *gradient)
{
    size_t layer_index;

    if (gradient == NULL ||
        !neural_gradient_is_compatible(gradient, model)) {
        return 0;
    }
    for (layer_index = 0U;
         layer_index < neural_model_layer_count(model);
         layer_index++) {
        const neural_real *weights;
        const neural_real *biases;
        size_t weight_count;
        size_t bias_count;
        size_t index;

        weights = neural_gradient_layer_weights_const(
            gradient, layer_index, &weight_count);
        biases = neural_gradient_layer_biases_const(
            gradient, layer_index, &bias_count);
        if (weights == NULL || biases == NULL ||
            fprintf(stream, "layer %zu\nweights\n", layer_index) < 0) {
            return 0;
        }
        for (index = 0U; index < weight_count; index++) {
            if (!isfinite(weights[index]) ||
                fprintf(stream,
                        "%.*g\n",
                        DBL_DECIMAL_DIG,
                        weights[index]) < 0) {
                return 0;
            }
        }
        if (fputs("biases\n", stream) == EOF) {
            return 0;
        }
        for (index = 0U; index < bias_count; index++) {
            if (!isfinite(biases[index]) ||
                fprintf(stream,
                        "%.*g\n",
                        DBL_DECIMAL_DIG,
                        biases[index]) < 0) {
                return 0;
            }
        }
        if (fputs("end_layer\n", stream) == EOF) {
            return 0;
        }
    }
    return fputs("end\n", stream) != EOF;
}

static int write_optimizer_state(FILE *stream,
                                 const NeuralModel *model,
                                 const NeuralOptimizer *optimizer)
{
    NeuralOptimizerKind kind = neural_optimizer_kind(optimizer);

    if (!neural_optimizer_requires_checkpoint_state(optimizer)) {
        return 0;
    }
    if (fprintf(
            stream,
            "optimizer_timestep %zu\n"
            "learning_rate_schedule %s\n"
            "schedule_completed_epochs %zu\n"
            "schedule_current_rate %.*g\n"
            "schedule_next_transition %zu\n"
            "schedule_has_best %d\n"
            "schedule_stale_epochs %zu\n",
            neural_optimizer_timestep(optimizer),
            neural_learning_rate_schedule_name(
                neural_optimizer_schedule_kind(optimizer)),
            neural_optimizer_schedule_completed_epochs(optimizer),
            DBL_DECIMAL_DIG,
            neural_optimizer_current_learning_rate(optimizer),
            neural_optimizer_schedule_next_transition(optimizer),
            neural_optimizer_schedule_has_best(optimizer),
            neural_optimizer_schedule_stale_epochs(optimizer)) < 0) {
        return 0;
    }
    if (neural_optimizer_schedule_has_best(optimizer) &&
        fprintf(stream,
                "schedule_best_metric %.*g\n",
                DBL_DECIMAL_DIG,
                neural_optimizer_schedule_best_metric(optimizer)) < 0) {
        return 0;
    }
    if (fprintf(stream,
                "convergence_has_best %d\n"
                "convergence_stale_epochs %zu\n"
                "convergence_reason %d\n",
                neural_optimizer_convergence_has_best(optimizer),
                neural_optimizer_convergence_stale_epochs(optimizer),
                (int)neural_optimizer_convergence_reason(optimizer)) < 0) {
        return 0;
    }
    if (neural_optimizer_convergence_has_best(optimizer) &&
        fprintf(stream,
                "convergence_best_loss %.*g\n",
                DBL_DECIMAL_DIG,
                neural_optimizer_convergence_best_loss(optimizer)) < 0) {
        return 0;
    }
    if (kind == NEURAL_OPTIMIZER_ADAM &&
        fprintf(stream,
                "optimizer_beta1_power %.*g\n"
                "optimizer_beta2_power %.*g\n",
                DBL_DECIMAL_DIG,
                neural_optimizer_beta1_power(optimizer),
                DBL_DECIMAL_DIG,
                neural_optimizer_beta2_power(optimizer)) < 0) {
        return 0;
    }
    if (kind == NEURAL_OPTIMIZER_GRADIENT_DESCENT) {
        return 1;
    }
    if (fputs("optimizer_state1\n", stream) == EOF ||
        !write_gradient_payload(stream,
                                model,
                                neural_optimizer_state1(optimizer))) {
        return 0;
    }
    if (kind == NEURAL_OPTIMIZER_ADAM &&
        (fputs("optimizer_state2\n", stream) == EOF ||
         !write_gradient_payload(stream,
                                 model,
                                 neural_optimizer_state2(optimizer)))) {
        return 0;
    }
    return 1;
}

static int write_persistence(FILE *stream,
                             enum persistence_kind kind,
                             const NeuralModel *model,
                             const NeuralWeightsMetadata *weights,
                             const NeuralCheckpointMetadata *checkpoint,
                             const NeuralOptimizer *optimizer)
{
    const NeuralProjectDigests *digests =
        kind == PERSISTENCE_WEIGHTS ? &weights->digests : &checkpoint->digests;
    size_t completed_epochs = kind == PERSISTENCE_WEIGHTS
                                  ? weights->completed_epochs
                                  : checkpoint->completed_epochs;

    size_t format_version = kind == PERSISTENCE_CHECKPOINT &&
                                    optimizer != NULL &&
                                    neural_optimizer_requires_checkpoint_state(
                                        optimizer)
                                ? NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION
                                : (size_t)NEURAL_FORMAT_VERSION;

    if (fprintf(stream,
                "%s %s %zu\n",
                NEURAL_FORMAT_MAGIC,
                persistence_kind_name(kind),
                format_version) < 0 ||
        !write_digest(stream, "model_digest", digests->model) ||
        !write_digest(stream, "dataset_digest", digests->dataset) ||
        !write_digest(stream, "training_digest", digests->training) ||
        fprintf(stream, "completed_epochs %zu\n", completed_epochs) < 0) {
        return 0;
    }
    if (kind == PERSISTENCE_CHECKPOINT &&
        fprintf(stream,
                "target_epochs %zu\n"
                "optimizer %s\n"
                "rng_state %" PRIu64 "\n",
                checkpoint->target_epochs,
                neural_optimizer_name(checkpoint->optimizer),
                checkpoint->rng_state) < 0) {
        return 0;
    }
    if (format_version == NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION &&
        (!write_optimizer_state(stream, model, optimizer) ||
         fputs("model\n", stream) == EOF)) {
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
    size_t format_version =
        request->kind == PERSISTENCE_CHECKPOINT &&
                request->optimizer != NULL &&
                neural_optimizer_requires_checkpoint_state(request->optimizer)
            ? NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION
            : 2U;

    if (fprintf(stream,
                "%s %s %zu\n",
                NEURAL_FORMAT_MAGIC,
                persistence_kind_name(request->kind),
                format_version) < 0 ||
        !write_digest(stream, "model_digest", digests->model) ||
        !write_digest(stream, "dataset_digest", digests->dataset) ||
        !write_digest(stream, "training_digest", digests->training) ||
        fprintf(stream, "completed_epochs %zu\n", completed_epochs) < 0) {
        return 0;
    }
    if (request->kind == PERSISTENCE_WEIGHTS) {
        const char *reason;

        switch (request->weights->completion_reason) {
        case NEURAL_COMPLETION_TARGET:
            reason = "target";
            break;
        case NEURAL_COMPLETION_EARLY_STOPPING:
            reason = "early_stopping";
            break;
        case NEURAL_COMPLETION_LOSS_TARGET:
            reason = "loss_target";
            break;
        case NEURAL_COMPLETION_NO_IMPROVEMENT:
            reason = "no_improvement";
            break;
        default:
            return 0;
        }

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
                "optimizer %s\n"
                "rng_state %" PRIu64 "\n"
                "best_epoch %zu\n"
                "best_loss %.*g\n"
                "stale_epochs %zu\n",
                request->checkpoint->target_epochs,
                neural_optimizer_name(request->checkpoint->optimizer),
                request->checkpoint->rng_state,
                request->checkpoint->best_epoch,
                DBL_DECIMAL_DIG,
                request->checkpoint->best_loss,
                request->checkpoint->stale_epochs) < 0 ||
        (format_version == NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION &&
         !write_optimizer_state(stream,
                                request->model,
                                request->optimizer)) ||
        fputs("current_model\n", stream) == EOF ||
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
                                    request->checkpoint,
                                    request->optimizer);
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
                       const NeuralOptimizer *optimizer,
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
    if (kind == PERSISTENCE_CHECKPOINT &&
        checkpoint->optimizer != NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
        (optimizer == NULL ||
         neural_optimizer_kind(optimizer) != checkpoint->optimizer)) {
        neural_error_set(error,
                         "checkpoint optimizer state does not match metadata");
        return 0;
    }
    request.path = path;
    request.kind = kind;
    request.model = model;
    request.best_model = NULL;
    request.weights = weights;
    request.checkpoint = checkpoint;
    request.optimizer = optimizer;
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
        (version != (size_t)NEURAL_FORMAT_VERSION && version != 2U &&
         version != NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION)) {
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
        if (!neural_optimizer_from_name(reader->tokens[1],
                                        &checkpoint->optimizer)) {
            neural_error_set(error,
                             "%s:%zu: unknown optimizer '%s'",
                             reader->path,
                             reader->line_number,
                             reader->tokens[1]);
            return 0;
        }
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
        } else if (strcmp(reader->tokens[1], "loss_target") == 0) {
            weights->completion_reason = NEURAL_COMPLETION_LOSS_TARGET;
        } else if (strcmp(reader->tokens[1], "no_improvement") == 0) {
            weights->completion_reason = NEURAL_COMPLETION_NO_IMPROVEMENT;
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
                 NEURAL_COMPLETION_EARLY_STOPPING &&
             weights->completion_reason !=
                 NEURAL_COMPLETION_LOSS_TARGET &&
             weights->completion_reason !=
                 NEURAL_COMPLETION_NO_IMPROVEMENT) ||
            (weights->completion_reason == NEURAL_COMPLETION_TARGET &&
             weights->completed_epochs != weights->target_epochs) ||
            (weights->completion_reason != NEURAL_COMPLETION_TARGET &&
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
    if (kind == PERSISTENCE_CHECKPOINT) {
        checkpoint->format_version = format_version;
        if (format_version != NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION &&
            checkpoint->optimizer != NEURAL_OPTIMIZER_GRADIENT_DESCENT) {
            neural_error_set(error,
                             "%s:%zu: stateful optimizer requires checkpoint version 3",
                             reader->path,
                             reader->line_number);
            return 0;
        }
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

static int staging_to_gradient(const NeuralModel *model,
                               const ParameterStaging *staging,
                               NeuralGradient *gradient,
                               NeuralError *error)
{
    size_t layer_index;

    if (staging == NULL || gradient == NULL ||
        !neural_gradient_is_compatible(gradient, model)) {
        neural_error_set(error, "optimizer staging is incompatible");
        return 0;
    }
    for (layer_index = 0U; layer_index < staging->layer_count; layer_index++) {
        size_t weight_count;
        size_t bias_count;
        neural_real *weights = neural_gradient_layer_weights(
            gradient, layer_index, &weight_count);
        neural_real *biases = neural_gradient_layer_biases(
            gradient, layer_index, &bias_count);

        memcpy(weights,
               staging->weights[layer_index],
               weight_count * sizeof(*weights));
        memcpy(biases,
               staging->biases[layer_index],
               bias_count * sizeof(*biases));
    }
    return 1;
}

static int parse_optimizer_state(PersistenceReader *reader,
                                 const NeuralModel *model,
                                 NeuralOptimizerKind kind,
                                 NeuralLearningRateScheduleKind expected_schedule,
                                 size_t *timestep,
                                 size_t *schedule_completed_epochs,
                                 neural_real *schedule_current_rate,
                                 size_t *schedule_next_transition,
                                 int *schedule_has_best,
                                 neural_real *schedule_best_metric,
                                 size_t *schedule_stale_epochs,
                                 int *convergence_has_best,
                                 neural_real *convergence_best_loss,
                                 size_t *convergence_stale_epochs,
                                 NeuralConvergenceReason *convergence_reason,
                                 neural_real *beta1_power,
                                 neural_real *beta2_power,
                                 ParameterStaging *state1,
                                 ParameterStaging *state2,
                                 NeuralError *error)
{
    NeuralLearningRateScheduleKind parsed_schedule;

    if (!parse_size_line(reader,
                         "optimizer_timestep",
                         timestep,
                         error)) {
        return 0;
    }
    if (!require_line(reader, 2U, "learning_rate_schedule", error) ||
        !neural_learning_rate_schedule_from_name(reader->tokens[1],
                                                 &parsed_schedule) ||
        parsed_schedule != expected_schedule ||
        !parse_size_line(reader,
                         "schedule_completed_epochs",
                         schedule_completed_epochs,
                         error) ||
        !parse_real_line(reader,
                         "schedule_current_rate",
                         schedule_current_rate,
                         error) ||
        !parse_size_line(reader,
                         "schedule_next_transition",
                         schedule_next_transition,
                         error) ||
        !require_line(reader, 2U, "schedule_has_best", error)) {
        return 0;
    }
    if (strcmp(reader->tokens[1], "0") == 0) {
        *schedule_has_best = 0;
    } else if (strcmp(reader->tokens[1], "1") == 0) {
        *schedule_has_best = 1;
    } else {
        neural_error_set(error,
                         "%s:%zu: schedule_has_best must be 0 or 1",
                         reader->path,
                         reader->line_number);
        return 0;
    }
    if (!parse_size_line(reader,
                         "schedule_stale_epochs",
                         schedule_stale_epochs,
                         error)) {
        return 0;
    }
    *schedule_best_metric = 0.0;
    if (*schedule_has_best &&
        !parse_real_line(reader,
                         "schedule_best_metric",
                         schedule_best_metric,
                         error)) {
        return 0;
    }
    if (!require_line(reader, 2U, "convergence_has_best", error)) {
        return 0;
    }
    if (strcmp(reader->tokens[1], "0") == 0) {
        *convergence_has_best = 0;
    } else if (strcmp(reader->tokens[1], "1") == 0) {
        *convergence_has_best = 1;
    } else {
        neural_error_set(error,
                         "%s:%zu: convergence_has_best must be 0 or 1",
                         reader->path,
                         reader->line_number);
        return 0;
    }
    {
        size_t parsed_reason;

        if (!parse_size_line(reader,
                             "convergence_stale_epochs",
                             convergence_stale_epochs,
                             error) ||
            !parse_size_line(reader,
                             "convergence_reason",
                             &parsed_reason,
                             error) ||
            parsed_reason >
                (size_t)NEURAL_CONVERGENCE_NO_IMPROVEMENT) {
            return 0;
        }
        *convergence_reason = (NeuralConvergenceReason)parsed_reason;
    }
    *convergence_best_loss = 0.0;
    if (*convergence_has_best &&
        !parse_real_line(reader,
                         "convergence_best_loss",
                         convergence_best_loss,
                         error)) {
        return 0;
    }
    *beta1_power = 1.0;
    *beta2_power = 1.0;
    if (kind == NEURAL_OPTIMIZER_ADAM &&
        (!parse_real_line(reader,
                          "optimizer_beta1_power",
                          beta1_power,
                          error) ||
         !parse_real_line(reader,
                          "optimizer_beta2_power",
                          beta2_power,
                          error))) {
        return 0;
    }
    if (kind == NEURAL_OPTIMIZER_GRADIENT_DESCENT) {
        return 1;
    }
    if (!require_line(reader, 1U, "optimizer_state1", error) ||
        !parse_payload(reader, model, state1, 0, error)) {
        return 0;
    }
    if (kind == NEURAL_OPTIMIZER_ADAM &&
        (!require_line(reader, 1U, "optimizer_state2", error) ||
         !parse_payload(reader, model, state2, 0, error))) {
        return 0;
    }
    return 1;
}

static int load_persistence(const char *path,
                            enum persistence_kind kind,
                            NeuralModel *model,
                            NeuralOptimizer *optimizer,
                            const NeuralProjectDigests *expected_digests,
                            NeuralWeightsMetadata *weights,
                            NeuralCheckpointMetadata *checkpoint,
                            NeuralError *error)
{
    PersistenceReader reader = {0};
    ParameterStaging staging;
    ParameterStaging state1_staging = {0};
    ParameterStaging state2_staging = {0};
    NeuralGradient *state1 = NULL;
    NeuralGradient *state2 = NULL;
    NeuralWeightsMetadata loaded_weights = {0};
    NeuralCheckpointMetadata loaded_checkpoint = {0};
    const NeuralProjectDigests *loaded_digests;
    size_t format_version = 0U;
    size_t optimizer_timestep = 0U;
    neural_real beta1_power = 1.0;
    neural_real beta2_power = 1.0;
    size_t schedule_completed_epochs = 0U;
    neural_real schedule_current_rate = 0.0;
    size_t schedule_next_transition = 0U;
    int schedule_has_best = 0;
    neural_real schedule_best_metric = 0.0;
    size_t schedule_stale_epochs = 0U;
    int convergence_has_best = 0;
    neural_real convergence_best_loss = 0.0;
    size_t convergence_stale_epochs = 0U;
    NeuralConvergenceReason convergence_reason = NEURAL_CONVERGENCE_NONE;
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
    if ((kind == PERSISTENCE_CHECKPOINT &&
         format_version != 1U &&
         format_version != NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION) ||
        (kind == PERSISTENCE_WEIGHTS &&
         format_version != 1U && format_version != 2U)) {
        neural_error_set(error,
                         "%s:%zu: expected '%s %s %d'",
                         path,
                         reader.line_number,
                         NEURAL_FORMAT_MAGIC,
                         persistence_kind_name(kind),
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
    if (!verify_digests(path, loaded_digests, expected_digests, error)) {
        goto cleanup;
    }
    if (kind == PERSISTENCE_CHECKPOINT &&
        format_version == NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION) {
        if (optimizer == NULL ||
            neural_optimizer_kind(optimizer) != loaded_checkpoint.optimizer ||
            (loaded_checkpoint.optimizer !=
                 NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
             !staging_create(model, &state1_staging, error)) ||
            (loaded_checkpoint.optimizer == NEURAL_OPTIMIZER_ADAM &&
             !staging_create(model, &state2_staging, error)) ||
            !parse_optimizer_state(&reader,
                                   model,
                                   loaded_checkpoint.optimizer,
                                   neural_optimizer_schedule_kind(optimizer),
                                   &optimizer_timestep,
                                   &schedule_completed_epochs,
                                   &schedule_current_rate,
                                   &schedule_next_transition,
                                   &schedule_has_best,
                                   &schedule_best_metric,
                                   &schedule_stale_epochs,
                                   &convergence_has_best,
                                   &convergence_best_loss,
                                   &convergence_stale_epochs,
                                   &convergence_reason,
                                   &beta1_power,
                                   &beta2_power,
                                   &state1_staging,
                                   &state2_staging,
                                   error) ||
            !require_line(&reader, 1U, "model", error)) {
            goto cleanup;
        }
    } else if (kind == PERSISTENCE_CHECKPOINT && optimizer != NULL &&
               neural_optimizer_requires_checkpoint_state(optimizer)) {
        neural_error_set(error,
                         "%s: checkpoint has no state for configured optimizer",
                         path);
        goto cleanup;
    }
    if (!parse_payload(&reader, model, &staging, 1, error)) {
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
    if (kind == PERSISTENCE_CHECKPOINT &&
        format_version == NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION) {
        if ((loaded_checkpoint.optimizer !=
                 NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
             (!neural_gradient_create(model, &state1, error) ||
              !staging_to_gradient(model,
                                   &state1_staging,
                                   state1,
                                   error))) ||
            (loaded_checkpoint.optimizer == NEURAL_OPTIMIZER_ADAM &&
             (!neural_gradient_create(model, &state2, error) ||
              !staging_to_gradient(model,
                                   &state2_staging,
                                   state2,
                                   error))) ||
            (loaded_checkpoint.optimizer !=
                 NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
             !neural_optimizer_restore(optimizer,
                                       optimizer_timestep,
                                       beta1_power,
                                       beta2_power,
                                       state1,
                                       state2,
                                       error)) ||
            !neural_optimizer_restore_schedule(
                optimizer,
                schedule_completed_epochs,
                schedule_current_rate,
                schedule_next_transition,
                schedule_has_best,
                schedule_best_metric,
                schedule_stale_epochs,
                error) ||
            !neural_optimizer_restore_convergence(
                optimizer,
                convergence_has_best,
                convergence_best_loss,
                convergence_stale_epochs,
                convergence_reason,
                error)) {
            goto cleanup;
        }
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
    staging_free(&state2_staging);
    staging_free(&state1_staging);
    neural_gradient_free(state2);
    neural_gradient_free(state1);
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
                            NULL,
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
                       NULL,
                       error);
}

int neural_checkpoint_save_atomic_with_optimizer(
    const char *path,
    const NeuralModel *model,
    const NeuralOptimizer *optimizer,
    const NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    return save_atomic(path,
                       PERSISTENCE_CHECKPOINT,
                       model,
                       NULL,
                       metadata,
                       optimizer,
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
                            NULL,
                            expected_digests,
                            NULL,
                            metadata,
                            error);
}

int neural_checkpoint_load_with_optimizer(
    const char *path,
    NeuralModel *model,
    NeuralOptimizer *optimizer,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    return load_persistence(path,
                            PERSISTENCE_CHECKPOINT,
                            model,
                            optimizer,
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
                             const NeuralOptimizer *optimizer,
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
            (weights->completion_reason != NEURAL_COMPLETION_TARGET &&
             weights->completed_epochs >= weights->target_epochs)) {
            neural_error_set(error, "early-stopping weights metadata is invalid");
            return 0;
        }
    } else if ((checkpoint->format_version != 2U &&
                checkpoint->format_version !=
                    NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION) ||
               checkpoint->best_epoch == 0U ||
               checkpoint->best_epoch > checkpoint->completed_epochs ||
               checkpoint->stale_epochs > checkpoint->completed_epochs ||
               !isfinite(checkpoint->best_loss) ||
               checkpoint->rng_state != neural_model_random_state(model)) {
        neural_error_set(error, "early-stopping checkpoint metadata is invalid");
        return 0;
    }
    if (kind == PERSISTENCE_CHECKPOINT &&
        checkpoint->optimizer != NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
        (optimizer == NULL ||
         neural_optimizer_kind(optimizer) != checkpoint->optimizer)) {
        neural_error_set(error,
                         "early checkpoint optimizer state does not match metadata");
        return 0;
    }
    request.path = path;
    request.kind = kind;
    request.model = model;
    request.best_model = best_model;
    request.weights = weights;
    request.checkpoint = checkpoint;
    request.optimizer = optimizer;
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
                             NULL,
                             error);
}

int neural_early_checkpoint_save_atomic_with_optimizer(
    const char *path,
    const NeuralModel *current_model,
    const NeuralModel *best_model,
    const NeuralOptimizer *optimizer,
    const NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    return save_early_atomic(path,
                             PERSISTENCE_CHECKPOINT,
                             current_model,
                             best_model,
                             NULL,
                             metadata,
                             optimizer,
                             error);
}

static int load_early_checkpoint(
    const char *path,
    NeuralModel *current_model,
    NeuralModel *best_model,
    NeuralOptimizer *optimizer,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    PersistenceReader reader = {0};
    ParameterStaging current_staging = {0};
    ParameterStaging best_staging = {0};
    ParameterStaging state1_staging = {0};
    ParameterStaging state2_staging = {0};
    NeuralGradient *state1 = NULL;
    NeuralGradient *state2 = NULL;
    NeuralWeightsMetadata unused_weights = {0};
    NeuralCheckpointMetadata loaded = {0};
    size_t format_version = 0U;
    size_t optimizer_timestep = 0U;
    neural_real beta1_power = 1.0;
    neural_real beta2_power = 1.0;
    size_t schedule_completed_epochs = 0U;
    neural_real schedule_current_rate = 0.0;
    size_t schedule_next_transition = 0U;
    int schedule_has_best = 0;
    neural_real schedule_best_metric = 0.0;
    size_t schedule_stale_epochs = 0U;
    int convergence_has_best = 0;
    neural_real convergence_best_loss = 0.0;
    size_t convergence_stale_epochs = 0U;
    NeuralConvergenceReason convergence_reason = NEURAL_CONVERGENCE_NONE;
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
        (format_version != 2U &&
         format_version != NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION) ||
        !parse_metadata(&reader,
                        PERSISTENCE_CHECKPOINT,
                        &unused_weights,
                        &loaded,
                        format_version,
                        error) ||
        !parse_size_line(&reader, "best_epoch", &loaded.best_epoch, error) ||
        !parse_real_line(&reader, "best_loss", &loaded.best_loss, error) ||
        !parse_size_line(&reader, "stale_epochs", &loaded.stale_epochs, error) ||
        !verify_digests(path, &loaded.digests, expected_digests, error)) {
        goto cleanup;
    }
    if (format_version == NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION) {
        if (optimizer == NULL ||
            neural_optimizer_kind(optimizer) != loaded.optimizer ||
            (loaded.optimizer != NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
             !staging_create(current_model, &state1_staging, error)) ||
            (loaded.optimizer == NEURAL_OPTIMIZER_ADAM &&
             !staging_create(current_model, &state2_staging, error)) ||
            !parse_optimizer_state(&reader,
                                   current_model,
                                   loaded.optimizer,
                                   neural_optimizer_schedule_kind(optimizer),
                                   &optimizer_timestep,
                                   &schedule_completed_epochs,
                                   &schedule_current_rate,
                                   &schedule_next_transition,
                                   &schedule_has_best,
                                   &schedule_best_metric,
                                   &schedule_stale_epochs,
                                   &convergence_has_best,
                                   &convergence_best_loss,
                                   &convergence_stale_epochs,
                                   &convergence_reason,
                                   &beta1_power,
                                   &beta2_power,
                                   &state1_staging,
                                   &state2_staging,
                                   error)) {
            goto cleanup;
        }
    } else if (optimizer != NULL &&
               neural_optimizer_requires_checkpoint_state(optimizer)) {
        neural_error_set(error,
                         "%s: checkpoint has no state for configured optimizer",
                         path);
        goto cleanup;
    }
    if (!require_line(&reader, 1U, "current_model", error) ||
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
    if (format_version == NEURAL_STATEFUL_CHECKPOINT_FORMAT_VERSION &&
        ((loaded.optimizer != NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
          (!neural_gradient_create(current_model, &state1, error) ||
           !staging_to_gradient(current_model,
                                &state1_staging,
                                state1,
                                error))) ||
         (loaded.optimizer == NEURAL_OPTIMIZER_ADAM &&
          (!neural_gradient_create(current_model, &state2, error) ||
           !staging_to_gradient(current_model,
                                &state2_staging,
                                state2,
                                error))) ||
         (loaded.optimizer != NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
          !neural_optimizer_restore(optimizer,
                                    optimizer_timestep,
                                    beta1_power,
                                    beta2_power,
                                    state1,
                                    state2,
                                    error)) ||
         !neural_optimizer_restore_schedule(
             optimizer,
             schedule_completed_epochs,
             schedule_current_rate,
             schedule_next_transition,
             schedule_has_best,
             schedule_best_metric,
             schedule_stale_epochs,
             error) ||
         !neural_optimizer_restore_convergence(
             optimizer,
             convergence_has_best,
             convergence_best_loss,
             convergence_stale_epochs,
             convergence_reason,
             error))) {
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
    staging_free(&state2_staging);
    staging_free(&state1_staging);
    neural_gradient_free(state2);
    neural_gradient_free(state1);
    return success;
}

int neural_early_checkpoint_load(
    const char *path,
    NeuralModel *current_model,
    NeuralModel *best_model,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    return load_early_checkpoint(path,
                                 current_model,
                                 best_model,
                                 NULL,
                                 expected_digests,
                                 metadata,
                                 error);
}

int neural_early_checkpoint_load_with_optimizer(
    const char *path,
    NeuralModel *current_model,
    NeuralModel *best_model,
    NeuralOptimizer *optimizer,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error)
{
    return load_early_checkpoint(path,
                                 current_model,
                                 best_model,
                                 optimizer,
                                 expected_digests,
                                 metadata,
                                 error);
}
