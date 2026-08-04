#define _POSIX_C_SOURCE 200809L

#include "neural/project.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "neural/defaults.h"
#include "neural/parse.h"
#include "neural/version.h"
#include "path.h"

#if NEURAL_DEFAULT_TEXT_INITIAL_CAPACITY < 2U
#error "NEURAL_DEFAULT_TEXT_INITIAL_CAPACITY must be at least 2"
#endif

#if NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH < NEURAL_DEFAULT_TEXT_INITIAL_CAPACITY
#error "NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH must not be smaller than the initial capacity"
#endif

#if NEURAL_DEFAULT_TOKEN_CAPACITY < 1U
#error "NEURAL_DEFAULT_TOKEN_CAPACITY must be positive"
#endif

#if NEURAL_DEFAULT_LAYER_CAPACITY < 1U
#error "NEURAL_DEFAULT_LAYER_CAPACITY must be positive"
#endif

#if NEURAL_DEFAULT_SAMPLE_CAPACITY < 1U
#error "NEURAL_DEFAULT_SAMPLE_CAPACITY must be positive"
#endif

#if NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR < 2U
#error "NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR must be at least 2"
#endif

_Static_assert(sizeof(NEURAL_DEFAULT_MODEL_FILENAME) > 1U,
               "NEURAL_DEFAULT_MODEL_FILENAME must not be empty");
_Static_assert(sizeof(NEURAL_DEFAULT_PROJECT_FILENAME) > 1U,
               "NEURAL_DEFAULT_PROJECT_FILENAME must not be empty");
_Static_assert(sizeof(NEURAL_DEFAULT_DATASET_FILENAME) > 1U,
               "NEURAL_DEFAULT_DATASET_FILENAME must not be empty");
_Static_assert(sizeof(NEURAL_DEFAULT_WEIGHTS_FILENAME) > 1U,
               "NEURAL_DEFAULT_WEIGHTS_FILENAME must not be empty");
_Static_assert(sizeof(NEURAL_DEFAULT_CHECKPOINT_FILENAME) > 1U,
               "NEURAL_DEFAULT_CHECKPOINT_FILENAME must not be empty");

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} TokenList;

static int checked_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0U && right > SIZE_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static void token_list_free(TokenList *tokens)
{
    free(tokens->items);
    memset(tokens, 0, sizeof(*tokens));
}

static int token_list_add(TokenList *tokens, char *token, NeuralError *error)
{
    if (tokens->count == tokens->capacity) {
        size_t new_capacity;
        char **new_items;

        if (tokens->capacity == 0U) {
            new_capacity = NEURAL_DEFAULT_TOKEN_CAPACITY;
        } else if (tokens->capacity >
                   SIZE_MAX / NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR) {
            neural_error_set(error, "too many tokens on one line");
            return 0;
        } else {
            new_capacity = tokens->capacity *
                           NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR;
        }
        if (new_capacity < tokens->capacity ||
            new_capacity > SIZE_MAX / sizeof(*new_items)) {
            neural_error_set(error, "too many tokens on one line");
            return 0;
        }
        new_items = realloc(tokens->items,
                            new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            neural_error_set(error, "unable to allocate parser tokens");
            return 0;
        }
        tokens->items = new_items;
        tokens->capacity = new_capacity;
    }
    tokens->items[tokens->count] = token;
    tokens->count++;
    return 1;
}

static int tokenize(char *line, TokenList *tokens, NeuralError *error)
{
    char *cursor = line;

    tokens->count = 0U;
    while (*cursor != '\0') {
        char *start;

        while (isspace((unsigned char)*cursor) != 0) {
            cursor++;
        }
        if (*cursor == '\0' || *cursor == '#') {
            break;
        }

        start = cursor;
        while (*cursor != '\0' && *cursor != '#' &&
               isspace((unsigned char)*cursor) == 0) {
            cursor++;
        }
        if (*cursor == '#') {
            *cursor = '\0';
        } else if (*cursor != '\0') {
            *cursor = '\0';
            cursor++;
        }
        if (!token_list_add(tokens, start, error)) {
            return 0;
        }
    }
    return 1;
}

static int read_line(FILE *stream,
                     char **buffer,
                     size_t *capacity,
                     size_t *length,
                     NeuralError *error)
{
    int character;

    *length = 0U;
    if (*buffer == NULL) {
        *buffer = malloc(NEURAL_DEFAULT_TEXT_INITIAL_CAPACITY);
        if (*buffer == NULL) {
            neural_error_set(error, "unable to allocate line buffer");
            return -1;
        }
        *capacity = NEURAL_DEFAULT_TEXT_INITIAL_CAPACITY;
    }

    while ((character = fgetc(stream)) != EOF) {
        if (character == '\n') {
            break;
        }
        if (*length + 1U >= *capacity) {
            size_t new_capacity;
            char *new_buffer;

            if (*capacity >= NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH) {
                neural_error_set(error,
                                 "line exceeds maximum length of %zu bytes",
                                 (size_t)NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH);
                return -1;
            }
            if (*capacity > NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH /
                                NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR) {
                new_capacity = NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH;
            } else {
                new_capacity = *capacity *
                               NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR;
            }
            new_buffer = realloc(*buffer, new_capacity);
            if (new_buffer == NULL) {
                neural_error_set(error, "unable to grow line buffer");
                return -1;
            }
            *buffer = new_buffer;
            *capacity = new_capacity;
        }
        (*buffer)[*length] = (char)character;
        (*length)++;
    }

    if (ferror(stream) != 0) {
        neural_error_set(error, "unable to read input file");
        return -1;
    }
    if (character == EOF && *length == 0U) {
        return 0;
    }
    if (*length > 0U && (*buffer)[*length - 1U] == '\r') {
        (*length)--;
    }
    (*buffer)[*length] = '\0';
    return 1;
}

static int next_content_line(FILE *stream,
                             const char *path,
                             char **line,
                             size_t *line_capacity,
                             size_t *line_number,
                             TokenList *tokens,
                             NeuralError *error)
{
    for (;;) {
        size_t length;
        int status = read_line(stream,
                               line,
                               line_capacity,
                               &length,
                               error);
        (void)length;
        if (status <= 0) {
            if (status < 0 && error != NULL) {
                char detail[NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY];
                (void)snprintf(detail, sizeof(detail), "%s", error->message);
                neural_error_set(error,
                                 "%s:%zu: %s",
                                 path,
                                 *line_number + 1U,
                                 detail);
            }
            return status;
        }
        (*line_number)++;
        if (!tokenize(*line, tokens, error)) {
            if (error != NULL) {
                char detail[NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY];
                (void)snprintf(detail, sizeof(detail), "%s", error->message);
                neural_error_set(error,
                                 "%s:%zu: %s",
                                 path,
                                 *line_number,
                                 detail);
            }
            return -1;
        }
        if (tokens->count != 0U) {
            return 1;
        }
    }
}

static int parse_header(FILE *stream,
                        const char *path,
                        const char *kind,
                        char **line,
                        size_t *line_capacity,
                        size_t *line_number,
                        TokenList *tokens,
                        NeuralError *error)
{
    int status = next_content_line(stream,
                                   path,
                                   line,
                                   line_capacity,
                                   line_number,
                                   tokens,
                                   error);

    if (status == 0) {
        neural_error_set(error, "%s: file is empty", path);
        return 0;
    }
    if (status < 0) {
        return 0;
    }
    {
        char expected_version[32];
        (void)snprintf(expected_version,
                       sizeof(expected_version),
                       "%d",
                       NEURAL_FORMAT_VERSION);
        if (tokens->count != 3U ||
            strcmp(tokens->items[0], NEURAL_FORMAT_MAGIC) != 0 ||
            strcmp(tokens->items[1], kind) != 0 ||
            strcmp(tokens->items[2], expected_version) != 0) {
            neural_error_set(error,
                             "%s:%zu: expected 'neural-c %s %d'",
                             path,
                             *line_number,
                             kind,
                             NEURAL_FORMAT_VERSION);
            return 0;
        }
    }
    return 1;
}


int neural_model_spec_validate(const NeuralModelSpec *model,
                               NeuralError *error)
{
    size_t layer_index;
    size_t previous_count;

    if (model == NULL || model->input_count == 0U ||
        model->layer_count == 0U || model->layers == NULL) {
        neural_error_set(error,
                         "model requires positive inputs and at least one layer");
        return 0;
    }
    previous_count = model->input_count;
    for (layer_index = 0U; layer_index < model->layer_count; layer_index++) {
        const NeuralLayerSpec *layer = &model->layers[layer_index];
        size_t weight_count;

        if (layer->neuron_count == 0U) {
            neural_error_set(error,
                             "layer %zu must contain at least one neuron",
                             layer_index);
            return 0;
        }
        if (!neural_activation_spec_validate(&layer->activation, error)) {
            if (error != NULL) {
                char detail[NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY];
                (void)snprintf(detail, sizeof(detail), "%s", error->message);
                neural_error_set(error,
                                 "layer %zu has an invalid activation: %s",
                                 layer_index,
                                 detail);
            }
            return 0;
        }
        if (!checked_multiply(previous_count,
                              layer->neuron_count,
                              &weight_count) ||
            weight_count > SIZE_MAX / sizeof(neural_real) ||
            layer->neuron_count > SIZE_MAX / sizeof(neural_real)) {
            neural_error_set(error,
                             "layer %zu dimensions exceed addressable memory",
                             layer_index);
            return 0;
        }
        previous_count = layer->neuron_count;
    }
    return 1;
}

int neural_training_config_validate(const NeuralTrainingConfig *config,
                                    NeuralError *error)
{
    if (config == NULL || config->epochs == 0U) {
        neural_error_set(error, "epochs must be a positive integer");
        return 0;
    }
    if (!isfinite(config->learning_rate) || config->learning_rate <= 0.0) {
        neural_error_set(error, "learning_rate must be finite and positive");
        return 0;
    }
    if (strcmp(neural_loss_name(config->loss), "unknown") == 0) {
        neural_error_set(error, "training configuration has an invalid loss");
        return 0;
    }
    if (!isfinite(config->early_stopping_min_delta) ||
        config->early_stopping_min_delta < 0.0) {
        neural_error_set(error,
                         "early_stopping_min_delta must be finite and non-negative");
        return 0;
    }
    if (config->early_stopping_patience == 0U &&
        config->early_stopping_min_delta != 0.0) {
        neural_error_set(error,
                         "early_stopping_min_delta requires positive patience");
        return 0;
    }
    return 1;
}

static int append_layer(NeuralModelSpec *model,
                        NeuralLayerSpec layer,
                        size_t *capacity,
                        NeuralError *error)
{
    if (model->layer_count == *capacity) {
        size_t new_capacity;
        NeuralLayerSpec *new_layers;

        if (*capacity == 0U) {
            new_capacity = NEURAL_DEFAULT_LAYER_CAPACITY;
        } else if (*capacity >
                   SIZE_MAX / NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR) {
            neural_error_set(error, "model contains too many layers");
            return 0;
        } else {
            new_capacity = *capacity *
                           NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR;
        }
        if (new_capacity < *capacity ||
            new_capacity > SIZE_MAX / sizeof(*new_layers)) {
            neural_error_set(error, "model contains too many layers");
            return 0;
        }
        new_layers = realloc(model->layers,
                             new_capacity * sizeof(*new_layers));
        if (new_layers == NULL) {
            neural_error_set(error, "unable to allocate model layers");
            return 0;
        }
        model->layers = new_layers;
        *capacity = new_capacity;
    }
    model->layers[model->layer_count] = layer;
    model->layer_count++;
    return 1;
}

void neural_model_spec_free(NeuralModelSpec *model)
{
    if (model != NULL) {
        size_t layer_index;
        for (layer_index = 0U;
             layer_index < model->layer_count;
             layer_index++) {
            neural_activation_spec_free(&model->layers[layer_index].activation);
        }
        free(model->layers);
        memset(model, 0, sizeof(*model));
    }
}

int neural_model_spec_load(const char *path,
                           NeuralModelSpec *model,
                           NeuralError *error)
{
    FILE *stream;
    char *line = NULL;
    size_t line_capacity = 0U;
    size_t line_number = 0U;
    size_t layer_capacity = 0U;
    size_t previous_count = 0U;
    int has_input = 0;
    int success = 0;
    TokenList tokens = {0};

    if (path == NULL || model == NULL) {
        neural_error_set(error, "invalid model loader arguments");
        return 0;
    }
    memset(model, 0, sizeof(*model));
    neural_error_clear(error);
    stream = fopen(path, "r");
    if (stream == NULL) {
        neural_error_set(error, "%s: unable to open file: %s", path, strerror(errno));
        return 0;
    }
    if (!parse_header(stream,
                      path,
                      "model",
                      &line,
                      &line_capacity,
                      &line_number,
                      &tokens,
                      error)) {
        goto cleanup;
    }

    for (;;) {
        int status = next_content_line(stream,
                                       path,
                                       &line,
                                       &line_capacity,
                                       &line_number,
                                       &tokens,
                                       error);
        if (status < 0) {
            goto cleanup;
        }
        if (status == 0) {
            break;
        }

        if (strcmp(tokens.items[0], "input") == 0) {
            if (tokens.count != 2U) {
                neural_error_set(error,
                                 "%s:%zu: input requires exactly one size",
                                 path,
                                 line_number);
                goto cleanup;
            }
            if (has_input) {
                neural_error_set(error,
                                 "%s:%zu: input was specified more than once",
                                 path,
                                 line_number);
                goto cleanup;
            }
            if (model->layer_count != 0U) {
                neural_error_set(error,
                                 "%s:%zu: input must precede all layers",
                                 path,
                                 line_number);
                goto cleanup;
            }
            if (!neural_parse_size(tokens.items[1], &model->input_count) ||
                model->input_count == 0U) {
                neural_error_set(error,
                                 "%s:%zu: input size must be a positive integer",
                                 path,
                                 line_number);
                goto cleanup;
            }
            has_input = 1;
            previous_count = model->input_count;
        } else if (strcmp(tokens.items[0], "dense") == 0) {
            NeuralLayerSpec layer = {0};
            size_t weight_count;

            if (!has_input) {
                neural_error_set(error,
                                 "%s:%zu: input must be declared before layers",
                                 path,
                                 line_number);
                goto cleanup;
            }
            if (tokens.count < 3U) {
                neural_error_set(error,
                                 "%s:%zu: dense requires a size and activation",
                                 path,
                                 line_number);
                goto cleanup;
            }
            if (!neural_parse_size(tokens.items[1], &layer.neuron_count) ||
                layer.neuron_count == 0U) {
                neural_error_set(error,
                                 "%s:%zu: neuron count must be a positive integer",
                                 path,
                                 line_number);
                goto cleanup;
            }
            if (!neural_activation_kind_from_name(tokens.items[2],
                                                  &layer.activation.kind)) {
                neural_error_set(error,
                                 "%s:%zu: unknown activation '%s'",
                                 path,
                                 line_number,
                                 tokens.items[2]);
                goto cleanup;
            }
            {
                size_t parameter_index;
                for (parameter_index = 3U;
                     parameter_index < tokens.count;
                     parameter_index++) {
                    char *token = tokens.items[parameter_index];
                    char *separator = strchr(token, '=');
                    NeuralActivationParameterKind parameter_kind;
                    neural_real parameter_value;
                    NeuralError activation_error;

                    if (separator == NULL || separator == token ||
                        separator[1] == '\0' ||
                        strchr(separator + 1, '=') != NULL) {
                        neural_error_set(
                            error,
                            "%s:%zu: invalid activation parameter '%s'",
                            path,
                            line_number,
                            token);
                        neural_activation_spec_free(&layer.activation);
                        goto cleanup;
                    }
                    *separator = '\0';
                    if (!neural_activation_parameter_from_name(
                            token,
                            &parameter_kind)) {
                        neural_error_set(
                            error,
                            "%s:%zu: unknown activation parameter '%s'",
                            path,
                            line_number,
                            token);
                        neural_activation_spec_free(&layer.activation);
                        goto cleanup;
                    }
                    if (!neural_parse_real(separator + 1,
                                           &parameter_value)) {
                        neural_error_set(
                            error,
                            "%s:%zu: invalid activation parameter value '%s'",
                            path,
                            line_number,
                            separator + 1);
                        neural_activation_spec_free(&layer.activation);
                        goto cleanup;
                    }
                    if (!neural_activation_spec_set_parameter(
                            &layer.activation,
                            parameter_kind,
                            parameter_value,
                            &activation_error)) {
                        neural_error_set(error,
                                         "%s:%zu: %s",
                                         path,
                                         line_number,
                                         activation_error.message);
                        neural_activation_spec_free(&layer.activation);
                        goto cleanup;
                    }
                }
            }
            {
                NeuralError activation_error;

                if (!neural_activation_spec_validate(&layer.activation,
                                                     &activation_error)) {
                    neural_error_set(error,
                                     "%s:%zu: %s",
                                     path,
                                     line_number,
                                     activation_error.message);
                    neural_activation_spec_free(&layer.activation);
                    goto cleanup;
                }
            }
            if (!checked_multiply(previous_count,
                                  layer.neuron_count,
                                  &weight_count) ||
                weight_count > SIZE_MAX / sizeof(neural_real) ||
                layer.neuron_count > SIZE_MAX / sizeof(neural_real)) {
                neural_error_set(error,
                                 "%s:%zu: layer dimensions exceed addressable memory",
                                 path,
                                 line_number);
                neural_activation_spec_free(&layer.activation);
                goto cleanup;
            }
            if (!append_layer(model, layer, &layer_capacity, error)) {
                neural_activation_spec_free(&layer.activation);
                goto cleanup;
            }
            previous_count = layer.neuron_count;
        } else {
            neural_error_set(error,
                             "%s:%zu: unknown model directive '%s'",
                             path,
                             line_number,
                             tokens.items[0]);
            goto cleanup;
        }
    }

    if (!has_input) {
        neural_error_set(error, "%s: model does not declare its inputs", path);
        goto cleanup;
    }
    if (model->layer_count == 0U) {
        neural_error_set(error, "%s: model must contain at least one layer", path);
        goto cleanup;
    }
    success = 1;

cleanup:
    free(line);
    token_list_free(&tokens);
    (void)fclose(stream);
    if (!success) {
        neural_model_spec_free(model);
    }
    return success;
}

int neural_training_config_load(const char *path,
                                NeuralTrainingConfig *config,
                                NeuralError *error)
{
    FILE *stream;
    char *line = NULL;
    size_t line_capacity = 0U;
    size_t line_number = 0U;
    unsigned int fields = 0U;
    int success = 0;
    TokenList tokens = {0};
    enum {
        FIELD_EPOCHS = 1U,
        FIELD_RATE = 2U,
        FIELD_SEED = 4U,
        FIELD_LOSS = 8U,
        FIELD_CHECKPOINT_INTERVAL = 16U,
        FIELD_EARLY_PATIENCE = 32U,
        FIELD_EARLY_MIN_DELTA = 64U,
        REQUIRED_FIELDS = FIELD_EPOCHS | FIELD_RATE | FIELD_SEED | FIELD_LOSS |
                          FIELD_CHECKPOINT_INTERVAL,
        EARLY_FIELDS = FIELD_EARLY_PATIENCE | FIELD_EARLY_MIN_DELTA
    };

    if (path == NULL || config == NULL) {
        neural_error_set(error, "invalid configuration loader arguments");
        return 0;
    }
    memset(config, 0, sizeof(*config));
    neural_error_clear(error);
    stream = fopen(path, "r");
    if (stream == NULL) {
        neural_error_set(error, "%s: unable to open file: %s", path, strerror(errno));
        return 0;
    }
    if (!parse_header(stream,
                      path,
                      "project",
                      &line,
                      &line_capacity,
                      &line_number,
                      &tokens,
                      error)) {
        goto cleanup;
    }

    for (;;) {
        unsigned int field;
        int status = next_content_line(stream,
                                       path,
                                       &line,
                                       &line_capacity,
                                       &line_number,
                                       &tokens,
                                       error);
        if (status < 0) {
            goto cleanup;
        }
        if (status == 0) {
            break;
        }
        if (tokens.count != 2U) {
            neural_error_set(error,
                             "%s:%zu: configuration entries require one value",
                             path,
                             line_number);
            goto cleanup;
        }

        if (strcmp(tokens.items[0], "epochs") == 0) {
            field = FIELD_EPOCHS;
            if (!neural_parse_size(tokens.items[1], &config->epochs) ||
                config->epochs == 0U) {
                neural_error_set(error,
                                 "%s:%zu: epochs must be a positive integer",
                                 path,
                                 line_number);
                goto cleanup;
            }
        } else if (strcmp(tokens.items[0], "learning_rate") == 0) {
            field = FIELD_RATE;
            if (!neural_parse_real(tokens.items[1], &config->learning_rate) ||
                config->learning_rate <= 0.0) {
                neural_error_set(error,
                                 "%s:%zu: learning_rate must be finite and positive",
                                 path,
                                 line_number);
                goto cleanup;
            }
        } else if (strcmp(tokens.items[0], "seed") == 0) {
            field = FIELD_SEED;
            if (!neural_parse_uint64(tokens.items[1], &config->seed)) {
                neural_error_set(error,
                                 "%s:%zu: seed must be an unsigned 64-bit integer",
                                 path,
                                 line_number);
                goto cleanup;
            }
        } else if (strcmp(tokens.items[0], "loss") == 0) {
            field = FIELD_LOSS;
            if (strcmp(tokens.items[1], "mse") != 0) {
                neural_error_set(error,
                                 "%s:%zu: unknown loss '%s'",
                                 path,
                                 line_number,
                                 tokens.items[1]);
                goto cleanup;
            }
            config->loss = NEURAL_LOSS_MSE;
        } else if (strcmp(tokens.items[0], "checkpoint_interval") == 0) {
            field = FIELD_CHECKPOINT_INTERVAL;
            if (!neural_parse_size(tokens.items[1],
                                   &config->checkpoint_interval)) {
                neural_error_set(
                    error,
                    "%s:%zu: checkpoint_interval must be a "
                    "non-negative integer",
                    path,
                    line_number);
                goto cleanup;
            }
        } else if (strcmp(tokens.items[0], "early_stopping_patience") == 0) {
            field = FIELD_EARLY_PATIENCE;
            if (!neural_parse_size(tokens.items[1],
                                   &config->early_stopping_patience)) {
                neural_error_set(
                    error,
                    "%s:%zu: early_stopping_patience must be non-negative",
                    path,
                    line_number);
                goto cleanup;
            }
        } else if (strcmp(tokens.items[0],
                          "early_stopping_min_delta") == 0) {
            field = FIELD_EARLY_MIN_DELTA;
            if (!neural_parse_real(tokens.items[1],
                                   &config->early_stopping_min_delta) ||
                config->early_stopping_min_delta < 0.0) {
                neural_error_set(
                    error,
                    "%s:%zu: early_stopping_min_delta must be finite and "
                    "non-negative",
                    path,
                    line_number);
                goto cleanup;
            }
        } else {
            neural_error_set(error,
                             "%s:%zu: unknown configuration property '%s'",
                             path,
                             line_number,
                             tokens.items[0]);
            goto cleanup;
        }

        if ((fields & field) != 0U) {
            neural_error_set(error,
                             "%s:%zu: property '%s' was specified more than once",
                             path,
                             line_number,
                             tokens.items[0]);
            goto cleanup;
        }
        fields |= field;
    }

    if ((fields & REQUIRED_FIELDS) != REQUIRED_FIELDS) {
        neural_error_set(error,
                         "%s: required properties are epochs, learning_rate, "
                         "seed, loss, and checkpoint_interval",
                         path);
        goto cleanup;
    }
    if ((fields & EARLY_FIELDS) != 0U &&
        (fields & EARLY_FIELDS) != EARLY_FIELDS) {
        neural_error_set(
            error,
            "%s: early_stopping_patience and early_stopping_min_delta "
            "must be specified together",
            path);
        goto cleanup;
    }
    if (!neural_training_config_validate(config, error)) {
        goto cleanup;
    }
    success = 1;

cleanup:
    free(line);
    token_list_free(&tokens);
    (void)fclose(stream);
    if (!success) {
        memset(config, 0, sizeof(*config));
    }
    return success;
}

static int reserve_dataset(NeuralDataset *dataset,
                           size_t *capacity,
                           size_t required,
                           NeuralError *error)
{
    size_t new_capacity;
    size_t input_elements;
    size_t output_elements;
    neural_real *new_values;

    if (required <= *capacity) {
        return 1;
    }
    new_capacity = *capacity == 0U
                       ? NEURAL_DEFAULT_SAMPLE_CAPACITY
                       : *capacity;
    while (new_capacity < required) {
        if (new_capacity >
            SIZE_MAX / NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR) {
            neural_error_set(error, "dataset contains too many samples");
            return 0;
        }
        new_capacity *= NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR;
    }
    if (!checked_multiply(new_capacity,
                          dataset->input_count,
                          &input_elements) ||
        input_elements > SIZE_MAX / sizeof(*dataset->inputs) ||
        !checked_multiply(new_capacity,
                          dataset->output_count,
                          &output_elements) ||
        output_elements > SIZE_MAX / sizeof(*dataset->outputs)) {
        neural_error_set(error, "dataset dimensions exceed addressable memory");
        return 0;
    }

    new_values = realloc(dataset->inputs,
                         input_elements * sizeof(*dataset->inputs));
    if (new_values == NULL) {
        neural_error_set(error, "unable to allocate dataset inputs");
        return 0;
    }
    dataset->inputs = new_values;
    new_values = realloc(dataset->outputs,
                         output_elements * sizeof(*dataset->outputs));
    if (new_values == NULL) {
        neural_error_set(error, "unable to allocate dataset outputs");
        return 0;
    }
    dataset->outputs = new_values;
    *capacity = new_capacity;
    return 1;
}

void neural_dataset_free(NeuralDataset *dataset)
{
    if (dataset != NULL) {
        free(dataset->inputs);
        free(dataset->outputs);
        memset(dataset, 0, sizeof(*dataset));
    }
}

int neural_dataset_load(const char *path,
                        size_t input_count,
                        size_t output_count,
                        NeuralDataset *dataset,
                        NeuralError *error)
{
    FILE *stream;
    char *line = NULL;
    size_t line_capacity = 0U;
    size_t line_number = 0U;
    size_t sample_capacity = 0U;
    size_t expected_tokens;
    int success = 0;
    TokenList tokens = {0};

    if (path == NULL || dataset == NULL || input_count == 0U ||
        output_count == 0U || output_count >= SIZE_MAX ||
        input_count > SIZE_MAX - output_count - 1U) {
        neural_error_set(error, "invalid dataset loader arguments");
        return 0;
    }
    memset(dataset, 0, sizeof(*dataset));
    dataset->input_count = input_count;
    dataset->output_count = output_count;
    expected_tokens = input_count + output_count + 1U;
    neural_error_clear(error);
    stream = fopen(path, "r");
    if (stream == NULL) {
        neural_error_set(error, "%s: unable to open file: %s", path, strerror(errno));
        return 0;
    }
    if (!parse_header(stream,
                      path,
                      "dataset",
                      &line,
                      &line_capacity,
                      &line_number,
                      &tokens,
                      error)) {
        goto cleanup;
    }

    for (;;) {
        size_t value_index;
        size_t input_offset;
        size_t output_offset;
        int status = next_content_line(stream,
                                       path,
                                       &line,
                                       &line_capacity,
                                       &line_number,
                                       &tokens,
                                       error);
        if (status < 0) {
            goto cleanup;
        }
        if (status == 0) {
            break;
        }
        if (tokens.count != expected_tokens ||
            strcmp(tokens.items[input_count], "->") != 0) {
            neural_error_set(error,
                             "%s:%zu: expected %zu inputs, '->', and %zu outputs",
                             path,
                             line_number,
                             input_count,
                             output_count);
            goto cleanup;
        }
        if (dataset->sample_count == SIZE_MAX ||
            !reserve_dataset(dataset,
                             &sample_capacity,
                             dataset->sample_count + 1U,
                             error)) {
            goto cleanup;
        }
        input_offset = dataset->sample_count * input_count;
        output_offset = dataset->sample_count * output_count;
        for (value_index = 0U; value_index < input_count; value_index++) {
            if (!neural_parse_real(tokens.items[value_index],
                                   &dataset->inputs[input_offset + value_index])) {
                neural_error_set(error,
                                 "%s:%zu: invalid finite input value '%s'",
                                 path,
                                 line_number,
                                 tokens.items[value_index]);
                goto cleanup;
            }
        }
        for (value_index = 0U; value_index < output_count; value_index++) {
            const char *text = tokens.items[input_count + 1U + value_index];
            if (!neural_parse_real(text,
                                   &dataset->outputs[output_offset + value_index])) {
                neural_error_set(error,
                                 "%s:%zu: invalid finite output value '%s'",
                                 path,
                                 line_number,
                                 text);
                goto cleanup;
            }
        }
        dataset->sample_count++;
    }

    if (dataset->sample_count == 0U) {
        neural_error_set(error, "%s: dataset must contain at least one sample", path);
        goto cleanup;
    }
    success = 1;

cleanup:
    free(line);
    token_list_free(&tokens);
    (void)fclose(stream);
    if (!success) {
        neural_dataset_free(dataset);
    }
    return success;
}

void neural_project_free(NeuralProject *project)
{
    if (project != NULL) {
        neural_model_spec_free(&project->model);
        neural_dataset_free(&project->dataset);
        neural_dataset_free(&project->validation);
        neural_preprocessing_free(&project->preprocessing);
        memset(&project->training, 0, sizeof(project->training));
        project->has_validation = 0;
        project->has_preprocessing = 0;
    }
}

int neural_project_load(const char *directory,
                        NeuralProject *project,
                        NeuralError *error)
{
    char *model_path = NULL;
    char *config_path = NULL;
    char *dataset_path = NULL;
    char *validation_path = NULL;
    char *preprocessing_path = NULL;
    struct stat preprocessing_status;
    size_t output_count;
    int success = 0;

    if (directory == NULL || directory[0] == '\0' || project == NULL) {
        neural_error_set(error, "invalid project loader arguments");
        return 0;
    }
    memset(project, 0, sizeof(*project));
    model_path = neural_path_join(directory,
                                  NEURAL_DEFAULT_MODEL_FILENAME,
                                  error);
    config_path = neural_path_join(directory,
                                   NEURAL_DEFAULT_PROJECT_FILENAME,
                                   error);
    dataset_path = neural_path_join(directory,
                                    NEURAL_DEFAULT_DATASET_FILENAME,
                                    error);
    if (model_path == NULL || config_path == NULL || dataset_path == NULL) {
        goto cleanup;
    }
    if (!neural_model_spec_load(model_path, &project->model, error)) {
        goto cleanup;
    }
    if (!neural_training_config_load(config_path,
                                     &project->training,
                                     error)) {
        goto cleanup;
    }
    preprocessing_path = neural_path_join(
        directory, NEURAL_DEFAULT_PREPROCESSING_FILENAME, error);
    if (preprocessing_path == NULL) {
        goto cleanup;
    }
    if (lstat(preprocessing_path, &preprocessing_status) == 0) {
        if (!S_ISREG(preprocessing_status.st_mode) ||
            !neural_preprocessing_load(preprocessing_path,
                                       &project->preprocessing,
                                       error) ||
            project->preprocessing.input_count != project->model.input_count) {
            if (error != NULL && error->message[0] == '\0') {
                neural_error_set(error,
                                 "%s: preprocessing input width does not match model",
                                 preprocessing_path);
            }
            goto cleanup;
        }
        project->has_preprocessing = 1;
    } else if (errno != ENOENT) {
        neural_error_set(error,
                         "%s: unable to inspect file: %s",
                         preprocessing_path,
                         strerror(errno));
        goto cleanup;
    }
    output_count = project->model.layers[project->model.layer_count - 1U]
                       .neuron_count;
    if (!neural_dataset_load(dataset_path,
                             project->model.input_count,
                             output_count,
                             &project->dataset,
                             error)) {
        goto cleanup;
    }
    if (project->training.early_stopping_patience != 0U) {
        validation_path = neural_path_join(directory,
                                           NEURAL_DEFAULT_VALIDATION_FILENAME,
                                           error);
        if (validation_path == NULL ||
            !neural_dataset_load(validation_path,
                                 project->model.input_count,
                                 output_count,
                                 &project->validation,
                                 error)) {
            goto cleanup;
        }
        project->has_validation = 1;
    }
    success = 1;

cleanup:
    free(model_path);
    free(config_path);
    free(dataset_path);
    free(validation_path);
    free(preprocessing_path);
    if (!success) {
        neural_project_free(project);
    }
    return success;
}

const char *neural_loss_name(NeuralLoss loss)
{
    switch (loss) {
    case NEURAL_LOSS_MSE:
        return "mse";
    }
    return "unknown";
}

int neural_loss_from_name(const char *name, NeuralLoss *loss)
{
    if (name == NULL || loss == NULL) {
        return 0;
    }
    if (strcmp(name, "mse") == 0) {
        *loss = NEURAL_LOSS_MSE;
        return 1;
    }
    return 0;
}
