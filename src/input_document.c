#define _POSIX_C_SOURCE 200809L

#include "neural/input_document.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/defaults.h"
#include "neural/parse.h"
#include "neural/version.h"

struct NeuralInputDocument {
    FILE *stream;
    const char *path;
    char *line;
    size_t line_capacity;
    size_t line_number;
    char **tokens;
    size_t token_capacity;
    size_t token_count;
    size_t sample_count;
    size_t input_count;
    size_t next_sample;
    int owns_stream;
    int complete;
};

static int document_error(NeuralInputDocument *document,
                          NeuralError *error,
                          const char *message)
{
    neural_error_set(error,
                     "%s:%zu: %s",
                     document->path,
                     document->line_number,
                     message);
    return -1;
}

static int reserve_tokens(NeuralInputDocument *document,
                          size_t required,
                          NeuralError *error)
{
    size_t capacity = document->token_capacity == 0U
                          ? NEURAL_DEFAULT_TOKEN_CAPACITY
                          : document->token_capacity;
    char **tokens;

    while (capacity < required) {
        if (capacity > SIZE_MAX / NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR) {
            neural_error_set(error, "input document contains too many fields");
            return 0;
        }
        capacity *= NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR;
    }
    if (capacity > SIZE_MAX / sizeof(*tokens)) {
        neural_error_set(error, "input document fields exceed addressable memory");
        return 0;
    }
    tokens = realloc(document->tokens, capacity * sizeof(*tokens));
    if (tokens == NULL) {
        neural_error_set(error, "unable to allocate input document fields");
        return 0;
    }
    document->tokens = tokens;
    document->token_capacity = capacity;
    return 1;
}

static int next_content_line(NeuralInputDocument *document,
                             NeuralError *error)
{
    for (;;) {
        ssize_t length = getline(&document->line,
                                 &document->line_capacity,
                                 document->stream);
        char *cursor;

        if (length < 0) {
            if (ferror(document->stream) != 0) {
                document->line_number++;
                return document_error(document, error, "unable to read file");
            }
            return 0;
        }
        document->line_number++;
        if ((size_t)length > NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH) {
            return document_error(document,
                                  error,
                                  "line exceeds maximum length");
        }
        while (length > 0 &&
               (document->line[(size_t)length - 1U] == '\n' ||
                document->line[(size_t)length - 1U] == '\r')) {
            length--;
            document->line[(size_t)length] = '\0';
        }
        document->token_count = 0U;
        cursor = document->line;
        while (*cursor != '\0') {
            char *start;

            while (isspace((unsigned char)*cursor) != 0) {
                cursor++;
            }
            if (*cursor == '\0' || *cursor == '#') {
                break;
            }
            if (!reserve_tokens(document,
                                document->token_count + 1U,
                                error)) {
                return -1;
            }
            start = cursor;
            while (*cursor != '\0' && *cursor != '#' &&
                   isspace((unsigned char)*cursor) == 0) {
                cursor++;
            }
            document->tokens[document->token_count++] = start;
            if (*cursor == '#') {
                *cursor = '\0';
                break;
            }
            if (*cursor != '\0') {
                *cursor = '\0';
                cursor++;
            }
        }
        if (document->token_count != 0U) {
            return 1;
        }
    }
}

static int require_line(NeuralInputDocument *document,
                        size_t count,
                        const char *first,
                        NeuralError *error)
{
    int status = next_content_line(document, error);

    if (status < 0) {
        return 0;
    }
    if (status == 0) {
        neural_error_set(error,
                         "%s:%zu: expected '%s' before end of file",
                         document->path,
                         document->line_number + 1U,
                         first);
        return 0;
    }
    if (document->token_count != count ||
        strcmp(document->tokens[0], first) != 0) {
        neural_error_set(error,
                         "%s:%zu: expected '%s'",
                         document->path,
                         document->line_number,
                         first);
        return 0;
    }
    return 1;
}

int neural_input_document_open(const char *path,
                               NeuralInputDocument **document,
                               NeuralError *error)
{
    NeuralInputDocument *opened = NULL;
    size_t version;
    int success = 0;

    neural_error_clear(error);
    if (path == NULL || path[0] == '\0' || document == NULL) {
        neural_error_set(error, "input document path is required");
        return 0;
    }
    *document = NULL;
    opened = calloc(1U, sizeof(*opened));
    if (opened == NULL) {
        neural_error_set(error, "unable to allocate input document reader");
        return 0;
    }
    opened->path = strcmp(path, "-") == 0 ? "<stdin>" : path;
    if (strcmp(path, "-") == 0) {
        opened->stream = stdin;
    } else {
        opened->stream = fopen(path, "r");
        opened->owns_stream = 1;
    }
    if (opened->stream == NULL) {
        neural_error_set(error,
                         "%s: unable to open file: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    if (!require_line(opened, 3U, NEURAL_FORMAT_MAGIC, error) ||
        strcmp(opened->tokens[1], "inputs") != 0 ||
        !neural_parse_size(opened->tokens[2], &version) ||
        version != 1U) {
        if (error->message[0] == '\0') {
            neural_error_set(error,
                             "%s:%zu: expected '%s inputs 1'",
                             opened->path,
                             opened->line_number,
                             NEURAL_FORMAT_MAGIC);
        }
        goto cleanup;
    }
    if (!require_line(opened, 2U, "samples", error) ||
        !neural_parse_size(opened->tokens[1], &opened->sample_count) ||
        opened->sample_count == 0U) {
        neural_error_set(error,
                         "%s:%zu: samples must be a positive integer",
                         opened->path,
                         opened->line_number);
        goto cleanup;
    }
    if (!require_line(opened, 2U, "inputs", error) ||
        !neural_parse_size(opened->tokens[1], &opened->input_count) ||
        opened->input_count == 0U) {
        neural_error_set(error,
                         "%s:%zu: inputs must be a positive integer",
                         opened->path,
                         opened->line_number);
        goto cleanup;
    }
    *document = opened;
    opened = NULL;
    success = 1;

cleanup:
    neural_input_document_close(opened);
    return success;
}

size_t neural_input_document_sample_count(const NeuralInputDocument *document)
{
    return document == NULL ? 0U : document->sample_count;
}

size_t neural_input_document_input_count(const NeuralInputDocument *document)
{
    return document == NULL ? 0U : document->input_count;
}

int neural_input_document_read(NeuralInputDocument *document,
                               neural_real *inputs,
                               size_t batch_capacity,
                               size_t *sample_count,
                               int *complete,
                               NeuralError *error)
{
    size_t loaded = 0U;

    neural_error_clear(error);
    if (document == NULL || inputs == NULL || batch_capacity == 0U ||
        sample_count == NULL || complete == NULL || document->complete) {
        neural_error_set(error, "input document read arguments are invalid");
        return 0;
    }
    *sample_count = 0U;
    *complete = 0;
    while (loaded < batch_capacity &&
           document->next_sample < document->sample_count) {
        size_t parsed_index;
        size_t value_index;

        if (!require_line(document,
                          document->input_count + 2U,
                          "sample",
                          error) ||
            !neural_parse_size(document->tokens[1], &parsed_index) ||
            parsed_index != document->next_sample) {
            neural_error_set(error,
                             "%s:%zu: expected sample %zu with %zu inputs",
                             document->path,
                             document->line_number,
                             document->next_sample,
                             document->input_count);
            return 0;
        }
        for (value_index = 0U;
             value_index < document->input_count;
             value_index++) {
            const char *text = document->tokens[value_index + 2U];

            if (strcmp(text, "?") == 0) {
                inputs[loaded * document->input_count + value_index] = NAN;
            } else if (!neural_parse_real(
                           text,
                           &inputs[loaded * document->input_count + value_index])) {
                neural_error_set(error,
                                 "%s:%zu: invalid finite input %zu '%s'",
                                 document->path,
                                 document->line_number,
                                 value_index,
                                 text);
                return 0;
            }
        }
        loaded++;
        document->next_sample++;
    }
    if (document->next_sample == document->sample_count) {
        int status;

        if (!require_line(document, 1U, "end", error)) {
            return 0;
        }
        status = next_content_line(document, error);
        if (status < 0) {
            return 0;
        }
        if (status != 0) {
            neural_error_set(error,
                             "%s:%zu: unexpected content after end",
                             document->path,
                             document->line_number);
            return 0;
        }
        document->complete = 1;
        *complete = 1;
    }
    *sample_count = loaded;
    return 1;
}

void neural_input_document_close(NeuralInputDocument *document)
{
    if (document != NULL) {
        if (document->owns_stream && document->stream != NULL) {
            (void)fclose(document->stream);
        }
        free(document->tokens);
        free(document->line);
        free(document);
    }
}
