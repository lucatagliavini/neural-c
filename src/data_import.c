#define _POSIX_C_SOURCE 200809L

#include "neural/data_import.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/defaults.h"
#include "neural/parse.h"
#include "neural/project.h"
#include "neural/random.h"
#include "neural/version.h"
#include "path.h"
#include "project_lock.h"
#include "sha256.h"

typedef struct {
    char *name;
    neural_real *outputs;
} CsvClass;

typedef struct {
    size_t column_count;
    int has_header;
    size_t *input_columns;
    size_t input_count;
    size_t *target_columns;
    size_t target_count;
    size_t label_column;
    int categorical;
    CsvClass *classes;
    size_t class_count;
} CsvSchema;

typedef struct {
    neural_real *inputs;
    neural_real *outputs;
    size_t *classes;
    size_t sample_count;
    size_t capacity;
    size_t input_count;
    size_t output_count;
} ImportedRows;

typedef struct {
    char *final_path;
    char *new_path;
    char *old_path;
    int wanted;
    int had_old;
    int installed;
} ImportFile;

enum {
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_TEST
};

static void schema_free(CsvSchema *schema)
{
    size_t index;

    if (schema == NULL) {
        return;
    }
    for (index = 0U; index < schema->class_count; index++) {
        free(schema->classes[index].name);
        free(schema->classes[index].outputs);
    }
    free(schema->classes);
    free(schema->target_columns);
    free(schema->input_columns);
    memset(schema, 0, sizeof(*schema));
}

static void rows_free(ImportedRows *rows)
{
    if (rows != NULL) {
        free(rows->classes);
        free(rows->outputs);
        free(rows->inputs);
        memset(rows, 0, sizeof(*rows));
    }
}

static char *trim(char *text)
{
    char *end;

    while (isspace((unsigned char)*text) != 0) {
        text++;
    }
    end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1]) != 0) {
        *--end = '\0';
    }
    return text;
}

static size_t split_words(char *line, char **tokens, size_t capacity)
{
    size_t count = 0U;
    char *cursor = line;

    while (*cursor != '\0') {
        while (isspace((unsigned char)*cursor) != 0) {
            cursor++;
        }
        if (*cursor == '\0' || *cursor == '#') {
            break;
        }
        if (count == capacity) {
            return SIZE_MAX;
        }
        tokens[count++] = cursor;
        while (*cursor != '\0' && *cursor != '#' &&
               isspace((unsigned char)*cursor) == 0) {
            cursor++;
        }
        if (*cursor == '#') {
            *cursor = '\0';
            break;
        }
        if (*cursor != '\0') {
            *cursor++ = '\0';
        }
    }
    return count;
}

static int allocate_indices(char **tokens,
                            size_t count,
                            size_t **indices,
                            NeuralError *error)
{
    size_t *loaded;
    size_t index;

    if (count == 0U || count > SIZE_MAX / sizeof(*loaded)) {
        neural_error_set(error, "schema column list is invalid");
        return 0;
    }
    loaded = malloc(count * sizeof(*loaded));
    if (loaded == NULL) {
        neural_error_set(error, "unable to allocate schema columns");
        return 0;
    }
    for (index = 0U; index < count; index++) {
        size_t previous;

        if (!neural_parse_size(tokens[index], &loaded[index])) {
            neural_error_set(error, "invalid schema column '%s'", tokens[index]);
            free(loaded);
            return 0;
        }
        for (previous = 0U; previous < index; previous++) {
            if (loaded[previous] == loaded[index]) {
                neural_error_set(error,
                                 "schema column %zu is repeated",
                                 loaded[index]);
                free(loaded);
                return 0;
            }
        }
    }
    *indices = loaded;
    return 1;
}

static int schema_validate(const CsvSchema *schema,
                           size_t model_inputs,
                           size_t model_outputs,
                           NeuralError *error)
{
    size_t index;

    if (schema->column_count == 0U ||
        schema->input_count != model_inputs ||
        (!schema->categorical && schema->target_count != model_outputs) ||
        (schema->categorical && schema->class_count == 0U)) {
        neural_error_set(error,
                         "schema dimensions do not match model (%zu inputs, %zu outputs)",
                         model_inputs,
                         model_outputs);
        return 0;
    }
    for (index = 0U; index < schema->input_count; index++) {
        size_t target;

        if (schema->input_columns[index] >= schema->column_count) {
            neural_error_set(error, "input column %zu is out of range",
                             schema->input_columns[index]);
            return 0;
        }
        if (schema->categorical &&
            schema->input_columns[index] == schema->label_column) {
            neural_error_set(error,
                             "label column %zu is also an input column",
                             schema->label_column);
            return 0;
        }
        for (target = 0U; !schema->categorical &&
             target < schema->target_count; target++) {
            if (schema->input_columns[index] ==
                schema->target_columns[target]) {
                neural_error_set(error,
                                 "column %zu is both input and target",
                                 schema->input_columns[index]);
                return 0;
            }
        }
    }
    if (schema->categorical) {
        if (schema->label_column >= schema->column_count) {
            neural_error_set(error, "label column is out of range");
            return 0;
        }
    } else {
        for (index = 0U; index < schema->target_count; index++) {
            if (schema->target_columns[index] >= schema->column_count) {
                neural_error_set(error, "target column %zu is out of range",
                                 schema->target_columns[index]);
                return 0;
            }
        }
    }
    return 1;
}

static int schema_load(const char *path,
                       size_t model_inputs,
                       size_t model_outputs,
                       CsvSchema *schema,
                       NeuralError *error)
{
    FILE *stream = NULL;
    char *line = NULL;
    size_t capacity = 0U;
    size_t line_number = 0U;
    int saw_header = 0;
    int saw_header_setting = 0;
    int saw_end = 0;
    int success = 0;

    stream = fopen(path, "r");
    if (stream == NULL) {
        neural_error_set(error, "%s: unable to open schema: %s",
                         path, strerror(errno));
        return 0;
    }
    while (!saw_end) {
        ssize_t length = getline(&line, &capacity, stream);
        char *tokens[256];
        size_t count;
        char *content;

        if (length < 0) {
            if (ferror(stream) != 0) {
                neural_error_set(error, "%s:%zu: unable to read schema",
                                 path, line_number + 1U);
            } else {
                neural_error_set(error, "%s:%zu: expected end",
                                 path, line_number + 1U);
            }
            goto cleanup;
        }
        line_number++;
        if ((size_t)length > NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH) {
            neural_error_set(error, "%s:%zu: line is too long", path, line_number);
            goto cleanup;
        }
        content = trim(line);
        count = split_words(content, tokens,
                            sizeof(tokens) / sizeof(tokens[0]));
        if (count == 0U) {
            continue;
        }
        if (count == SIZE_MAX) {
            neural_error_set(error, "%s:%zu: too many schema fields",
                             path, line_number);
            goto cleanup;
        }
        if (!saw_header) {
            if (count != 3U || strcmp(tokens[0], NEURAL_FORMAT_MAGIC) != 0 ||
                strcmp(tokens[1], "csv-schema") != 0 ||
                strcmp(tokens[2], "1") != 0) {
                neural_error_set(error,
                                 "%s:%zu: expected '%s csv-schema 1'",
                                 path, line_number, NEURAL_FORMAT_MAGIC);
                goto cleanup;
            }
            saw_header = 1;
        } else if (strcmp(tokens[0], "columns") == 0 && count == 2U &&
                   schema->column_count == 0U) {
            if (!neural_parse_size(tokens[1], &schema->column_count) ||
                schema->column_count == 0U) {
                neural_error_set(error, "%s:%zu: invalid column count",
                                 path, line_number);
                goto cleanup;
            }
        } else if (strcmp(tokens[0], "header") == 0 && count == 2U) {
            if (saw_header_setting) {
                neural_error_set(error, "%s:%zu: duplicate header setting",
                                 path, line_number);
                goto cleanup;
            }
            if (strcmp(tokens[1], "yes") == 0) {
                schema->has_header = 1;
            } else if (strcmp(tokens[1], "no") != 0) {
                neural_error_set(error, "%s:%zu: header must be yes or no",
                                 path, line_number);
                goto cleanup;
            }
            saw_header_setting = 1;
        } else if (strcmp(tokens[0], "inputs") == 0 && count > 1U &&
                   schema->input_columns == NULL) {
            if (!allocate_indices(tokens + 1U, count - 1U,
                                  &schema->input_columns, error)) {
                goto cleanup;
            }
            schema->input_count = count - 1U;
        } else if (strcmp(tokens[0], "targets") == 0 && count > 1U &&
                   schema->target_columns == NULL && !schema->categorical) {
            if (!allocate_indices(tokens + 1U, count - 1U,
                                  &schema->target_columns, error)) {
                goto cleanup;
            }
            schema->target_count = count - 1U;
        } else if (strcmp(tokens[0], "label") == 0 && count == 2U &&
                   schema->target_columns == NULL && !schema->categorical) {
            if (!neural_parse_size(tokens[1], &schema->label_column)) {
                neural_error_set(error, "%s:%zu: invalid label column",
                                 path, line_number);
                goto cleanup;
            }
            schema->categorical = 1;
        } else if (strcmp(tokens[0], "class") == 0 &&
                   count == model_outputs + 2U && schema->categorical) {
            CsvClass *classes;
            CsvClass *item;
            size_t index;

            if (schema->class_count == SIZE_MAX / sizeof(*classes)) {
                neural_error_set(error, "too many categorical classes");
                goto cleanup;
            }
            classes = realloc(schema->classes,
                              (schema->class_count + 1U) * sizeof(*classes));
            if (classes == NULL) {
                neural_error_set(error, "unable to allocate class mapping");
                goto cleanup;
            }
            schema->classes = classes;
            item = &schema->classes[schema->class_count];
            for (index = 0U; index < schema->class_count; index++) {
                if (strcmp(schema->classes[index].name, tokens[1]) == 0) {
                    neural_error_set(error,
                                     "%s:%zu: duplicate class label '%s'",
                                     path, line_number, tokens[1]);
                    goto cleanup;
                }
            }
            memset(item, 0, sizeof(*item));
            item->name = strdup(tokens[1]);
            item->outputs = malloc(model_outputs * sizeof(*item->outputs));
            if (item->name == NULL || item->outputs == NULL) {
                neural_error_set(error, "unable to allocate class mapping");
                free(item->name);
                free(item->outputs);
                memset(item, 0, sizeof(*item));
                goto cleanup;
            }
            schema->class_count++;
            for (index = 0U; index < model_outputs; index++) {
                if (!neural_parse_real(tokens[index + 2U],
                                       &item->outputs[index])) {
                    neural_error_set(error,
                                     "%s:%zu: invalid class output '%s'",
                                     path, line_number, tokens[index + 2U]);
                    goto cleanup;
                }
            }
        } else if (strcmp(tokens[0], "end") == 0 && count == 1U) {
            saw_end = 1;
        } else {
            neural_error_set(error, "%s:%zu: invalid or duplicate schema entry",
                             path, line_number);
            goto cleanup;
        }
    }
    while (getline(&line, &capacity, stream) >= 0) {
        line_number++;
        if (*trim(line) != '\0' && *trim(line) != '#') {
            neural_error_set(error, "%s:%zu: content after end",
                             path, line_number);
            goto cleanup;
        }
    }
    if (!saw_header_setting) {
        neural_error_set(error, "%s: schema must explicitly set header yes|no",
                         path);
        goto cleanup;
    }
    if (!schema_validate(schema, model_inputs, model_outputs, error)) {
        goto cleanup;
    }
    success = 1;

cleanup:
    free(line);
    (void)fclose(stream);
    if (!success) {
        schema_free(schema);
    }
    return success;
}

static int parse_csv_line(char *line,
                          char ***output_fields,
                          size_t *output_count,
                          NeuralError *error)
{
    char **fields = NULL;
    size_t count = 0U;
    size_t capacity = 0U;
    char *read = line;
    char *write = line;

    for (;;) {
        char *start;

        if (count == capacity) {
            size_t next = capacity == 0U ? 16U : capacity * 2U;
            char **resized;

            if (next < capacity || next > SIZE_MAX / sizeof(*resized)) {
                neural_error_set(error, "CSV row has too many columns");
                free(fields);
                return 0;
            }
            resized = realloc(fields, next * sizeof(*resized));
            if (resized == NULL) {
                neural_error_set(error, "unable to allocate CSV columns");
                free(fields);
                return 0;
            }
            fields = resized;
            capacity = next;
        }
        start = write;
        if (*read == '"') {
            int closed = 0;

            read++;
            while (*read != '\0') {
                if (*read == '"') {
                    if (read[1] == '"') {
                        *write++ = '"';
                        read += 2;
                    } else {
                        read++;
                        closed = 1;
                        break;
                    }
                } else {
                    *write++ = *read++;
                }
            }
            if (!closed || (*read != ',' && *read != '\0')) {
                neural_error_set(error, "invalid quoted CSV field");
                free(fields);
                return 0;
            }
        } else {
            while (*read != ',' && *read != '\0') {
                if (*read == '"') {
                    neural_error_set(error, "quote in unquoted CSV field");
                    free(fields);
                    return 0;
                }
                *write++ = *read++;
            }
        }
        {
            int has_more = *read == ',';

            if (has_more) {
                read++;
            }
            *write++ = '\0';
            fields[count++] = start;
            if (!has_more) {
                break;
            }
        }
        if (*read == '\0') {
            if (count == capacity) {
                char **resized = realloc(fields,
                    (capacity + 1U) * sizeof(*resized));
                if (resized == NULL) {
                    neural_error_set(error, "unable to allocate CSV columns");
                    free(fields);
                    return 0;
                }
                fields = resized;
            }
            fields[count++] = write;
            *write = '\0';
            break;
        }
    }
    *output_fields = fields;
    *output_count = count;
    return 1;
}

static int rows_reserve(ImportedRows *rows, NeuralError *error)
{
    size_t next = rows->capacity == 0U ? 64U : rows->capacity * 2U;
    neural_real *inputs;
    neural_real *outputs;
    size_t *classes;

    if (next < rows->capacity || next > SIZE_MAX / rows->input_count ||
        next * rows->input_count > SIZE_MAX / sizeof(*inputs) ||
        next > SIZE_MAX / rows->output_count ||
        next * rows->output_count > SIZE_MAX / sizeof(*outputs)) {
        neural_error_set(error, "CSV dataset is too large");
        return 0;
    }
    inputs = realloc(rows->inputs,
                     next * rows->input_count * sizeof(*inputs));
    if (inputs == NULL) {
        neural_error_set(error, "unable to allocate imported inputs");
        return 0;
    }
    rows->inputs = inputs;
    outputs = realloc(rows->outputs,
                      next * rows->output_count * sizeof(*outputs));
    if (outputs == NULL) {
        neural_error_set(error, "unable to allocate imported outputs");
        return 0;
    }
    rows->outputs = outputs;
    classes = realloc(rows->classes, next * sizeof(*classes));
    if (classes == NULL) {
        neural_error_set(error, "unable to allocate imported labels");
        return 0;
    }
    rows->classes = classes;
    rows->capacity = next;
    return 1;
}

static int csv_load(const char *path,
                    const CsvSchema *schema,
                    size_t output_count,
                    ImportedRows *rows,
                    NeuralError *error)
{
    FILE *stream = fopen(path, "r");
    char *line = NULL;
    size_t capacity = 0U;
    size_t line_number = 0U;
    int skipped_header = 0;
    int success = 0;

    if (stream == NULL) {
        neural_error_set(error, "%s: unable to open CSV: %s",
                         path, strerror(errno));
        return 0;
    }
    rows->input_count = schema->input_count;
    rows->output_count = output_count;
    while (1) {
        ssize_t length = getline(&line, &capacity, stream);
        char **fields = NULL;
        size_t field_count = 0U;
        size_t index;

        if (length < 0) {
            if (ferror(stream) != 0) {
                neural_error_set(error, "%s:%zu: unable to read CSV",
                                 path, line_number + 1U);
                goto cleanup;
            }
            break;
        }
        line_number++;
        if ((size_t)length > NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH) {
            neural_error_set(error, "%s:%zu: CSV row is too long",
                             path, line_number);
            goto cleanup;
        }
        while (length > 0 &&
               (line[(size_t)length - 1U] == '\n' ||
                line[(size_t)length - 1U] == '\r')) {
            line[--length] = '\0';
        }
        if (schema->has_header && !skipped_header) {
            skipped_header = 1;
            if (!parse_csv_line(line, &fields, &field_count, error) ||
                field_count != schema->column_count) {
                if (error->message[0] == '\0') {
                    neural_error_set(error,
                                     "%s:%zu: header has %zu columns; expected %zu",
                                     path, line_number, field_count,
                                     schema->column_count);
                }
                free(fields);
                goto cleanup;
            }
            free(fields);
            continue;
        }
        if (length == 0) {
            neural_error_set(error, "%s:%zu: empty CSV row", path, line_number);
            goto cleanup;
        }
        if (!parse_csv_line(line, &fields, &field_count, error)) {
            char detail[NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY];

            (void)snprintf(detail, sizeof(detail), "%s", error->message);
            neural_error_set(error, "%s:%zu: %s", path, line_number, detail);
            goto cleanup;
        }
        if (field_count != schema->column_count) {
            neural_error_set(error,
                             "%s:%zu: row has %zu columns; expected %zu",
                             path, line_number, field_count,
                             schema->column_count);
            free(fields);
            goto cleanup;
        }
        if (rows->sample_count == rows->capacity &&
            !rows_reserve(rows, error)) {
            free(fields);
            goto cleanup;
        }
        for (index = 0U; index < schema->input_count; index++) {
            const char *value = fields[schema->input_columns[index]];
            neural_real *destination = &rows->inputs[
                rows->sample_count * rows->input_count + index];

            if (value[0] == '\0' || strcmp(value, "?") == 0) {
                *destination = NAN;
            } else if (!neural_parse_real(value, destination)) {
                neural_error_set(error,
                                 "%s:%zu: invalid input column %zu value '%s'",
                                 path, line_number,
                                 schema->input_columns[index], value);
                free(fields);
                goto cleanup;
            }
        }
        if (schema->categorical) {
            const char *label = fields[schema->label_column];
            size_t class_index;

            for (class_index = 0U; class_index < schema->class_count;
                 class_index++) {
                if (strcmp(label, schema->classes[class_index].name) == 0) {
                    break;
                }
            }
            if (class_index == schema->class_count) {
                neural_error_set(error,
                                 "%s:%zu: unmapped label '%s' in column %zu",
                                 path, line_number, label, schema->label_column);
                free(fields);
                goto cleanup;
            }
            memcpy(&rows->outputs[rows->sample_count * output_count],
                   schema->classes[class_index].outputs,
                   output_count * sizeof(*rows->outputs));
            rows->classes[rows->sample_count] = class_index;
        } else {
            for (index = 0U; index < schema->target_count; index++) {
                const char *value = fields[schema->target_columns[index]];
                if (value[0] == '\0' || strcmp(value, "?") == 0 ||
                    !neural_parse_real(value,
                        &rows->outputs[rows->sample_count * output_count + index])) {
                    neural_error_set(error,
                                     "%s:%zu: invalid target column %zu value '%s'",
                                     path, line_number,
                                     schema->target_columns[index], value);
                    free(fields);
                    goto cleanup;
                }
            }
            rows->classes[rows->sample_count] = 0U;
        }
        rows->sample_count++;
        free(fields);
    }
    if (schema->has_header && !skipped_header) {
        neural_error_set(error, "%s: CSV header is missing", path);
        goto cleanup;
    }
    if (rows->sample_count == 0U) {
        neural_error_set(error, "%s: CSV contains no data rows", path);
        goto cleanup;
    }
    success = 1;

cleanup:
    free(line);
    (void)fclose(stream);
    if (!success) {
        rows_free(rows);
    }
    return success;
}

static void shuffle_indices(size_t *indices,
                            size_t count,
                            NeuralRandom *random)
{
    while (count > 1U) {
        size_t selected = (size_t)(neural_random_next_uint64(random) % count);
        size_t temporary = indices[count - 1U];

        indices[count - 1U] = indices[selected];
        indices[selected] = temporary;
        count--;
    }
}

static size_t ratio_count(size_t count, neural_real ratio)
{
    return (size_t)floor((neural_real)count * ratio);
}

static int create_split(const ImportedRows *rows,
                        const CsvSchema *schema,
                        const NeuralDataImportConfig *config,
                        unsigned char **assignment_output,
                        NeuralDataImportResult *result,
                        NeuralError *error)
{
    unsigned char *assignment = calloc(rows->sample_count, 1U);
    NeuralRandom random;
    size_t group_count = schema->categorical ? schema->class_count : 1U;
    size_t group;

    if (assignment == NULL) {
        neural_error_set(error, "unable to allocate split assignments");
        return 0;
    }
    neural_random_init(&random, config->split_seed);
    for (group = 0U; group < group_count; group++) {
        size_t count = 0U;
        size_t *indices;
        size_t index;
        size_t test_count;
        size_t validation_count;

        for (index = 0U; index < rows->sample_count; index++) {
            if (!schema->categorical || rows->classes[index] == group) {
                count++;
            }
        }
        if (count == 0U) {
            neural_error_set(error, "categorical class %zu has no CSV rows", group);
            free(assignment);
            return 0;
        }
        indices = malloc(count * sizeof(*indices));
        if (indices == NULL) {
            neural_error_set(error, "unable to allocate split order");
            free(assignment);
            return 0;
        }
        count = 0U;
        for (index = 0U; index < rows->sample_count; index++) {
            if (!schema->categorical || rows->classes[index] == group) {
                indices[count++] = index;
            }
        }
        shuffle_indices(indices, count, &random);
        test_count = ratio_count(count, config->test_ratio);
        validation_count = ratio_count(count, config->validation_ratio);
        if (test_count + validation_count >= count) {
            neural_error_set(error, "split leaves no training samples in group %zu",
                             group);
            free(indices);
            free(assignment);
            return 0;
        }
        for (index = 0U; index < test_count; index++) {
            assignment[indices[index]] = SPLIT_TEST;
        }
        for (; index < test_count + validation_count; index++) {
            assignment[indices[index]] = SPLIT_VALIDATION;
        }
        free(indices);
    }
    memset(result, 0, sizeof(*result));
    result->total_samples = rows->sample_count;
    result->stratified = schema->categorical;
    for (group = 0U; group < rows->sample_count; group++) {
        if (assignment[group] == SPLIT_TEST) {
            result->test_samples++;
        } else if (assignment[group] == SPLIT_VALIDATION) {
            result->validation_samples++;
        } else {
            result->training_samples++;
        }
    }
    if (result->training_samples == 0U ||
        (config->test_ratio > 0.0 && result->test_samples == 0U) ||
        (config->validation_ratio > 0.0 && result->validation_samples == 0U)) {
        neural_error_set(error,
                         "requested split cannot be represented by %zu samples",
                         rows->sample_count);
        free(assignment);
        return 0;
    }
    *assignment_output = assignment;
    return 1;
}

static int fit_preprocessing(ImportedRows *rows,
                             const unsigned char *assignment,
                             const NeuralDataImportConfig *config,
                             NeuralPreprocessing *preprocessing,
                             NeuralError *error)
{
    size_t feature;
    size_t sample;

    preprocessing->input_count = rows->input_count;
    preprocessing->normalization = config->normalization;
    preprocessing->missing_policy = config->missing_policy;
    preprocessing->offsets = calloc(rows->input_count,
                                    sizeof(*preprocessing->offsets));
    preprocessing->scales = malloc(rows->input_count *
                                   sizeof(*preprocessing->scales));
    preprocessing->imputations = malloc(rows->input_count *
                                        sizeof(*preprocessing->imputations));
    if (preprocessing->offsets == NULL || preprocessing->scales == NULL ||
        preprocessing->imputations == NULL) {
        neural_error_set(error, "unable to allocate preprocessing statistics");
        return 0;
    }
    for (feature = 0U; feature < rows->input_count; feature++) {
        neural_real sum = 0.0;
        size_t count = 0U;

        for (sample = 0U; sample < rows->sample_count; sample++) {
            neural_real value = rows->inputs[sample * rows->input_count + feature];
            if (assignment[sample] != SPLIT_TRAIN) {
                continue;
            }
            if (isnan(value)) {
                if (config->missing_policy == NEURAL_MISSING_REJECT) {
                    neural_error_set(error,
                                     "missing training input at sample %zu feature %zu",
                                     sample, feature);
                    return 0;
                }
                continue;
            }
            sum += value;
            count++;
        }
        if (count == 0U || !isfinite(sum)) {
            neural_error_set(error,
                             "training feature %zu has no finite values",
                             feature);
            return 0;
        }
        preprocessing->imputations[feature] = sum / (neural_real)count;
        preprocessing->scales[feature] = 1.0;
        if (config->normalization == NEURAL_NORMALIZATION_STANDARDIZE) {
            neural_real squared = 0.0;
            neural_real mean = preprocessing->imputations[feature];
            size_t training_count = 0U;

            for (sample = 0U; sample < rows->sample_count; sample++) {
                neural_real value;
                neural_real delta;
                if (assignment[sample] != SPLIT_TRAIN) {
                    continue;
                }
                value = rows->inputs[sample * rows->input_count + feature];
                if (isnan(value)) {
                    value = mean;
                }
                delta = value - mean;
                squared += delta * delta;
                training_count++;
            }
            preprocessing->offsets[feature] = mean;
            preprocessing->scales[feature] =
                sqrt(squared / (neural_real)training_count);
            if (preprocessing->scales[feature] == 0.0) {
                preprocessing->scales[feature] = 1.0;
            }
        } else if (config->normalization == NEURAL_NORMALIZATION_MINMAX) {
            neural_real minimum = preprocessing->imputations[feature];
            neural_real maximum = minimum;

            for (sample = 0U; sample < rows->sample_count; sample++) {
                neural_real value;
                if (assignment[sample] != SPLIT_TRAIN) {
                    continue;
                }
                value = rows->inputs[sample * rows->input_count + feature];
                if (isnan(value)) {
                    value = preprocessing->imputations[feature];
                }
                if (value < minimum) minimum = value;
                if (value > maximum) maximum = value;
            }
            preprocessing->offsets[feature] = minimum;
            preprocessing->scales[feature] = maximum - minimum;
            if (preprocessing->scales[feature] == 0.0) {
                preprocessing->scales[feature] = 1.0;
            }
        }
    }
    return neural_preprocessing_apply(preprocessing,
                                      rows->inputs,
                                      rows->sample_count,
                                      error);
}

static int digest_file(const char *path,
                       char output[NEURAL_SHA256_TEXT_CAPACITY],
                       NeuralError *error)
{
    static const char hex[] = "0123456789abcdef";
    FILE *stream = fopen(path, "rb");
    NeuralSha256 context;
    unsigned char buffer[8192];
    unsigned char digest[32];
    size_t index;

    if (stream == NULL) {
        neural_error_set(error, "%s: unable to hash file: %s",
                         path, strerror(errno));
        return 0;
    }
    neural_sha256_init(&context);
    for (;;) {
        size_t count = fread(buffer, 1U, sizeof(buffer), stream);
        if (count != 0U) neural_sha256_update(&context, buffer, count);
        if (count < sizeof(buffer)) {
            if (ferror(stream) != 0) {
                neural_error_set(error, "%s: unable to read file for digest", path);
                (void)fclose(stream);
                return 0;
            }
            break;
        }
    }
    (void)fclose(stream);
    neural_sha256_final(&context, digest);
    for (index = 0U; index < sizeof(digest); index++) {
        output[index * 2U] = hex[digest[index] >> 4U];
        output[index * 2U + 1U] = hex[digest[index] & 15U];
    }
    output[NEURAL_SHA256_HEX_LENGTH] = '\0';
    return 1;
}

static int write_dataset(FILE *stream,
                         const ImportedRows *rows,
                         const unsigned char *assignment,
                         unsigned char wanted,
                         NeuralError *error)
{
    locale_t locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    locale_t previous;
    size_t sample;
    int success = 1;

    if (locale == (locale_t)0) {
        neural_error_set(error, "unable to create dataset numeric locale");
        return 0;
    }
    previous = uselocale(locale);
    if (previous == (locale_t)0) {
        freelocale(locale);
        neural_error_set(error, "unable to select dataset numeric locale");
        return 0;
    }
    if (fprintf(stream, "%s dataset 1\n", NEURAL_FORMAT_MAGIC) < 0) {
        success = 0;
    }
    for (sample = 0U; success && sample < rows->sample_count; sample++) {
        size_t index;
        if (assignment[sample] != wanted) continue;
        for (index = 0U; index < rows->input_count; index++) {
            if (index != 0U && fputc(' ', stream) == EOF) success = 0;
            if (success && fprintf(stream, "%.*g", DBL_DECIMAL_DIG,
                    rows->inputs[sample * rows->input_count + index]) < 0) {
                success = 0;
            }
        }
        if (success && fputs(" ->", stream) == EOF) success = 0;
        for (index = 0U; success && index < rows->output_count; index++) {
            if (fprintf(stream, " %.*g", DBL_DECIMAL_DIG,
                    rows->outputs[sample * rows->output_count + index]) < 0) {
                success = 0;
            }
        }
        if (success && fputc('\n', stream) == EOF) success = 0;
    }
    if (uselocale(previous) == (locale_t)0) success = 0;
    freelocale(locale);
    if (!success) neural_error_set(error, "unable to serialize imported dataset");
    return success;
}

static void import_files_free(ImportFile *files, size_t count)
{
    size_t index;
    for (index = 0U; index < count; index++) {
        free(files[index].final_path);
        free(files[index].new_path);
        free(files[index].old_path);
    }
}

static int stage_file(ImportFile *file,
                      const char *directory,
                      const char *name,
                      size_t ordinal,
                      int wanted,
                      const ImportedRows *rows,
                      const unsigned char *assignment,
                      unsigned char split,
                      const NeuralPreprocessing *preprocessing,
                      NeuralError *error)
{
    char temporary[128];
    int descriptor;
    FILE *stream;
    int success;

    file->final_path = neural_path_join(directory, name, error);
    if (file->final_path == NULL) return 0;
    (void)snprintf(temporary, sizeof(temporary),
                   ".neural-c-import-%ld-%zu-new", (long)getpid(), ordinal);
    file->new_path = neural_path_join(directory, temporary, error);
    (void)snprintf(temporary, sizeof(temporary),
                   ".neural-c-import-%ld-%zu-old", (long)getpid(), ordinal);
    file->old_path = neural_path_join(directory, temporary, error);
    if (file->new_path == NULL || file->old_path == NULL) return 0;
    {
        struct stat status;

        if (lstat(file->old_path, &status) == 0) {
            neural_error_set(error,
                             "%s: stale import recovery file exists",
                             file->old_path);
            return 0;
        }
        if (errno != ENOENT) {
            neural_error_set(error, "%s: unable to inspect staged import: %s",
                             file->old_path, strerror(errno));
            return 0;
        }
    }
    file->wanted = wanted;
    if (!wanted) return 1;
    descriptor = open(file->new_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        neural_error_set(error, "%s: unable to create staged import: %s",
                         file->new_path, strerror(errno));
        return 0;
    }
    stream = fdopen(descriptor, "w");
    if (stream == NULL) {
        (void)close(descriptor);
        (void)unlink(file->new_path);
        neural_error_set(error, "unable to open staged import stream");
        return 0;
    }
    if (preprocessing != NULL) {
        success = neural_preprocessing_write(stream, preprocessing, error);
    } else {
        success = write_dataset(stream, rows, assignment, split, error);
    }
    if (success && (fflush(stream) != 0 || fsync(fileno(stream)) != 0)) {
        neural_error_set(error, "unable to sync staged import file");
        success = 0;
    }
    if (fclose(stream) != 0 && success) {
        neural_error_set(error, "unable to close staged import file");
        success = 0;
    }
    if (!success) (void)unlink(file->new_path);
    return success;
}

static int sync_directory(const char *directory, NeuralError *error)
{
    int descriptor = open(directory, O_RDONLY);

    if (descriptor < 0 || fsync(descriptor) != 0) {
        int saved_errno = errno;

        if (descriptor >= 0) (void)close(descriptor);
        neural_error_set(error, "%s: unable to sync import directory: %s",
                         directory, strerror(saved_errno));
        return 0;
    }
    if (close(descriptor) != 0) {
        neural_error_set(error, "%s: unable to close import directory",
                         directory);
        return 0;
    }
    return 1;
}

static int install_files(const char *directory,
                         ImportFile *files,
                         size_t count,
                         NeuralError *error)
{
    size_t index;
    int success = 0;

    for (index = 0U; index < count; index++) {
        struct stat status;
        if (lstat(files[index].final_path, &status) == 0) {
            if (!S_ISREG(status.st_mode) ||
                rename(files[index].final_path, files[index].old_path) != 0) {
                neural_error_set(error, "%s: unable to stage previous file: %s",
                                 files[index].final_path, strerror(errno));
                goto rollback;
            }
            files[index].had_old = 1;
        } else if (errno != ENOENT) {
            neural_error_set(error, "%s: unable to inspect destination: %s",
                             files[index].final_path, strerror(errno));
            goto rollback;
        }
        if (files[index].wanted) {
            if (rename(files[index].new_path, files[index].final_path) != 0) {
                neural_error_set(error, "%s: unable to install import: %s",
                                 files[index].final_path, strerror(errno));
                goto rollback;
            }
            files[index].installed = 1;
        }
    }
    if (!sync_directory(directory, error)) {
        index = count;
        goto rollback;
    }
    success = 1;

rollback:
    if (!success) {
        if (index < count) {
            index++;
        }
        while (index > 0U) {
            index--;
            if (files[index].installed) {
                (void)unlink(files[index].final_path);
                files[index].installed = 0;
            }
            if (files[index].had_old) {
                if (rename(files[index].old_path,
                           files[index].final_path) == 0) {
                    files[index].had_old = 0;
                }
            }
        }
        (void)sync_directory(directory, NULL);
    } else {
        for (index = 0U; index < count; index++) {
            if (files[index].had_old) (void)unlink(files[index].old_path);
        }
    }
    for (index = 0U; index < count; index++) {
        if (files[index].wanted && files[index].new_path != NULL) {
            (void)unlink(files[index].new_path);
        }
    }
    return success;
}

static int path_exists_regular(const char *directory,
                               const char *name,
                               int *exists,
                               NeuralError *error)
{
    char *path = neural_path_join(directory, name, error);
    struct stat status;
    int success = 0;

    if (path == NULL) return 0;
    if (lstat(path, &status) == 0) {
        if (!S_ISREG(status.st_mode)) {
            neural_error_set(error, "%s: expected a regular file", path);
            goto cleanup;
        }
        *exists = 1;
    } else if (errno == ENOENT) {
        *exists = 0;
    } else {
        neural_error_set(error, "%s: unable to inspect file: %s",
                         path, strerror(errno));
        goto cleanup;
    }
    success = 1;
cleanup:
    free(path);
    return success;
}

int neural_data_import_csv(const char *project_directory,
                           const char *csv_path,
                           const NeuralDataImportConfig *config,
                           NeuralDataImportResult *result,
                           NeuralError *error)
{
    NeuralProjectLock lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralModelSpec model = {0};
    NeuralTrainingConfig training = {0};
    CsvSchema schema = {0};
    ImportedRows rows = {0};
    NeuralPreprocessing preprocessing = {0};
    unsigned char *assignment = NULL;
    ImportFile files[4] = {{0}};
    char *model_path = NULL;
    char *config_path = NULL;
    char source_digest_after[NEURAL_SHA256_TEXT_CAPACITY];
    char schema_digest_after[NEURAL_SHA256_TEXT_CAPACITY];
    size_t output_count;
    int weights_exists;
    int checkpoint_exists;
    int success = 0;

    neural_error_clear(error);
    if (project_directory == NULL || csv_path == NULL || config == NULL ||
        config->schema_path == NULL || result == NULL ||
        !isfinite(config->validation_ratio) ||
        !isfinite(config->test_ratio) || config->validation_ratio < 0.0 ||
        config->test_ratio < 0.0 ||
        config->validation_ratio + config->test_ratio >= 1.0) {
        neural_error_set(error, "CSV import configuration is invalid");
        return 0;
    }
    if (!neural_project_lock_acquire(project_directory,
                                     NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                     &lock, error)) {
        goto cleanup;
    }
    model_path = neural_path_join(project_directory,
                                  NEURAL_DEFAULT_MODEL_FILENAME, error);
    config_path = neural_path_join(project_directory,
                                   NEURAL_DEFAULT_PROJECT_FILENAME, error);
    if (model_path == NULL || config_path == NULL ||
        !neural_model_spec_load(model_path, &model, error) ||
        !neural_training_config_load(config_path, &training, error)) {
        goto cleanup;
    }
    output_count = model.layers[model.layer_count - 1U].neuron_count;
    if (!path_exists_regular(project_directory,
                             NEURAL_DEFAULT_WEIGHTS_FILENAME,
                             &weights_exists, error) ||
        !path_exists_regular(project_directory,
                             NEURAL_DEFAULT_CHECKPOINT_FILENAME,
                             &checkpoint_exists, error)) {
        goto cleanup;
    }
    if (weights_exists || checkpoint_exists) {
        neural_error_set(error,
                         "CSV import would invalidate weights/checkpoint; archive or remove them first");
        goto cleanup;
    }
    if (!digest_file(csv_path, preprocessing.source_digest, error) ||
        !digest_file(config->schema_path, preprocessing.schema_digest, error) ||
        !schema_load(config->schema_path, model.input_count, output_count,
                     &schema, error) ||
        !csv_load(csv_path, &schema, output_count, &rows, error) ||
        !create_split(&rows, &schema, config, &assignment, result, error)) {
        goto cleanup;
    }
    if (!digest_file(csv_path, source_digest_after, error) ||
        !digest_file(config->schema_path, schema_digest_after, error)) {
        goto cleanup;
    }
    if (strcmp(preprocessing.source_digest, source_digest_after) != 0 ||
        strcmp(preprocessing.schema_digest, schema_digest_after) != 0) {
        neural_error_set(error,
                         "CSV or schema changed while it was being imported");
        goto cleanup;
    }
    if (training.early_stopping_patience != 0U &&
        result->validation_samples == 0U) {
        neural_error_set(error,
                         "project early stopping requires a non-empty validation split");
        goto cleanup;
    }
    preprocessing.split_seed = config->split_seed;
    preprocessing.validation_ratio = config->validation_ratio;
    preprocessing.test_ratio = config->test_ratio;
    preprocessing.stratified = schema.categorical;
    if (!fit_preprocessing(&rows, assignment, config, &preprocessing, error) ||
        !neural_preprocessing_validate(&preprocessing, error)) {
        goto cleanup;
    }
    if (!stage_file(&files[0], project_directory,
                    NEURAL_DEFAULT_DATASET_FILENAME, 0U, 1,
                    &rows, assignment, SPLIT_TRAIN, NULL, error) ||
        !stage_file(&files[1], project_directory,
                    NEURAL_DEFAULT_VALIDATION_FILENAME, 1U,
                    result->validation_samples != 0U,
                    &rows, assignment, SPLIT_VALIDATION, NULL, error) ||
        !stage_file(&files[2], project_directory,
                    NEURAL_DEFAULT_TEST_FILENAME, 2U,
                    result->test_samples != 0U,
                    &rows, assignment, SPLIT_TEST, NULL, error) ||
        !stage_file(&files[3], project_directory,
                    NEURAL_DEFAULT_PREPROCESSING_FILENAME, 3U, 1,
                    NULL, NULL, 0U, &preprocessing, error) ||
        !install_files(project_directory, files, 4U, error)) {
        goto cleanup;
    }
    success = 1;

cleanup:
    import_files_free(files, 4U);
    free(assignment);
    neural_preprocessing_free(&preprocessing);
    rows_free(&rows);
    schema_free(&schema);
    neural_model_spec_free(&model);
    free(config_path);
    free(model_path);
    neural_project_lock_release(&lock);
    return success;
}
