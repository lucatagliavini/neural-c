#define _POSIX_C_SOURCE 200809L

#include "neural/preprocessing.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural/defaults.h"
#include "neural/parse.h"
#include "neural/version.h"
#include "atomic_file.h"
#include "sha256.h"

typedef struct {
    FILE *stream;
    const char *path;
    char *line;
    size_t capacity;
    size_t line_number;
    char *tokens[8];
    size_t token_count;
} PreprocessingReader;

static int is_digest(const char *text)
{
    size_t index;

    if (text == NULL || strncmp(text, "sha256:", 7U) != 0 ||
        strlen(text + 7U) != NEURAL_SHA256_HEX_LENGTH) {
        return 0;
    }
    for (index = 0U; index < NEURAL_SHA256_HEX_LENGTH; index++) {
        unsigned char value = (unsigned char)text[index + 7U];

        if (isdigit(value) == 0 && !(value >= 'a' && value <= 'f')) {
            return 0;
        }
    }
    return 1;
}

const char *neural_normalization_name(NeuralNormalization normalization)
{
    switch (normalization) {
    case NEURAL_NORMALIZATION_NONE:
        return "none";
    case NEURAL_NORMALIZATION_STANDARDIZE:
        return "standardize";
    case NEURAL_NORMALIZATION_MINMAX:
        return "minmax";
    }
    return "unknown";
}

int neural_normalization_from_name(const char *name,
                                   NeuralNormalization *normalization)
{
    if (name == NULL || normalization == NULL) {
        return 0;
    }
    if (strcmp(name, "none") == 0) {
        *normalization = NEURAL_NORMALIZATION_NONE;
    } else if (strcmp(name, "standardize") == 0) {
        *normalization = NEURAL_NORMALIZATION_STANDARDIZE;
    } else if (strcmp(name, "minmax") == 0) {
        *normalization = NEURAL_NORMALIZATION_MINMAX;
    } else {
        return 0;
    }
    return 1;
}

const char *neural_missing_policy_name(NeuralMissingPolicy policy)
{
    switch (policy) {
    case NEURAL_MISSING_REJECT:
        return "reject";
    case NEURAL_MISSING_MEAN:
        return "mean";
    }
    return "unknown";
}

int neural_missing_policy_from_name(const char *name,
                                    NeuralMissingPolicy *policy)
{
    if (name == NULL || policy == NULL) {
        return 0;
    }
    if (strcmp(name, "reject") == 0) {
        *policy = NEURAL_MISSING_REJECT;
    } else if (strcmp(name, "mean") == 0) {
        *policy = NEURAL_MISSING_MEAN;
    } else {
        return 0;
    }
    return 1;
}

void neural_preprocessing_free(NeuralPreprocessing *preprocessing)
{
    if (preprocessing != NULL) {
        free(preprocessing->offsets);
        free(preprocessing->scales);
        free(preprocessing->imputations);
        memset(preprocessing, 0, sizeof(*preprocessing));
    }
}

int neural_preprocessing_validate(const NeuralPreprocessing *preprocessing,
                                  NeuralError *error)
{
    size_t index;

    if (preprocessing == NULL || preprocessing->input_count == 0U ||
        preprocessing->offsets == NULL || preprocessing->scales == NULL ||
        preprocessing->imputations == NULL ||
        strcmp(neural_normalization_name(preprocessing->normalization),
               "unknown") == 0 ||
        strcmp(neural_missing_policy_name(preprocessing->missing_policy),
               "unknown") == 0) {
        neural_error_set(error, "preprocessing metadata is incomplete");
        return 0;
    }
    if (strlen(preprocessing->source_digest) != NEURAL_SHA256_HEX_LENGTH ||
        strlen(preprocessing->schema_digest) != NEURAL_SHA256_HEX_LENGTH) {
        neural_error_set(error, "preprocessing digests are invalid");
        return 0;
    }
    for (index = 0U; index < NEURAL_SHA256_HEX_LENGTH; index++) {
        unsigned char source = (unsigned char)preprocessing->source_digest[index];
        unsigned char schema = (unsigned char)preprocessing->schema_digest[index];

        if ((isdigit(source) == 0 && !(source >= 'a' && source <= 'f')) ||
            (isdigit(schema) == 0 && !(schema >= 'a' && schema <= 'f'))) {
            neural_error_set(error, "preprocessing digests are invalid");
            return 0;
        }
    }
    if (!isfinite(preprocessing->validation_ratio) ||
        !isfinite(preprocessing->test_ratio) ||
        preprocessing->validation_ratio < 0.0 ||
        preprocessing->test_ratio < 0.0 ||
        preprocessing->validation_ratio + preprocessing->test_ratio >= 1.0 ||
        (preprocessing->stratified != 0 && preprocessing->stratified != 1)) {
        neural_error_set(error, "preprocessing split metadata is invalid");
        return 0;
    }
    for (index = 0U; index < preprocessing->input_count; index++) {
        if (!isfinite(preprocessing->offsets[index]) ||
            !isfinite(preprocessing->scales[index]) ||
            preprocessing->scales[index] <= 0.0 ||
            !isfinite(preprocessing->imputations[index])) {
            neural_error_set(error,
                             "preprocessing feature %zu is invalid",
                             index);
            return 0;
        }
        if (preprocessing->normalization == NEURAL_NORMALIZATION_NONE &&
            (preprocessing->offsets[index] != 0.0 ||
             preprocessing->scales[index] != 1.0)) {
            neural_error_set(error,
                             "unnormalized feature %zu must use offset 0 and scale 1",
                             index);
            return 0;
        }
    }
    return 1;
}

int neural_preprocessing_copy(const NeuralPreprocessing *source,
                              NeuralPreprocessing *destination,
                              NeuralError *error)
{
    NeuralPreprocessing copied = {0};
    size_t bytes;

    if (destination == NULL ||
        !neural_preprocessing_validate(source, error) ||
        source->input_count > SIZE_MAX / sizeof(*source->offsets)) {
        return 0;
    }
    bytes = source->input_count * sizeof(*source->offsets);
    copied = *source;
    copied.offsets = malloc(bytes);
    copied.scales = malloc(bytes);
    copied.imputations = malloc(bytes);
    if (copied.offsets == NULL || copied.scales == NULL ||
        copied.imputations == NULL) {
        neural_error_set(error, "unable to copy preprocessing metadata");
        neural_preprocessing_free(&copied);
        return 0;
    }
    memcpy(copied.offsets, source->offsets, bytes);
    memcpy(copied.scales, source->scales, bytes);
    memcpy(copied.imputations, source->imputations, bytes);
    *destination = copied;
    return 1;
}

int neural_preprocessing_apply(const NeuralPreprocessing *preprocessing,
                               neural_real *inputs,
                               size_t sample_count,
                               NeuralError *error)
{
    size_t sample_index;

    if (!neural_preprocessing_validate(preprocessing, error) ||
        inputs == NULL || sample_count == 0U ||
        sample_count > SIZE_MAX / preprocessing->input_count) {
        if (error != NULL && error->message[0] == '\0') {
            neural_error_set(error, "preprocessing inputs are invalid");
        }
        return 0;
    }
    for (sample_index = 0U; sample_index < sample_count; sample_index++) {
        size_t input_index;

        for (input_index = 0U;
             input_index < preprocessing->input_count;
             input_index++) {
            size_t offset = sample_index * preprocessing->input_count +
                            input_index;
            neural_real value = inputs[offset];

            if (isnan(value)) {
                if (preprocessing->missing_policy == NEURAL_MISSING_REJECT) {
                    neural_error_set(error,
                                     "missing input at sample %zu feature %zu",
                                     sample_index,
                                     input_index);
                    return 0;
                }
                value = preprocessing->imputations[input_index];
            } else if (!isfinite(value)) {
                neural_error_set(error,
                                 "non-finite input at sample %zu feature %zu",
                                 sample_index,
                                 input_index);
                return 0;
            }
            value = (value - preprocessing->offsets[input_index]) /
                    preprocessing->scales[input_index];
            if (!isfinite(value)) {
                neural_error_set(error,
                                 "transformed input at sample %zu feature %zu "
                                 "is not finite",
                                 sample_index,
                                 input_index);
                return 0;
            }
            inputs[offset] = value;
        }
    }
    return 1;
}

static int next_line(PreprocessingReader *reader, NeuralError *error)
{
    for (;;) {
        ssize_t length = getline(&reader->line,
                                 &reader->capacity,
                                 reader->stream);
        char *cursor;

        if (length < 0) {
            if (ferror(reader->stream) != 0) {
                neural_error_set(error,
                                 "%s:%zu: unable to read file",
                                 reader->path,
                                 reader->line_number + 1U);
                return -1;
            }
            return 0;
        }
        reader->line_number++;
        if ((size_t)length > NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH) {
            neural_error_set(error,
                             "%s:%zu: line exceeds maximum length",
                             reader->path,
                             reader->line_number);
            return -1;
        }
        while (length > 0 &&
               (reader->line[(size_t)length - 1U] == '\n' ||
                reader->line[(size_t)length - 1U] == '\r')) {
            reader->line[--length] = '\0';
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
                neural_error_set(error,
                                 "%s:%zu: too many fields",
                                 reader->path,
                                 reader->line_number);
                return -1;
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

static int require(PreprocessingReader *reader,
                   size_t count,
                   const char *first,
                   NeuralError *error)
{
    int status = next_line(reader, error);

    if (status <= 0) {
        if (status == 0) {
            neural_error_set(error,
                             "%s:%zu: expected '%s' before end of file",
                             reader->path,
                             reader->line_number + 1U,
                             first);
        }
        return 0;
    }
    if (reader->token_count != count ||
        strcmp(reader->tokens[0], first) != 0) {
        neural_error_set(error,
                         "%s:%zu: expected '%s'",
                         reader->path,
                         reader->line_number,
                         first);
        return 0;
    }
    return 1;
}

int neural_preprocessing_load(const char *path,
                              NeuralPreprocessing *preprocessing,
                              NeuralError *error)
{
    PreprocessingReader reader = {0};
    NeuralPreprocessing loaded = {0};
    size_t version;
    size_t index;
    int success = 0;

    neural_error_clear(error);
    if (path == NULL || preprocessing == NULL) {
        neural_error_set(error, "preprocessing path and output are required");
        return 0;
    }
    memset(preprocessing, 0, sizeof(*preprocessing));
    reader.path = path;
    reader.stream = fopen(path, "r");
    if (reader.stream == NULL) {
        neural_error_set(error, "%s: unable to open file: %s", path, strerror(errno));
        goto cleanup;
    }
    if (!require(&reader, 3U, NEURAL_FORMAT_MAGIC, error) ||
        strcmp(reader.tokens[1], "preprocessing") != 0 ||
        !neural_parse_size(reader.tokens[2], &version) || version != 1U) {
        if (error->message[0] == '\0') {
            neural_error_set(error,
                             "%s:%zu: expected '%s preprocessing 1'",
                             path,
                             reader.line_number,
                             NEURAL_FORMAT_MAGIC);
        }
        goto cleanup;
    }
    if (!require(&reader, 2U, "inputs", error) ||
        !neural_parse_size(reader.tokens[1], &loaded.input_count) ||
        loaded.input_count == 0U ||
        !require(&reader, 2U, "normalization", error) ||
        !neural_normalization_from_name(reader.tokens[1],
                                        &loaded.normalization) ||
        !require(&reader, 2U, "missing", error) ||
        !neural_missing_policy_from_name(reader.tokens[1],
                                         &loaded.missing_policy) ||
        !require(&reader, 2U, "source_digest", error) ||
        !is_digest(reader.tokens[1])) {
        if (error->message[0] == '\0') {
            neural_error_set(error,
                             "%s:%zu: invalid preprocessing metadata",
                             path,
                             reader.line_number);
        }
        goto cleanup;
    }
    memcpy(loaded.source_digest,
           reader.tokens[1] + 7U,
           NEURAL_SHA256_TEXT_CAPACITY);
    if (!require(&reader, 2U, "schema_digest", error) ||
        !is_digest(reader.tokens[1])) {
        if (error->message[0] == '\0') {
            neural_error_set(error,
                             "%s:%zu: invalid schema digest",
                             path,
                             reader.line_number);
        }
        goto cleanup;
    }
    memcpy(loaded.schema_digest,
           reader.tokens[1] + 7U,
           NEURAL_SHA256_TEXT_CAPACITY);
    if (!require(&reader, 2U, "split_seed", error) ||
        !neural_parse_uint64(reader.tokens[1], &loaded.split_seed) ||
        !require(&reader, 2U, "validation_ratio", error) ||
        !neural_parse_real(reader.tokens[1], &loaded.validation_ratio) ||
        !require(&reader, 2U, "test_ratio", error) ||
        !neural_parse_real(reader.tokens[1], &loaded.test_ratio) ||
        !require(&reader, 2U, "stratified", error) ||
        (strcmp(reader.tokens[1], "yes") != 0 &&
         strcmp(reader.tokens[1], "no") != 0)) {
        if (error->message[0] == '\0') {
            neural_error_set(error,
                             "%s:%zu: invalid split metadata",
                             path,
                             reader.line_number);
        }
        goto cleanup;
    }
    loaded.stratified = strcmp(reader.tokens[1], "yes") == 0;
    if (loaded.input_count > SIZE_MAX / sizeof(*loaded.offsets)) {
        neural_error_set(error, "preprocessing dimensions are too large");
        goto cleanup;
    }
    loaded.offsets = malloc(loaded.input_count * sizeof(*loaded.offsets));
    loaded.scales = malloc(loaded.input_count * sizeof(*loaded.scales));
    loaded.imputations =
        malloc(loaded.input_count * sizeof(*loaded.imputations));
    if (loaded.offsets == NULL || loaded.scales == NULL ||
        loaded.imputations == NULL) {
        neural_error_set(error, "unable to allocate preprocessing features");
        goto cleanup;
    }
    for (index = 0U; index < loaded.input_count; index++) {
        size_t parsed_index;

        if (!require(&reader, 8U, "feature", error) ||
            !neural_parse_size(reader.tokens[1], &parsed_index) ||
            parsed_index != index || strcmp(reader.tokens[2], "offset") != 0 ||
            !neural_parse_real(reader.tokens[3], &loaded.offsets[index]) ||
            strcmp(reader.tokens[4], "scale") != 0 ||
            !neural_parse_real(reader.tokens[5], &loaded.scales[index]) ||
            strcmp(reader.tokens[6], "impute") != 0 ||
            !neural_parse_real(reader.tokens[7], &loaded.imputations[index])) {
            neural_error_set(error,
                             "%s:%zu: invalid feature %zu metadata",
                             path,
                             reader.line_number,
                             index);
            goto cleanup;
        }
    }
    if (!require(&reader, 1U, "end", error) || next_line(&reader, error) != 0 ||
        !neural_preprocessing_validate(&loaded, error)) {
        goto cleanup;
    }
    *preprocessing = loaded;
    memset(&loaded, 0, sizeof(loaded));
    success = 1;

cleanup:
    free(reader.line);
    if (reader.stream != NULL) {
        (void)fclose(reader.stream);
    }
    neural_preprocessing_free(&loaded);
    return success;
}

int neural_preprocessing_write(FILE *stream,
                               const NeuralPreprocessing *preprocessing,
                               NeuralError *error)
{
    locale_t locale;
    locale_t previous;
    size_t index;
    int success = 0;

    neural_error_clear(error);
    if (stream == NULL ||
        !neural_preprocessing_validate(preprocessing, error)) {
        return 0;
    }
    locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    if (locale == (locale_t)0) {
        neural_error_set(error, "unable to create preprocessing numeric locale");
        return 0;
    }
    previous = uselocale(locale);
    if (previous == (locale_t)0) {
        freelocale(locale);
        neural_error_set(error, "unable to select preprocessing numeric locale");
        return 0;
    }
    if (fprintf(stream,
                "%s preprocessing 1\n"
                "inputs %zu\n"
                "normalization %s\n"
                "missing %s\n"
                "source_digest sha256:%s\n"
                "schema_digest sha256:%s\n"
                "split_seed %" PRIu64 "\n"
                "validation_ratio %.*g\n"
                "test_ratio %.*g\n"
                "stratified %s\n",
                NEURAL_FORMAT_MAGIC,
                preprocessing->input_count,
                neural_normalization_name(preprocessing->normalization),
                neural_missing_policy_name(preprocessing->missing_policy),
                preprocessing->source_digest,
                preprocessing->schema_digest,
                preprocessing->split_seed,
                DBL_DECIMAL_DIG,
                preprocessing->validation_ratio,
                DBL_DECIMAL_DIG,
                preprocessing->test_ratio,
                preprocessing->stratified ? "yes" : "no") < 0) {
        goto cleanup;
    }
    for (index = 0U; index < preprocessing->input_count; index++) {
        if (fprintf(stream,
                    "feature %zu offset %.*g scale %.*g impute %.*g\n",
                    index,
                    DBL_DECIMAL_DIG,
                    preprocessing->offsets[index],
                    DBL_DECIMAL_DIG,
                    preprocessing->scales[index],
                    DBL_DECIMAL_DIG,
                    preprocessing->imputations[index]) < 0) {
            goto cleanup;
        }
    }
    success = fputs("end\n", stream) != EOF;

cleanup:
    if (uselocale(previous) == (locale_t)0) {
        success = 0;
        neural_error_set(error, "unable to restore preprocessing numeric locale");
    }
    freelocale(locale);
    if (!success && error->message[0] == '\0') {
        neural_error_set(error, "unable to serialize preprocessing metadata");
    }
    return success;
}

static int write_preprocessing_atomic(FILE *stream,
                                      void *context,
                                      NeuralError *error)
{
    return neural_preprocessing_write(stream, context, error);
}

int neural_preprocessing_save_atomic(const char *path,
                                     const NeuralPreprocessing *preprocessing,
                                     NeuralError *error)
{
    neural_error_clear(error);
    if (path == NULL || !neural_preprocessing_validate(preprocessing, error)) {
        return 0;
    }
    return neural_atomic_file_write(path,
                                    write_preprocessing_atomic,
                                    (void *)preprocessing,
                                    error);
}

static void update_u64(NeuralSha256 *context, uint64_t value)
{
    unsigned char encoded[8];
    size_t index;

    for (index = 0U; index < sizeof(encoded); index++) {
        encoded[index] = (unsigned char)(value >>
            (unsigned int)((7U - index) * 8U));
    }
    neural_sha256_update(context, encoded, sizeof(encoded));
}

static void update_text(NeuralSha256 *context, const char *text)
{
    size_t length = strlen(text);

    update_u64(context, (uint64_t)length);
    neural_sha256_update(context, text, length);
}

static void update_real(NeuralSha256 *context, neural_real value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    update_u64(context, bits);
}

int neural_preprocessing_digest(
    const NeuralPreprocessing *preprocessing,
    char output[NEURAL_SHA256_TEXT_CAPACITY],
    NeuralError *error)
{
    static const char hexadecimal[] = "0123456789abcdef";
    NeuralSha256 context;
    unsigned char digest[32];
    size_t index;

    if (output == NULL || !neural_preprocessing_validate(preprocessing, error)) {
        return 0;
    }
    neural_sha256_init(&context);
    update_text(&context, NEURAL_FORMAT_MAGIC ":canonical:preprocessing");
    update_u64(&context, UINT64_C(1));
    update_u64(&context, (uint64_t)preprocessing->input_count);
    update_text(&context, neural_normalization_name(preprocessing->normalization));
    update_text(&context, neural_missing_policy_name(preprocessing->missing_policy));
    update_text(&context, preprocessing->source_digest);
    update_text(&context, preprocessing->schema_digest);
    update_u64(&context, preprocessing->split_seed);
    update_real(&context, preprocessing->validation_ratio);
    update_real(&context, preprocessing->test_ratio);
    update_u64(&context, (uint64_t)preprocessing->stratified);
    for (index = 0U; index < preprocessing->input_count; index++) {
        update_real(&context, preprocessing->offsets[index]);
        update_real(&context, preprocessing->scales[index]);
        update_real(&context, preprocessing->imputations[index]);
    }
    neural_sha256_final(&context, digest);
    for (index = 0U; index < sizeof(digest); index++) {
        output[index * 2U] = hexadecimal[digest[index] >> 4U];
        output[index * 2U + 1U] = hexadecimal[digest[index] & 0x0fU];
    }
    output[NEURAL_SHA256_HEX_LENGTH] = '\0';
    return 1;
}
