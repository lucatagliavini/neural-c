#ifndef NEURAL_CLI_OPTIONS_H
#define NEURAL_CLI_OPTIONS_H

#include <stddef.h>

#include "neural/error.h"

typedef enum {
    NEURAL_OPTION_FLAG,
    NEURAL_OPTION_VALUE
} NeuralOptionKind;

typedef struct {
    const char *long_name;
    char short_name;
    NeuralOptionKind kind;
    int repeatable;
} NeuralOptionDefinition;

typedef struct {
    size_t count;
    const char **values;
} NeuralOptionOccurrences;

typedef struct {
    size_t definition_count;
    NeuralOptionOccurrences *options;
    size_t positional_count;
    const char **positionals;
} NeuralParsedOptions;

/* The result owns its arrays after success and must be released with free. */
int neural_options_parse(int argc,
                         char **argv,
                         const NeuralOptionDefinition *definitions,
                         size_t definition_count,
                         NeuralParsedOptions *result,
                         NeuralError *error);
void neural_options_free(NeuralParsedOptions *options);
int neural_option_is_present(const NeuralParsedOptions *options, size_t index);
size_t neural_option_count(const NeuralParsedOptions *options, size_t index);
const char *neural_option_value(const NeuralParsedOptions *options,
                                size_t index);
const char *neural_option_value_at(const NeuralParsedOptions *options,
                                   size_t index,
                                   size_t occurrence);

#endif
