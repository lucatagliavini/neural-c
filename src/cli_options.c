#include "neural/cli_options.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int validate_definitions(const NeuralOptionDefinition *definitions,
                                size_t count,
                                NeuralError *error)
{
    size_t current;

    for (current = 0U; current < count; current++) {
        size_t previous;
        const NeuralOptionDefinition *definition = &definitions[current];

        if (definition->long_name == NULL || definition->long_name[0] == '\0' ||
            strchr(definition->long_name, '=') != NULL ||
            definition->long_name[0] == '-' ||
            (definition->kind != NEURAL_OPTION_FLAG &&
             definition->kind != NEURAL_OPTION_VALUE) ||
            (definition->repeatable &&
             definition->kind != NEURAL_OPTION_VALUE)) {
            neural_error_set(error, "invalid option definition at index %zu", current);
            return 0;
        }
        if (definition->short_name == '-') {
            neural_error_set(error, "invalid short option '-' at index %zu", current);
            return 0;
        }
        for (previous = 0U; previous < current; previous++) {
            if (strcmp(definition->long_name,
                       definitions[previous].long_name) == 0 ||
                (definition->short_name != '\0' &&
                 definition->short_name == definitions[previous].short_name)) {
                neural_error_set(error,
                                 "duplicate option definition at index %zu",
                                 current);
                return 0;
            }
        }
    }
    return 1;
}

static int find_long_option(const NeuralOptionDefinition *definitions,
                            size_t count,
                            const char *name,
                            size_t name_length,
                            size_t *index)
{
    size_t current;

    for (current = 0U; current < count; current++) {
        if (strlen(definitions[current].long_name) == name_length &&
            strncmp(definitions[current].long_name, name, name_length) == 0) {
            *index = current;
            return 1;
        }
    }
    return 0;
}

static int find_short_option(const NeuralOptionDefinition *definitions,
                             size_t count,
                             char name,
                             size_t *index)
{
    size_t current;

    for (current = 0U; current < count; current++) {
        if (definitions[current].short_name == name) {
            *index = current;
            return 1;
        }
    }
    return 0;
}

static int record_option(const NeuralOptionDefinition *definitions,
                         size_t index,
                         const char *inline_value,
                         int *argument_index,
                         int argc,
                         char **argv,
                         NeuralParsedOptions *result,
                         NeuralError *error)
{
    const NeuralOptionDefinition *definition = &definitions[index];
    NeuralOptionOccurrences *occurrences = &result->options[index];
    const char *value = NULL;
    const char **new_values;

    if (occurrences->count != 0U && !definition->repeatable) {
        neural_error_set(error,
                         "option '--%s' was specified more than once",
                         definition->long_name);
        return 0;
    }

    if (definition->kind == NEURAL_OPTION_FLAG) {
        if (inline_value != NULL) {
            neural_error_set(error,
                             "option '--%s' does not accept a value",
                             definition->long_name);
            return 0;
        }
    } else if (inline_value != NULL) {
        if (inline_value[0] == '\0') {
            neural_error_set(error,
                             "option '--%s' requires a value",
                             definition->long_name);
            return 0;
        }
        value = inline_value;
    } else {
        if (*argument_index + 1 >= argc) {
            neural_error_set(error,
                             "option '--%s' requires a value",
                             definition->long_name);
            return 0;
        }
        (*argument_index)++;
        value = argv[*argument_index];
    }

    if (definition->kind == NEURAL_OPTION_VALUE) {
        if (occurrences->count >= SIZE_MAX / sizeof(*new_values)) {
            neural_error_set(error,
                             "option '--%s' was repeated too many times",
                             definition->long_name);
            return 0;
        }
        new_values = realloc(occurrences->values,
                             (occurrences->count + 1U) *
                                 sizeof(*new_values));
        if (new_values == NULL) {
            neural_error_set(error,
                             "unable to record option '--%s'",
                             definition->long_name);
            return 0;
        }
        occurrences->values = new_values;
        occurrences->values[occurrences->count] = value;
    }
    occurrences->count++;
    return 1;
}

void neural_options_free(NeuralParsedOptions *options)
{
    if (options == NULL) {
        return;
    }
    if (options->options != NULL) {
        size_t index;
        for (index = 0U; index < options->definition_count; index++) {
            free(options->options[index].values);
        }
    }
    free(options->options);
    free(options->positionals);
    memset(options, 0, sizeof(*options));
}

int neural_options_parse(int argc,
                         char **argv,
                         const NeuralOptionDefinition *definitions,
                         size_t definition_count,
                         NeuralParsedOptions *result,
                         NeuralError *error)
{
    int argument_index;
    int options_finished = 0;

    if (argc < 0 || (argc > 0 && argv == NULL) || result == NULL ||
        (definition_count != 0U && definitions == NULL)) {
        neural_error_set(error, "invalid arguments passed to option parser");
        return 0;
    }

    memset(result, 0, sizeof(*result));
    neural_error_clear(error);
    if (!validate_definitions(definitions, definition_count, error)) {
        return 0;
    }
    result->definition_count = definition_count;
    result->options = calloc(definition_count, sizeof(*result->options));
    result->positionals = calloc((size_t)argc, sizeof(*result->positionals));
    if ((definition_count != 0U && result->options == NULL) ||
        (argc != 0 && result->positionals == NULL)) {
        neural_error_set(error, "unable to allocate command-line parser state");
        neural_options_free(result);
        return 0;
    }

    for (argument_index = 0; argument_index < argc; argument_index++) {
        const char *argument = argv[argument_index];
        size_t option_index;

        if (!options_finished && strcmp(argument, "--") == 0) {
            options_finished = 1;
            continue;
        }

        if (!options_finished && strncmp(argument, "--", 2U) == 0 &&
            argument[2] != '\0') {
            const char *name = argument + 2;
            const char *separator = strchr(name, '=');
            size_t name_length = separator == NULL
                                     ? strlen(name)
                                     : (size_t)(separator - name);
            const char *value = separator == NULL ? NULL : separator + 1;

            if (!find_long_option(definitions,
                                  definition_count,
                                  name,
                                  name_length,
                                  &option_index)) {
                neural_error_set(error,
                                 "unknown option '%s'",
                                 argument);
                neural_options_free(result);
                return 0;
            }
            if (!record_option(definitions,
                               option_index,
                               value,
                               &argument_index,
                               argc,
                               argv,
                               result,
                               error)) {
                neural_options_free(result);
                return 0;
            }
            continue;
        }

        if (!options_finished && argument[0] == '-' &&
            argument[1] != '\0') {
            if (argument[2] != '\0') {
                neural_error_set(error,
                                 "short options cannot be grouped: '%s'",
                                 argument);
                neural_options_free(result);
                return 0;
            }
            if (!find_short_option(definitions,
                                   definition_count,
                                   argument[1],
                                   &option_index)) {
                neural_error_set(error,
                                 "unknown option '%s'",
                                 argument);
                neural_options_free(result);
                return 0;
            }
            if (!record_option(definitions,
                               option_index,
                               NULL,
                               &argument_index,
                               argc,
                               argv,
                               result,
                               error)) {
                neural_options_free(result);
                return 0;
            }
            continue;
        }

        result->positionals[result->positional_count] = argument;
        result->positional_count++;
    }

    return 1;
}

int neural_option_is_present(const NeuralParsedOptions *options, size_t index)
{
    return options != NULL && index < options->definition_count &&
           options->options[index].count != 0U;
}

size_t neural_option_count(const NeuralParsedOptions *options, size_t index)
{
    if (options == NULL || index >= options->definition_count) {
        return 0U;
    }
    return options->options[index].count;
}

const char *neural_option_value(const NeuralParsedOptions *options,
                                size_t index)
{
    if (options == NULL || index >= options->definition_count) {
        return NULL;
    }
    return neural_option_value_at(options, index, 0U);
}

const char *neural_option_value_at(const NeuralParsedOptions *options,
                                   size_t index,
                                   size_t occurrence)
{
    if (options == NULL || index >= options->definition_count ||
        occurrence >= options->options[index].count ||
        options->options[index].values == NULL) {
        return NULL;
    }
    return options->options[index].values[occurrence];
}
