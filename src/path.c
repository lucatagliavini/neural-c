#include "path.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *neural_path_join(const char *directory,
                       const char *filename,
                       NeuralError *error)
{
    size_t directory_length;
    size_t filename_length;
    int needs_separator;
    size_t total_length;
    char *path;

    if (directory == NULL || filename == NULL) {
        neural_error_set(error, "invalid path components");
        return NULL;
    }
    directory_length = strlen(directory);
    filename_length = strlen(filename);
    needs_separator = directory_length != 0U &&
                      directory[directory_length - 1U] != '/';
    if (directory_length > SIZE_MAX - filename_length) {
        neural_error_set(error, "project path is too long");
        return NULL;
    }
    total_length = directory_length + filename_length;
    if (total_length > SIZE_MAX - (needs_separator ? 1U : 0U) - 1U) {
        neural_error_set(error, "project path is too long");
        return NULL;
    }
    total_length += (needs_separator ? 1U : 0U) + 1U;
    path = malloc(total_length);
    if (path == NULL) {
        neural_error_set(error, "unable to allocate project path");
        return NULL;
    }
    (void)snprintf(path,
                   total_length,
                   "%s%s%s",
                   directory,
                   needs_separator ? "/" : "",
                   filename);
    return path;
}
