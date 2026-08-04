#define _POSIX_C_SOURCE 200809L

#include "atomic_file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "neural/defaults.h"

_Static_assert(sizeof(NEURAL_DEFAULT_ATOMIC_TEMP_SUFFIX) >= 7U,
               "atomic temporary suffix must end in six X characters");

static char *parent_directory(const char *path, NeuralError *error)
{
    const char *separator = strrchr(path, '/');
    size_t length;
    char *directory;

    if (separator == NULL || separator == path) {
        length = 1U;
    } else {
        length = (size_t)(separator - path);
    }
    directory = malloc(length + 1U);
    if (directory == NULL) {
        neural_error_set(error, "unable to allocate atomic-write directory");
        return NULL;
    }
    if (separator == NULL) {
        directory[0] = '.';
    } else {
        memcpy(directory, path, length);
    }
    directory[length] = '\0';
    return directory;
}

static int synchronize_parent(const char *path, NeuralError *error)
{
    char *directory = parent_directory(path, error);
    int descriptor;
    int success = 0;

    if (directory == NULL) {
        return 0;
    }
    descriptor = open(directory, O_RDONLY);
    if (descriptor < 0) {
        neural_error_set(error,
                         "%s: unable to open parent directory: %s",
                         path,
                         strerror(errno));
    } else if (fsync(descriptor) != 0) {
        neural_error_set(error,
                         "%s: unable to synchronize parent directory: %s",
                         path,
                         strerror(errno));
    } else if (close(descriptor) != 0) {
        descriptor = -1;
        neural_error_set(error,
                         "%s: unable to close parent directory: %s",
                         path,
                         strerror(errno));
    } else {
        descriptor = -1;
        success = 1;
    }
    if (descriptor >= 0) {
        (void)close(descriptor);
    }
    free(directory);
    return success;
}

int neural_atomic_file_write(const char *path,
                             NeuralAtomicFileWriter writer,
                             void *context,
                             NeuralError *error)
{
    size_t path_length;
    size_t suffix_length = sizeof(NEURAL_DEFAULT_ATOMIC_TEMP_SUFFIX) - 1U;
    char *temporary_path;
    int descriptor = -1;
    FILE *stream = NULL;
    int success = 0;

    if (path == NULL || path[0] == '\0' || writer == NULL) {
        neural_error_set(error, "atomic-write path and writer are required");
        return 0;
    }
    path_length = strlen(path);
    if (path_length > SIZE_MAX - suffix_length - 1U) {
        neural_error_set(error, "atomic-write path is too long");
        return 0;
    }
    temporary_path = malloc(path_length + suffix_length + 1U);
    if (temporary_path == NULL) {
        neural_error_set(error, "unable to allocate atomic-write path");
        return 0;
    }
    memcpy(temporary_path, path, path_length);
    memcpy(temporary_path + path_length,
           NEURAL_DEFAULT_ATOMIC_TEMP_SUFFIX,
           suffix_length + 1U);

    descriptor = mkstemp(temporary_path);
    if (descriptor < 0) {
        neural_error_set(error,
                         "%s: unable to create temporary file: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    stream = fdopen(descriptor, "w");
    if (stream == NULL) {
        neural_error_set(error,
                         "%s: unable to open temporary stream: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    descriptor = -1;
    if (!writer(stream, context, error)) {
        goto cleanup;
    }
    if (fflush(stream) != 0 || fsync(fileno(stream)) != 0) {
        neural_error_set(error,
                         "%s: unable to synchronize temporary file: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    if (fclose(stream) != 0) {
        stream = NULL;
        neural_error_set(error,
                         "%s: unable to close temporary file: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    stream = NULL;
    if (rename(temporary_path, path) != 0) {
        neural_error_set(error,
                         "%s: unable to install atomic file: %s",
                         path,
                         strerror(errno));
        goto cleanup;
    }
    if (!synchronize_parent(path, error)) {
        goto cleanup;
    }
    success = 1;

cleanup:
    if (stream != NULL) {
        (void)fclose(stream);
    } else if (descriptor >= 0) {
        (void)close(descriptor);
    }
    if (!success) {
        (void)unlink(temporary_path);
    }
    free(temporary_path);
    return success;
}

int neural_atomic_file_remove(const char *path,
                              int allow_missing,
                              NeuralError *error)
{
    if (path == NULL || path[0] == '\0') {
        neural_error_set(error, "atomic-remove path is required");
        return 0;
    }
    if (unlink(path) != 0) {
        if (allow_missing && errno == ENOENT) {
            return 1;
        }
        neural_error_set(error,
                         "%s: unable to remove file: %s",
                         path,
                         strerror(errno));
        return 0;
    }
    return synchronize_parent(path, error);
}
