#define _GNU_SOURCE

#include "project_lock.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/defaults.h"
#include "path.h"

static int open_lock_file(const char *path,
                          int *created,
                          NeuralError *error)
{
    int descriptor;

    *created = 0;
    descriptor = open(path,
                      O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                      0600);
    if (descriptor >= 0) {
        *created = 1;
        return descriptor;
    }
    if (errno != EEXIST) {
        neural_error_set(error,
                         "%s: unable to create project lock: %s",
                         path,
                         strerror(errno));
        return -1;
    }
    descriptor = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        neural_error_set(error,
                         "%s: unable to open project lock: %s",
                         path,
                         strerror(errno));
    }
    return descriptor;
}

int neural_project_lock_acquire(const char *directory,
                                NeuralProjectLockMode mode,
                                NeuralProjectLock *lock,
                                NeuralError *error)
{
    struct stat status;
    char *path;
    int descriptor;
    int created;
    int operation;

    if (directory == NULL || directory[0] == '\0' || lock == NULL ||
        (mode != NEURAL_PROJECT_LOCK_SHARED &&
         mode != NEURAL_PROJECT_LOCK_EXCLUSIVE)) {
        neural_error_set(error, "project lock configuration is invalid");
        return 0;
    }
    lock->descriptor = -1;
    lock->path = NULL;
    path = neural_path_join(directory, NEURAL_DEFAULT_LOCK_FILENAME, error);
    if (path == NULL) {
        return 0;
    }
    descriptor = open_lock_file(path, &created, error);
    if (descriptor < 0) {
        free(path);
        return 0;
    }
    if (fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
        neural_error_set(error, "%s: project lock is not a regular file", path);
        (void)close(descriptor);
        free(path);
        return 0;
    }
    operation = mode == NEURAL_PROJECT_LOCK_SHARED ? LOCK_SH : LOCK_EX;
    while (flock(descriptor, operation | LOCK_NB) != 0) {
        if (errno == EINTR) {
            continue;
        }
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
            neural_error_set(
                error,
                "%s: project is busy; another command holds its lock",
                directory);
        } else {
            neural_error_set(error,
                             "%s: unable to acquire project lock: %s",
                             path,
                             strerror(errno));
        }
        (void)close(descriptor);
        free(path);
        return 0;
    }
    if (created && fchmod(descriptor, 0600) != 0) {
        neural_error_set(error,
                         "%s: unable to set project lock permissions: %s",
                         path,
                         strerror(errno));
        (void)close(descriptor);
        free(path);
        return 0;
    }
    lock->descriptor = descriptor;
    lock->path = path;
    return 1;
}

void neural_project_lock_release(NeuralProjectLock *lock)
{
    if (lock == NULL) {
        return;
    }
    if (lock->descriptor >= 0) {
        (void)close(lock->descriptor);
    }
    free(lock->path);
    lock->descriptor = -1;
    lock->path = NULL;
}

void neural_project_lock_discard(NeuralProjectLock *lock)
{
    if (lock != NULL && lock->path != NULL) {
        (void)unlink(lock->path);
    }
    neural_project_lock_release(lock);
}
