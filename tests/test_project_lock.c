#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/parallel.h"
#include "neural/training.h"
#include "../src/project_lock.h"
#include "../src/train_project.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static void test_shared_and_exclusive_modes(void)
{
    static const char *const directory = "build/tests/project-lock";
    static const char *const path = "build/tests/project-lock/.neural-c.lock";
    NeuralProjectLock first_shared = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralProjectLock second_shared = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralProjectLock exclusive = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralProjectLock rejected = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralError error;
    struct stat status;

    (void)remove(path);
    (void)rmdir(directory);
    if (mkdir(directory, 0700) != 0 && errno != EEXIST) {
        check(0, "project lock directory must be created");
        return;
    }
    check(neural_project_lock_acquire(directory,
                                      NEURAL_PROJECT_LOCK_SHARED,
                                      &first_shared,
                                      &error) &&
              neural_project_lock_acquire(directory,
                                          NEURAL_PROJECT_LOCK_SHARED,
                                          &second_shared,
                                          &error),
          "multiple readers must acquire shared project locks");
    check(!neural_project_lock_acquire(directory,
                                       NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                       &rejected,
                                       &error) &&
              strstr(error.message, "project is busy") != NULL,
          "a writer must fail immediately while readers hold the lock");
    neural_project_lock_release(&rejected);
    neural_project_lock_release(&second_shared);
    neural_project_lock_release(&first_shared);

    check(neural_project_lock_acquire(directory,
                                      NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                      &exclusive,
                                      &error),
          "a writer must acquire the released project lock");
    check(!neural_project_lock_acquire(directory,
                                       NEURAL_PROJECT_LOCK_SHARED,
                                       &rejected,
                                       &error) &&
              strstr(error.message, "project is busy") != NULL,
          "a reader must fail immediately while a writer holds the lock");
    neural_project_lock_release(&rejected);
    neural_project_lock_release(&exclusive);

    check(lstat(path, &status) == 0 && S_ISREG(status.st_mode) &&
              (status.st_mode & 0777) == 0600,
          "the permanent project lock file must use owner-only permissions");
    check(neural_project_lock_acquire(directory,
                                      NEURAL_PROJECT_LOCK_SHARED,
                                      &first_shared,
                                      &error),
          "closing the descriptor must release the project lock");
    neural_project_lock_release(&first_shared);
    check(lstat(path, &status) == 0,
          "releasing a lock must not remove its permanent file");

    (void)remove(path);
    (void)rmdir(directory);
}

static void test_training_respects_project_lock(void)
{
    NeuralProjectLock reader = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralExecutionConfig execution = {1U};
    NeuralTrainingResult result = {
        99U, 99U, 99.0, 99.0, 99.0, 99U, NEURAL_TRAINING_IN_PROGRESS
    };
    NeuralError error;

    check(neural_project_lock_acquire("projects/xor",
                                      NEURAL_PROJECT_LOCK_SHARED,
                                      &reader,
                                      &error),
          "training contention fixture must hold a shared lock");
    if (reader.descriptor >= 0) {
        check(!neural_project_train_fresh("projects/xor",
                                          &execution,
                                          &result,
                                          &error) &&
                  result.completed_epochs == 0U &&
                  strstr(error.message, "project is busy") != NULL,
              "fresh training must fail before work when the project is busy");
        check(!neural_project_train_resume("projects/xor",
                                           &execution,
                                           &result,
                                           &error) &&
                  result.completed_epochs == 0U &&
                  strstr(error.message, "project is busy") != NULL,
              "resume must fail before state checks when the project is busy");
    }
    neural_project_lock_release(&reader);
}

int main(void)
{
    test_shared_and_exclusive_modes();
    test_training_respects_project_lock();

    if (failures != 0) {
        fprintf(stderr, "%d project-lock test(s) failed\n", failures);
        return 1;
    }
    puts("All project-lock tests passed");
    return 0;
}
