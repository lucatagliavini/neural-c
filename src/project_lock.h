#ifndef NEURAL_INTERNAL_PROJECT_LOCK_H
#define NEURAL_INTERNAL_PROJECT_LOCK_H

#include "neural/error.h"

typedef enum {
    NEURAL_PROJECT_LOCK_SHARED,
    NEURAL_PROJECT_LOCK_EXCLUSIVE
} NeuralProjectLockMode;

typedef struct {
    int descriptor;
    char *path;
} NeuralProjectLock;

#define NEURAL_PROJECT_LOCK_INITIALIZER {-1, NULL}

int neural_project_lock_acquire(const char *directory,
                                NeuralProjectLockMode mode,
                                NeuralProjectLock *lock,
                                NeuralError *error);
void neural_project_lock_release(NeuralProjectLock *lock);
void neural_project_lock_discard(NeuralProjectLock *lock);

#endif
