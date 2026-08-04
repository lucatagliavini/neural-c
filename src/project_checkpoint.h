#ifndef NEURAL_INTERNAL_PROJECT_CHECKPOINT_H
#define NEURAL_INTERNAL_PROJECT_CHECKPOINT_H

#include <signal.h>

#include "neural/checkpoint.h"
#include "neural/model.h"
#include "neural/training.h"

typedef struct {
    const char *path;
    const NeuralModel *model;
    size_t interval;
    const volatile sig_atomic_t *stop_request;
    int interrupted_signal;
    NeuralCheckpointMetadata metadata;
} NeuralProjectCheckpointObserver;

int neural_project_checkpoint_observer_initialize(
    NeuralProjectCheckpointObserver *observer,
    const char *path,
    const NeuralModel *model,
    const NeuralProjectDigests *digests,
    size_t interval,
    size_t target_epochs,
    NeuralError *error);

void neural_project_checkpoint_observer_set_stop_request(
    NeuralProjectCheckpointObserver *observer,
    const volatile sig_atomic_t *stop_request);

int neural_project_checkpoint_observe(const NeuralEpochReport *report,
                                      void *context,
                                      NeuralError *error);

#endif
