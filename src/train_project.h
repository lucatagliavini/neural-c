#ifndef NEURAL_INTERNAL_TRAIN_PROJECT_H
#define NEURAL_INTERNAL_TRAIN_PROJECT_H

#include <signal.h>

#include "neural/error.h"
#include "neural/parallel.h"
#include "neural/training.h"

int neural_project_train_fresh(const char *directory,
                               const NeuralExecutionConfig *execution,
                               NeuralTrainingResult *result,
                               NeuralError *error);
int neural_project_train_resume(const char *directory,
                                const NeuralExecutionConfig *execution,
                                NeuralTrainingResult *result,
                                NeuralError *error);
int neural_project_train_additional(const char *directory,
                                    const NeuralExecutionConfig *execution,
                                    size_t additional_epochs,
                                    NeuralTrainingResult *result,
                                    NeuralError *error);
int neural_project_train_fresh_controlled(
    const char *directory,
    const NeuralExecutionConfig *execution,
    const volatile sig_atomic_t *stop_request,
    NeuralEpochObserver observer,
    void *observer_context,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error);
int neural_project_train_resume_controlled(
    const char *directory,
    const NeuralExecutionConfig *execution,
    const volatile sig_atomic_t *stop_request,
    NeuralEpochObserver observer,
    void *observer_context,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error);
int neural_project_train_additional_controlled(
    const char *directory,
    const NeuralExecutionConfig *execution,
    size_t additional_epochs,
    const volatile sig_atomic_t *stop_request,
    NeuralEpochObserver observer,
    void *observer_context,
    int *interrupted_signal,
    NeuralTrainingResult *result,
    NeuralError *error);

#endif
