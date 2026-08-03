#ifndef NEURAL_PARALLEL_H
#define NEURAL_PARALLEL_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/gradient.h"
#include "neural/model.h"

typedef struct NeuralWorkerContext NeuralWorkerContext;

typedef struct {
    size_t thread_count;
} NeuralExecutionConfig;

typedef struct {
    size_t sample_count;
    size_t worker_count;
    size_t task_count;
} NeuralExecutionPlan;

int neural_execution_config_validate(const NeuralExecutionConfig *config,
                                     NeuralError *error);
int neural_execution_plan_create(size_t sample_count,
                                 const NeuralExecutionConfig *config,
                                 NeuralExecutionPlan *plan,
                                 NeuralError *error);
int neural_execution_plan_task_range(const NeuralExecutionPlan *plan,
                                     size_t task_index,
                                     size_t *begin,
                                     size_t *end,
                                     NeuralError *error);

int neural_worker_context_create(const NeuralModel *model,
                                 NeuralWorkerContext **context,
                                 NeuralError *error);
void neural_worker_context_free(NeuralWorkerContext *context);
NeuralWorkspace *neural_worker_context_workspace(
    NeuralWorkerContext *context);
NeuralGradient *neural_worker_context_gradient(NeuralWorkerContext *context);

#endif
