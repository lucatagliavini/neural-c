#ifndef NEURAL_PARALLEL_H
#define NEURAL_PARALLEL_H

#include <stddef.h>

#include "neural/batch.h"
#include "neural/error.h"
#include "neural/gradient.h"
#include "neural/model.h"
#include "neural/project.h"

typedef struct NeuralWorkerContext NeuralWorkerContext;
typedef struct NeuralParallelExecutor NeuralParallelExecutor;

typedef struct {
    size_t thread_count;
} NeuralExecutionConfig;

typedef struct {
    size_t sample_begin;
    size_t sample_end;
    size_t worker_count;
    size_t wave_count;
} NeuralExecutionPlan;

int neural_execution_config_validate(const NeuralExecutionConfig *config,
                                     NeuralError *error);
int neural_execution_plan_create(size_t sample_begin,
                                 size_t sample_end,
                                 const NeuralExecutionConfig *config,
                                 NeuralExecutionPlan *plan,
                                 NeuralError *error);
int neural_execution_plan_wave_range(const NeuralExecutionPlan *plan,
                                     size_t wave_index,
                                     size_t *sample_begin,
                                     size_t *sample_end,
                                     NeuralError *error);

int neural_worker_context_create(const NeuralModel *model,
                                 NeuralWorkerContext **context,
                                 NeuralError *error);
void neural_worker_context_free(NeuralWorkerContext *context);
NeuralWorkspace *neural_worker_context_workspace(
    NeuralWorkerContext *context);
NeuralGradient *neural_worker_context_gradient(NeuralWorkerContext *context);

int neural_parallel_executor_create(
    const NeuralModel *model,
    const NeuralDataset *dataset,
    NeuralLoss loss,
    const NeuralExecutionConfig *config,
    NeuralParallelExecutor **executor,
    NeuralError *error);
void neural_parallel_executor_free(NeuralParallelExecutor *executor);
size_t neural_parallel_executor_worker_count(
    const NeuralParallelExecutor *executor);

/* Model and dataset must outlive the executor and remain unchanged during a
 * batch operation. The returned gradient is owned by the executor and valid
 * until its next batch operation or destruction. One coordinator may drive
 * an executor. */
int neural_parallel_executor_batch_gradient(
    NeuralParallelExecutor *executor,
    size_t sample_begin,
    size_t sample_end,
    const NeuralGradient **gradient,
    NeuralError *error);

/* The half-open range addresses logical positions in a complete order whose
 * sample count matches the executor dataset. Reduction follows that range. */
int neural_parallel_executor_ordered_batch_gradient(
    NeuralParallelExecutor *executor,
    const NeuralSampleOrder *order,
    size_t logical_begin,
    size_t logical_end,
    const NeuralGradient **gradient,
    NeuralError *error);

#endif
