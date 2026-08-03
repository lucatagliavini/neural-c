#ifndef NEURAL_BATCH_H
#define NEURAL_BATCH_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/gradient.h"
#include "neural/model.h"

typedef struct {
    size_t sample_count;
    size_t batch_size;
    size_t batch_count;
} NeuralBatchPlan;

typedef struct NeuralBatchAccumulator NeuralBatchAccumulator;

int neural_batch_plan_create(size_t sample_count,
                             size_t batch_size,
                             NeuralBatchPlan *plan,
                             NeuralError *error);
int neural_batch_plan_range(const NeuralBatchPlan *plan,
                            size_t batch_index,
                            size_t *sample_begin,
                            size_t *sample_end,
                            NeuralError *error);

int neural_batch_accumulator_create(const NeuralModel *model,
                                    NeuralBatchAccumulator **accumulator,
                                    NeuralError *error);
void neural_batch_accumulator_free(NeuralBatchAccumulator *accumulator);
int neural_batch_accumulator_reset(NeuralBatchAccumulator *accumulator,
                                   size_t sample_begin,
                                   size_t sample_end,
                                   NeuralError *error);
int neural_batch_accumulator_add(NeuralBatchAccumulator *accumulator,
                                 size_t sample_index,
                                 const NeuralGradient *sample_gradient,
                                 NeuralError *error);
int neural_batch_accumulator_finalize(NeuralBatchAccumulator *accumulator,
                                      NeuralError *error);
const NeuralGradient *neural_batch_accumulator_gradient(
    const NeuralBatchAccumulator *accumulator);
size_t neural_batch_accumulator_sample_count(
    const NeuralBatchAccumulator *accumulator);

#endif
