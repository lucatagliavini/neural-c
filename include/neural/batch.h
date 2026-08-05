#ifndef NEURAL_BATCH_H
#define NEURAL_BATCH_H

#include <stddef.h>
#include <stdint.h>

#include "neural/error.h"
#include "neural/gradient.h"
#include "neural/model.h"

typedef struct {
    size_t sample_count;
    size_t batch_size;
    size_t batch_count;
} NeuralBatchPlan;

typedef struct NeuralBatchAccumulator NeuralBatchAccumulator;
typedef struct NeuralSampleOrder NeuralSampleOrder;

/* The order owns one source index per logical position. Preparation is a pure
 * function of the training seed, zero-based absolute epoch, and shuffle flag. */
int neural_sample_order_create(size_t sample_count,
                               NeuralSampleOrder **order,
                               NeuralError *error);
void neural_sample_order_free(NeuralSampleOrder *order);
int neural_sample_order_prepare(NeuralSampleOrder *order,
                                uint64_t training_seed,
                                uint64_t epoch_index,
                                int shuffle,
                                NeuralError *error);
size_t neural_sample_order_count(const NeuralSampleOrder *order);
int neural_sample_order_index(const NeuralSampleOrder *order,
                              size_t logical_index,
                              size_t *sample_index,
                              NeuralError *error);

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
