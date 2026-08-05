#include "neural/batch.h"

#include <stdint.h>
#include <stdlib.h>

#include "neural/random.h"
#include "gradient_internal.h"

struct NeuralSampleOrder {
    size_t sample_count;
    size_t *indices;
};

struct NeuralBatchAccumulator {
    NeuralGradient *gradient;
    NeuralGradient *compensation;
    size_t next_sample_index;
    size_t sample_end;
    size_t sample_count;
    int active;
    int finalized;
};

static uint64_t mix_uint64(uint64_t value)
{
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

static uint64_t epoch_random_seed(uint64_t training_seed,
                                  uint64_t epoch_index)
{
    const uint64_t domain = UINT64_C(0x6e657572616c2d73);
    const uint64_t increment = UINT64_C(0x9e3779b97f4a7c15);

    return mix_uint64((training_seed ^ domain) +
                      increment * (epoch_index + UINT64_C(1)));
}

static uint64_t random_bounded(NeuralRandom *random, uint64_t bound)
{
    const uint64_t threshold = (UINT64_C(0) - bound) % bound;
    uint64_t value;

    do {
        value = neural_random_next_uint64(random);
    } while (value < threshold);
    return value % bound;
}

int neural_sample_order_create(size_t sample_count,
                               NeuralSampleOrder **order,
                               NeuralError *error)
{
    NeuralSampleOrder *created;

    if (order == NULL || sample_count == 0U ||
        sample_count > SIZE_MAX / sizeof(*created->indices)) {
        neural_error_set(error, "sample order requires a valid sample count");
        return 0;
    }
    *order = NULL;
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate sample order");
        return 0;
    }
    created->indices = malloc(sample_count * sizeof(*created->indices));
    if (created->indices == NULL) {
        neural_error_set(error, "unable to allocate sample-order indices");
        free(created);
        return 0;
    }
    created->sample_count = sample_count;
    *order = created;
    return 1;
}

void neural_sample_order_free(NeuralSampleOrder *order)
{
    if (order != NULL) {
        free(order->indices);
        free(order);
    }
}

int neural_sample_order_prepare(NeuralSampleOrder *order,
                                uint64_t training_seed,
                                uint64_t epoch_index,
                                int shuffle,
                                NeuralError *error)
{
    size_t logical_index;

    if (order == NULL || order->sample_count == 0U ||
        order->indices == NULL || (shuffle != 0 && shuffle != 1)) {
        neural_error_set(error, "invalid sample-order request");
        return 0;
    }
    for (logical_index = 0U;
         logical_index < order->sample_count;
         logical_index++) {
        order->indices[logical_index] = logical_index;
    }
    if (!shuffle) {
        return 1;
    }
    {
        NeuralRandom random;

        neural_random_init(&random,
                           epoch_random_seed(training_seed, epoch_index));
        for (logical_index = order->sample_count;
             logical_index > 1U;
             logical_index--) {
            size_t selected = (size_t)random_bounded(
                &random, (uint64_t)logical_index);
            size_t temporary = order->indices[logical_index - 1U];

            order->indices[logical_index - 1U] = order->indices[selected];
            order->indices[selected] = temporary;
        }
    }
    return 1;
}

size_t neural_sample_order_count(const NeuralSampleOrder *order)
{
    return order == NULL ? 0U : order->sample_count;
}

int neural_sample_order_index(const NeuralSampleOrder *order,
                              size_t logical_index,
                              size_t *sample_index,
                              NeuralError *error)
{
    if (order == NULL || order->indices == NULL || sample_index == NULL ||
        logical_index >= order->sample_count) {
        neural_error_set(error, "sample-order index is outside the plan");
        return 0;
    }
    *sample_index = order->indices[logical_index];
    return 1;
}

static int batch_plan_is_valid(const NeuralBatchPlan *plan)
{
    size_t expected_count;

    if (plan == NULL || plan->sample_count == 0U || plan->batch_size == 0U ||
        plan->batch_size > plan->sample_count) {
        return 0;
    }
    expected_count = plan->sample_count / plan->batch_size;
    if (plan->sample_count % plan->batch_size != 0U) {
        expected_count++;
    }
    return plan->batch_count == expected_count;
}

int neural_batch_plan_create(size_t sample_count,
                             size_t batch_size,
                             NeuralBatchPlan *plan,
                             NeuralError *error)
{
    if (plan == NULL || sample_count == 0U || batch_size == 0U ||
        batch_size > sample_count) {
        neural_error_set(error, "batch size must be within the dataset");
        return 0;
    }
    plan->sample_count = sample_count;
    plan->batch_size = batch_size;
    plan->batch_count = sample_count / batch_size;
    if (sample_count % batch_size != 0U) {
        plan->batch_count++;
    }
    return 1;
}

int neural_batch_plan_range(const NeuralBatchPlan *plan,
                            size_t batch_index,
                            size_t *sample_begin,
                            size_t *sample_end,
                            NeuralError *error)
{
    size_t begin;
    size_t remaining;

    if (!batch_plan_is_valid(plan) || sample_begin == NULL ||
        sample_end == NULL || batch_index >= plan->batch_count) {
        neural_error_set(error, "invalid batch range request");
        return 0;
    }
    begin = batch_index * plan->batch_size;
    remaining = plan->sample_count - begin;
    *sample_begin = begin;
    *sample_end = begin +
                  (remaining < plan->batch_size ? remaining : plan->batch_size);
    return 1;
}

int neural_batch_accumulator_create(const NeuralModel *model,
                                    NeuralBatchAccumulator **accumulator,
                                    NeuralError *error)
{
    NeuralBatchAccumulator *created;

    if (model == NULL || accumulator == NULL) {
        neural_error_set(error, "model and batch accumulator are required");
        return 0;
    }
    *accumulator = NULL;
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate batch accumulator");
        return 0;
    }
    if (!neural_gradient_create(model, &created->gradient, error) ||
        !neural_gradient_create(model, &created->compensation, error)) {
        neural_gradient_free(created->gradient);
        free(created);
        return 0;
    }
    *accumulator = created;
    return 1;
}

void neural_batch_accumulator_free(NeuralBatchAccumulator *accumulator)
{
    if (accumulator == NULL) {
        return;
    }
    neural_gradient_free(accumulator->compensation);
    neural_gradient_free(accumulator->gradient);
    free(accumulator);
}

int neural_batch_accumulator_reset(NeuralBatchAccumulator *accumulator,
                                   size_t sample_begin,
                                   size_t sample_end,
                                   NeuralError *error)
{
    if (accumulator == NULL || sample_begin >= sample_end) {
        neural_error_set(error, "invalid batch accumulator reset");
        return 0;
    }
    if (!neural_gradient_zero(accumulator->gradient, error) ||
        !neural_gradient_zero(accumulator->compensation, error)) {
        return 0;
    }
    accumulator->next_sample_index = sample_begin;
    accumulator->sample_end = sample_end;
    accumulator->sample_count = 0U;
    accumulator->active = 1;
    accumulator->finalized = 0;
    return 1;
}

int neural_batch_accumulator_add(NeuralBatchAccumulator *accumulator,
                                 size_t sample_index,
                                 const NeuralGradient *sample_gradient,
                                 NeuralError *error)
{
    if (accumulator == NULL || !accumulator->active ||
        accumulator->finalized ||
        accumulator->next_sample_index >= accumulator->sample_end ||
        sample_index != accumulator->next_sample_index) {
        neural_error_set(error,
                         "sample gradients must be accumulated in batch order");
        return 0;
    }
    if (!neural_gradient_accumulate_compensated(accumulator->gradient,
                                                accumulator->compensation,
                                                sample_gradient,
                                                error)) {
        return 0;
    }
    accumulator->next_sample_index++;
    accumulator->sample_count++;
    return 1;
}

int neural_batch_accumulator_finalize(NeuralBatchAccumulator *accumulator,
                                      NeuralError *error)
{
    neural_real divisor;

    if (accumulator == NULL || !accumulator->active ||
        accumulator->finalized || accumulator->sample_count == 0U ||
        accumulator->next_sample_index != accumulator->sample_end) {
        neural_error_set(error, "complete non-empty batch is required");
        return 0;
    }
    divisor = (neural_real)accumulator->sample_count;
    if (!neural_gradient_finish_compensated(accumulator->gradient,
                                            accumulator->compensation,
                                            1.0 / divisor,
                                            error)) {
        return 0;
    }
    accumulator->finalized = 1;
    return 1;
}

const NeuralGradient *neural_batch_accumulator_gradient(
    const NeuralBatchAccumulator *accumulator)
{
    if (accumulator == NULL || !accumulator->finalized) {
        return NULL;
    }
    return accumulator->gradient;
}

size_t neural_batch_accumulator_sample_count(
    const NeuralBatchAccumulator *accumulator)
{
    return accumulator == NULL ? 0U : accumulator->sample_count;
}
