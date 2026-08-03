#include "neural/batch.h"

#include <stdint.h>
#include <stdlib.h>

#include "gradient_internal.h"

struct NeuralBatchAccumulator {
    NeuralGradient *gradient;
    NeuralGradient *compensation;
    size_t next_sample_index;
    size_t sample_end;
    size_t sample_count;
    int active;
    int finalized;
};

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
