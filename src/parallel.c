#include "neural/parallel.h"

#include <stdlib.h>

#include "neural/defaults.h"

#if NEURAL_DEFAULT_THREAD_COUNT < 1U
#error "NEURAL_DEFAULT_THREAD_COUNT must be positive"
#endif

struct NeuralWorkerContext {
    NeuralWorkspace *workspace;
    NeuralGradient *gradient;
};

int neural_execution_config_validate(const NeuralExecutionConfig *config,
                                     NeuralError *error)
{
    if (config == NULL || config->thread_count == 0U) {
        neural_error_set(error, "thread count must be a positive integer");
        return 0;
    }
    return 1;
}

static int execution_plan_is_valid(const NeuralExecutionPlan *plan)
{
    size_t sample_count;
    size_t expected_wave_count;

    if (plan == NULL || plan->sample_begin >= plan->sample_end) {
        return 0;
    }
    sample_count = plan->sample_end - plan->sample_begin;
    if (plan->worker_count == 0U || plan->worker_count > sample_count) {
        return 0;
    }
    expected_wave_count = sample_count / plan->worker_count;
    if (sample_count % plan->worker_count != 0U) {
        expected_wave_count++;
    }
    return plan->wave_count == expected_wave_count;
}

int neural_execution_plan_create(size_t sample_begin,
                                 size_t sample_end,
                                 const NeuralExecutionConfig *config,
                                 NeuralExecutionPlan *plan,
                                 NeuralError *error)
{
    size_t sample_count;

    if (plan == NULL || sample_begin >= sample_end ||
        !neural_execution_config_validate(config, error)) {
        if (plan == NULL || sample_begin >= sample_end) {
            neural_error_set(error,
                             "execution wave plan requires a sample range");
        }
        return 0;
    }
    sample_count = sample_end - sample_begin;
    plan->sample_begin = sample_begin;
    plan->sample_end = sample_end;
    plan->worker_count = config->thread_count < sample_count
                             ? config->thread_count
                             : sample_count;
    plan->wave_count = sample_count / plan->worker_count;
    if (sample_count % plan->worker_count != 0U) {
        plan->wave_count++;
    }
    return 1;
}

int neural_execution_plan_wave_range(const NeuralExecutionPlan *plan,
                                     size_t wave_index,
                                     size_t *sample_begin,
                                     size_t *sample_end,
                                     NeuralError *error)
{
    size_t begin;
    size_t remaining;

    if (!execution_plan_is_valid(plan) || sample_begin == NULL ||
        sample_end == NULL || wave_index >= plan->wave_count) {
        neural_error_set(error, "invalid execution wave range request");
        return 0;
    }
    begin = plan->sample_begin + wave_index * plan->worker_count;
    remaining = plan->sample_end - begin;
    *sample_begin = begin;
    *sample_end = begin + (remaining < plan->worker_count
                               ? remaining
                               : plan->worker_count);
    return 1;
}

void neural_worker_context_free(NeuralWorkerContext *context)
{
    if (context != NULL) {
        neural_gradient_free(context->gradient);
        neural_workspace_free(context->workspace);
        free(context);
    }
}

int neural_worker_context_create(const NeuralModel *model,
                                 NeuralWorkerContext **context,
                                 NeuralError *error)
{
    NeuralWorkerContext *created;

    if (model == NULL || context == NULL) {
        neural_error_set(error, "model and worker context output are required");
        return 0;
    }
    *context = NULL;
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate worker context");
        return 0;
    }
    if (!neural_workspace_create(model, &created->workspace, error) ||
        !neural_gradient_create(model, &created->gradient, error)) {
        neural_worker_context_free(created);
        return 0;
    }
    *context = created;
    return 1;
}

NeuralWorkspace *neural_worker_context_workspace(
    NeuralWorkerContext *context)
{
    return context == NULL ? NULL : context->workspace;
}

NeuralGradient *neural_worker_context_gradient(NeuralWorkerContext *context)
{
    return context == NULL ? NULL : context->gradient;
}
