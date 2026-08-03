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

int neural_execution_plan_create(size_t sample_count,
                                 const NeuralExecutionConfig *config,
                                 NeuralExecutionPlan *plan,
                                 NeuralError *error)
{
    if (plan == NULL || sample_count == 0U ||
        !neural_execution_config_validate(config, error)) {
        if (plan == NULL || sample_count == 0U) {
            neural_error_set(error,
                             "execution plan requires samples and an output");
        }
        return 0;
    }
    plan->sample_count = sample_count;
    plan->worker_count = config->thread_count < sample_count
                             ? config->thread_count
                             : sample_count;
    plan->task_count = sample_count;
    return 1;
}

int neural_execution_plan_task_range(const NeuralExecutionPlan *plan,
                                     size_t task_index,
                                     size_t *begin,
                                     size_t *end,
                                     NeuralError *error)
{
    if (plan == NULL || begin == NULL || end == NULL ||
        plan->sample_count == 0U ||
        plan->task_count != plan->sample_count ||
        plan->worker_count == 0U || task_index >= plan->task_count) {
        neural_error_set(error, "invalid deterministic execution task");
        return 0;
    }
    *begin = task_index;
    *end = task_index + 1U;
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
