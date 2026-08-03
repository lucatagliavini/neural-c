#include <stdio.h>
#include <string.h>
#include <threads.h>

#include "neural/defaults.h"
#include "neural/gradient.h"
#include "neural/model.h"
#include "neural/parallel.h"

enum {
    TEST_THREAD_COUNT = 4,
    TEST_ITERATIONS = 500
};

typedef struct {
    const NeuralModel *model;
    NeuralWorkerContext *context;
    size_t sample_index;
    int success;
} WorkerArguments;

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int worker_run(void *opaque)
{
    WorkerArguments *arguments = opaque;
    const neural_real inputs[] = {1.0, -1.0};
    neural_real expected_outputs[2] = {0.0, 0.0};
    NeuralWorkspace *workspace =
        neural_worker_context_workspace(arguments->context);
    NeuralGradient *gradient =
        neural_worker_context_gradient(arguments->context);
    neural_real *weight_gradients;
    neural_real *bias_gradients;
    size_t weight_count;
    size_t bias_count;
    size_t iteration;
    NeuralError error;

    arguments->success = 0;
    for (iteration = 0U; iteration < TEST_ITERATIONS; iteration++) {
        neural_real outputs[2];

        if (!neural_model_forward(arguments->model,
                                  workspace,
                                  inputs,
                                  2U,
                                  outputs,
                                  2U,
                                  &error)) {
            return -1;
        }
        if (iteration == 0U) {
            memcpy(expected_outputs, outputs, sizeof(outputs));
        } else if (memcmp(expected_outputs, outputs, sizeof(outputs)) != 0) {
            return -1;
        }
    }
    weight_gradients = neural_gradient_layer_weights(gradient,
                                                      0U,
                                                      &weight_count);
    bias_gradients = neural_gradient_layer_biases(gradient,
                                                  0U,
                                                  &bias_count);
    if (weight_gradients == NULL || bias_gradients == NULL) {
        return -1;
    }
    for (iteration = 0U; iteration < weight_count; iteration++) {
        weight_gradients[iteration] =
            (neural_real)(arguments->sample_index + 1U);
    }
    for (iteration = 0U; iteration < bias_count; iteration++) {
        bias_gradients[iteration] =
            (neural_real)(arguments->sample_index + 1U);
    }
    arguments->success = 1;
    return 0;
}

static void test_execution_plan(void)
{
    NeuralExecutionConfig serial = {1U};
    NeuralExecutionConfig parallel = {4U};
    NeuralExecutionPlan serial_plan;
    NeuralExecutionPlan parallel_plan;
    NeuralError error;
    size_t task_index;

    check(neural_execution_plan_create(7U,
                                       &serial,
                                       &serial_plan,
                                       &error) &&
              neural_execution_plan_create(7U,
                                           &parallel,
                                           &parallel_plan,
                                           &error),
          "serial and parallel plans must be created");
    check(serial_plan.task_count == parallel_plan.task_count &&
              serial_plan.worker_count == 1U &&
              parallel_plan.worker_count == 4U,
          "thread count must change workers but not logical tasks");
    for (task_index = 0U; task_index < serial_plan.task_count; task_index++) {
        size_t serial_begin;
        size_t serial_end;
        size_t parallel_begin;
        size_t parallel_end;

        check(neural_execution_plan_task_range(&serial_plan,
                                               task_index,
                                               &serial_begin,
                                               &serial_end,
                                               &error) &&
                  neural_execution_plan_task_range(&parallel_plan,
                                                   task_index,
                                                   &parallel_begin,
                                                   &parallel_end,
                                                   &error) &&
                  serial_begin == parallel_begin &&
                  serial_end == parallel_end &&
                  serial_end == serial_begin + 1U,
              "logical sample tasks must be thread-count independent");
    }
}

static void test_parallel_forward_and_reduction(void)
{
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 1U, layers};
    const neural_real weights[] = {1.0, 2.0, -1.0, 0.5};
    const neural_real biases[] = {0.25, -0.5};
    NeuralModel *model = NULL;
    NeuralWorkerContext *contexts[TEST_THREAD_COUNT] = {0};
    NeuralGradient *sample_gradients[TEST_THREAD_COUNT] = {0};
    NeuralGradient *reduced = NULL;
    WorkerArguments arguments[TEST_THREAD_COUNT];
    thrd_t threads[TEST_THREAD_COUNT];
    int started[TEST_THREAD_COUNT] = {0};
    NeuralError error;
    size_t index;
    int workers_ready = 1;
    int model_prepared;

    model_prepared = neural_model_create(&spec,
                                         UINT64_C(9),
                                         &model,
                                         &error);
    if (model_prepared) {
        model_prepared = neural_model_set_layer_parameters(model,
                                                           0U,
                                                           weights,
                                                           4U,
                                                           biases,
                                                           2U,
                                                           &error);
    }
    check(model_prepared, "parallel test model must be prepared");
    if (!model_prepared) {
        neural_model_free(model);
        return;
    }
    for (index = 0U; index < TEST_THREAD_COUNT; index++) {
        int context_created =
            neural_worker_context_create(model, &contexts[index], &error);

        check(context_created, "each worker must receive private state");
        if (!context_created) {
            workers_ready = 0;
            continue;
        }
        arguments[index].model = model;
        arguments[index].context = contexts[index];
        arguments[index].sample_index = index;
        arguments[index].success = 0;
        started[index] =
            thrd_create(&threads[index], worker_run, &arguments[index]) ==
            thrd_success;
        check(started[index], "worker thread must start");
        if (!started[index]) {
            workers_ready = 0;
        }
    }
    for (index = 0U; index < TEST_THREAD_COUNT; index++) {
        int result = -1;

        if (started[index]) {
            int joined = thrd_join(threads[index], &result) == thrd_success;

            check(joined && result == 0 && arguments[index].success,
                  "worker thread must finish without shared-state races");
            if (!joined || result != 0 || !arguments[index].success) {
                workers_ready = 0;
            }
        }
        if (workers_ready && contexts[index] != NULL) {
            int copied = neural_gradient_create(model,
                                                &sample_gradients[index],
                                                &error) &&
                         neural_gradient_copy(
                             sample_gradients[index],
                             neural_worker_context_gradient(contexts[index]),
                             &error);

            check(copied,
                  "worker scratch gradient must copy into its sample slot");
            if (!copied) {
                workers_ready = 0;
            }
        }
    }
    if (workers_ready) {
        check(neural_gradient_create(model, &reduced, &error) &&
                  neural_gradient_reduce_ordered(reduced,
                                                 sample_gradients,
                                                 TEST_THREAD_COUNT,
                                                 &error),
              "sample gradients must reduce in sample order");
    }
    if (reduced != NULL) {
        neural_real *reduced_weights;
        size_t count;

        reduced_weights = neural_gradient_layer_weights(reduced, 0U, &count);
        check(count == 4U && reduced_weights[0] == 10.0 &&
                  reduced_weights[3] == 10.0,
              "ordered reduction must include every private gradient");
        check(neural_gradient_scale(reduced,
                                    1.0 / (neural_real)TEST_THREAD_COUNT,
                                    &error),
              "ordered gradient sum must form a batch mean");
        check(neural_model_apply_gradient(model, reduced, 0.01, &error),
              "one coordinated model update must succeed after reduction");
        {
            const neural_real *updated_weights;

            updated_weights = neural_model_layer_weights(model, 0U, &count);
            check(updated_weights[0] == 0.975 &&
                      updated_weights[3] == 0.475,
                  "model update must apply the reduced batch mean once");
        }
    }

    neural_gradient_free(reduced);
    for (index = 0U; index < TEST_THREAD_COUNT; index++) {
        neural_gradient_free(sample_gradients[index]);
        neural_worker_context_free(contexts[index]);
    }
    neural_model_free(model);
}

int main(void)
{
    check(NEURAL_DEFAULT_THREAD_COUNT >= 1U,
          "default thread count must be positive");
    test_execution_plan();
    test_parallel_forward_and_reduction();

    if (failures != 0) {
        fprintf(stderr, "%d parallel test(s) failed\n", failures);
        return 1;
    }
    puts("All parallel tests passed");
    return 0;
}
