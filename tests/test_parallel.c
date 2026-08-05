#include <float.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

static int gradients_equal(const NeuralGradient *left,
                           const NeuralGradient *right)
{
    size_t layer_index;

    for (layer_index = 0U; ; layer_index++) {
        size_t left_weight_count;
        size_t right_weight_count;
        size_t left_bias_count;
        size_t right_bias_count;
        const neural_real *left_weights =
            neural_gradient_layer_weights_const(left,
                                                layer_index,
                                                &left_weight_count);
        const neural_real *right_weights =
            neural_gradient_layer_weights_const(right,
                                                layer_index,
                                                &right_weight_count);
        const neural_real *left_biases =
            neural_gradient_layer_biases_const(left,
                                               layer_index,
                                               &left_bias_count);
        const neural_real *right_biases =
            neural_gradient_layer_biases_const(right,
                                               layer_index,
                                               &right_bias_count);

        if (left_weights == NULL || right_weights == NULL ||
            left_biases == NULL || right_biases == NULL) {
            return layer_index > 0U && left_weights == right_weights &&
                   left_biases == right_biases;
        }
        if (left_weight_count != right_weight_count ||
            left_bias_count != right_bias_count ||
            memcmp(left_weights,
                   right_weights,
                   left_weight_count * sizeof(*left_weights)) != 0 ||
            memcmp(left_biases,
                   right_biases,
                   left_bias_count * sizeof(*left_biases)) != 0) {
            return 0;
        }
    }
}

static void *worker_run(void *opaque)
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
            return NULL;
        }
        if (iteration == 0U) {
            memcpy(expected_outputs, outputs, sizeof(outputs));
        } else if (memcmp(expected_outputs, outputs, sizeof(outputs)) != 0) {
            return NULL;
        }
    }
    weight_gradients = neural_gradient_layer_weights(gradient,
                                                      0U,
                                                      &weight_count);
    bias_gradients = neural_gradient_layer_biases(gradient,
                                                  0U,
                                                  &bias_count);
    if (weight_gradients == NULL || bias_gradients == NULL) {
        return NULL;
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
    return NULL;
}

static void test_execution_plan(void)
{
    NeuralExecutionConfig serial = {1U};
    NeuralExecutionConfig parallel = {3U};
    NeuralExecutionConfig oversized = {20U};
    NeuralExecutionConfig invalid_config = {0U};
    NeuralExecutionPlan plan;
    NeuralExecutionPlan invalid;
    NeuralError error;
    size_t begin;
    size_t end;

    check(neural_execution_plan_create(4U,
                                       11U,
                                       &parallel,
                                       &plan,
                                       &error) &&
              plan.worker_count == 3U && plan.wave_count == 3U,
          "parallel batch must be split into bounded waves");
    check(neural_execution_plan_wave_range(&plan, 0U, &begin, &end, &error) &&
              begin == 4U && end == 7U &&
              neural_execution_plan_wave_range(&plan,
                                               1U,
                                               &begin,
                                               &end,
                                               &error) &&
              begin == 7U && end == 10U &&
              neural_execution_plan_wave_range(&plan,
                                               2U,
                                               &begin,
                                               &end,
                                               &error) &&
              begin == 10U && end == 11U,
          "execution waves must cover the batch contiguously");
    check(neural_execution_plan_create(4U,
                                       11U,
                                       &serial,
                                       &plan,
                                       &error) &&
              plan.worker_count == 1U && plan.wave_count == 7U,
          "serial wave plan must retain every logical sample");
    check(neural_execution_plan_create(4U,
                                       11U,
                                       &oversized,
                                       &plan,
                                       &error) &&
              plan.worker_count == 7U && plan.wave_count == 1U,
          "worker count must be capped by the batch sample count");
    check(neural_execution_plan_create(SIZE_MAX - 5U,
                                       SIZE_MAX,
                                       &parallel,
                                       &plan,
                                       &error) &&
              neural_execution_plan_wave_range(&plan,
                                               1U,
                                               &begin,
                                               &end,
                                               &error) &&
              begin == SIZE_MAX - 2U && end == SIZE_MAX,
          "wave ranges near SIZE_MAX must not overflow");
    check(!neural_execution_plan_create(4U,
                                        4U,
                                        &parallel,
                                        &plan,
                                        &error) &&
              !neural_execution_plan_create(5U,
                                             4U,
                                             &parallel,
                                             &plan,
                                             &error) &&
              !neural_execution_plan_create(4U,
                                             11U,
                                             &parallel,
                                             NULL,
                                             &error) &&
              !neural_execution_plan_create(4U,
                                             11U,
                                             &invalid_config,
                                             &plan,
                                             &error) &&
              !neural_execution_plan_create(4U,
                                             11U,
                                             NULL,
                                             &plan,
                                             &error),
          "invalid execution wave plans must be rejected");
    invalid.sample_begin = 4U;
    invalid.sample_end = 11U;
    invalid.worker_count = 3U;
    invalid.wave_count = 2U;
    check(!neural_execution_plan_wave_range(&invalid,
                                            0U,
                                            &begin,
                                            &end,
                                            &error) &&
              !neural_execution_plan_wave_range(&plan,
                                                2U,
                                                &begin,
                                                &end,
                                                &error) &&
              !neural_execution_plan_wave_range(&plan,
                                                0U,
                                                NULL,
                                                &end,
                                                &error),
          "corrupt wave plans and invalid range requests must be rejected");
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
    pthread_t threads[TEST_THREAD_COUNT];
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
        started[index] = pthread_create(&threads[index],
                                        NULL,
                                        worker_run,
                                        &arguments[index]) == 0;
        check(started[index], "worker thread must start");
        if (!started[index]) {
            workers_ready = 0;
        }
    }
    for (index = 0U; index < TEST_THREAD_COUNT; index++) {
        if (started[index]) {
            int joined = pthread_join(threads[index], NULL) == 0;

            check(joined && arguments[index].success,
                  "worker thread must finish without shared-state races");
            if (!joined || !arguments[index].success) {
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

static void test_persistent_executor_equivalence(void)
{
    NeuralLayerSpec layers[] = {
        {3U, {NEURAL_ACTIVATION_TANH, 0U, NULL}},
        {1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    const neural_real first_weights[] = {
        0.2, -0.3,
        0.4, 0.1,
        -0.5, 0.6
    };
    const neural_real first_biases[] = {0.1, -0.2, 0.3};
    const neural_real second_weights[] = {0.7, -0.4, 0.2};
    const neural_real second_biases[] = {0.05};
    neural_real inputs[] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        -1.0, 0.5,
        0.25, -0.75,
        2.0, -1.0
    };
    neural_real outputs[] = {0.0, 1.0, 1.0, 0.0, -0.5, 0.75, 1.5};
    NeuralDataset dataset = {7U, 2U, 1U, inputs, outputs};
    NeuralDataset invalid_dataset = dataset;
    NeuralExecutionConfig serial_config = {1U};
    NeuralExecutionConfig parallel_config = {3U};
    NeuralExecutionConfig oversized_config = {20U};
    NeuralExecutionConfig invalid_config = {0U};
    NeuralModel *model = NULL;
    NeuralParallelExecutor *serial = NULL;
    NeuralParallelExecutor *parallel = NULL;
    NeuralParallelExecutor *oversized = NULL;
    NeuralParallelExecutor *invalid = NULL;
    NeuralSampleOrder *order = NULL;
    NeuralSampleOrder *wrong_order = NULL;
    const NeuralGradient *serial_gradient = NULL;
    const NeuralGradient *parallel_gradient = NULL;
    const NeuralGradient *oversized_gradient = NULL;
    NeuralError error;
    size_t count;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(31), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  first_weights,
                                                  6U,
                                                  first_biases,
                                                  3U,
                                                  &error) &&
               neural_model_set_layer_parameters(model,
                                                  1U,
                                                  second_weights,
                                                  3U,
                                                  second_biases,
                                                  1U,
                                                  &error);
    check(prepared, "persistent executor model must be prepared");
    if (!prepared) {
        neural_model_free(model);
        return;
    }
    invalid_dataset.output_count = 2U;
    check(!neural_parallel_executor_create(model,
                                           &invalid_dataset,
                                           NEURAL_LOSS_MSE,
                                           &serial_config,
                                           &invalid,
                                           &error) &&
              invalid == NULL &&
              !neural_parallel_executor_create(model,
                                               &dataset,
                                               NEURAL_LOSS_MSE,
                                               &invalid_config,
                                               &invalid,
                                               &error),
          "executor creation must reject invalid dataset and thread settings");
    prepared = neural_parallel_executor_create(model,
                                               &dataset,
                                               NEURAL_LOSS_MSE,
                                               &serial_config,
                                               &serial,
                                               &error) &&
               neural_parallel_executor_create(model,
                                               &dataset,
                                               NEURAL_LOSS_MSE,
                                               &parallel_config,
                                               &parallel,
                                               &error) &&
               neural_parallel_executor_create(model,
                                               &dataset,
                                               NEURAL_LOSS_MSE,
                                               &oversized_config,
                                               &oversized,
                                               &error) &&
               neural_sample_order_create(7U, &order, &error) &&
               neural_sample_order_prepare(order,
                                           UINT64_C(31),
                                           UINT64_C(4),
                                           1,
                                           &error) &&
               neural_sample_order_create(6U, &wrong_order, &error);
    check(prepared, "persistent executors must be created");
    if (prepared) {
        check(neural_parallel_executor_worker_count(serial) == 1U &&
                  neural_parallel_executor_worker_count(parallel) == 3U &&
                  neural_parallel_executor_worker_count(oversized) == 7U,
              "executor worker count must be capped by the dataset");
        check(neural_parallel_executor_batch_gradient(serial,
                                                      1U,
                                                      7U,
                                                      &serial_gradient,
                                                      &error) &&
                  neural_parallel_executor_batch_gradient(parallel,
                                                          1U,
                                                          7U,
                                                          &parallel_gradient,
                                                          &error) &&
                  gradients_equal(serial_gradient, parallel_gradient),
              "serial and multi-wave gradients must be bit-identical");
        check(neural_parallel_executor_batch_gradient(serial,
                                                      0U,
                                                      7U,
                                                      &serial_gradient,
                                                      &error) &&
                  neural_parallel_executor_batch_gradient(parallel,
                                                          0U,
                                                          7U,
                                                          &parallel_gradient,
                                                          &error) &&
                  neural_parallel_executor_batch_gradient(oversized,
                                                          0U,
                                                          7U,
                                                          &oversized_gradient,
                                                          &error) &&
                  gradients_equal(serial_gradient, parallel_gradient) &&
                  gradients_equal(serial_gradient, oversized_gradient),
              "persistent pool reuse must be independent of worker count");
        check(neural_parallel_executor_ordered_batch_gradient(
                  serial, order, 0U, 7U, &serial_gradient, &error) &&
                  neural_parallel_executor_ordered_batch_gradient(
                      parallel,
                      order,
                      0U,
                      7U,
                      &parallel_gradient,
                      &error) &&
                  gradients_equal(serial_gradient, parallel_gradient),
              "ordered executor gradients must be bit-identical across workers");
        check(!neural_parallel_executor_ordered_batch_gradient(
                  serial,
                  wrong_order,
                  0U,
                  6U,
                  &serial_gradient,
                  &error) &&
                  serial_gradient == NULL,
              "executor must reject an order for a different dataset");
        check(!neural_parallel_executor_batch_gradient(serial,
                                                       7U,
                                                       7U,
                                                       &serial_gradient,
                                                       &error) &&
                  serial_gradient == NULL &&
                  !neural_parallel_executor_batch_gradient(serial,
                                                           0U,
                                                           8U,
                                                           &serial_gradient,
                                                           &error),
              "executor must reject invalid batch ranges");
        check(memcmp(neural_model_layer_weights(model, 0U, &count),
                     first_weights,
                     sizeof(first_weights)) == 0 &&
                  count == 6U,
              "parallel gradient execution must not update the model");
    }
    neural_sample_order_free(wrong_order);
    neural_sample_order_free(order);
    neural_parallel_executor_free(invalid);
    neural_parallel_executor_free(oversized);
    neural_parallel_executor_free(parallel);
    neural_parallel_executor_free(serial);
    neural_model_free(model);
}

static void test_executor_error_order_and_recovery(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    neural_real inputs[] = {0.0, 2.0, 3.0, 0.0};
    neural_real outputs[] = {0.0, 0.0, 0.0, 0.0};
    NeuralDataset dataset = {4U, 1U, 1U, inputs, outputs};
    NeuralExecutionConfig config = {3U};
    const neural_real overflowing_weights[] = {DBL_MAX};
    const neural_real valid_weights[] = {1.0};
    const neural_real biases[] = {0.0};
    NeuralModel *model = NULL;
    NeuralParallelExecutor *executor = NULL;
    const NeuralGradient *gradient = NULL;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(32), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                  0U,
                                                  overflowing_weights,
                                                  1U,
                                                  biases,
                                                  1U,
                                                  &error) &&
               neural_parallel_executor_create(model,
                                               &dataset,
                                               NEURAL_LOSS_MSE,
                                               &config,
                                               &executor,
                                               &error);
    check(prepared, "executor error fixture must be prepared");
    if (prepared) {
        check(!neural_parallel_executor_batch_gradient(executor,
                                                       0U,
                                                       4U,
                                                       &gradient,
                                                       &error) &&
                  gradient == NULL &&
                  strstr(error.message, "sample 1 failed") != NULL,
              "executor must report the lowest failing sample index");
        check(neural_model_set_layer_parameters(model,
                                                0U,
                                                valid_weights,
                                                1U,
                                                biases,
                                                1U,
                                                &error) &&
                  neural_parallel_executor_batch_gradient(executor,
                                                          0U,
                                                          4U,
                                                          &gradient,
                                                          &error) &&
                  gradient != NULL,
              "persistent executor must recover on the next batch request");
    }
    neural_parallel_executor_free(executor);
    neural_model_free(model);
}

int main(void)
{
    check(NEURAL_DEFAULT_THREAD_COUNT >= 1U,
          "default thread count must be positive");
    test_execution_plan();
    test_parallel_forward_and_reduction();
    test_persistent_executor_equivalence();
    test_executor_error_order_and_recovery();

    if (failures != 0) {
        fprintf(stderr, "%d parallel test(s) failed\n", failures);
        return 1;
    }
    puts("All parallel tests passed");
    return 0;
}
