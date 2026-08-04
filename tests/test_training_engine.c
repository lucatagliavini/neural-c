#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "neural/activation.h"
#include "neural/loss.h"
#include "neural/model.h"
#include "neural/training.h"

typedef struct {
    size_t next_epoch;
    size_t call_count;
    size_t fail_epoch;
    neural_real last_loss;
} ObserverState;

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int observe_epoch(const NeuralEpochReport *report,
                         void *opaque,
                         NeuralError *error)
{
    ObserverState *state = opaque;

    if (report == NULL || state == NULL || !isfinite(report->loss) ||
        report->completed_epochs != state->next_epoch) {
        neural_error_set(error, "observer received an invalid epoch report");
        return 0;
    }
    state->call_count++;
    state->next_epoch++;
    state->last_loss = report->loss;
    if (report->completed_epochs == state->fail_epoch) {
        neural_error_set(error,
                         "observer stopped at epoch %zu",
                         report->completed_epochs);
        return 0;
    }
    return 1;
}

static int models_equal(const NeuralModel *left, const NeuralModel *right)
{
    size_t layer_index;

    if (neural_model_layer_count(left) != neural_model_layer_count(right)) {
        return 0;
    }
    for (layer_index = 0U;
         layer_index < neural_model_layer_count(left);
         layer_index++) {
        size_t left_weight_count;
        size_t right_weight_count;
        size_t left_bias_count;
        size_t right_bias_count;
        const neural_real *left_weights = neural_model_layer_weights(
            left, layer_index, &left_weight_count);
        const neural_real *right_weights = neural_model_layer_weights(
            right, layer_index, &right_weight_count);
        const neural_real *left_biases = neural_model_layer_biases(
            left, layer_index, &left_bias_count);
        const neural_real *right_biases = neural_model_layer_biases(
            right, layer_index, &right_bias_count);

        if (left_weights == NULL || right_weights == NULL ||
            left_biases == NULL || right_biases == NULL ||
            left_weight_count != right_weight_count ||
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
    return 1;
}

static int evaluate_loss(const NeuralModel *model,
                         const NeuralDataset *dataset,
                         neural_real *mean,
                         NeuralError *error)
{
    NeuralWorkspace *workspace = NULL;
    neural_real predicted[1];
    neural_real sum = 0.0;
    size_t sample_index;
    int success = 0;

    if (!neural_workspace_create(model, &workspace, error)) {
        return 0;
    }
    for (sample_index = 0U;
         sample_index < dataset->sample_count;
         sample_index++) {
        neural_real loss;

        if (!neural_model_forward(
                model,
                workspace,
                dataset->inputs + sample_index * dataset->input_count,
                dataset->input_count,
                predicted,
                dataset->output_count,
                error) ||
            !neural_loss_evaluate(
                NEURAL_LOSS_MSE,
                predicted,
                dataset->outputs + sample_index * dataset->output_count,
                dataset->output_count,
                &loss,
                error)) {
            goto cleanup;
        }
        sum += loss;
    }
    *mean = sum / (neural_real)dataset->sample_count;
    success = 1;

cleanup:
    neural_workspace_free(workspace);
    return success;
}

static void test_xor_training_and_determinism(void)
{
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}},
        {1U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    neural_real inputs[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    neural_real outputs[] = {0.0, 1.0, 1.0, 0.0};
    NeuralDataset dataset = {4U, 2U, 1U, inputs, outputs};
    NeuralTrainingConfig training = {
        10000U, 0.5, UINT64_C(42), NEURAL_LOSS_MSE, 100U
    };
    NeuralExecutionConfig serial = {1U};
    NeuralExecutionConfig parallel = {4U};
    NeuralModel *serial_model = NULL;
    NeuralModel *parallel_model = NULL;
    NeuralTrainingResult serial_result;
    NeuralTrainingResult parallel_result;
    ObserverState observer = {1U, 0U, 0U, 0.0};
    NeuralError error;
    neural_real verified_loss = 0.0;
    int prepared;

    prepared = neural_model_create(&spec,
                                   training.seed,
                                   &serial_model,
                                   &error) &&
               neural_model_create(&spec,
                                   training.seed,
                                   &parallel_model,
                                   &error);
    check(prepared, "XOR training models must be prepared");
    if (prepared) {
        check(neural_model_train_full_batch(serial_model,
                                            &dataset,
                                            &training,
                                            &serial,
                                            observe_epoch,
                                            &observer,
                                            &serial_result,
                                            &error),
              "serial XOR training must complete");
        check(neural_model_train_full_batch(parallel_model,
                                            &dataset,
                                            &training,
                                            &parallel,
                                            NULL,
                                            NULL,
                                            &parallel_result,
                                            &error),
              "parallel XOR training must complete");
        check(serial_result.completed_epochs == training.epochs &&
                  parallel_result.completed_epochs == training.epochs &&
                  serial_result.worker_count == 1U &&
                  parallel_result.worker_count == 4U &&
                  observer.call_count == training.epochs &&
                  observer.last_loss == serial_result.final_loss,
              "training results and epoch reports must be complete");
        check(models_equal(serial_model, parallel_model) &&
                  serial_result.final_loss == parallel_result.final_loss,
              "training must be bit-identical across worker counts");
        check(evaluate_loss(serial_model,
                            &dataset,
                            &verified_loss,
                            &error) &&
                  fabs(verified_loss - serial_result.final_loss) < 1e-15 &&
                  serial_result.final_loss < 0.001,
              "reported final loss must describe the converged model");
    }
    neural_model_free(parallel_model);
    neural_model_free(serial_model);
}

static void test_observer_failure(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    neural_real inputs[] = {1.0};
    neural_real outputs[] = {0.0};
    NeuralDataset dataset = {1U, 1U, 1U, inputs, outputs};
    NeuralTrainingConfig training = {
        3U, 0.1, UINT64_C(7), NEURAL_LOSS_MSE, 1U
    };
    NeuralExecutionConfig execution = {1U};
    NeuralTrainingResult result = {99U, 99U, 99.0};
    ObserverState observer = {1U, 0U, 2U, 0.0};
    NeuralModel *model = NULL;
    NeuralError error;

    check(neural_model_create(&spec, training.seed, &model, &error),
          "observer failure model must be prepared");
    if (model != NULL) {
        check(!neural_model_train_full_batch(model,
                                             &dataset,
                                             &training,
                                             &execution,
                                             observe_epoch,
                                             &observer,
                                             &result,
                                             &error) &&
                  observer.call_count == 2U &&
                  result.completed_epochs == 0U &&
                  strstr(error.message, "observer stopped at epoch 2") != NULL,
              "observer failure must stop training without a success result");
    }
    neural_model_free(model);
}

static void test_absolute_epoch_ranges(void)
{
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}},
        {1U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    neural_real inputs[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    neural_real outputs[] = {0.0, 1.0, 1.0, 0.0};
    NeuralDataset dataset = {4U, 2U, 1U, inputs, outputs};
    NeuralTrainingConfig training = {
        4U, 0.5, UINT64_C(42), NEURAL_LOSS_MSE, 2U
    };
    NeuralExecutionConfig serial = {1U};
    NeuralExecutionConfig parallel = {4U};
    NeuralModel *continuous = NULL;
    NeuralModel *resumed = NULL;
    NeuralTrainingResult continuous_result;
    NeuralTrainingResult partial_result;
    NeuralTrainingResult resumed_result;
    NeuralTrainingResult no_work_result;
    NeuralTrainingResult invalid_result = {99U, 99U, 99.0};
    ObserverState no_work_observer = {5U, 0U, 0U, 0.0};
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec,
                                   training.seed,
                                   &continuous,
                                   &error) &&
               neural_model_create(&spec,
                                   training.seed,
                                   &resumed,
                                   &error);
    check(prepared, "epoch-range models must be prepared");
    if (prepared) {
        check(neural_model_train_full_batch(continuous,
                                            &dataset,
                                            &training,
                                            &serial,
                                            NULL,
                                            NULL,
                                            &continuous_result,
                                            &error),
              "continuous epoch-range reference must train");
        check(neural_model_train_full_batch_range(resumed,
                                                  &dataset,
                                                  &training,
                                                  &serial,
                                                  0U,
                                                  2U,
                                                  NULL,
                                                  NULL,
                                                  &partial_result,
                                                  &error) &&
                  partial_result.completed_epochs == 2U,
              "the first absolute epoch range must complete");
        check(neural_model_train_full_batch_range(resumed,
                                                  &dataset,
                                                  &training,
                                                  &parallel,
                                                  2U,
                                                  4U,
                                                  NULL,
                                                  NULL,
                                                  &resumed_result,
                                                  &error) &&
                  resumed_result.completed_epochs == 4U &&
                  resumed_result.worker_count == 4U &&
                  models_equal(continuous, resumed) &&
                  resumed_result.final_loss == continuous_result.final_loss,
              "resumed ranges must match continuous training across workers");
        check(neural_model_train_full_batch_range(
                  resumed,
                  &dataset,
                  &training,
                  &parallel,
                  4U,
                  4U,
                  observe_epoch,
                  &no_work_observer,
                  &no_work_result,
                  &error) &&
                  no_work_observer.call_count == 0U &&
                  no_work_result.completed_epochs == 4U &&
                  no_work_result.final_loss == resumed_result.final_loss,
              "an already-complete range must only evaluate final loss");
        check(!neural_model_train_full_batch_range(resumed,
                                                   &dataset,
                                                   &training,
                                                   &serial,
                                                   5U,
                                                   4U,
                                                   NULL,
                                                   NULL,
                                                   &invalid_result,
                                                   &error) &&
                  invalid_result.completed_epochs == 0U &&
                  strstr(error.message, "range is invalid") != NULL,
              "descending epoch ranges must fail without a result");
    }
    neural_model_free(resumed);
    neural_model_free(continuous);
}

int main(void)
{
    test_xor_training_and_determinism();
    test_observer_failure();
    test_absolute_epoch_ranges();

    if (failures != 0) {
        fprintf(stderr, "%d training-engine test(s) failed\n", failures);
        return 1;
    }
    puts("All training-engine tests passed");
    return 0;
}
