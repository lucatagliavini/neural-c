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
        !isfinite(report->objective) || report->objective < report->loss ||
        !isfinite(report->max_gradient_norm) ||
        report->max_gradient_norm < 0.0 ||
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
                         NeuralLoss loss_kind,
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
            !neural_loss_evaluate_with_logits(
                loss_kind,
                neural_model_output_activation(model),
                neural_workspace_layer_pre_activations(
                    workspace,
                    neural_model_layer_count(model) - 1U,
                    NULL),
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
        10000U, 0.5, UINT64_C(42), NEURAL_LOSS_MSE, 100U, 0U, 0.0,
        0U, 0, 0.0, 0.0, 0.0, 0
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
        check(neural_model_train(serial_model,
                                            &dataset,
                                            &training,
                                            &serial,
                                            observe_epoch,
                                            &observer,
                                            &serial_result,
                                            &error),
              "serial XOR training must complete");
        check(neural_model_train(parallel_model,
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
                  serial_result.final_loss == parallel_result.final_loss &&
                  serial_result.final_objective == serial_result.final_loss &&
                  parallel_result.final_objective == parallel_result.final_loss,
              "training must be bit-identical across worker counts");
        check(evaluate_loss(serial_model,
                            &dataset,
                            training.loss,
                            &verified_loss,
                            &error) &&
                  fabs(verified_loss - serial_result.final_loss) < 1e-15 &&
                  serial_result.final_loss < 0.001,
              "reported final loss must describe the converged model");
    }
    neural_model_free(parallel_model);
    neural_model_free(serial_model);
}

static void test_binary_cross_entropy_training(void)
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
        5000U, 0.5, UINT64_C(42),
        NEURAL_LOSS_BINARY_CROSS_ENTROPY, 0U, 0U, 0.0, 0U, 0, 0.0,
        0.0, 0.0, 0
    };
    NeuralExecutionConfig serial = {1U};
    NeuralExecutionConfig parallel = {4U};
    NeuralModel *serial_model = NULL;
    NeuralModel *parallel_model = NULL;
    NeuralTrainingResult serial_result;
    NeuralTrainingResult parallel_result;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec, training.seed,
                                   &serial_model, &error) &&
               neural_model_create(&spec, training.seed,
                                   &parallel_model, &error);
    check(prepared, "binary cross-entropy training models must be prepared");
    if (prepared) {
        check(neural_model_train(serial_model,
                                            &dataset,
                                            &training,
                                            &serial,
                                            NULL,
                                            NULL,
                                            &serial_result,
                                            &error) &&
                  neural_model_train(parallel_model,
                                                &dataset,
                                                &training,
                                                &parallel,
                                                NULL,
                                                NULL,
                                                &parallel_result,
                                                &error),
              "binary cross-entropy XOR training must complete");
        check(models_equal(serial_model, parallel_model) &&
                  serial_result.final_loss == parallel_result.final_loss,
              "binary cross-entropy training must be thread-independent");
        check(serial_result.final_loss < 0.01,
              "binary cross-entropy training must converge on XOR");
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
        3U, 0.1, UINT64_C(7), NEURAL_LOSS_MSE, 1U, 0U, 0.0,
        0U, 0, 0.0, 0.0, 0.0, 0
    };
    NeuralExecutionConfig execution = {1U};
    NeuralTrainingResult result = {99U, 99U, 99.0, 99.0, 99.0, 99U};
    ObserverState observer = {1U, 0U, 2U, 0.0};
    NeuralModel *model = NULL;
    NeuralError error;

    check(neural_model_create(&spec, training.seed, &model, &error),
          "observer failure model must be prepared");
    if (model != NULL) {
        check(!neural_model_train(model,
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
        4U, 0.5, UINT64_C(42), NEURAL_LOSS_MSE, 2U, 0U, 0.0,
        0U, 0, 0.0, 0.0, 0.0, 0
    };
    NeuralExecutionConfig serial = {1U};
    NeuralExecutionConfig parallel = {4U};
    NeuralModel *continuous = NULL;
    NeuralModel *resumed = NULL;
    NeuralTrainingResult continuous_result;
    NeuralTrainingResult partial_result;
    NeuralTrainingResult resumed_result;
    NeuralTrainingResult no_work_result;
    NeuralTrainingResult invalid_result = {
        99U, 99U, 99.0, 99.0, 99.0, 99U
    };
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
        check(neural_model_train(continuous,
                                            &dataset,
                                            &training,
                                            &serial,
                                            NULL,
                                            NULL,
                                            &continuous_result,
                                            &error),
              "continuous epoch-range reference must train");
        check(neural_model_train_range(resumed,
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
        check(neural_model_train_range(resumed,
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
        check(neural_model_train_range(
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
        check(!neural_model_train_range(resumed,
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

static void test_mini_batch_determinism_and_continuation(void)
{
    NeuralLayerSpec layers[] = {
        {3U, {NEURAL_ACTIVATION_TANH, 0U, NULL}},
        {1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}}
    };
    NeuralModelSpec spec = {1U, 2U, layers};
    neural_real inputs[] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    neural_real outputs[] = {-1.5, -0.5, 0.5, 1.5, 2.5};
    NeuralDataset dataset = {5U, 1U, 1U, inputs, outputs};
    NeuralTrainingConfig mini_batch = {
        5U, 0.05, UINT64_C(91), NEURAL_LOSS_MSE, 0U, 0U, 0.0,
        2U, 1, 0.0, 0.0, 0.0, 0
    };
    NeuralTrainingConfig full_batch = mini_batch;
    NeuralTrainingConfig oversized_batch = mini_batch;
    NeuralTrainingConfig source_order_config = mini_batch;
    NeuralTrainingConfig clipped_config = mini_batch;
    NeuralTrainingConfig regularized_config = mini_batch;
    NeuralExecutionConfig serial = {1U};
    NeuralExecutionConfig parallel = {4U};
    NeuralModel *continuous = NULL;
    NeuralModel *threaded = NULL;
    NeuralModel *resumed = NULL;
    NeuralModel *full = NULL;
    NeuralModel *oversized = NULL;
    NeuralModel *source_order = NULL;
    NeuralModel *clipped_continuous = NULL;
    NeuralModel *clipped_threaded = NULL;
    NeuralModel *clipped_resumed = NULL;
    NeuralModel *regularized_continuous = NULL;
    NeuralModel *regularized_threaded = NULL;
    NeuralModel *regularized_resumed = NULL;
    NeuralTrainingResult continuous_result;
    NeuralTrainingResult threaded_result;
    NeuralTrainingResult partial_result;
    NeuralTrainingResult resumed_result;
    NeuralTrainingResult full_result;
    NeuralTrainingResult oversized_result;
    NeuralTrainingResult source_order_result;
    NeuralTrainingResult clipped_continuous_result;
    NeuralTrainingResult clipped_threaded_result;
    NeuralTrainingResult clipped_partial_result;
    NeuralTrainingResult clipped_resumed_result;
    NeuralTrainingResult regularized_continuous_result;
    NeuralTrainingResult regularized_threaded_result;
    NeuralTrainingResult regularized_partial_result;
    NeuralTrainingResult regularized_resumed_result;
    NeuralError error;
    int prepared;

    full_batch.batch_size = 0U;
    oversized_batch.batch_size = 99U;
    source_order_config.shuffle = 0;
    clipped_config.gradient_clip_norm = 1e-6;
    regularized_config.gradient_clip_norm = 0.1;
    regularized_config.l1_regularization = 0.01;
    regularized_config.l2_regularization = 0.02;
    regularized_config.regularize_biases = 1;
    prepared = neural_model_create(&spec, mini_batch.seed,
                                   &continuous, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &threaded, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &resumed, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &full, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &oversized, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &source_order, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &clipped_continuous, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &clipped_threaded, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &clipped_resumed, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &regularized_continuous, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &regularized_threaded, &error) &&
               neural_model_create(&spec, mini_batch.seed,
                                   &regularized_resumed, &error);
    check(prepared, "mini-batch training models must be prepared");
    if (prepared) {
        check(neural_model_train(continuous,
                                 &dataset,
                                 &mini_batch,
                                 &serial,
                                 NULL,
                                 NULL,
                                 &continuous_result,
                                 &error) &&
                  neural_model_train(threaded,
                                     &dataset,
                                     &mini_batch,
                                     &parallel,
                                     NULL,
                                     NULL,
                                     &threaded_result,
                                     &error),
              "incomplete mini-batches must train across worker counts");
        check(models_equal(continuous, threaded) &&
                  continuous_result.final_loss == threaded_result.final_loss,
              "mini-batch training must be bit-identical across workers");
        check(neural_model_train_range(resumed,
                                       &dataset,
                                       &mini_batch,
                                       &serial,
                                       0U,
                                       2U,
                                       NULL,
                                       NULL,
                                       &partial_result,
                                       &error) &&
                  neural_model_train_range(resumed,
                                           &dataset,
                                           &mini_batch,
                                           &parallel,
                                           2U,
                                           mini_batch.epochs,
                                           NULL,
                                           NULL,
                                           &resumed_result,
                                           &error) &&
                  models_equal(continuous, resumed) &&
                  resumed_result.final_loss == continuous_result.final_loss,
              "mini-batch epoch ranges must continue exactly");
        check(neural_model_train(full,
                                 &dataset,
                                 &full_batch,
                                 &serial,
                                 NULL,
                                 NULL,
                                 &full_result,
                                 &error) &&
                  !models_equal(continuous, full),
              "positive batch size must select mini-batch updates");
        check(neural_model_train(oversized,
                                 &dataset,
                                 &oversized_batch,
                                 &parallel,
                                 NULL,
                                 NULL,
                                 &oversized_result,
                                 &error) &&
                  models_equal(full, oversized) &&
                  full_result.final_loss == oversized_result.final_loss,
              "oversized training batches must resolve to the full dataset");
        check(neural_model_train(source_order,
                                 &dataset,
                                 &source_order_config,
                                 &parallel,
                                 NULL,
                                 NULL,
                                 &source_order_result,
                                 &error) &&
                  !models_equal(continuous, source_order),
              "enabled epoch shuffle must change mini-batch training order");
        check(neural_model_train(clipped_continuous,
                                 &dataset,
                                 &clipped_config,
                                 &serial,
                                 NULL,
                                 NULL,
                                 &clipped_continuous_result,
                                 &error) &&
                  neural_model_train(clipped_threaded,
                                     &dataset,
                                     &clipped_config,
                                     &parallel,
                                     NULL,
                                     NULL,
                                     &clipped_threaded_result,
                                     &error) &&
                  models_equal(clipped_continuous, clipped_threaded) &&
                  clipped_continuous_result.final_loss ==
                      clipped_threaded_result.final_loss &&
                  clipped_continuous_result.final_max_gradient_norm ==
                      clipped_threaded_result.final_max_gradient_norm &&
                  clipped_continuous_result.clipped_batch_count == 3U &&
                  clipped_threaded_result.clipped_batch_count == 3U &&
                  clipped_continuous_result.final_max_gradient_norm >
                      clipped_config.gradient_clip_norm &&
                  !models_equal(continuous, clipped_continuous),
              "gradient clipping must report and update identically across workers");
        check(neural_model_train_range(clipped_resumed,
                                       &dataset,
                                       &clipped_config,
                                       &parallel,
                                       0U,
                                       2U,
                                       NULL,
                                       NULL,
                                       &clipped_partial_result,
                                       &error) &&
                  neural_model_train_range(clipped_resumed,
                                           &dataset,
                                           &clipped_config,
                                           &serial,
                                           2U,
                                           clipped_config.epochs,
                                           NULL,
                                           NULL,
                                           &clipped_resumed_result,
                                           &error) &&
                  models_equal(clipped_continuous, clipped_resumed) &&
                  clipped_continuous_result.final_loss ==
                      clipped_resumed_result.final_loss &&
                  clipped_continuous_result.final_max_gradient_norm ==
                      clipped_resumed_result.final_max_gradient_norm &&
                  clipped_continuous_result.clipped_batch_count ==
                      clipped_resumed_result.clipped_batch_count,
              "clipped absolute epoch ranges must continue exactly");
        check(neural_model_train(regularized_continuous,
                                 &dataset,
                                 &regularized_config,
                                 &serial,
                                 NULL,
                                 NULL,
                                 &regularized_continuous_result,
                                 &error) &&
                  neural_model_train(regularized_threaded,
                                     &dataset,
                                     &regularized_config,
                                     &parallel,
                                     NULL,
                                     NULL,
                                     &regularized_threaded_result,
                                     &error) &&
                  models_equal(regularized_continuous,
                               regularized_threaded) &&
                  regularized_continuous_result.final_loss ==
                      regularized_threaded_result.final_loss &&
                  regularized_continuous_result.final_objective ==
                      regularized_threaded_result.final_objective &&
                  regularized_continuous_result.final_objective >
                      regularized_continuous_result.final_loss &&
                  regularized_continuous_result.clipped_batch_count > 0U &&
                  !models_equal(clipped_continuous,
                                regularized_continuous),
              "regularization before clipping must be thread-independent");
        check(neural_model_train_range(regularized_resumed,
                                       &dataset,
                                       &regularized_config,
                                       &parallel,
                                       0U,
                                       2U,
                                       NULL,
                                       NULL,
                                       &regularized_partial_result,
                                       &error) &&
                  neural_model_train_range(regularized_resumed,
                                           &dataset,
                                           &regularized_config,
                                           &serial,
                                           2U,
                                           regularized_config.epochs,
                                           NULL,
                                           NULL,
                                           &regularized_resumed_result,
                                           &error) &&
                  models_equal(regularized_continuous,
                               regularized_resumed) &&
                  regularized_continuous_result.final_loss ==
                      regularized_resumed_result.final_loss &&
                  regularized_continuous_result.final_objective ==
                      regularized_resumed_result.final_objective &&
                  regularized_continuous_result.final_max_gradient_norm ==
                      regularized_resumed_result.final_max_gradient_norm &&
                  regularized_continuous_result.clipped_batch_count ==
                      regularized_resumed_result.clipped_batch_count,
              "regularized absolute epoch ranges must continue exactly");
    }
    neural_model_free(regularized_resumed);
    neural_model_free(regularized_threaded);
    neural_model_free(regularized_continuous);
    neural_model_free(clipped_resumed);
    neural_model_free(clipped_threaded);
    neural_model_free(clipped_continuous);
    neural_model_free(source_order);
    neural_model_free(oversized);
    neural_model_free(full);
    neural_model_free(resumed);
    neural_model_free(threaded);
    neural_model_free(continuous);
}

static void test_regularization_clipping_order(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    neural_real inputs[] = {1.0};
    neural_real outputs[] = {0.0};
    NeuralDataset dataset = {1U, 1U, 1U, inputs, outputs};
    NeuralTrainingConfig training = {
        1U, 0.1, UINT64_C(92), NEURAL_LOSS_MSE, 0U, 0U, 0.0,
        0U, 0, 1.0, 0.5, 0.25, 0
    };
    NeuralExecutionConfig execution = {1U};
    neural_real initial_weights[] = {2.0};
    neural_real initial_biases[] = {3.0};
    NeuralModel *model = NULL;
    NeuralTrainingResult result;
    NeuralError error;
    neural_real norm = sqrt(221.0);
    const neural_real *weights;
    const neural_real *biases;
    int prepared;

    prepared = neural_model_create(&spec, training.seed, &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                 0U,
                                                 initial_weights,
                                                 1U,
                                                 initial_biases,
                                                 1U,
                                                 &error);
    check(prepared, "regularization clipping-order model must be prepared");
    if (prepared) {
        check(neural_model_train(model,
                                 &dataset,
                                 &training,
                                 &execution,
                                 NULL,
                                 NULL,
                                 &result,
                                 &error),
              "regularized clipped update must complete");
        weights = neural_model_layer_weights(model, 0U, NULL);
        biases = neural_model_layer_biases(model, 0U, NULL);
        check(weights != NULL && biases != NULL &&
                  fabs(weights[0] - (2.0 - 0.1 * 11.0 / norm)) < 1e-15 &&
                  fabs(biases[0] - (3.0 - 0.1 * 10.0 / norm)) < 1e-15 &&
                  fabs(result.final_max_gradient_norm - norm) < 4e-15 &&
                  result.clipped_batch_count == 1U &&
                  result.final_objective > result.final_loss,
              "regularization must precede norm measurement and clipping");
    }
    neural_model_free(model);
}

int main(void)
{
    test_xor_training_and_determinism();
    test_binary_cross_entropy_training();
    test_observer_failure();
    test_absolute_epoch_ranges();
    test_mini_batch_determinism_and_continuation();
    test_regularization_clipping_order();

    if (failures != 0) {
        fprintf(stderr, "%d training-engine test(s) failed\n", failures);
        return 1;
    }
    puts("All training-engine tests passed");
    return 0;
}
