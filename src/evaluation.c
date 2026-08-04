#include "neural/evaluation.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/loss.h"
#include "compensated_sum.h"

void neural_evaluation_result_free(NeuralEvaluationResult *result)
{
    if (result != NULL) {
        free(result->confusion);
        free(result->precision);
        free(result->recall);
        free(result->f1);
        memset(result, 0, sizeof(*result));
    }
}

int neural_model_evaluate_dataset_loss(const NeuralModel *model,
                                       NeuralWorkspace *workspace,
                                       neural_real *predicted,
                                       const NeuralDataset *dataset,
                                       NeuralLoss loss,
                                       neural_real *value,
                                       NeuralError *error)
{
    neural_real sum = 0.0;
    neural_real compensation = 0.0;
    size_t sample_index;

    if (model == NULL || workspace == NULL || predicted == NULL ||
        dataset == NULL || dataset->sample_count == 0U || value == NULL) {
        neural_error_set(error, "dataset loss arguments are required");
        return 0;
    }
    for (sample_index = 0U;
         sample_index < dataset->sample_count;
         sample_index++) {
        const neural_real *inputs =
            dataset->inputs + sample_index * dataset->input_count;
        const neural_real *expected =
            dataset->outputs + sample_index * dataset->output_count;
        neural_real sample_loss;
        neural_real new_sum;
        neural_real new_compensation;

        if (!neural_model_forward(model,
                                  workspace,
                                  inputs,
                                  dataset->input_count,
                                  predicted,
                                  dataset->output_count,
                                  error) ||
            !neural_loss_evaluate_with_logits(
                loss,
                neural_model_output_activation(model),
                neural_workspace_layer_pre_activations(
                    workspace,
                    neural_model_layer_count(model) - 1U,
                    NULL),
                predicted,
                expected,
                dataset->output_count,
                &sample_loss,
                error) ||
            !neural_compensated_add(sum,
                                    compensation,
                                    sample_loss,
                                    &new_sum,
                                    &new_compensation)) {
            if (error != NULL && error->message[0] == '\0') {
                neural_error_set(error, "dataset loss is not finite");
            }
            return 0;
        }
        sum = new_sum;
        compensation = new_compensation;
    }
    *value = (sum + compensation) / (neural_real)dataset->sample_count;
    if (!isfinite(*value)) {
        neural_error_set(error, "dataset mean loss is not finite");
        return 0;
    }
    return 1;
}

static int targets_are_binary(const neural_real *expected,
                              size_t sample_count)
{
    size_t sample_index;

    for (sample_index = 0U; sample_index < sample_count; sample_index++) {
        if (expected[sample_index] != 0.0 &&
            expected[sample_index] != 1.0) {
            return 0;
        }
    }
    return 1;
}

static int targets_are_one_hot(const neural_real *expected,
                               size_t sample_count,
                               size_t output_count)
{
    size_t sample_index;

    for (sample_index = 0U; sample_index < sample_count; sample_index++) {
        size_t output_index;
        size_t selected = 0U;

        for (output_index = 0U;
             output_index < output_count;
             output_index++) {
            neural_real value =
                expected[sample_index * output_count + output_index];

            if (value == 1.0) {
                selected++;
            } else if (value != 0.0) {
                return 0;
            }
        }
        if (selected != 1U) {
            return 0;
        }
    }
    return 1;
}

static size_t maximum_index(const neural_real *values, size_t count)
{
    size_t selected = 0U;
    size_t index;

    for (index = 1U; index < count; index++) {
        if (values[index] > values[selected]) {
            selected = index;
        }
    }
    return selected;
}

static int allocate_classification(NeuralEvaluationResult *result,
                                   size_t class_count,
                                   NeuralError *error)
{
    size_t confusion_count;

    if (class_count == 0U || class_count > SIZE_MAX / class_count) {
        neural_error_set(error, "evaluation class dimensions are too large");
        return 0;
    }
    confusion_count = class_count * class_count;
    if (confusion_count > SIZE_MAX / sizeof(*result->confusion) ||
        class_count > SIZE_MAX / sizeof(*result->precision)) {
        neural_error_set(error, "evaluation metric dimensions are too large");
        return 0;
    }
    result->confusion = calloc(confusion_count, sizeof(*result->confusion));
    result->precision = calloc(class_count, sizeof(*result->precision));
    result->recall = calloc(class_count, sizeof(*result->recall));
    result->f1 = calloc(class_count, sizeof(*result->f1));
    if (result->confusion == NULL || result->precision == NULL ||
        result->recall == NULL || result->f1 == NULL) {
        neural_error_set(error, "unable to allocate evaluation metrics");
        return 0;
    }
    result->class_count = class_count;
    result->is_classification = 1;
    return 1;
}

static void finalize_classification(NeuralEvaluationResult *result)
{
    size_t class_index;

    result->accuracy = (neural_real)result->correct_count /
                       (neural_real)result->sample_count;
    for (class_index = 0U;
         class_index < result->class_count;
         class_index++) {
        size_t predicted_total = 0U;
        size_t actual_total = 0U;
        size_t other;
        size_t true_positive =
            result->confusion[class_index * result->class_count + class_index];

        for (other = 0U; other < result->class_count; other++) {
            actual_total +=
                result->confusion[class_index * result->class_count + other];
            predicted_total +=
                result->confusion[other * result->class_count + class_index];
        }
        result->precision[class_index] = predicted_total == 0U
                                             ? 0.0
                                             : (neural_real)true_positive /
                                                   (neural_real)predicted_total;
        result->recall[class_index] = actual_total == 0U
                                          ? 0.0
                                          : (neural_real)true_positive /
                                                (neural_real)actual_total;
        if (result->precision[class_index] != 0.0 ||
            result->recall[class_index] != 0.0) {
            result->f1[class_index] =
                2.0 * result->precision[class_index] *
                result->recall[class_index] /
                (result->precision[class_index] + result->recall[class_index]);
        }
    }
}

int neural_evaluation_compute(NeuralLoss loss,
                              NeuralActivationKind output_activation,
                              const neural_real *predicted,
                              const neural_real *expected,
                              size_t sample_count,
                              size_t output_count,
                              NeuralEvaluationResult *result,
                              NeuralError *error)
{
    NeuralEvaluationResult computed = {0};
    neural_real loss_sum = 0.0;
    neural_real compensation = 0.0;
    size_t sample_index;
    int binary;
    int multiclass;

    neural_error_clear(error);
    if (predicted == NULL || expected == NULL || sample_count == 0U ||
        output_count == 0U || result == NULL ||
        sample_count > SIZE_MAX / output_count) {
        neural_error_set(error, "evaluation values and dimensions are required");
        return 0;
    }
    *result = computed;
    if (!neural_loss_validate_output(loss,
                                     output_activation,
                                     output_count,
                                     error) ||
        !neural_loss_validate_targets(loss,
                                      expected,
                                      sample_count,
                                      output_count,
                                      error)) {
        return 0;
    }
    computed.sample_count = sample_count;
    computed.output_count = output_count;
    binary = output_count == 1U &&
             output_activation == NEURAL_ACTIVATION_SIGMOID &&
             targets_are_binary(expected, sample_count);
    multiclass = output_count >= 2U &&
                 output_activation == NEURAL_ACTIVATION_SOFTMAX &&
                 targets_are_one_hot(expected, sample_count, output_count);
    if ((binary && !allocate_classification(&computed, 2U, error)) ||
        (multiclass &&
         !allocate_classification(&computed, output_count, error))) {
        goto failure;
    }
    for (sample_index = 0U; sample_index < sample_count; sample_index++) {
        const neural_real *sample_prediction =
            predicted + sample_index * output_count;
        const neural_real *sample_expected =
            expected + sample_index * output_count;
        neural_real sample_loss;
        neural_real new_sum;
        neural_real new_compensation;

        if (!neural_loss_evaluate(loss,
                                  sample_prediction,
                                  sample_expected,
                                  output_count,
                                  &sample_loss,
                                  error) ||
            !neural_compensated_add(loss_sum,
                                    compensation,
                                    sample_loss,
                                    &new_sum,
                                    &new_compensation)) {
            if (error != NULL && error->message[0] == '\0') {
                neural_error_set(error, "evaluation loss is not finite");
            }
            goto failure;
        }
        loss_sum = new_sum;
        compensation = new_compensation;
        if (computed.is_classification) {
            size_t actual;
            size_t selected;

            if (binary) {
                actual = sample_expected[0] == 1.0 ? 1U : 0U;
                selected = sample_prediction[0] >= 0.5 ? 1U : 0U;
            } else {
                actual = maximum_index(sample_expected, output_count);
                selected = maximum_index(sample_prediction, output_count);
            }
            computed.confusion[actual * computed.class_count + selected]++;
            if (actual == selected) {
                computed.correct_count++;
            }
        }
    }
    computed.loss = (loss_sum + compensation) / (neural_real)sample_count;
    if (!isfinite(computed.loss)) {
        neural_error_set(error, "evaluation mean loss is not finite");
        goto failure;
    }
    if (computed.is_classification) {
        finalize_classification(&computed);
    }
    *result = computed;
    return 1;

failure:
    neural_evaluation_result_free(&computed);
    return 0;
}
