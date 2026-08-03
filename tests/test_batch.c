#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "neural/activation.h"
#include "neural/batch.h"
#include "neural/gradient.h"
#include "neural/model.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int fill_gradient(NeuralGradient *gradient, neural_real value)
{
    size_t layer_index;

    for (layer_index = 0U; ; layer_index++) {
        size_t weight_count;
        size_t bias_count;
        neural_real *weights = neural_gradient_layer_weights(
            gradient, layer_index, &weight_count);
        neural_real *biases = neural_gradient_layer_biases(
            gradient, layer_index, &bias_count);
        size_t index;

        if (weights == NULL || biases == NULL) {
            return layer_index > 0U;
        }
        for (index = 0U; index < weight_count; index++) {
            weights[index] = value;
        }
        for (index = 0U; index < bias_count; index++) {
            biases[index] = value;
        }
    }
}

static int gradient_equals(const NeuralGradient *gradient, neural_real expected)
{
    size_t layer_index;

    for (layer_index = 0U; ; layer_index++) {
        size_t weight_count;
        size_t bias_count;
        const neural_real *weights = neural_gradient_layer_weights_const(
            gradient, layer_index, &weight_count);
        const neural_real *biases = neural_gradient_layer_biases_const(
            gradient, layer_index, &bias_count);
        size_t index;

        if (weights == NULL || biases == NULL) {
            return layer_index > 0U;
        }
        for (index = 0U; index < weight_count; index++) {
            if (weights[index] != expected) {
                return 0;
            }
        }
        for (index = 0U; index < bias_count; index++) {
            if (biases[index] != expected) {
                return 0;
            }
        }
    }
}

static int gradient_nearly_equals(const NeuralGradient *gradient,
                                  neural_real expected,
                                  neural_real tolerance)
{
    size_t layer_index;

    for (layer_index = 0U; ; layer_index++) {
        size_t weight_count;
        size_t bias_count;
        const neural_real *weights = neural_gradient_layer_weights_const(
            gradient, layer_index, &weight_count);
        const neural_real *biases = neural_gradient_layer_biases_const(
            gradient, layer_index, &bias_count);
        size_t index;

        if (weights == NULL || biases == NULL) {
            return layer_index > 0U;
        }
        for (index = 0U; index < weight_count; index++) {
            if (fabs(weights[index] - expected) > tolerance) {
                return 0;
            }
        }
        for (index = 0U; index < bias_count; index++) {
            if (fabs(biases[index] - expected) > tolerance) {
                return 0;
            }
        }
    }
}

static void test_batch_plan(void)
{
    NeuralBatchPlan plan;
    NeuralBatchPlan invalid;
    NeuralError error;
    size_t begin = SIZE_MAX;
    size_t end = SIZE_MAX;

    check(neural_batch_plan_create(10U, 4U, &plan, &error) &&
              plan.batch_count == 3U,
          "partial batch plan must contain three batches");
    check(neural_batch_plan_range(&plan, 0U, &begin, &end, &error) &&
              begin == 0U && end == 4U,
          "first batch range must be contiguous");
    check(neural_batch_plan_range(&plan, 1U, &begin, &end, &error) &&
              begin == 4U && end == 8U,
          "middle batch range must be contiguous");
    check(neural_batch_plan_range(&plan, 2U, &begin, &end, &error) &&
              begin == 8U && end == 10U,
          "final batch range must use the remaining samples");
    check(neural_batch_plan_create(5U, 1U, &plan, &error) &&
              plan.batch_count == 5U,
          "unit batch plan must create one batch per sample");
    check(neural_batch_plan_create(5U, 5U, &plan, &error) &&
              plan.batch_count == 1U,
          "full batch plan must create one batch");
    check(!neural_batch_plan_create(0U, 1U, &plan, &error) &&
              !neural_batch_plan_create(5U, 0U, &plan, &error) &&
              !neural_batch_plan_create(5U, 6U, &plan, &error) &&
              !neural_batch_plan_create(5U, 1U, NULL, &error),
          "invalid batch dimensions must be rejected");
    invalid.sample_count = 5U;
    invalid.batch_size = 2U;
    invalid.batch_count = 2U;
    check(!neural_batch_plan_range(&invalid, 0U, &begin, &end, &error) &&
              !neural_batch_plan_range(&plan, 1U, &begin, &end, &error) &&
              !neural_batch_plan_range(&plan, 0U, NULL, &end, &error),
          "invalid plans, indices, and outputs must be rejected");
}

static void test_batch_accumulator(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    NeuralModel *model = NULL;
    NeuralModel *other_model = NULL;
    NeuralGradient *first = NULL;
    NeuralGradient *second = NULL;
    NeuralGradient *third = NULL;
    NeuralGradient *other = NULL;
    NeuralGradient *reduced = NULL;
    NeuralGradient *ordered[3];
    NeuralBatchAccumulator *accumulator = NULL;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(21), &model, &error) &&
               neural_model_create(&spec, UINT64_C(22), &other_model, &error) &&
               neural_gradient_create(model, &first, &error) &&
               neural_gradient_create(model, &second, &error) &&
               neural_gradient_create(model, &third, &error) &&
               neural_gradient_create(other_model, &other, &error) &&
               neural_gradient_create(model, &reduced, &error) &&
               neural_batch_accumulator_create(model, &accumulator, &error) &&
               fill_gradient(first, 1.0) && fill_gradient(second, 2.0) &&
               fill_gradient(third, 3.0) && fill_gradient(other, 4.0);
    check(prepared, "batch accumulator fixture must be prepared");
    if (prepared) {
        check(neural_batch_accumulator_gradient(accumulator) == NULL &&
                  neural_batch_accumulator_sample_count(accumulator) == 0U,
              "new accumulator must not expose an unfinished gradient");
        check(!neural_batch_accumulator_reset(accumulator, 4U, 4U, &error) &&
                  !neural_batch_accumulator_reset(accumulator, 5U, 4U, &error),
              "empty and reversed batch ranges must be rejected");
        check(neural_batch_accumulator_reset(accumulator, 4U, 7U, &error),
              "batch accumulator reset must establish global sample order");
        check(!neural_batch_accumulator_add(accumulator, 5U, first, &error) &&
                  neural_batch_accumulator_sample_count(accumulator) == 0U,
              "out-of-order sample must leave the accumulator unchanged");
        check(!neural_batch_accumulator_add(accumulator, 4U, other, &error) &&
                  neural_batch_accumulator_sample_count(accumulator) == 0U,
              "incompatible gradient must leave the accumulator unchanged");
        check(neural_batch_accumulator_add(accumulator, 4U, first, &error) &&
                  !neural_batch_accumulator_finalize(accumulator, &error) &&
                  neural_batch_accumulator_sample_count(accumulator) == 1U,
              "incomplete batch must not be finalizable");
        check(neural_batch_accumulator_add(accumulator, 5U, second, &error) &&
                  neural_batch_accumulator_add(accumulator, 6U, third, &error) &&
                  !neural_batch_accumulator_add(accumulator, 7U, first, &error),
              "ordered gradients must accumulate successfully");
        check(neural_batch_accumulator_finalize(accumulator, &error) &&
                  neural_batch_accumulator_sample_count(accumulator) == 3U &&
                  gradient_equals(
                      neural_batch_accumulator_gradient(accumulator), 2.0),
              "finalized gradient must use the actual partial-batch count");
        check(!neural_batch_accumulator_add(accumulator, 7U, first, &error) &&
                  !neural_batch_accumulator_finalize(accumulator, &error),
              "finalized accumulator must reject further state changes");
        check(neural_batch_accumulator_reset(accumulator, 0U, 1U, &error) &&
                  neural_batch_accumulator_gradient(accumulator) == NULL,
              "reset must make the accumulator reusable");
        check(neural_batch_accumulator_add(accumulator, 0U, third, &error) &&
                  neural_batch_accumulator_finalize(accumulator, &error) &&
                  gradient_equals(
                      neural_batch_accumulator_gradient(accumulator), 3.0),
              "single-sample batch must retain its gradient");
        check(fill_gradient(first, 1e16) && fill_gradient(second, 1.0) &&
                  fill_gradient(third, -1e16) &&
                  neural_batch_accumulator_reset(accumulator, 10U, 13U, &error) &&
                  neural_batch_accumulator_add(accumulator, 10U, first, &error) &&
                  neural_batch_accumulator_add(accumulator, 11U, second, &error) &&
                  neural_batch_accumulator_add(accumulator, 12U, third, &error) &&
                  neural_batch_accumulator_finalize(accumulator, &error) &&
                  gradient_nearly_equals(
                      neural_batch_accumulator_gradient(accumulator),
                      1.0 / 3.0,
                      1e-16),
              "compensated accumulation must preserve a small gradient");
        ordered[0] = first;
        ordered[1] = second;
        ordered[2] = third;
        check(neural_gradient_reduce_ordered(reduced, ordered, 3U, &error) &&
                  gradient_equals(reduced, 1.0),
              "ordered reduction must use the same compensated arithmetic");
        check(fill_gradient(first, DBL_MAX) &&
                  fill_gradient(second, DBL_MAX) &&
                  neural_batch_accumulator_reset(accumulator, 20U, 22U, &error) &&
                  neural_batch_accumulator_add(accumulator, 20U, first, &error) &&
                  !neural_batch_accumulator_add(accumulator,
                                                21U,
                                                second,
                                                &error) &&
                  neural_batch_accumulator_sample_count(accumulator) == 1U &&
                  !neural_batch_accumulator_finalize(accumulator, &error),
              "overflowing compensated add must preserve accumulator state");
        check(fill_gradient(first, 2.0) &&
                  neural_batch_accumulator_reset(accumulator,
                                                 SIZE_MAX - 1U,
                                                 SIZE_MAX,
                                                 &error) &&
                  neural_batch_accumulator_add(accumulator,
                                               SIZE_MAX - 1U,
                                               first,
                                               &error) &&
                  neural_batch_accumulator_finalize(accumulator, &error) &&
                  gradient_equals(
                      neural_batch_accumulator_gradient(accumulator), 2.0),
              "half-open batch range must support the maximum end index");
    }
    neural_batch_accumulator_free(accumulator);
    neural_gradient_free(reduced);
    neural_gradient_free(other);
    neural_gradient_free(third);
    neural_gradient_free(second);
    neural_gradient_free(first);
    neural_model_free(other_model);
    neural_model_free(model);
}

static void test_transactional_gradient_add(void)
{
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}},
        {1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}}
    };
    NeuralModelSpec spec = {1U, 2U, layers};
    NeuralModel *model = NULL;
    NeuralGradient *destination = NULL;
    NeuralGradient *source = NULL;
    NeuralError error;
    neural_real *late_bias;
    neural_real *late_destination_bias;
    const neural_real *first_weights;
    neural_real snapshot[2] = {0.0, 0.0};
    size_t weight_count;
    size_t bias_count;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(23), &model, &error) &&
               neural_gradient_create(model, &destination, &error) &&
               neural_gradient_create(model, &source, &error) &&
               fill_gradient(destination, 5.0) && fill_gradient(source, 1.0);
    check(prepared, "transactional gradient fixture must be prepared");
    if (prepared) {
        first_weights = neural_gradient_layer_weights_const(destination,
                                                            0U,
                                                            &weight_count);
        late_bias = neural_gradient_layer_biases(source, 1U, &bias_count);
        if (first_weights != NULL && weight_count == 2U) {
            memcpy(snapshot, first_weights, sizeof(snapshot));
        }
        if (late_bias != NULL && bias_count == 1U) {
            late_bias[0] = INFINITY;
        }
        check(first_weights != NULL && weight_count == 2U &&
                  late_bias != NULL && bias_count == 1U &&
                  !neural_gradient_add(destination, source, &error) &&
                  memcmp(first_weights, snapshot, sizeof(snapshot)) == 0,
              "failed gradient addition must not modify earlier layers");
        if (late_bias != NULL && bias_count == 1U) {
            late_bias[0] = 1.0;
        }
        check(neural_gradient_add(destination, source, &error) &&
                  gradient_equals(destination, 6.0),
              "finite compatible gradients must be added");
        if (first_weights != NULL && weight_count == 2U) {
            memcpy(snapshot, first_weights, sizeof(snapshot));
        }
        late_destination_bias = neural_gradient_layer_biases(destination,
                                                              1U,
                                                              NULL);
        if (late_bias != NULL && bias_count == 1U) {
            late_bias[0] = DBL_MAX;
        }
        if (late_destination_bias != NULL) {
            late_destination_bias[0] = DBL_MAX;
        }
        check(late_bias != NULL && late_destination_bias != NULL &&
                  !neural_gradient_add(destination, source, &error) &&
                  late_destination_bias[0] == DBL_MAX &&
                  memcmp(first_weights, snapshot, sizeof(snapshot)) == 0,
              "overflowing sum must not modify any gradient layer");
    }
    neural_gradient_free(source);
    neural_gradient_free(destination);
    neural_model_free(model);
}

int main(void)
{
    test_batch_plan();
    test_batch_accumulator();
    test_transactional_gradient_add();

    if (failures != 0) {
        fprintf(stderr, "%d batch test(s) failed\n", failures);
        return 1;
    }
    printf("All batch tests passed\n");
    return 0;
}
