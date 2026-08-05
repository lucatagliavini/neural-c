#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "neural/activation.h"
#include "neural/batch.h"
#include "neural/gradient.h"
#include "neural/model.h"
#include "neural/optimizer.h"

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

static void test_sample_order(void)
{
    const size_t expected[] = {
        15U, 3U, 8U, 6U, 10U, 7U, 4U, 2U, 16U,
        14U, 1U, 9U, 0U, 11U, 13U, 12U, 5U
    };
    NeuralSampleOrder *order = NULL;
    NeuralSampleOrder *repeated = NULL;
    NeuralError error;
    size_t first[17];
    size_t second[17];
    unsigned char seen[17] = {0};
    size_t index;
    size_t sample_index;
    int prepared;

    check(!neural_sample_order_create(0U, &order, &error) &&
              !neural_sample_order_create(1U, NULL, &error),
          "sample order must reject invalid creation arguments");
    prepared = neural_sample_order_create(17U, &order, &error) &&
               neural_sample_order_create(17U, &repeated, &error);
    check(prepared, "sample-order fixtures must be allocated");
    if (prepared) {
        check(neural_sample_order_count(order) == 17U &&
                  neural_sample_order_prepare(order,
                                              UINT64_C(42),
                                              UINT64_C(0),
                                              0,
                                              &error),
              "disabled shuffle must prepare a complete identity plan");
        for (index = 0U; index < 17U; index++) {
            check(neural_sample_order_index(order,
                                            index,
                                            &sample_index,
                                            &error) &&
                      sample_index == index,
                  "disabled shuffle must retain source order");
        }
        check(!neural_sample_order_prepare(order,
                                           UINT64_C(42),
                                           UINT64_C(0),
                                           2,
                                           &error) &&
                  !neural_sample_order_index(order,
                                             17U,
                                             &sample_index,
                                             &error) &&
                  !neural_sample_order_index(NULL,
                                             0U,
                                             &sample_index,
                                             &error),
              "sample order must reject invalid requests");
        check(neural_sample_order_prepare(order,
                                          UINT64_C(42),
                                          UINT64_C(0),
                                          1,
                                          &error) &&
                  neural_sample_order_prepare(repeated,
                                              UINT64_C(42),
                                              UINT64_C(0),
                                              1,
                                              &error),
              "shuffle must prepare repeated epoch plans");
        for (index = 0U; index < 17U; index++) {
            if (neural_sample_order_index(order,
                                          index,
                                          &first[index],
                                          &error) &&
                neural_sample_order_index(repeated,
                                          index,
                                          &second[index],
                                          &error) &&
                first[index] < 17U) {
                seen[first[index]] = 1U;
            }
        }
        check(memcmp(first, expected, sizeof(expected)) == 0 &&
                  memcmp(first, second, sizeof(first)) == 0,
              "epoch zero shuffle must match the portable reference plan");
        for (index = 0U; index < 17U; index++) {
            check(seen[index] != 0U,
                  "shuffle plan must contain every source sample once");
        }
        check(neural_sample_order_prepare(repeated,
                                          UINT64_C(42),
                                          UINT64_C(1),
                                          1,
                                          &error),
              "next absolute epoch plan must be generated");
        for (index = 0U; index < 17U; index++) {
            (void)neural_sample_order_index(repeated,
                                            index,
                                            &second[index],
                                            &error);
        }
        check(memcmp(first, second, sizeof(first)) != 0,
              "different absolute epochs must produce different plans");
    }
    neural_sample_order_free(repeated);
    neural_sample_order_free(order);
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

static void test_gradient_norm_and_clipping(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    NeuralModel *model = NULL;
    NeuralGradient *gradient = NULL;
    NeuralError error;
    neural_real *weight;
    neural_real *bias;
    neural_real norm = -1.0;
    int clipped = -1;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(25), &model, &error) &&
               neural_gradient_create(model, &gradient, &error);
    check(prepared, "gradient norm fixture must be prepared");
    if (prepared) {
        weight = neural_gradient_layer_weights(gradient, 0U, NULL);
        bias = neural_gradient_layer_biases(gradient, 0U, NULL);
        weight[0] = 3.0;
        bias[0] = 4.0;
        check(neural_gradient_norm(gradient, &norm, &error) && norm == 5.0,
              "gradient L2 norm must include weights and biases");
        check(neural_gradient_clip_norm(gradient,
                                        0.0,
                                        &norm,
                                        &clipped,
                                        &error) &&
                  norm == 5.0 && clipped == 0 &&
                  weight[0] == 3.0 && bias[0] == 4.0,
              "zero clipping threshold must report without mutation");
        check(neural_gradient_clip_norm(gradient,
                                        5.0,
                                        &norm,
                                        &clipped,
                                        &error) &&
                  norm == 5.0 && clipped == 0 &&
                  weight[0] == 3.0 && bias[0] == 4.0,
              "gradient exactly at the threshold must not be clipped");
        check(neural_gradient_clip_norm(gradient,
                                        2.5,
                                        &norm,
                                        &clipped,
                                        &error) &&
                  norm == 5.0 && clipped == 1 &&
                  weight[0] == 1.5 && bias[0] == 2.0 &&
                  neural_gradient_norm(gradient, &norm, &error) &&
                  norm == 2.5,
              "gradient above the threshold must scale to the maximum norm");
        check(neural_gradient_zero(gradient, &error) &&
                  neural_gradient_clip_norm(gradient,
                                            1.0,
                                            &norm,
                                            &clipped,
                                            &error) &&
                  norm == 0.0 && clipped == 0,
              "zero gradient must have zero norm and remain unclipped");
        weight[0] = DBL_MAX;
        bias[0] = DBL_MAX;
        check(!neural_gradient_norm(gradient, &norm, &error) &&
                  weight[0] == DBL_MAX && bias[0] == DBL_MAX,
              "unrepresentable gradient norm must fail without mutation");
        weight[0] = 1.0;
        bias[0] = 1.0;
        check(!neural_gradient_clip_norm(gradient,
                                         -1.0,
                                         &norm,
                                         &clipped,
                                         &error) &&
                  weight[0] == 1.0 && bias[0] == 1.0,
              "invalid clipping threshold must fail without mutation");
    }
    neural_gradient_free(gradient);
    neural_model_free(model);
}

static void test_regularization(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {2U, 1U, &layer};
    neural_real weights[] = {-2.0, 3.0};
    neural_real biases[] = {4.0};
    NeuralModel *model = NULL;
    NeuralGradient *gradient = NULL;
    NeuralError error;
    neural_real *gradient_weights;
    neural_real *gradient_biases;
    neural_real penalty = -1.0;
    neural_real positive_penalty = 0.0;
    neural_real negative_penalty = 0.0;
    const neural_real epsilon = 1e-6;
    int prepared;

    prepared = neural_model_create(&spec, UINT64_C(26), &model, &error) &&
               neural_model_set_layer_parameters(model,
                                                 0U,
                                                 weights,
                                                 2U,
                                                 biases,
                                                 1U,
                                                 &error) &&
               neural_gradient_create(model, &gradient, &error) &&
               fill_gradient(gradient, 1.0);
    check(prepared, "regularization fixture must be prepared");
    if (prepared) {
        gradient_weights = neural_gradient_layer_weights(gradient, 0U, NULL);
        gradient_biases = neural_gradient_layer_biases(gradient, 0U, NULL);
        check(neural_model_regularization_penalty(model,
                                                  0.5,
                                                  0.25,
                                                  0,
                                                  &penalty,
                                                  &error) &&
                  penalty == 4.125,
              "regularization penalty must use L1 plus half-scaled L2");
        check(neural_gradient_add_regularization(gradient,
                                                 model,
                                                 0.5,
                                                 0.25,
                                                 0,
                                                 &error) &&
                  gradient_weights[0] == 0.0 &&
                  gradient_weights[1] == 2.25 &&
                  gradient_biases[0] == 1.0,
              "regularization gradient must exclude biases by default");
        check(fill_gradient(gradient, 1.0) &&
                  neural_model_regularization_penalty(model,
                                                      0.5,
                                                      0.25,
                                                      1,
                                                      &penalty,
                                                      &error) &&
                  penalty == 8.125 &&
                  neural_gradient_add_regularization(gradient,
                                                     model,
                                                     0.5,
                                                     0.25,
                                                     1,
                                                     &error) &&
                  gradient_biases[0] == 2.5,
              "explicit bias regularization must affect penalty and gradient");
        weights[1] = 3.0 + epsilon;
        check(neural_model_set_layer_parameters(model,
                                                0U,
                                                weights,
                                                2U,
                                                biases,
                                                1U,
                                                &error) &&
                  neural_model_regularization_penalty(model,
                                                      0.5,
                                                      0.25,
                                                      0,
                                                      &positive_penalty,
                                                      &error),
              "positive finite-difference penalty must evaluate");
        weights[1] = 3.0 - epsilon;
        check(neural_model_set_layer_parameters(model,
                                                0U,
                                                weights,
                                                2U,
                                                biases,
                                                1U,
                                                &error) &&
                  neural_model_regularization_penalty(model,
                                                      0.5,
                                                      0.25,
                                                      0,
                                                      &negative_penalty,
                                                      &error) &&
                  fabs((positive_penalty - negative_penalty) /
                           (2.0 * epsilon) -
                       1.25) < 1e-9,
              "regularization gradient must match a finite difference");
        weights[1] = 3.0;
        check(neural_model_set_layer_parameters(model,
                                                0U,
                                                weights,
                                                2U,
                                                biases,
                                                1U,
                                                &error),
              "finite-difference fixture must restore model parameters");
        check(fill_gradient(gradient, 1.0),
              "transactional regularization fixture must reset");
        gradient_biases[0] = DBL_MAX;
        check(!neural_gradient_add_regularization(gradient,
                                                  model,
                                                  DBL_MAX,
                                                  DBL_MAX,
                                                  1,
                                                  &error) &&
                  gradient_weights[0] == 1.0 &&
                  gradient_weights[1] == 1.0 &&
                  gradient_biases[0] == DBL_MAX,
              "overflowing regularization must leave the gradient unchanged");
        check(!neural_model_regularization_penalty(model,
                                                   -1.0,
                                                   0.0,
                                                   0,
                                                   &penalty,
                                                   &error),
              "negative regularization coefficients must be rejected");
    }
    neural_gradient_free(gradient);
    neural_model_free(model);
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
    return 1;
}

static void test_optimizer_abstraction(void)
{
    NeuralLayerSpec layers[] = {
        {2U, {NEURAL_ACTIVATION_TANH, 0U, NULL}},
        {1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}}
    };
    NeuralModelSpec spec = {2U, 2U, layers};
    NeuralModel *direct_model = NULL;
    NeuralModel *optimized_model = NULL;
    NeuralGradient *direct_gradient = NULL;
    NeuralGradient *optimized_gradient = NULL;
    NeuralOptimizer *optimizer = NULL;
    NeuralOptimizerOptions options = {
        NEURAL_OPTIMIZER_GRADIENT_DESCENT, 0.0, 0.0, 0.0, 0.0,
        0.1, NEURAL_LR_SCHEDULE_CONSTANT, 0.0, 0U, 0U, 0.0,
        0.0, -1.0, 0U, 0.0
    };
    NeuralOptimizerKind kind = (NeuralOptimizerKind)-1;
    NeuralError error;
    int prepared;

    check(strcmp(neural_optimizer_name(NEURAL_OPTIMIZER_GRADIENT_DESCENT),
                 "gradient_descent") == 0 &&
              neural_optimizer_from_name("gradient_descent", &kind) &&
              kind == NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
              neural_optimizer_from_name("momentum", &kind) &&
              kind == NEURAL_OPTIMIZER_MOMENTUM &&
              neural_optimizer_from_name("adam", &kind) &&
              kind == NEURAL_OPTIMIZER_ADAM &&
              !neural_optimizer_from_name("rmsprop", &kind),
          "optimizer names must round-trip and reject unsupported values");
    prepared = neural_model_create(&spec, UINT64_C(17), &direct_model, &error) &&
               neural_model_create(&spec,
                                   UINT64_C(17),
                                   &optimized_model,
                                   &error) &&
               neural_gradient_create(direct_model,
                                      &direct_gradient,
                                      &error) &&
               neural_gradient_create(optimized_model,
                                      &optimized_gradient,
                                      &error) &&
               fill_gradient(direct_gradient, 0.375) &&
               fill_gradient(optimized_gradient, 0.375) &&
               neural_optimizer_create(optimized_model,
                                       &options,
                                       &optimizer,
                                       &error);
    check(prepared, "optimizer abstraction fixtures must be prepared");
    if (prepared) {
        check(neural_optimizer_kind(optimizer) ==
                  NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
                  neural_optimizer_version(optimizer) ==
                      NEURAL_OPTIMIZER_ABSTRACTION_VERSION,
              "optimizer instances must expose stable identity and version");
        check(!neural_optimizer_step(optimizer,
                                     direct_model,
                                     direct_gradient,
                                     0.125,
                                     &error),
              "optimizer instances must reject a different model");
        check(neural_model_apply_gradient(direct_model,
                                          direct_gradient,
                                          0.125,
                                          &error) &&
                  neural_optimizer_step(optimizer,
                                        optimized_model,
                                        optimized_gradient,
                                        0.125,
                                        &error) &&
                  models_equal(direct_model, optimized_model),
              "gradient descent abstraction must preserve exact updates");
    }
    neural_optimizer_free(optimizer);
    neural_gradient_free(optimized_gradient);
    neural_gradient_free(direct_gradient);
    neural_model_free(optimized_model);
    neural_model_free(direct_model);
}

static void test_stateful_optimizers(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    NeuralOptimizerOptions momentum_options = {
        NEURAL_OPTIMIZER_MOMENTUM, 0.5, 0.0, 0.0, 0.0,
        0.1, NEURAL_LR_SCHEDULE_CONSTANT, 0.0, 0U, 0U, 0.0,
        0.0, -1.0, 0U, 0.0
    };
    NeuralOptimizerOptions adam_options = {
        NEURAL_OPTIMIZER_ADAM, 0.0, 0.9, 0.999, 1e-8,
        0.1, NEURAL_LR_SCHEDULE_CONSTANT, 0.0, 0U, 0U, 0.0,
        0.0, -1.0, 0U, 0.0
    };
    neural_real initial_weight[] = {1.0};
    neural_real initial_bias[] = {-1.0};
    NeuralModel *momentum_model = NULL;
    NeuralModel *adam_model = NULL;
    NeuralModel *adam_reference = NULL;
    NeuralGradient *momentum_gradient = NULL;
    NeuralGradient *adam_gradient = NULL;
    NeuralGradient *adam_reference_gradient = NULL;
    NeuralOptimizer *momentum = NULL;
    NeuralOptimizer *adam = NULL;
    NeuralOptimizer *adam_reference_optimizer = NULL;
    NeuralError error;
    int prepared;

    prepared = neural_model_create(&spec,
                                   UINT64_C(21),
                                   &momentum_model,
                                   &error) &&
               neural_model_create(&spec, UINT64_C(21), &adam_model, &error) &&
               neural_model_create(&spec,
                                   UINT64_C(21),
                                   &adam_reference,
                                   &error) &&
               neural_model_set_layer_parameters(momentum_model,
                                                 0U,
                                                 initial_weight,
                                                 1U,
                                                 initial_bias,
                                                 1U,
                                                 &error) &&
               neural_model_set_layer_parameters(adam_model,
                                                 0U,
                                                 initial_weight,
                                                 1U,
                                                 initial_bias,
                                                 1U,
                                                 &error) &&
               neural_model_set_layer_parameters(adam_reference,
                                                 0U,
                                                 initial_weight,
                                                 1U,
                                                 initial_bias,
                                                 1U,
                                                 &error) &&
               neural_gradient_create(momentum_model,
                                      &momentum_gradient,
                                      &error) &&
               neural_gradient_create(adam_model, &adam_gradient, &error) &&
               neural_gradient_create(adam_reference,
                                      &adam_reference_gradient,
                                      &error) &&
               neural_optimizer_create(momentum_model,
                                       &momentum_options,
                                       &momentum,
                                       &error) &&
               neural_optimizer_create(adam_model,
                                       &adam_options,
                                       &adam,
                                       &error) &&
               neural_optimizer_create(adam_reference,
                                       &adam_options,
                                       &adam_reference_optimizer,
                                       &error);
    check(prepared, "stateful optimizer fixtures must be prepared");
    if (prepared) {
        neural_real *momentum_weights = neural_gradient_layer_weights(
            momentum_gradient, 0U, NULL);
        neural_real *momentum_biases = neural_gradient_layer_biases(
            momentum_gradient, 0U, NULL);
        neural_real *adam_weights = neural_gradient_layer_weights(
            adam_gradient, 0U, NULL);
        neural_real *adam_biases = neural_gradient_layer_biases(
            adam_gradient, 0U, NULL);
        neural_real *reference_weights = neural_gradient_layer_weights(
            adam_reference_gradient, 0U, NULL);
        neural_real *reference_biases = neural_gradient_layer_biases(
            adam_reference_gradient, 0U, NULL);
        const neural_real *model_weights;
        const neural_real *model_biases;

        momentum_weights[0] = 2.0;
        momentum_biases[0] = -4.0;
        check(neural_optimizer_step(momentum,
                                    momentum_model,
                                    momentum_gradient,
                                    0.1,
                                    &error) &&
                  neural_optimizer_step(momentum,
                                        momentum_model,
                                        momentum_gradient,
                                        0.1,
                                        &error),
              "momentum must apply two finite deterministic steps");
        model_weights = neural_model_layer_weights(momentum_model, 0U, NULL);
        model_biases = neural_model_layer_biases(momentum_model, 0U, NULL);
        check(fabs(model_weights[0] - 0.5) < 1e-15 &&
                  fabs(model_biases[0]) < 1e-15,
              "momentum velocity must follow its specified recurrence");

        adam_weights[0] = DBL_MAX;
        adam_biases[0] = 1.0;
        check(!neural_optimizer_step(adam,
                                     adam_model,
                                     adam_gradient,
                                     0.1,
                                     &error) &&
                  models_equal(adam_model, adam_reference),
              "non-finite Adam candidates must not mutate model or state");
        adam_weights[0] = 2.0;
        adam_biases[0] = -4.0;
        reference_weights[0] = 2.0;
        reference_biases[0] = -4.0;
        check(neural_optimizer_step(adam,
                                    adam_model,
                                    adam_gradient,
                                    0.1,
                                    &error) &&
                  neural_optimizer_step(adam_reference_optimizer,
                                        adam_reference,
                                        adam_reference_gradient,
                                        0.1,
                                        &error) &&
                  models_equal(adam_model, adam_reference),
              "failed Adam steps must leave the next valid step exact");
    }
    neural_optimizer_free(adam_reference_optimizer);
    neural_optimizer_free(adam);
    neural_optimizer_free(momentum);
    neural_gradient_free(adam_reference_gradient);
    neural_gradient_free(adam_gradient);
    neural_gradient_free(momentum_gradient);
    neural_model_free(adam_reference);
    neural_model_free(adam_model);
    neural_model_free(momentum_model);
}

static void test_learning_rate_schedules(void)
{
    NeuralLayerSpec layer = {
        1U, {NEURAL_ACTIVATION_LINEAR, 0U, NULL}
    };
    NeuralModelSpec spec = {1U, 1U, &layer};
    NeuralOptimizerOptions options = {
        NEURAL_OPTIMIZER_GRADIENT_DESCENT, 0.0, 0.0, 0.0, 0.0,
        1.0, NEURAL_LR_SCHEDULE_STEP, 0.5, 2U, 0U, 0.0,
        0.0, -1.0, 0U, 0.0
    };
    NeuralModel *model = NULL;
    NeuralOptimizer *schedule = NULL;
    NeuralError error;

    check(neural_model_create(&spec, UINT64_C(29), &model, &error) &&
              neural_optimizer_create(model, &options, &schedule, &error),
          "step schedule fixture must be prepared");
    if (schedule != NULL) {
        check(neural_optimizer_observe_epoch(schedule, 4.0, &error) &&
                  neural_optimizer_current_learning_rate(schedule) == 1.0 &&
                  neural_optimizer_schedule_next_transition(schedule) == 2U &&
                  neural_optimizer_observe_epoch(schedule, 3.0, &error) &&
                  neural_optimizer_current_learning_rate(schedule) == 0.5 &&
                  neural_optimizer_schedule_next_transition(schedule) == 4U,
              "step schedule must transition after exact epoch intervals");
    }
    neural_optimizer_free(schedule);
    schedule = NULL;

    options.schedule = NEURAL_LR_SCHEDULE_EXPONENTIAL;
    options.schedule_decay = 0.25;
    check(neural_optimizer_create(model, &options, &schedule, &error) &&
              neural_optimizer_observe_epoch(schedule, 4.0, &error) &&
              neural_optimizer_current_learning_rate(schedule) == 0.25 &&
              neural_optimizer_observe_epoch(schedule, 3.0, &error) &&
              neural_optimizer_current_learning_rate(schedule) == 0.0625,
          "exponential schedule must decay after every completed epoch");
    neural_optimizer_free(schedule);
    schedule = NULL;

    options.schedule = NEURAL_LR_SCHEDULE_PLATEAU;
    options.schedule_decay = 0.5;
    options.schedule_plateau_patience = 2U;
    options.schedule_plateau_min_delta = 0.1;
    check(neural_optimizer_create(model, &options, &schedule, &error) &&
              neural_optimizer_observe_epoch(schedule, 1.0, &error) &&
              neural_optimizer_observe_epoch(schedule, 0.95, &error) &&
              neural_optimizer_current_learning_rate(schedule) == 1.0 &&
              neural_optimizer_schedule_stale_epochs(schedule) == 1U &&
              neural_optimizer_observe_epoch(schedule, 0.96, &error) &&
              neural_optimizer_current_learning_rate(schedule) == 0.5 &&
              neural_optimizer_schedule_stale_epochs(schedule) == 0U &&
              neural_optimizer_observe_epoch(schedule, 0.8, &error) &&
              neural_optimizer_schedule_best_metric(schedule) == 0.8,
          "plateau schedule must use patience and minimum improvement");
    neural_optimizer_free(schedule);
    neural_model_free(model);
}

int main(void)
{
    test_batch_plan();
    test_sample_order();
    test_batch_accumulator();
    test_transactional_gradient_add();
    test_gradient_norm_and_clipping();
    test_regularization();
    test_optimizer_abstraction();
    test_stateful_optimizers();
    test_learning_rate_schedules();

    if (failures != 0) {
        fprintf(stderr, "%d batch test(s) failed\n", failures);
        return 1;
    }
    printf("All batch tests passed\n");
    return 0;
}
