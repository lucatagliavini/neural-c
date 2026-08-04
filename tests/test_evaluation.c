#include <math.h>
#include <stdio.h>

#include "neural/evaluation.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static void test_multiclass_metrics(void)
{
    static const neural_real predicted[] = {
        0.8, 0.1, 0.1,
        0.1, 0.3, 0.6,
        0.1, 0.2, 0.7,
        0.1, 0.7, 0.2
    };
    static const neural_real expected[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0
    };
    NeuralEvaluationResult result = {0};
    NeuralError error;

    check(neural_evaluation_compute(NEURAL_LOSS_MSE,
                                    NEURAL_ACTIVATION_SOFTMAX,
                                    predicted,
                                    expected,
                                    4U,
                                    3U,
                                    &result,
                                    &error),
          "multiclass evaluation must succeed");
    check(result.is_classification && result.class_count == 3U &&
              result.correct_count == 3U && result.accuracy == 0.75,
          "multiclass accuracy must be exact");
    check(result.confusion[0] == 1U && result.confusion[1] == 0U &&
              result.confusion[2] == 0U && result.confusion[3] == 0U &&
              result.confusion[4] == 1U && result.confusion[5] == 1U &&
              result.confusion[6] == 0U && result.confusion[7] == 0U &&
              result.confusion[8] == 1U,
          "multiclass confusion matrix must use truth-major rows");
    check(result.precision[0] == 1.0 && result.recall[0] == 1.0 &&
              result.f1[0] == 1.0 && result.precision[1] == 1.0 &&
              result.recall[1] == 0.5 &&
              fabs(result.f1[1] - 2.0 / 3.0) < 1e-15 &&
              result.precision[2] == 0.5 && result.recall[2] == 1.0 &&
              fabs(result.f1[2] - 2.0 / 3.0) < 1e-15,
          "multiclass precision, recall, and F1 must be correct");
    check(isfinite(result.loss) && result.loss > 0.0,
          "multiclass loss must be finite and positive");
    neural_evaluation_result_free(&result);
}

static void test_binary_and_regression_detection(void)
{
    static const neural_real binary_predicted[] = {0.1, 0.8, 0.7, 0.2};
    static const neural_real binary_expected[] = {0.0, 1.0, 0.0, 0.0};
    static const neural_real regression_predicted[] = {0.2, 0.8};
    static const neural_real regression_expected[] = {0.0, 1.0};
    NeuralEvaluationResult result = {0};
    NeuralError error;

    check(neural_evaluation_compute(NEURAL_LOSS_MSE,
                                    NEURAL_ACTIVATION_SIGMOID,
                                    binary_predicted,
                                    binary_expected,
                                    4U,
                                    1U,
                                    &result,
                                    &error) &&
              result.is_classification && result.class_count == 2U &&
              result.correct_count == 3U && result.accuracy == 0.75,
          "binary sigmoid targets must produce classification metrics");
    neural_evaluation_result_free(&result);

    check(neural_evaluation_compute(NEURAL_LOSS_MSE,
                                    NEURAL_ACTIVATION_LINEAR,
                                    regression_predicted,
                                    regression_expected,
                                    2U,
                                    1U,
                                    &result,
                                    &error) &&
              !result.is_classification && result.class_count == 0U &&
              result.confusion == NULL && fabs(result.loss - 0.04) < 1e-15,
          "linear outputs must remain regression even with binary values");
    neural_evaluation_result_free(&result);
}

static void test_invalid_values(void)
{
    static const neural_real predicted[] = {NAN};
    static const neural_real expected[] = {0.0};
    NeuralEvaluationResult result = {0};
    NeuralError error;

    check(!neural_evaluation_compute(NEURAL_LOSS_MSE,
                                     NEURAL_ACTIVATION_SIGMOID,
                                     predicted,
                                     expected,
                                     1U,
                                     1U,
                                     &result,
                                     &error) &&
              error.message[0] != '\0' && result.confusion == NULL,
          "non-finite predictions must fail without leaking results");
}

int main(void)
{
    test_multiclass_metrics();
    test_binary_and_regression_detection();
    test_invalid_values();

    if (failures != 0) {
        fprintf(stderr, "%d evaluation test(s) failed\n", failures);
        return 1;
    }
    puts("All evaluation tests passed");
    return 0;
}
