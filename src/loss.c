#include "neural/loss.h"

#include <math.h>

static int validate_arguments(const neural_real *predicted,
                              const neural_real *expected,
                              size_t count,
                              NeuralError *error)
{
    size_t index;

    if (predicted == NULL || expected == NULL || count == 0U) {
        neural_error_set(error, "invalid loss buffers");
        return 0;
    }
    for (index = 0U; index < count; index++) {
        if (!isfinite(predicted[index]) || !isfinite(expected[index])) {
            neural_error_set(error, "loss values must be finite");
            return 0;
        }
    }
    return 1;
}

int neural_loss_evaluate(NeuralLoss loss,
                         const neural_real *predicted,
                         const neural_real *expected,
                         size_t count,
                         neural_real *value,
                         NeuralError *error)
{
    neural_real sum = 0.0;
    size_t index;

    if (value == NULL || loss != NEURAL_LOSS_MSE ||
        !validate_arguments(predicted, expected, count, error)) {
        if (value == NULL) {
            neural_error_set(error, "loss output is required");
        } else if (loss != NEURAL_LOSS_MSE) {
            neural_error_set(error, "unsupported loss kind");
        }
        return 0;
    }
    for (index = 0U; index < count; index++) {
        neural_real difference = predicted[index] - expected[index];

        sum += difference * difference;
        if (!isfinite(sum)) {
            neural_error_set(error, "loss result is not finite");
            return 0;
        }
    }
    *value = sum / (neural_real)count;
    return isfinite(*value);
}

int neural_loss_gradient(NeuralLoss loss,
                         const neural_real *predicted,
                         const neural_real *expected,
                         size_t count,
                         neural_real *gradient,
                         NeuralError *error)
{
    neural_real factor;
    size_t index;

    if (gradient == NULL || loss != NEURAL_LOSS_MSE ||
        !validate_arguments(predicted, expected, count, error)) {
        if (gradient == NULL) {
            neural_error_set(error, "loss gradient output is required");
        } else if (loss != NEURAL_LOSS_MSE) {
            neural_error_set(error, "unsupported loss kind");
        }
        return 0;
    }
    factor = 2.0 / (neural_real)count;
    for (index = 0U; index < count; index++) {
        gradient[index] = factor * (predicted[index] - expected[index]);
        if (!isfinite(gradient[index])) {
            neural_error_set(error, "loss gradient is not finite");
            return 0;
        }
    }
    return 1;
}
