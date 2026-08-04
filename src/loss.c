#include "neural/loss.h"

#include <float.h>
#include <math.h>
#include <string.h>

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

static int validate_probabilities(NeuralLoss loss,
                                  const neural_real *predicted,
                                  size_t count,
                                  NeuralError *error)
{
    size_t index;

    if (loss == NEURAL_LOSS_MSE) {
        return 1;
    }
    for (index = 0U; index < count; index++) {
        if (predicted[index] < 0.0 || predicted[index] > 1.0) {
            neural_error_set(error,
                             "cross-entropy predictions must be probabilities");
            return 0;
        }
    }
    return 1;
}

int neural_loss_validate_output(NeuralLoss loss,
                                NeuralActivationKind activation,
                                size_t output_count,
                                NeuralError *error)
{
    if (output_count == 0U) {
        neural_error_set(error, "loss requires at least one model output");
        return 0;
    }
    switch (loss) {
    case NEURAL_LOSS_MSE:
        return 1;
    case NEURAL_LOSS_BINARY_CROSS_ENTROPY:
        if (activation != NEURAL_ACTIVATION_SIGMOID) {
            neural_error_set(error,
                             "binary_cross_entropy requires sigmoid output");
            return 0;
        }
        return 1;
    case NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY:
        if (activation != NEURAL_ACTIVATION_SOFTMAX || output_count < 2U) {
            neural_error_set(
                error,
                "categorical_cross_entropy requires at least two softmax outputs");
            return 0;
        }
        return 1;
    }
    neural_error_set(error, "unsupported loss kind");
    return 0;
}

int neural_loss_validate_targets(NeuralLoss loss,
                                 const neural_real *expected,
                                 size_t sample_count,
                                 size_t output_count,
                                 NeuralError *error)
{
    size_t sample;

    if (expected == NULL || sample_count == 0U || output_count == 0U ||
        sample_count > SIZE_MAX / output_count) {
        neural_error_set(error, "loss targets and dimensions are required");
        return 0;
    }
    if (strcmp(neural_loss_name(loss), "unknown") == 0) {
        neural_error_set(error, "unsupported loss kind");
        return 0;
    }
    for (sample = 0U; sample < sample_count; sample++) {
        size_t output;
        size_t selected = 0U;

        for (output = 0U; output < output_count; output++) {
            neural_real target = expected[sample * output_count + output];

            if (!isfinite(target)) {
                neural_error_set(error, "loss targets must be finite");
                return 0;
            }
            if (loss == NEURAL_LOSS_BINARY_CROSS_ENTROPY ||
                loss == NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY) {
                if (target != 0.0 && target != 1.0) {
                    neural_error_set(
                        error,
                        "%s targets must contain only zero or one",
                        neural_loss_name(loss));
                    return 0;
                }
                if (target == 1.0) {
                    selected++;
                }
            }
        }
        if (loss == NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY &&
            selected != 1U) {
            neural_error_set(error,
                             "categorical_cross_entropy targets must be one-hot");
            return 0;
        }
    }
    return 1;
}

static neural_real bounded_probability(neural_real probability)
{
    if (probability < DBL_MIN) {
        return DBL_MIN;
    }
    if (probability > 1.0 - DBL_EPSILON) {
        return 1.0 - DBL_EPSILON;
    }
    return probability;
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

    if (value == NULL || !validate_arguments(predicted, expected, count, error) ||
        !validate_probabilities(loss, predicted, count, error) ||
        !neural_loss_validate_targets(loss, expected, 1U, count, error)) {
        if (value == NULL) {
            neural_error_set(error, "loss output is required");
        }
        return 0;
    }
    for (index = 0U; index < count; index++) {
        neural_real term;

        if (loss == NEURAL_LOSS_MSE) {
            neural_real difference = predicted[index] - expected[index];

            term = difference * difference;
        } else if (loss == NEURAL_LOSS_BINARY_CROSS_ENTROPY) {
            neural_real probability = bounded_probability(predicted[index]);

            term = expected[index] == 1.0
                       ? -log(probability) : -log1p(-probability);
        } else if (loss == NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY) {
            term = expected[index] == 1.0
                       ? -log(bounded_probability(predicted[index])) : 0.0;
        } else {
            neural_error_set(error, "unsupported loss kind");
            return 0;
        }
        sum += term;
        if (!isfinite(sum)) {
            neural_error_set(error, "loss result is not finite");
            return 0;
        }
    }
    *value = loss == NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY
                 ? sum : sum / (neural_real)count;
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

    if (gradient == NULL || !validate_arguments(predicted, expected, count, error) ||
        !validate_probabilities(loss, predicted, count, error) ||
        !neural_loss_validate_targets(loss, expected, 1U, count, error)) {
        if (gradient == NULL) {
            neural_error_set(error, "loss gradient output is required");
        }
        return 0;
    }
    factor = loss == NEURAL_LOSS_MSE ? 2.0 / (neural_real)count
                                     : 1.0 / (neural_real)count;
    for (index = 0U; index < count; index++) {
        if (loss == NEURAL_LOSS_MSE) {
            gradient[index] = factor * (predicted[index] - expected[index]);
        } else if (loss == NEURAL_LOSS_BINARY_CROSS_ENTROPY &&
                   predicted[index] > 0.0 && predicted[index] < 1.0) {
            gradient[index] = factor *
                ((1.0 - expected[index]) / (1.0 - predicted[index]) -
                 expected[index] / predicted[index]);
        } else if (loss == NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY &&
                   (expected[index] == 0.0 || predicted[index] > 0.0)) {
            gradient[index] = expected[index] == 0.0
                                  ? 0.0 : -1.0 / predicted[index];
        } else {
            neural_error_set(error,
                             "cross-entropy probability gradient is singular");
            return 0;
        }
        if (!isfinite(gradient[index])) {
            neural_error_set(error, "loss gradient is not finite");
            return 0;
        }
    }
    return 1;
}

int neural_loss_evaluate_with_logits(NeuralLoss loss,
                                     NeuralActivationKind activation,
                                     const neural_real *logits,
                                     const neural_real *predicted,
                                     const neural_real *expected,
                                     size_t count,
                                     neural_real *value,
                                     NeuralError *error)
{
    neural_real sum = 0.0;
    size_t index;

    if (!neural_loss_validate_output(loss, activation, count, error) ||
        !neural_loss_validate_targets(loss, expected, 1U, count, error) ||
        (predicted != NULL &&
         !validate_probabilities(loss, predicted, count, error))) {
        return 0;
    }
    if (loss == NEURAL_LOSS_MSE) {
        return neural_loss_evaluate(loss, predicted, expected, count,
                                    value, error);
    }
    if (logits == NULL || predicted == NULL || value == NULL) {
        neural_error_set(error, "cross-entropy logits and output are required");
        return 0;
    }
    if (loss == NEURAL_LOSS_BINARY_CROSS_ENTROPY) {
        for (index = 0U; index < count; index++) {
            neural_real logit = logits[index];
            neural_real term;

            if (!isfinite(logit) || !isfinite(predicted[index])) {
                neural_error_set(error, "cross-entropy values must be finite");
                return 0;
            }
            term = fmax(logit, 0.0) - logit * expected[index] +
                   log1p(exp(-fabs(logit)));
            sum += term;
        }
        *value = sum / (neural_real)count;
    } else {
        neural_real maximum = logits[0];
        neural_real selected_logit = 0.0;

        for (index = 0U; index < count; index++) {
            if (!isfinite(logits[index]) || !isfinite(predicted[index])) {
                neural_error_set(error, "cross-entropy values must be finite");
                return 0;
            }
            if (logits[index] > maximum) {
                maximum = logits[index];
            }
            if (expected[index] == 1.0) {
                selected_logit = logits[index];
            }
        }
        for (index = 0U; index < count; index++) {
            sum += exp(logits[index] - maximum);
        }
        *value = maximum + log(sum) - selected_logit;
    }
    if (!isfinite(*value)) {
        neural_error_set(error, "loss result is not finite");
        return 0;
    }
    return 1;
}

int neural_loss_pre_activation_gradient(NeuralLoss loss,
                                        NeuralActivationKind activation,
                                        const neural_real *predicted,
                                        const neural_real *expected,
                                        size_t count,
                                        neural_real *gradient,
                                        NeuralError *error)
{
    neural_real factor;
    size_t index;

    if (loss == NEURAL_LOSS_MSE || gradient == NULL ||
        !neural_loss_validate_output(loss, activation, count, error) ||
        !validate_arguments(predicted, expected, count, error) ||
        !validate_probabilities(loss, predicted, count, error) ||
        !neural_loss_validate_targets(loss, expected, 1U, count, error)) {
        if (loss == NEURAL_LOSS_MSE) {
            neural_error_set(error, "MSE does not have a fused output gradient");
        } else if (gradient == NULL) {
            neural_error_set(error, "loss gradient output is required");
        }
        return 0;
    }
    factor = loss == NEURAL_LOSS_BINARY_CROSS_ENTROPY
                 ? 1.0 / (neural_real)count : 1.0;
    for (index = 0U; index < count; index++) {
        gradient[index] = factor * (predicted[index] - expected[index]);
        if (!isfinite(gradient[index])) {
            neural_error_set(error, "loss gradient is not finite");
            return 0;
        }
    }
    return 1;
}
