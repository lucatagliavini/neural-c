#include "neural/training.h"

int neural_training_request_validate(const NeuralTrainingRequest *request,
                                     NeuralError *error)
{
    if (request == NULL) {
        neural_error_set(error, "training request is required");
        return 0;
    }
    switch (request->mode) {
    case NEURAL_TRAIN_FRESH:
    case NEURAL_TRAIN_RESUME:
        if (request->additional_epochs != 0U) {
            neural_error_set(error,
                             "additional epochs are valid only in additional mode");
            return 0;
        }
        return 1;
    case NEURAL_TRAIN_ADDITIONAL:
        if (request->additional_epochs == 0U) {
            neural_error_set(error,
                             "additional epochs must be a positive integer");
            return 0;
        }
        return 1;
    }
    neural_error_set(error, "unknown training mode");
    return 0;
}

const char *neural_training_mode_name(NeuralTrainingMode mode)
{
    switch (mode) {
    case NEURAL_TRAIN_FRESH:
        return "fresh";
    case NEURAL_TRAIN_RESUME:
        return "resume";
    case NEURAL_TRAIN_ADDITIONAL:
        return "additional";
    }
    return "unknown";
}
