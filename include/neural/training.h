#ifndef NEURAL_TRAINING_H
#define NEURAL_TRAINING_H

#include <stddef.h>

#include "neural/error.h"

typedef enum {
    NEURAL_TRAIN_FRESH,
    NEURAL_TRAIN_RESUME,
    NEURAL_TRAIN_ADDITIONAL
} NeuralTrainingMode;

typedef struct {
    NeuralTrainingMode mode;
    size_t additional_epochs;
} NeuralTrainingRequest;

int neural_training_request_validate(const NeuralTrainingRequest *request,
                                     NeuralError *error);
const char *neural_training_mode_name(NeuralTrainingMode mode);

#endif
