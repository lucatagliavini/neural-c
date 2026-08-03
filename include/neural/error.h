#ifndef NEURAL_ERROR_H
#define NEURAL_ERROR_H

#include <stddef.h>

#include "neural/defaults.h"

#if NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY < 2U
#error "NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY must be at least 2"
#endif

typedef struct {
    char message[NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY];
} NeuralError;

void neural_error_clear(NeuralError *error);
void neural_error_set(NeuralError *error, const char *format, ...);

#endif
