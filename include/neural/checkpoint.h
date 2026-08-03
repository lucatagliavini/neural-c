#ifndef NEURAL_CHECKPOINT_H
#define NEURAL_CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

#define NEURAL_SHA256_HEX_LENGTH 64U
#define NEURAL_SHA256_TEXT_CAPACITY (NEURAL_SHA256_HEX_LENGTH + 1U)

typedef enum {
    NEURAL_OPTIMIZER_GRADIENT_DESCENT
} NeuralOptimizer;

typedef struct {
    size_t completed_epochs;
    size_t target_epochs;
    uint64_t rng_state;
    NeuralOptimizer optimizer;
    char model_digest[NEURAL_SHA256_TEXT_CAPACITY];
    char dataset_digest[NEURAL_SHA256_TEXT_CAPACITY];
    char training_digest[NEURAL_SHA256_TEXT_CAPACITY];
} NeuralCheckpointMetadata;

#endif
