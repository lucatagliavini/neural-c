#ifndef NEURAL_CHECKPOINT_H
#define NEURAL_CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

#include "neural/types.h"

#define NEURAL_SHA256_HEX_LENGTH 64U
#define NEURAL_SHA256_TEXT_CAPACITY (NEURAL_SHA256_HEX_LENGTH + 1U)

typedef enum {
    NEURAL_OPTIMIZER_GRADIENT_DESCENT
} NeuralOptimizer;

typedef enum {
    NEURAL_COMPLETION_TARGET,
    NEURAL_COMPLETION_EARLY_STOPPING
} NeuralCompletionReason;

typedef struct {
    char model[NEURAL_SHA256_TEXT_CAPACITY];
    char dataset[NEURAL_SHA256_TEXT_CAPACITY];
    char training[NEURAL_SHA256_TEXT_CAPACITY];
} NeuralProjectDigests;

typedef struct {
    size_t completed_epochs;
    NeuralProjectDigests digests;
    size_t selected_epoch;
    size_t target_epochs;
    NeuralCompletionReason completion_reason;
    size_t format_version;
} NeuralWeightsMetadata;

typedef struct {
    size_t completed_epochs;
    size_t target_epochs;
    uint64_t rng_state;
    NeuralOptimizer optimizer;
    NeuralProjectDigests digests;
    size_t best_epoch;
    neural_real best_loss;
    size_t stale_epochs;
    size_t format_version;
} NeuralCheckpointMetadata;

#endif
