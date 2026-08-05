#ifndef NEURAL_PERSISTENCE_H
#define NEURAL_PERSISTENCE_H

#include "neural/checkpoint.h"
#include "neural/error.h"
#include "neural/model.h"

int neural_weights_save_atomic(const char *path,
                               const NeuralModel *model,
                               const NeuralWeightsMetadata *metadata,
                               NeuralError *error);
int neural_weights_load(const char *path,
                        NeuralModel *model,
                        const NeuralProjectDigests *expected_digests,
                        NeuralWeightsMetadata *metadata,
                        NeuralError *error);

int neural_checkpoint_save_atomic(const char *path,
                                  const NeuralModel *model,
                                  const NeuralCheckpointMetadata *metadata,
                                  NeuralError *error);
int neural_checkpoint_load(const char *path,
                           NeuralModel *model,
                           const NeuralProjectDigests *expected_digests,
                           NeuralCheckpointMetadata *metadata,
                           NeuralError *error);
int neural_checkpoint_save_atomic_with_optimizer(
    const char *path,
    const NeuralModel *model,
    const NeuralOptimizer *optimizer,
    const NeuralCheckpointMetadata *metadata,
    NeuralError *error);
int neural_checkpoint_load_with_optimizer(
    const char *path,
    NeuralModel *model,
    NeuralOptimizer *optimizer,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error);

int neural_early_weights_save_atomic(
    const char *path,
    const NeuralModel *selected_model,
    const NeuralWeightsMetadata *metadata,
    NeuralError *error);
int neural_early_checkpoint_save_atomic(
    const char *path,
    const NeuralModel *current_model,
    const NeuralModel *best_model,
    const NeuralCheckpointMetadata *metadata,
    NeuralError *error);
int neural_early_checkpoint_load(
    const char *path,
    NeuralModel *current_model,
    NeuralModel *best_model,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error);
int neural_early_checkpoint_save_atomic_with_optimizer(
    const char *path,
    const NeuralModel *current_model,
    const NeuralModel *best_model,
    const NeuralOptimizer *optimizer,
    const NeuralCheckpointMetadata *metadata,
    NeuralError *error);
int neural_early_checkpoint_load_with_optimizer(
    const char *path,
    NeuralModel *current_model,
    NeuralModel *best_model,
    NeuralOptimizer *optimizer,
    const NeuralProjectDigests *expected_digests,
    NeuralCheckpointMetadata *metadata,
    NeuralError *error);

#endif
