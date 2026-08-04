#ifndef NEURAL_PROJECT_H
#define NEURAL_PROJECT_H

#include <stddef.h>
#include <stdint.h>

#include "neural/activation.h"
#include "neural/error.h"
#include "neural/preprocessing.h"
#include "neural/types.h"

typedef enum {
    NEURAL_LOSS_MSE,
    NEURAL_LOSS_BINARY_CROSS_ENTROPY,
    NEURAL_LOSS_CATEGORICAL_CROSS_ENTROPY
} NeuralLoss;

typedef struct {
    size_t neuron_count;
    NeuralActivationSpec activation;
} NeuralLayerSpec;

typedef struct {
    size_t input_count;
    size_t layer_count;
    NeuralLayerSpec *layers;
} NeuralModelSpec;

typedef struct {
    size_t epochs;
    neural_real learning_rate;
    uint64_t seed;
    NeuralLoss loss;
    size_t checkpoint_interval;
    size_t early_stopping_patience;
    neural_real early_stopping_min_delta;
    size_t batch_size;
} NeuralTrainingConfig;

typedef struct {
    size_t sample_count;
    size_t input_count;
    size_t output_count;
    neural_real *inputs;
    neural_real *outputs;
} NeuralDataset;

typedef struct {
    NeuralModelSpec model;
    NeuralTrainingConfig training;
    NeuralDataset dataset;
    NeuralDataset validation;
    int has_validation;
    NeuralPreprocessing preprocessing;
    int has_preprocessing;
} NeuralProject;

/* Load destinations must not already own data; release successful loads. */
int neural_model_spec_load(const char *path,
                           NeuralModelSpec *model,
                           NeuralError *error);
int neural_model_spec_validate(const NeuralModelSpec *model,
                               NeuralError *error);
void neural_model_spec_free(NeuralModelSpec *model);

int neural_training_config_load(const char *path,
                                NeuralTrainingConfig *config,
                                NeuralError *error);
int neural_training_config_validate(const NeuralTrainingConfig *config,
                                    NeuralError *error);

int neural_dataset_load(const char *path,
                        size_t input_count,
                        size_t output_count,
                        NeuralDataset *dataset,
                        NeuralError *error);
void neural_dataset_free(NeuralDataset *dataset);

int neural_project_load(const char *directory,
                        NeuralProject *project,
                        NeuralError *error);
void neural_project_free(NeuralProject *project);

const char *neural_loss_name(NeuralLoss loss);
int neural_loss_from_name(const char *name, NeuralLoss *loss);

#endif
