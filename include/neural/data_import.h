#ifndef NEURAL_DATA_IMPORT_H
#define NEURAL_DATA_IMPORT_H

#include <stddef.h>
#include <stdint.h>

#include "neural/error.h"
#include "neural/preprocessing.h"

typedef struct {
    const char *schema_path;
    neural_real validation_ratio;
    neural_real test_ratio;
    uint64_t split_seed;
    NeuralNormalization normalization;
    NeuralMissingPolicy missing_policy;
} NeuralDataImportConfig;

typedef struct {
    size_t total_samples;
    size_t training_samples;
    size_t validation_samples;
    size_t test_samples;
    int stratified;
} NeuralDataImportResult;

int neural_data_import_csv(const char *project_directory,
                           const char *csv_path,
                           const NeuralDataImportConfig *config,
                           NeuralDataImportResult *result,
                           NeuralError *error);

#endif
