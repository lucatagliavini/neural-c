#ifndef NEURAL_PREPROCESSING_H
#define NEURAL_PREPROCESSING_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "neural/checkpoint.h"
#include "neural/error.h"
#include "neural/types.h"

typedef enum {
    NEURAL_NORMALIZATION_NONE,
    NEURAL_NORMALIZATION_STANDARDIZE,
    NEURAL_NORMALIZATION_MINMAX
} NeuralNormalization;

typedef enum {
    NEURAL_MISSING_REJECT,
    NEURAL_MISSING_MEAN
} NeuralMissingPolicy;

typedef struct {
    size_t input_count;
    NeuralNormalization normalization;
    NeuralMissingPolicy missing_policy;
    neural_real *offsets;
    neural_real *scales;
    neural_real *imputations;
    char source_digest[NEURAL_SHA256_TEXT_CAPACITY];
    char schema_digest[NEURAL_SHA256_TEXT_CAPACITY];
    uint64_t split_seed;
    neural_real validation_ratio;
    neural_real test_ratio;
    int stratified;
} NeuralPreprocessing;

int neural_preprocessing_load(const char *path,
                              NeuralPreprocessing *preprocessing,
                              NeuralError *error);
int neural_preprocessing_save_atomic(const char *path,
                                     const NeuralPreprocessing *preprocessing,
                                     NeuralError *error);
int neural_preprocessing_write(FILE *stream,
                               const NeuralPreprocessing *preprocessing,
                               NeuralError *error);
int neural_preprocessing_validate(const NeuralPreprocessing *preprocessing,
                                  NeuralError *error);
int neural_preprocessing_copy(const NeuralPreprocessing *source,
                              NeuralPreprocessing *destination,
                              NeuralError *error);
int neural_preprocessing_apply(const NeuralPreprocessing *preprocessing,
                               neural_real *inputs,
                               size_t sample_count,
                               NeuralError *error);
int neural_preprocessing_digest(
    const NeuralPreprocessing *preprocessing,
    char output[NEURAL_SHA256_TEXT_CAPACITY],
    NeuralError *error);
void neural_preprocessing_free(NeuralPreprocessing *preprocessing);

const char *neural_normalization_name(NeuralNormalization normalization);
int neural_normalization_from_name(const char *name,
                                   NeuralNormalization *normalization);
const char *neural_missing_policy_name(NeuralMissingPolicy policy);
int neural_missing_policy_from_name(const char *name,
                                    NeuralMissingPolicy *policy);

#endif
