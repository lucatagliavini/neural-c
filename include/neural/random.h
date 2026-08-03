#ifndef NEURAL_RANDOM_H
#define NEURAL_RANDOM_H

#include <stdint.h>

#include "neural/types.h"

typedef struct {
    uint64_t state;
} NeuralRandom;

void neural_random_init(NeuralRandom *random, uint64_t seed);
uint64_t neural_random_next_uint64(NeuralRandom *random);
neural_real neural_random_next_unit(NeuralRandom *random);

#endif
