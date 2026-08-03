#include "neural/random.h"

#include <stddef.h>

void neural_random_init(NeuralRandom *random, uint64_t seed)
{
    if (random != NULL) {
        random->state = seed;
    }
}

uint64_t neural_random_next_uint64(NeuralRandom *random)
{
    uint64_t value;

    if (random == NULL) {
        return UINT64_C(0);
    }
    random->state += UINT64_C(0x9e3779b97f4a7c15);
    value = random->state;
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

neural_real neural_random_next_unit(NeuralRandom *random)
{
    const uint64_t mantissa = neural_random_next_uint64(random) >> 11U;
    return (neural_real)mantissa * 0x1.0p-53;
}
